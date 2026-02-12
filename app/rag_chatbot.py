import os
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PDF Processing
from pypdf import PdfReader

# Embeddings & Vector DB
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Re-ranking
from sentence_transformers import CrossEncoder

class RAGChatbot:
    def __init__(
            self,
            # embedding_model: HuggingFace model for embeddings
            embedding_model: str = "all-MiniLM-L6-v2",
            # rerank_model: Cross-encoder model for re-ranking
            rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
            # llm_provider: Which LLM to use ("openai", "anthropic", "cohere")
            llm_provider: str = "anthropic", #option: "openai", "anthropic", "cohere", "groq",
            #  api_key: API key for the LLM provider
            api_key: Optional[str] = None,
            # chunk_size: Size of text chunks
            chunk_size: int = 500,
            # chunk_overlap: Overlap between chunks
            chunk_overlap: int = 50,
            # max_chunks: Default limit
            max_chunks: int = 1000,
            # top_k: Number of chunks to retrieve initially
            top_k: int = 10,
            # top_n: Number of chunks after re-ranking 
            top_n: int = 3,
            # persist_directory: Directory to store ChromaDB
            persist_directory: str = "./chroma_db"
            ): 
        print("Initializing RAG Chatbot...")

        # Configuration
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.top_n = top_n
        self.llm_provider = llm_provider
        self.max_chunks = max_chunks

        # Validate parameters
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")
    
        if top_n > top_k:
            raise ValueError(f"top_n ({top_n}) cannot be greater than top_k ({top_k})")
        
        # Always Positive
        if max_chunks <= 0: 
            raise ValueError(f"max_chunks ({max_chunks}) must be greater then 0")

        # Load embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Load re-ranking model
        print(f"Loading re-ranking model: {rerank_model}")
        self.reranking_model = CrossEncoder(rerank_model)

        # ChromaDB
        print(f"Initializing ChromaDB at {persist_directory}")
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry = False,
                allow_reset = True
            )
        )

        # Create or Get Collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space":"cosine"}
        )

        # Initialize LLM
        print(f"Initializing LLM Provider: {llm_provider}")
        self._init_llm(api_key) # API Key

        print("RAG Chatbot Initialized Successfully!\n")

    def _init_llm(self, api_key: str):
        """Initialize the LLM based on provider."""
        if self.llm_provider == "openai":
            import openai
            self.llm_client = openai.OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            self.llm_model = "gpt-4"

        elif self.llm_provider == "anthropic":
            from anthropic import Anthropic
            self.llm_client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
            self.llm_model = "claude-3-5-sonnet-20241022"

        elif self.llm_provider == "cohere":
            import cohere
            self.llm_client = cohere.Client(api_key=api_key or os.getenv("COHERE_API_KEY"))
            self.llm_model = "command-r-plus"

        elif self.llm_provider == "groq":
            from groq import Groq
            self.llm_client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
            self.llm_model = "llama-3.3-70b-versatile"

        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")
        

    """PDF PROCESSING"""

    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Complete pipeline: Extract → Chunk → Embed → Store

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Dictionary with processing statistics
        """
        print(f"\n Processing PDF: {pdf_path}")

        # Step 1: Extract text from PDF
        text = self._extract_text_from_pdf(pdf_path)
        print(f"Extracted {len(text)} characters")

        # Step 2: Chunk the text
        chunks = self._chunk_text(text)
        print(f"Created {len(chunks)} chunks")

        if not chunks:
            raise ValueError(f"No valid chunks created from PDF: {pdf_path}")
        
        # Aplly max_chunks limit
        if len(chunks) > self.max_chunks:
            print(f"Warning: Limiting chunks to {self.max_chunks} for performance (PDF has {len(chunks)} chunks).")
            chunks = chunks[:self.max_chunks]

        # Step 3: Generate Embedding
        embeddings = self._generate_embeddings(chunks)
        print(f"Generated {len(embeddings)} embeddings")

        # Step 4: Store in ChromaDB
        source = Path(pdf_path).name
        stats = self._store_in_vectordb(chunks, embeddings, source)
        print(f"Stored in vector database")

        return stats
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            reader = PdfReader(pdf_path)
            text = ""

            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text() or ""
                text += f"\n Page {page_num} \n{page_text}"

            if not text.strip():
                raise ValueError(f"No text extracted from PDF: {pdf_path}")

            return text
        except Exception as e:
            raise RuntimeError(f"Error processing PDF {pdf_path}: {str(e)}") 
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Full text to chunk
        
        Return:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < text_length:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)

                if break_point > self.chunk_size * 0.5: # At least 50% chunk size
                    chunk = chunk[:break_point+1]
                    end = start + break_point + 1

            if chunk.strip():
                chunks.append(chunk.strip())

            # Always advance or move by (chunk_size - overlap)    
            start = start + self.chunk_size - self.chunk_overlap

            # To prevent infinite loop
            if start <= 0:
                start = self.chunk_size - self.chunk_overlap
                if start <= 0:
                    break

        return chunks

    def _generate_embeddings(self, chunks: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embedding for text chunks."""
        all_embeddings = []
    
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            embeddings = self.embedding_model.encode(
                batch,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            all_embeddings.extend(embeddings.tolist())
    
        return all_embeddings
    
    def _store_in_vectordb(
            self,
            chunks: List[str],
            embeddings: List[List[float]],
            source: str
    ) -> Dict:
        """Store chunks and embeddings in ChromaDB"""

        # Prepare Data
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {
                "source": source,
                "chunk_id": i,
                "chunk_size": len(chunk)
            }
            for i, chunk in enumerate(chunks)
        ]

        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )

        return {
            "total_chunks": len(chunks),
            "source": source,
            "collection_size": self.collection.count()
        }
    # QUERY PIPELINE
    def query(self, question: str, return_sources: bool = True) -> Dict:
        """
        Query Pipeline: Embed -> Retrieve -> Re-rank -> Generate Answer

        Args:
            question: User question
            return_sources: Return source chunks

        Return:
            Dictionary with answer and metadata
        """
        print(f"\n Query: {question}")

        # Step 5: Generate query embedding
        query_embedding = self._generate_query_embedding(question)
        print(f"Generated query embedding")

        # Step 6: Retrieve top-k chunks
        retrieved_chunks = self._retrieve_topk(query_embedding)
        print(f"Retrieved {len(retrieved_chunks)} chunks")

        if not retrieved_chunks:
            return {
                "question": question,
                "answer": "I don't have any relevant information in my database to answer this question.",
                "num_sources": 0,
                "sources": [] if return_sources else None
            }        

        # Step 7-8: Re-rank and select top-n 
        reranked_chunks = self._rerank_chunks(question, retrieved_chunks) 
        print(f"Re-ranked {len(reranked_chunks)} chunks")

        # Step 9: Build prompt
        prompt = self._build_prompt(question, reranked_chunks)

        # Step 10: Generate answer with LLM
        answer = self._generate_answer(prompt)
        print(f"Generated Answer")

        # Step 11: Return response
        response = {
            "question": question,
            "answer": answer,
            "num_sources": len(reranked_chunks)
        }

        if return_sources:
            response["sources"] = [
                {
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": chunk.get("rerank_score", chunk.get("similarity", 0))
                }
                for chunk in reranked_chunks
            ]

        return response

    def _generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for user query"""
        try:
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding for query: {str(e)}")

    def _retrieve_topk(self, query_embedding: List[float]) -> List[Dict]:
        """Retrieve top-k similar chunks from ChromaDB"""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k
        )

        # Check if any results were returned
        if not results['ids'][0]:
            print("Warning: No relevant document found.")
            return []

        chunks = []
        for i in range(len(results['ids'][0])):
            chunks.append({
                "ids": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "similarity": 1 - results['distances'][0][i] # Convert distance to similarity
            })

        return chunks
    
    def _rerank_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Re-rank chunks using cross-encoder model"""

        # Prepare pairs for re-ranking
        pairs = [[query, chunk["text"]] for chunk in chunks]

        # Get re-ranking scores
        scores = self.reranking_model.predict(pairs)

        # Add scores to chunk
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        # Sort by re-ranking score and select top-N
        reranked = sorted(chunks, key=lambda X: X["rerank_score"], reverse=True)
        return reranked[:self.top_n]
    
    def _build_prompt(self, question: str, context_chunks: List[Dict]) -> str:
        """Build Prompt with context for LLM"""

        # Combine context
        context = "\n\n".join([
            f"[Source{i+1}]:\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])

        # Build prompt
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context from a PDF document.
        
Context from PDF:
{context}

Question: {question}

Instructions:
    1. Answer the question based ONLY on the information provided in the context above
    2. If the context doesn't contain the enough information to answer the question, say so
    3. Be concise and accurate
    4. Cite the source number [Source X] when referencing specific information

Answer:"""
        
        return prompt
    
    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using LLM"""

        try:
            if self.llm_provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model = self.llm_model,
                    messages=[
                        {"role":"system", "content": "You are a helpful AI assistant"},
                        {"role":"user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                return response.choices[0].message.content
        
            elif self.llm_provider == "anthropic":
                response = self.llm_client.messages.create(
                    model = self.llm_model,
                    max_tokens = 1000,
                    temperature = 0.3,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
        
            elif self.llm_provider == "cohere":
                response = self.llm_client.generate(
                    model = self.llm_model,
                    prompt = prompt,
                    temperature = 0.3,
                    max_tokens = 1000
                )
                return response.generations[0].text
            
            elif self.llm_provider == "groq":
                response = self.llm_client.chat.completions.create(
                    model= self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens = 1000
                )
                return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
            
    # Utility Methods
    def reset_database(self):
        """Clear all documents from the database"""
        self.chroma_client.delete_collection("pdf_documents")
        self.collection = self.chroma_client.get_or_create_collection(
            name="pdf_documents",
            metadata={"hnsw:space": "cosine"}
        )
        print("Database reset successfully.")

    def get_stats(self) -> Dict:
        """Get Database Statistics."""
        all_data = self.collection.get()
        sources = set()
        if all_data['metadatas']:
            sources = {meta['source'] for meta in all_data['metadatas']}
    
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection.name,
            "unique_documents": len(sources),
            "document_sources": sorted(list(sources)),
            "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
            "retrieval_config": {
                "top_k": self.top_k,
                "top_n": self.top_n,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "max_chunks": self.max_chunks
            }
        }
    
# Main Program

def main():
    """
    Main program demonstrating the RAG chatbot.
    """

    print("=" * 70)
    print("RAG-BASED PDF CHATBOT")
    print("=" * 70)

    # Initialize chatbot
    # NOTE: Set your API key here or as environment variable
    chatbot = RAGChatbot(
        llm_provider="groq",  # Change to "openai" or "cohere" as needed
        api_key=None,  # Will use environment variable
        chunk_size=500,
        chunk_overlap=50,
        top_k=10,
        top_n=3
    )

    # ==================== PHASE 1: PROCESS PDF ====================
    print("\n" + "=" * 70)
    print("PHASE 1: PDF PROCESSING")
    print("=" * 70)

    # Auto-detect PDFs in uploads/pdf folder
    uploads_dir = Path("../uploads/pdf")
    
    if uploads_dir.exists():
        pdf_files = list(uploads_dir.glob("*.pdf"))
        
        if pdf_files:
            print(f"\n📚 Found {len(pdf_files)} PDF(s) in uploads/pdf folder:")
            for i, pdf in enumerate(pdf_files, 1):
                print(f"   {i}. {pdf.name}")
            
            choice = input("\n📁 Enter number to select, or enter custom path: ").strip()
            
            # Check if user entered a number
            if choice.isdigit() and 1 <= int(choice) <= len(pdf_files):
                pdf_path = str(pdf_files[int(choice) - 1])
            else:
                pdf_path = choice
        else:
            pdf_path = input("\n📁 Enter the path to your PDF file: ").strip()
    else:
        pdf_path = input("\n📁 Enter the path to your PDF file: ").strip()

    if not os.path.exists(pdf_path):
        print(f"❌ Error: File '{pdf_path}' not found!")
        return

    # Process the PDF
    stats = chatbot.process_pdf(pdf_path)

    print(f"\n📊 Processing Statistics:")
    print(f"   - Total chunks created: {stats['total_chunks']}")
    print(f"   - Total documents in DB: {stats['collection_size']}")

    # ==================== PHASE 2: QUERY LOOP ====================
    print("\n" + "=" * 70)
    print("PHASE 2: QUESTION ANSWERING")
    print("=" * 70)
    print("\n💡 You can now ask questions about the PDF!")
    print("   Type 'quit' to exit, 'stats' for database info, 'reset' to clear DB\n")

    while True:
        question = input("🤔 Your question: ").strip()

        if not question:
            print("Please enter a question.")
            continue

        if len(question) > 1000:  # Prevent abuse
            print("Question too long. Please keep it under 1000 characters.")
            continue

        if question.lower() == 'quit':
            print("\n👋 Goodbye!")
            break

        if question.lower() == 'stats':
            stats = chatbot.get_stats()
            print(f"\n📊 Database Stats: {stats}\n")
            continue

        if question.lower() == 'reset':
            chatbot.reset_database()
            print("💾 Database has been reset. Please upload a PDF again.\n")
            continue

        try:
            # Get answer
            result = chatbot.query(question, return_sources=True)

            # Display answer
            print("\n" + "=" * 70)
            print("🤖 ANSWER:")
            print("=" * 70)
            print(result["answer"])

            # Display sources
            print("\n📚 SOURCES:")
            for i, source in enumerate(result["sources"], 1):
                print(f"\n[Source {i}] (Score: {source['score']:.4f})")
                print(f"   {source['text'][:200]}...")

            print("\n" + "=" * 70 + "\n")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")


if __name__ == "__main__":
    main()
