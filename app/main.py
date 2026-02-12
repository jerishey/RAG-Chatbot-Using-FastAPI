from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict
import os
import shutil
from pathlib import Path
import tempfile
import uvicorn
from contextlib import asynccontextmanager

# Import your RAG chatbot class
from app.rag_chatbot import RAGChatbot

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# ==================== CONFIGURATION ====================

# Chatbot configuration from environment
CHATBOT_CONFIG = {
    "llm_provider": os.getenv("LLM_PROVIDER", "groq"),
    "chunk_size": int(os.getenv("CHUNK_SIZE", 500)),
    "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 50)),
    "top_k": int(os.getenv("TOP_K", 10)),
    "top_n": int(os.getenv("TOP_N", 3)),
    "max_chunks": int(os.getenv("MAX_CHUNKS", 1000))
}

# File upload constraints
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB default
ALLOWED_EXTENSIONS = {'.pdf'}

# CORS origins (should be set in production)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ==================== GLOBAL STATE ====================

# Initialize chatbot instance (will be done on startup)
chatbot: Optional[RAGChatbot] = None

# Directory for uploaded PDFs
UPLOAD_DIR = Path("../uploads/pdf")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ==================== LIFESPAN MANAGEMENT ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot
    print("Initializing RAG Chatbot...")

    try:
        # Get API key based on provider
        provider = CHATBOT_CONFIG["llm_provider"]
        api_key = None
        
        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY")
        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
        elif provider == "cohere":
            api_key = os.getenv("COHERE_API_KEY")
        
        if not api_key:
            raise ValueError(f"API key not found for provider: {provider}")
        
        print("RAG Chatbot initialized successfully!")

        chatbot = RAGChatbot(
            llm_provider=provider,
            api_key=api_key,
            chunk_size=500,
            chunk_overlap=50,
            top_k=10,
            top_n=3,
            max_chunks=1000
        )

    except Exception as e:
        raise ValueError(f"Failed to initialize chatbot: {str(e)}")

    yield

    print("Shutting down RAG Chatbot API...")

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="RAG PDF Chatbot API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== PYDANTIC MODELS ====================

class QueryRequest(BaseModel):
    """Request model for querying the chatbot"""
    question: str = Field(..., min_length=1, max_length=1000, description="Question to ask")
    return_sources: bool = Field(default=True, description="Whether to return source chunks")

    @field_validator('question')
    @classmethod
    def sanitize_question(cls, v):
        """Sanitize question to prevent injection attacks"""
        # Remove potentially dangerous characters
        dangerous_character = ['<', '>', '{', '}', '\\x00']
        for char in dangerous_character:
            if char in v:
                raise ValueError(f"Invalid Character '{char}' in question")
        return v.strip()

class QueryResponse(BaseModel):
    """Response model for query results"""
    question: str
    answer: str
    num_sources: int
    sources: Optional[List[Dict]] = None

class ProcessPDFResponse(BaseModel):
    """Response model for PDF processing"""
    message: str
    total_chunks: int
    source: str
    collection_size: int

class StatsResponse(BaseModel):
    """Response model for database statistics"""
    total_chunks: int
    collection_name: str
    unique_documents: int
    document_sources: List[str]
    embedding_dimension: int
    retrieval_config: Dict

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    chatbot_initialized: bool
    database_accessible: bool

class ResetResponse(BaseModel):
    """Reset database response"""
    message: str
    total_chunks: int

class ChatbotConfig(BaseModel):
    """Configuration model for chatbot initialization"""
    embedding_model: str = "all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_provider: str = "anthropic"
    chunk_size: int = Field(default=500, ge=100, le=2000, description="Chunk size for text splitting")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="Overlap between chunks")
    max_chunks: int = Field(default=1000, ge=100, le=10000, description="Maximum chunks to store")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of chunks to retrieve")
    top_n: int = Field(default=3, ge=1, le=20, description="Number of chunks after reranking")

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "RAG PDF Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "POST /upload": "Upload and process a PDF file",
            "POST /query": "Ask a question about uploaded PDFs",
            "GET /stats": "Get database statistics",
            "GET /documents": "List all uploaded documents",
            "POST /reset": "Reset the database",
            "GET /health": "Health check",
            "POST /process-local-pdf": "Process a PDF from local filesystem"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    return {
        "status": "healthy",
        "chatbot_initialized": chatbot is not None
    }

@app.post("/upload", response_model=ProcessPDFResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file
    
    - **file**: PDF file to upload and process
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    # Validate file type
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    temp_path = None
    saved_path = None
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            # Copy uploaded file to temp file
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Process the PDF
        stats = chatbot.process_pdf(temp_path)
        
        # Optionally save to uploads directory
        saved_path = UPLOAD_DIR / file.filename
        shutil.copy(temp_path, saved_path)
        
        return ProcessPDFResponse(
            message=f"Successfully processed {file.filename}",
            total_chunks=stats["total_chunks"],
            source=stats["source"],
            collection_size=stats["collection_size"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    finally:
        # Clean up temp file
        if file.file:
            file.file.close()
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Failed to cleanup temp file: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """
    Ask a question about the uploaded PDFs
    
    - **question**: The question to ask
    - **return_sources**: Whether to return source chunks (default: true)
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    # Check if database has documents
    stats = chatbot.get_stats()
    if stats["total_chunks"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents in database. Please upload a PDF first."
        )
    
    try:
        result = chatbot.query(
            question=request.question,
            return_sources=request.return_sources
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get database statistics
    
    Returns information about the current state of the vector database
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        stats = chatbot.get_stats()
        return StatsResponse(**stats)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/reset")
async def reset_database():
    """
    Reset the database
    
    Clears all documents from the vector database
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        chatbot.reset_database()
        return {
            "message": "Database reset successfully",
            "total_chunks": 0
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resetting database: {str(e)}")

@app.get("/documents")
async def list_documents():
    """
    List all uploaded documents
    
    Returns a list of all PDFs currently in the database
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        stats = chatbot.get_stats()
        return {
            "total_documents": stats["unique_documents"],
            "documents": stats["document_sources"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

@app.post("/process-local-pdf")
async def process_local_pdf(pdf_path: str = Query(..., description="Path to local PDF file")):
    """
    Process a PDF file from local filesystem
    
    - **pdf_path**: Path to the PDF file on the server
    """
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_path}")
    
    try:
        stats = chatbot.process_pdf(pdf_path)
        
        return ProcessPDFResponse(
            message=f"Successfully processed {Path(pdf_path).name}",
            total_chunks=stats["total_chunks"],
            source=stats["source"],
            collection_size=stats["collection_size"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# ==================== RUN SERVER ====================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes (disable in production)
        log_level="info"
    )