"""Main FastAPI application for legal document RAG system."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging

from api.endpoints import router
from helper_function.schemas import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Legal Document RAG API",
    description="Retrieval-Augmented Generation API for legal documents with Qdrant vector database",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=str(exc),
            status_code=500,
            timestamp=datetime.now().isoformat(),
            details={"path": str(request.url)}
        ).dict()
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Legal Document RAG",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Legal Document RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/api/docs",
            "ingest": "POST /api/ingest",
            "query": "POST /api/query",
            "list_documents": "GET /api/documents",
            "document_stats": "GET /api/documents/{doc_id}/stats"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
