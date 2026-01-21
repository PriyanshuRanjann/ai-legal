from pydantic import BaseModel, Field, validator, Field
from typing import List, Optional, Dict, Any


class PDFPageSchema(BaseModel):
    document_id: str = Field(..., description="Stable ID for the source document")
    page_number: int | float = Field(..., description="Page number within the document")
    text: str = Field("", description="Normalized page text")
    source_path: str
    extraction_method: str = Field("pymupdf", description="How the text was obtained")
    char_count: int | float = Field(0, ge=0)

    @validator("char_count", always=True)
    def compute_char_count(cls, value: int, values: dict) -> int:
        if value:
            return value
        return len(values.get("text", ""))


class ContextEnrichedChunk(BaseModel):
    document_id: str
    page_number: int
    source_path: str
    chunk_id: str
    text: str
    context_prev: str
    context_next: str
    char_count: int


class FilterParams(BaseModel):
    """Filter parameters for queries."""

    page_range: Optional[List[int]] = Field(
        None, description="[start_page, end_page] (0-indexed)"
    )
    date_before: Optional[str] = Field(None, description="ISO format date string")
    date_after: Optional[str] = Field(None, description="ISO format date string")


class QueryRequest(BaseModel):
    """Request schema for /query endpoint."""

    query: str = Field(..., description="User's search query")
    document_id: Optional[str] = Field(
        None, description="Specific document to search (optional)"
    )
    top_k: int = Field(
        default=3, ge=1, le=10, description="Number of top results to return"
    )
    filters: Optional[FilterParams] = Field(None, description="Optional filters")


class PassageResult(BaseModel):
    """Single passage retrieved from vector database."""

    text: str = Field(..., description="Exact passage text")
    page: int = Field(..., description="Page number (0-indexed)")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    document_id: str = Field(..., description="Source document ID")


class QueryResponse(BaseModel):
    """Response schema for /query endpoint."""

    answer: str = Field(..., description="Summarized response with citations")
    passages: List[PassageResult] = Field(..., description="Retrieved passages")
    total_instances: int = Field(..., description="Total matching instances found")
    pages_referenced: List[int] = Field(
        ..., description="Unique page numbers referenced"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0, description="Overall confidence"
    )
    filters_applied: Optional[Dict[str, Any]] = Field(
        None, description="Applied filters"
    )


class IngestRequest(BaseModel):
    """Request schema for document ingestion (multipart/form-data)."""

    file_name: str = Field(..., description="Name of the PDF file")


class IngestResponse(BaseModel):
    """Response schema for /ingest endpoint."""

    document_id: str = Field(..., description="Unique identifier for ingested document")
    file_name: str = Field(..., description="Original file name")
    total_pages: int = Field(..., description="Number of pages in PDF")
    total_chunks: int = Field(..., description="Total chunks created")
    ingestion_timestamp: str = Field(..., description="ISO format timestamp")
    status: str = Field(default="success", description="Ingestion status")


class DocumentInfo(BaseModel):
    """Information about a stored document."""

    document_id: str = Field(..., description="Document identifier")
    file_name: str = Field(..., description="Original file name")
    total_pages: int = Field(..., description="Number of pages")
    total_chunks: int = Field(..., description="Number of chunks")
    ingestion_date: str = Field(..., description="ISO format ingestion date")
    source_path: str = Field(..., description="Path to source PDF")


class DocumentListResponse(BaseModel):
    """Response schema for /documents endpoint."""

    documents: List[DocumentInfo] = Field(..., description="List of ingested documents")
    total_documents: int = Field(..., description="Total count")


class DocumentStats(BaseModel):
    """Document statistics."""

    document_id: str = Field(..., description="Document identifier")
    file_name: str = Field(..., description="File name")
    total_pages: int = Field(..., description="Total pages")
    total_chunks: int = Field(..., description="Total chunks")
    average_chunk_size: float = Field(..., description="Average characters per chunk")
    chunks_per_page: Dict[int, int] = Field(..., description="Chunks by page number")
    ingestion_timestamp: str = Field(..., description="When document was ingested")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: str = Field(..., description="ISO format timestamp")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
