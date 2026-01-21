"""API endpoints for the legal document RAG system."""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from qdrant_client import QdrantClient

from helper_function.schemas import (
    QueryRequest, QueryResponse, PassageResult,
    IngestRequest, IngestResponse, 
    DocumentListResponse, DocumentInfo,
    DocumentStats, ErrorResponse, FilterParams
)
from helper_function.llm import ollama_rag
from helper_function.qdrant_client import initialize_qdrant_client, index_chunks_to_qdrant
from helper_function.embeddings import load_json_chunks
from config import QDRANT_HOST, QDRANT_PORT, OUTPUT_DIR

router = APIRouter(prefix="/api", tags=["legal-rag"])

# In-memory store for document metadata (in production, use a database)
DOCUMENT_REGISTRY: Dict[str, Dict] = {}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    POST /ingest
    Upload and process a PDF document.
    
    Returns: document_id, total_pages, total_chunks
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Create unique document ID from filename
        doc_name = Path(file.filename).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        document_id = f"{doc_name}_{timestamp}"
        
        # Save uploaded file
        input_dir = Path("input")
        input_dir.mkdir(exist_ok=True)
        file_path = input_dir / file.filename
        
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        print(f"[INFO] Saved uploaded file to {file_path}")
        
        # Process and index the document
        print(f"[INFO] Indexing document {document_id}...")
        collection_name = f"collection_{document_id}"
        index_chunks_to_qdrant(str(file_path), collection_name)
        
        # Get document statistics
        chunks = load_json_chunks(str(file_path))
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks extracted from PDF")
        
        total_pages = len(set(chunk.get("page_number", 0) for chunk in chunks))
        total_chunks = len(chunks)
        
        # Store metadata
        DOCUMENT_REGISTRY[document_id] = {
            "file_name": file.filename,
            "collection_name": collection_name,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "source_path": str(file_path),
            "ingestion_timestamp": datetime.now().isoformat(),
            "chunks": chunks
        }
        
        print(f"[SUCCESS] Ingested {document_id}: {total_pages} pages, {total_chunks} chunks")
        
        return IngestResponse(
            document_id=document_id,
            file_name=file.filename,
            total_pages=total_pages,
            total_chunks=total_chunks,
            ingestion_timestamp=datetime.now().isoformat(),
            status="success"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    POST /query
    Query documents with optional filters.
    
    Input: { "query": "string", "filters": { "page_range": [1, 10], ... } }
    Return: { "answer": "...", "passages": [...], "total_instances": 3, "pages_referenced": [...] }
    """
    try:
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Determine which collection to search
        if request.document_id:
            if request.document_id not in DOCUMENT_REGISTRY:
                raise HTTPException(status_code=404, detail=f"Document {request.document_id} not found")
            collection_name = DOCUMENT_REGISTRY[request.document_id]["collection_name"]
            input_path = DOCUMENT_REGISTRY[request.document_id]["source_path"]
        else:
            # Search across all documents (use first available)
            if not DOCUMENT_REGISTRY:
                raise HTTPException(status_code=400, detail="No documents ingested yet")
            first_doc = list(DOCUMENT_REGISTRY.values())[0]
            collection_name = first_doc["collection_name"]
            input_path = first_doc["source_path"]
        
        # Execute RAG query
        answer = ollama_rag(
            input_path=input_path,
            query=request.query,
            collection_name=collection_name,
            top_k=request.top_k,
            skip_indexing=True  # Already indexed
        )
        
        # Get the full results with metadata from Qdrant
        client = initialize_qdrant_client(host=QDRANT_HOST, port=QDRANT_PORT)
        from helper_function.llm import search_qdrant
        
        results = search_qdrant(
            input_path=input_path,
            query=request.query,
            client=client,
            collection_name=collection_name,
            top_k=request.top_k,
            skip_indexing=True,
            prefer_early_pages=True
        )
        
        # Build passage results with filtering
        passages = []
        pages_set = set()
        
        for point in results:
            payload = point.payload if hasattr(point, 'payload') else point
            page = payload.get('page_number', 0)
            
            # Apply page range filter if specified
            if request.filters and request.filters.page_range:
                start, end = request.filters.page_range
                if not (start <= page <= end):
                    continue
            
            pages_set.add(page)
            passage = PassageResult(
                text=payload.get('text', '')[:500],  # Limit to 500 chars
                page=page,
                chunk_id=payload.get('chunk_id', 'unknown'),
                score=point.score if hasattr(point, 'score') else 0.5,
                document_id=payload.get('document_id', 'unknown')
            )
            passages.append(passage)
        
        confidence_score = sum(p.score for p in passages) / len(passages) if passages else 0
        
        return QueryResponse(
            answer=answer,
            passages=passages,
            total_instances=len(passages),
            pages_referenced=sorted(list(pages_set)),
            confidence_score=confidence_score,
            filters_applied=request.filters.dict() if request.filters else None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    GET /documents
    List all ingested documents.
    
    Return: { "documents": [...], "total_documents": 5 }
    """
    try:
        documents = []
        for doc_id, doc_info in DOCUMENT_REGISTRY.items():
            doc = DocumentInfo(
                document_id=doc_id,
                file_name=doc_info["file_name"],
                total_pages=doc_info["total_pages"],
                total_chunks=doc_info["total_chunks"],
                ingestion_date=doc_info["ingestion_timestamp"],
                source_path=doc_info["source_path"]
            )
            documents.append(doc)
        
        return DocumentListResponse(
            documents=documents,
            total_documents=len(documents)
        )
    
    except Exception as e:
        print(f"[ERROR] Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/documents/{doc_id}/stats", response_model=DocumentStats)
async def get_document_stats(doc_id: str):
    """
    GET /documents/{doc_id}/stats
    Return document statistics (pages, chunks, sections).
    
    Return: { "pages": 25, "chunks": 150, "average_chunk_size": 450, ... }
    """
    try:
        if doc_id not in DOCUMENT_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        doc_info = DOCUMENT_REGISTRY[doc_id]
        chunks = doc_info["chunks"]
        
        # Calculate statistics
        chunks_by_page: Dict[int, int] = {}
        total_chars = 0
        
        for chunk in chunks:
            page = chunk.get("page_number", 0)
            chunks_by_page[page] = chunks_by_page.get(page, 0) + 1
            total_chars += chunk.get("char_count", 0)
        
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        return DocumentStats(
            document_id=doc_id,
            file_name=doc_info["file_name"],
            total_pages=doc_info["total_pages"],
            total_chunks=doc_info["total_chunks"],
            average_chunk_size=avg_chunk_size,
            chunks_per_page=chunks_by_page,
            ingestion_timestamp=doc_info["ingestion_timestamp"]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to get document stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
