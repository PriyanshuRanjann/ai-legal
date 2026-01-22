"""API endpoints for the legal document RAG system."""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from qdrant_client import QdrantClient

from helper_function.schemas import (
    QueryRequest, QueryResponse, PassageResult, IngestResponse, 
    DocumentListResponse, DocumentInfo,
    DocumentStats
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
        
        # Create unique document ID from filename (without timestamp for data processing)
        doc_name = Path(file.filename).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use filename-based ID for data processing (matches parsed PDF filenames)
        base_document_id = doc_name.lower().replace(" ", "-")
        # Use timestamped ID for registry (for unique tracking)
        registry_document_id = f"{doc_name}_{timestamp}"
        
        print(f"[INFO] Starting ingestion for document: {registry_document_id}")
        print(f"[INFO] Base document ID: {base_document_id}")
        
        # Save uploaded file
        input_dir = Path("input")
        input_dir.mkdir(exist_ok=True)
        file_path = input_dir / file.filename
        
        print(f"[INFO] Saving uploaded file...")
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        print(f"[INFO] Saved uploaded file to {file_path}")
        
        # Process and index the document
        print(f"[INFO] Processing document: parsing, chunking, and indexing...")
        collection_name = f"collection_{registry_document_id}"
        
        # Direct pipeline: parse → chunk → embed → index
        print(f"[INFO] Parsing PDF directly...")
        from helper_function.pdf_parser import parse_pdf, save_pdf_pages_to_json
        from helper_function.chunker import process_parsed_pdfs_with_context_chunking
        from helper_function.embeddings import process_chunks_for_embedding
        from helper_function.qdrant_client import initialize_qdrant_client, create_qdrant_collection, format_points, upsert_points
        
        # 1. Parse PDF pages
        pages = parse_pdf(str(file_path), document_id=base_document_id)
        print(f"[INFO] Parsed {len(pages)} pages from PDF")
        
        # 2. Save parsed pages to JSON
        parsed_dir = "output/parsed_pdf"
        os.makedirs(parsed_dir, exist_ok=True)
        for page in pages:
            output_path = os.path.join(parsed_dir, f"{page.document_id}_page_{page.page_number}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(page.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved {len(pages)} parsed pages to {parsed_dir}")
        
        # 3. Chunk the parsed pages
        print(f"[INFO] Chunking pages...")
        process_parsed_pdfs_with_context_chunking(str(file_path), doc_id_filter=base_document_id)
        
        # 4. Get chunks for this document
        chunks = load_json_chunks(str(file_path), document_id=base_document_id)
        
        if not chunks:
            raise HTTPException(status_code=500, detail="Failed to extract chunks from PDF")
        
        print(f"[INFO] Generated {len(chunks)} chunks for embedding")
        
        # 5. Generate embeddings and index to Qdrant
        print(f"[INFO] Generating embeddings and indexing to Qdrant...")
        from helper_function.embeddings import process_chunks_for_embedding
        records = process_chunks_for_embedding(str(file_path), document_id=base_document_id)
        
        if not records:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings for chunks")
        
        print(f"[INFO] Generated {len(records)} embedding records")
        
        # 6. Index to Qdrant
        print(f"[INFO] Indexing to Qdrant collection: {collection_name}")
        from helper_function.qdrant_client import initialize_qdrant_client, create_qdrant_collection, format_points, upsert_points
        client = initialize_qdrant_client(host=QDRANT_HOST, port=QDRANT_PORT)
        if not client:
            raise HTTPException(status_code=500, detail="Failed to connect to Qdrant")
        
        create_qdrant_collection(client, collection_name)
        points = format_points(records)
        if not points:
            raise HTTPException(status_code=500, detail="Failed to format embedding points")
        
        upsert_points(client, collection_name, points)
        print(f"[INFO] Successfully indexed {len(points)} points to Qdrant")
        
        total_pages = len(set(chunk.get("page_number", 0) for chunk in chunks))
        total_chunks = len(chunks)
        
        # Store metadata using registry ID
        DOCUMENT_REGISTRY[registry_document_id] = {
            "file_name": file.filename,
            "collection_name": collection_name,
            "total_pages": total_pages,
            "total_chunks": total_chunks,
            "source_path": str(file_path),
            "ingestion_timestamp": datetime.now().isoformat(),
            "chunks": chunks,
            "base_document_id": base_document_id
        }
        
        print(f"[SUCCESS] Ingested {registry_document_id}: {total_pages} pages, {total_chunks} chunks")
        
        return IngestResponse(
            document_id=registry_document_id,
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
        import traceback
        traceback.print_exc()
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
        
        print(f"[INFO] Processing query: {request.query}")
        
        # Determine which collection to search
        if request.document_id:
            if request.document_id not in DOCUMENT_REGISTRY:
                raise HTTPException(status_code=404, detail=f"Document {request.document_id} not found")
            collection_name = DOCUMENT_REGISTRY[request.document_id]["collection_name"]
            input_path = DOCUMENT_REGISTRY[request.document_id]["source_path"]
            print(f"[INFO] Searching document {request.document_id}")
        else:
            # Search across all documents (use first available)
            if not DOCUMENT_REGISTRY:
                raise HTTPException(status_code=400, detail="No documents ingested yet")
            first_doc_id = list(DOCUMENT_REGISTRY.keys())[0]
            first_doc = DOCUMENT_REGISTRY[first_doc_id]
            collection_name = first_doc["collection_name"]
            input_path = first_doc["source_path"]
            print(f"[INFO] Searching all documents (using collection {collection_name})")
        
        # Execute RAG query
        print(f"[INFO] Running RAG query with top_k={request.top_k}...")
        answer = ollama_rag(
            input_path=input_path,
            query=request.query,
            collection_name=collection_name,
            top_k=request.top_k,
            skip_indexing=True  # Already indexed
        )
        
        if not answer:
            print("[WARN] RAG query returned empty answer")
            answer = "No relevant information found in documents."
        
        # Get the full results with metadata from Qdrant
        print(f"[INFO] Retrieving detailed passage results...")
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
        
        confidence_score = sum(p.score for p in passages) / len(passages) if passages else 0.0
        
        print(f"[SUCCESS] Query complete: {len(passages)} passages found, confidence={confidence_score:.2f}")
        
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
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    GET /documents
    List all ingested documents.
    
    Return: { "documents": [...], "total_documents": 5 }
    """
    try:
        print(f"[INFO] Listing {len(DOCUMENT_REGISTRY)} documents")
        documents = []
        
        for doc_id, doc_info in DOCUMENT_REGISTRY.items():
            try:
                doc = DocumentInfo(
                    document_id=doc_id,
                    file_name=doc_info["file_name"],
                    total_pages=doc_info["total_pages"],
                    total_chunks=doc_info["total_chunks"],
                    ingestion_date=doc_info["ingestion_timestamp"],
                    source_path=doc_info["source_path"]
                )
                documents.append(doc)
            except Exception as doc_error:
                print(f"[WARN] Error formatting document {doc_id}: {doc_error}")
                continue
        
        print(f"[SUCCESS] Retrieved {len(documents)} documents")
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
        
        print(f"[INFO] Retrieving stats for document {doc_id}")
        
        doc_info = DOCUMENT_REGISTRY[doc_id]
        chunks = doc_info.get("chunks", [])
        
        if not chunks:
            print(f"[WARN] Document {doc_id} has no chunks stored in registry")
            chunks = []
        
        # Calculate statistics
        chunks_by_page: Dict[int, int] = {}
        total_chars = 0
        
        for chunk in chunks:
            page = chunk.get("page_number", 0)
            chunks_by_page[page] = chunks_by_page.get(page, 0) + 1
            total_chars += chunk.get("char_count", 0)
        
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        print(f"[SUCCESS] Retrieved stats: {len(chunks)} chunks, avg size {avg_chunk_size:.1f} chars")
        
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.endpoints:router", host="127.0.0.1", port=8000)