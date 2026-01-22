"""API endpoints for the legal document RAG system."""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException
from qdrant_client import QdrantClient

from helper_function.schemas import (
    QueryRequest,
    QueryResponse,
    PassageResult,
    IngestResponse,
    DocumentListResponse,
    DocumentInfo,
    DocumentStats,
    DocumentIngestionError,
    DocumentQueryError,
)
from helper_function.llm import ollama_rag, search_qdrant
from helper_function.pdf_parser import parse_pdf
from helper_function.chunker import process_parsed_pdfs_with_context_chunking
from helper_function.embeddings import load_json_chunks, process_chunks_for_embedding
from helper_function.qdrant_client import (
    initialize_qdrant_client,
    create_qdrant_collection,
    format_points,
    upsert_points,
)
from config import QDRANT_HOST, QDRANT_PORT, OUTPUT_DIR, INPUT_DIR

router = APIRouter(prefix="/api", tags=["legal-rag"])

# In-memory store for document metadata (in production, use a database)
DOCUMENT_REGISTRY: Dict[str, Dict] = {}


def _validate_pdf_file(filename: str) -> None:
    """
    Validate that the uploaded file is a PDF.

    Args:
        filename: Name of the uploaded file

    Raises:
        DocumentIngestionError: If file is not a PDF
    """
    if not filename or not filename.lower().endswith(".pdf"):
        raise DocumentIngestionError("Only PDF files are supported")


def _generate_document_ids(filename: str) -> Tuple[str, str]:
    """
    Generate document IDs for processing and registry.

    Args:
        filename: Original filename from upload

    Returns:
        Tuple of (base_document_id, registry_document_id)
    """
    doc_name = Path(filename).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_document_id = doc_name.lower().replace(" ", "-")
    registry_document_id = f"{doc_name}_{timestamp}"

    return base_document_id, registry_document_id


def _save_uploaded_file(file_path: Path, content: bytes) -> None:
    """
    Save uploaded file to input directory.

    Args:
        file_path: Path where file should be saved
        content: File content bytes

    Raises:
        DocumentIngestionError: If file save fails
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(content)
        print(f"[INFO] Saved file to {file_path}")
    except IOError as e:
        raise DocumentIngestionError(f"Failed to save uploaded file: {str(e)}")


def _save_parsed_pages_to_json(pages: List, parsed_dir: Path) -> None:
    """
    Save parsed PDF pages to JSON files.

    Args:
        pages: List of parsed page objects
        parsed_dir: Directory to save JSON files

    Raises:
        DocumentIngestionError: If save operation fails
    """
    try:
        parsed_dir.mkdir(parents=True, exist_ok=True)
        for page in pages:
            output_path = (
                parsed_dir / f"{page.document_id}_page_{page.page_number}.json"
            )
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(page.model_dump(), f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved {len(pages)} parsed pages to {parsed_dir}")
    except (IOError, AttributeError) as e:
        raise DocumentIngestionError(f"Failed to save parsed pages: {str(e)}")


def _process_document_pipeline(
    file_path: Path,
    base_document_id: str,
    registry_document_id: str,
    qdrant_client: QdrantClient,
) -> Tuple[List[Dict], int, int]:
    """
    Execute the complete document processing pipeline.

    Args:
        file_path: Path to PDF file
        base_document_id: Base ID for data processing
        registry_document_id: ID for registry tracking
        qdrant_client: Initialized Qdrant client

    Returns:
        Tuple of (chunks, total_pages, total_chunks)

    Raises:
        DocumentIngestionError: If any processing step fails
    """
    try:
        print(f"[INFO] Starting document pipeline...")
        collection_name = f"collection_{registry_document_id}"

        # 1. Parse PDF pages
        print(f"[INFO] Parsing PDF...")
        pages = parse_pdf(str(file_path), document_id=base_document_id)
        if not pages:
            raise DocumentIngestionError("No pages extracted from PDF")
        print(f"[INFO] Parsed {len(pages)} pages")

        # 2. Save parsed pages
        parsed_dir = OUTPUT_DIR / "parsed_pdf"
        _save_parsed_pages_to_json(pages, parsed_dir)

        # 3. Chunk the parsed pages
        print(f"[INFO] Chunking pages...")
        process_parsed_pdfs_with_context_chunking(
            str(file_path), doc_id_filter=base_document_id
        )

        # 4. Load chunks
        print(f"[INFO] Loading chunks...")
        chunks = load_json_chunks(str(file_path), document_id=base_document_id)
        if not chunks:
            raise DocumentIngestionError("Failed to extract chunks from PDF")
        print(f"[INFO] Generated {len(chunks)} chunks")

        # 5. Generate embeddings
        print(f"[INFO] Generating embeddings...")
        records = process_chunks_for_embedding(
            str(file_path), document_id=base_document_id
        )
        if not records:
            raise DocumentIngestionError("Failed to generate embeddings")
        print(f"[INFO] Generated {len(records)} embedding records")

        # 6. Index to Qdrant
        print(f"[INFO] Indexing to Qdrant...")
        create_qdrant_collection(qdrant_client, collection_name)
        points = format_points(records)
        if not points:
            raise DocumentIngestionError("Failed to format embedding points")

        upsert_points(qdrant_client, collection_name, points)
        print(f"[INFO] Indexed {len(points)} points to Qdrant")

        total_pages = len(set(chunk.get("page_number", 0) for chunk in chunks))

        return chunks, total_pages, len(chunks)

    except DocumentIngestionError as e:
        raise e
    except Exception as e:
        raise DocumentIngestionError(f"Pipeline execution failed: {str(e)}")


def _initialize_qdrant() -> QdrantClient:
    """
    Initialize Qdrant client.

    Returns:
        Initialized QdrantClient

    Raises:
        DocumentIngestionError: If connection fails
    """
    try:
        client = initialize_qdrant_client(host=QDRANT_HOST, port=QDRANT_PORT)
        if not client:
            raise DocumentIngestionError("Failed to connect to Qdrant")
        return client
    except Exception as e:
        raise DocumentIngestionError(f"Qdrant initialization failed: {str(e)}")


def _register_document(
    registry_document_id: str,
    file_name: str,
    collection_name: str,
    total_pages: int,
    total_chunks: int,
    file_path: Path,
    chunks: List[Dict],
    base_document_id: str,
) -> None:
    """
    Register ingested document in the registry.

    Args:
        registry_document_id: Unique registry ID
        file_name: Original filename
        collection_name: Qdrant collection name
        total_pages: Number of pages
        total_chunks: Number of chunks
        file_path: Path to saved PDF
        chunks: List of chunk objects
        base_document_id: Base processing ID
    """
    DOCUMENT_REGISTRY[registry_document_id] = {
        "file_name": file_name,
        "collection_name": collection_name,
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "source_path": str(file_path),
        "ingestion_timestamp": datetime.now().isoformat(),
        "chunks": chunks,
        "base_document_id": base_document_id,
    }


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    POST /ingest
    Upload and process a PDF document.

    Args:
        file: PDF file to ingest

    Returns:
        IngestResponse with document metadata

    Raises:
        HTTPException: If validation or processing fails
    """
    try:
        # Validate file
        _validate_pdf_file(file.filename)

        # Generate IDs
        base_document_id, registry_document_id = _generate_document_ids(file.filename)
        print(f"[INFO] Ingesting: {registry_document_id}")

        # Save file
        file_path = INPUT_DIR / file.filename
        content = await file.read()
        _save_uploaded_file(file_path, content)

        # Initialize Qdrant
        qdrant_client = _initialize_qdrant()

        # Process document
        collection_name = f"collection_{registry_document_id}"
        chunks, total_pages, total_chunks = _process_document_pipeline(
            file_path, base_document_id, registry_document_id, qdrant_client
        )

        # Register document
        _register_document(
            registry_document_id,
            file.filename,
            collection_name,
            total_pages,
            total_chunks,
            file_path,
            chunks,
            base_document_id,
        )

        print(
            f"[SUCCESS] Ingested {registry_document_id}: {total_pages} pages, {total_chunks} chunks"
        )

        return IngestResponse(
            document_id=registry_document_id,
            file_name=file.filename,
            total_pages=total_pages,
            total_chunks=total_chunks,
            ingestion_timestamp=datetime.now().isoformat(),
            status="success",
        )

    except DocumentIngestionError as e:
        print(f"[ERROR] Ingestion validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] Ingestion failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Document ingestion failed")


def _get_collection_and_path(document_id: Optional[str]) -> Tuple[str, str]:
    """
    Determine which collection to search.

    Args:
        document_id: Optional specific document ID

    Returns:
        Tuple of (collection_name, input_path)

    Raises:
        DocumentQueryError: If document not found or no documents available
    """
    if document_id:
        if document_id not in DOCUMENT_REGISTRY:
            raise DocumentQueryError(f"Document {document_id} not found")
        doc_info = DOCUMENT_REGISTRY[document_id]
    else:
        if not DOCUMENT_REGISTRY:
            raise DocumentQueryError("No documents ingested yet")
        first_doc_id = list(DOCUMENT_REGISTRY.keys())[0]
        doc_info = DOCUMENT_REGISTRY[first_doc_id]

    return doc_info["collection_name"], doc_info["source_path"]


def _build_passage_results(
    results: List, filters: Optional[Dict] = None
) -> Tuple[List[PassageResult], set]:
    """
    Build passage results from Qdrant search results.

    Args:
        results: Search results from Qdrant
        filters: Optional filter parameters

    Returns:
        Tuple of (passages list, pages set)
    """
    passages = []
    pages_set = set()

    for point in results:
        payload = point.payload if hasattr(point, "payload") else point
        page = payload.get("page_number", 0)

        # Apply page range filter if specified
        if filters and filters.get("page_range"):
            start, end = filters["page_range"]
            if not (start <= page <= end):
                continue

        pages_set.add(page)
        passage = PassageResult(
            text=payload.get("text", "")[:500],
            page=page,
            chunk_id=payload.get("chunk_id", "unknown"),
            score=point.score if hasattr(point, "score") else 0.5,
            document_id=payload.get("document_id", "unknown"),
        )
        passages.append(passage)

    return passages, pages_set


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    POST /query
    Query documents with optional filters.

    Args:
        request: QueryRequest with query and optional filters

    Returns:
        QueryResponse with answer and passages

    Raises:
        HTTPException: If query validation or execution fails
    """
    try:
        # Validate query
        if not request.query or len(request.query.strip()) == 0:
            raise DocumentQueryError("Query cannot be empty")

        print(f"[INFO] Processing query: {request.query}")

        # Get collection and path
        collection_name, input_path = _get_collection_and_path(request.document_id)

        # Execute RAG query
        print(f"[INFO] Running RAG query with top_k={request.top_k}...")
        answer = ollama_rag(
            input_path=input_path,
            query=request.query,
            collection_name=collection_name,
            top_k=request.top_k,
            skip_indexing=True,
        )

        if not answer:
            print("[WARN] RAG query returned empty answer")
            answer = "No relevant information found in documents."

        # Retrieve detailed results
        print(f"[INFO] Retrieving passage results...")
        qdrant_client = _initialize_qdrant()

        results = search_qdrant(
            input_path=input_path,
            query=request.query,
            client=qdrant_client,
            collection_name=collection_name,
            top_k=request.top_k,
            skip_indexing=True,
            prefer_early_pages=True,
        )

        # Build passages
        filters_dict = request.filters.dict() if request.filters else None
        passages, pages_set = _build_passage_results(results, filters_dict)

        confidence_score = (
            sum(p.score for p in passages) / len(passages) if passages else 0.0
        )

        print(
            f"[SUCCESS] Query complete: {len(passages)} passages, confidence={confidence_score:.2f}"
        )

        return QueryResponse(
            answer=answer,
            passages=passages,
            total_instances=len(passages),
            pages_referenced=sorted(list(pages_set)),
            confidence_score=confidence_score,
            filters_applied=filters_dict,
        )

    except DocumentQueryError as e:
        print(f"[ERROR] Query validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[ERROR] Query failed: {str(e)}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Query execution failed")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    GET /documents
    List all ingested documents.

    Returns:
        DocumentListResponse with document list

    Raises:
        HTTPException: If operation fails
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
                    source_path=doc_info["source_path"],
                )
                documents.append(doc)
            except (KeyError, ValueError) as doc_error:
                print(f"[WARN] Error formatting document {doc_id}: {str(doc_error)}")
                continue

        print(f"[SUCCESS] Retrieved {len(documents)} documents")
        return DocumentListResponse(documents=documents, total_documents=len(documents))

    except Exception as e:
        print(f"[ERROR] Failed to list documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document list")


def _calculate_document_stats(chunks: List[Dict]) -> Tuple[Dict[int, int], float]:
    """
    Calculate statistics for document chunks.

    Args:
        chunks: List of chunk dictionaries

    Returns:
        Tuple of (chunks_per_page dict, average_chunk_size)
    """
    chunks_by_page: Dict[int, int] = {}
    total_chars = 0

    for chunk in chunks:
        page = chunk.get("page_number", 0)
        chunks_by_page[page] = chunks_by_page.get(page, 0) + 1
        total_chars += chunk.get("char_count", 0)

    avg_chunk_size = total_chars / len(chunks) if chunks else 0
    return chunks_by_page, avg_chunk_size


@router.get("/documents/{doc_id}/stats", response_model=DocumentStats)
async def get_document_stats(doc_id: str):
    """
    GET /documents/{doc_id}/stats
    Return document statistics.

    Args:
        doc_id: Document identifier

    Returns:
        DocumentStats with document statistics

    Raises:
        HTTPException: If document not found or operation fails
    """
    try:
        if doc_id not in DOCUMENT_REGISTRY:
            raise HTTPException(status_code=404, detail="Document not found")

        print(f"[INFO] Retrieving stats for document {doc_id}")

        doc_info = DOCUMENT_REGISTRY[doc_id]
        chunks = doc_info.get("chunks", [])

        if not chunks:
            print(f"[WARN] Document {doc_id} has no chunks")
            chunks = []

        chunks_by_page, avg_chunk_size = _calculate_document_stats(chunks)

        print(
            f"[SUCCESS] Retrieved stats: {len(chunks)} chunks, avg size {avg_chunk_size:.1f} chars"
        )

        return DocumentStats(
            document_id=doc_id,
            file_name=doc_info["file_name"],
            total_pages=doc_info["total_pages"],
            total_chunks=doc_info["total_chunks"],
            average_chunk_size=avg_chunk_size,
            chunks_per_page=chunks_by_page,
            ingestion_timestamp=doc_info["ingestion_timestamp"],
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to get document stats: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Failed to retrieve document statistics"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.endpoints:router", host="127.0.0.1", port=8000)
