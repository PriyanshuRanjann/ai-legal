import requests
import time
from helper_function.qdrant_client import (
    initialize_qdrant_client,
    index_chunks_to_qdrant,
)
from ollama import Client
from qdrant_client import QdrantClient
from config import OLLAMA_URL, OLLAMA_MODEL
from sentence_transformers import SentenceTransformer


def search_qdrant(input_path, query, client, collection_name, top_k=5, skip_indexing=False, prefer_early_pages=True):
    """Search Qdrant for top_k relevant chunks using a query vector.
    
    Args:
        input_path: Path to input PDF
        query: User query text
        client: Qdrant client
        collection_name: Name of Qdrant collection
        top_k: Number of top results to return
        skip_indexing: If True, skip re-indexing (faster for repeated queries)
        prefer_early_pages: If True, boost results from early pages (0-5)
    """
    try:
        # Only index if collection doesn't exist or skip_indexing is False
        if not skip_indexing:
            print("[DEBUG] Indexing chunks to Qdrant...")
            index_chunks_to_qdrant(input_path, collection_name)
        else:
            print("[DEBUG] Skipping re-indexing, using existing collection")
        
        # Encode query to vector using SAME model as used for chunks
        print("[DEBUG] Encoding query to vector embedding...")
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        query_vector = model.encode(query).tolist()
        
        # Query Qdrant vector database for similar embeddings
        # Retrieve more results so we can filter by page
        fetch_k = top_k * 3 if prefer_early_pages else top_k
        print(f"[DEBUG] Querying Qdrant for {fetch_k} results to filter...")
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,  # This searches the stored embeddings
            limit=fetch_k,
            with_payload=True,
            with_vectors=False,
        )
        
        result_points = results.points if hasattr(results, 'points') else results
        
        # Filter and prioritize by page number if requested
        if prefer_early_pages:
            print("[DEBUG] Prioritizing early pages (0-5)...")
            early_page_results = []
            late_page_results = []
            
            for point in result_points:
                page = point.payload.get('page_number', 999)
                if page <= 5:
                    early_page_results.append(point)
                else:
                    late_page_results.append(point)
            
            # Combine: early pages first, then late pages
            result_points = (early_page_results + late_page_results)[:top_k]
            print(f"[DEBUG] Selected {len(early_page_results)} from early pages, {len([p for p in result_points if p.payload.get('page_number', 999) > 5])} from later pages")
        else:
            result_points = result_points[:top_k]
        
        print(f"[DEBUG] Qdrant returned {len(result_points)} results from stored embeddings")
        if result_points:
            print(f"[DEBUG] Result details:")
            for i, point in enumerate(result_points):
                score = point.score if hasattr(point, 'score') else 'N/A'
                page = point.payload.get('page_number', '?')
                text_preview = point.payload.get('text', '')[:60]
                print(f"  Result {i}: page={page}, score={score}, text={text_preview}...")
        return result_points
    except Exception as e:
        print(f"[ERROR] Failed to search Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return []


def build_context(results, max_tokens=600):
    """Concatenate retrieved chunk texts for context with token limit."""
    try:
        texts = []
        token_count = 0
        max_tokens_per_chunk = 300  # Increased from 100
        
        print(f"[DEBUG] build_context called with {len(results)} results")
        
        for i, point in enumerate(results):
            try:
                # Handle both direct point objects and wrapped results
                payload = point.payload if hasattr(point, 'payload') else point
                
                if not payload:
                    print(f"[DEBUG] Result {i} has no payload")
                    continue
                
                # Extract text from payload dictionary
                text = payload.get("text", "") if isinstance(payload, dict) else getattr(payload, "text", "")
                
                if not text:
                    print(f"[DEBUG] Result {i} has no text in payload")
                    continue
                
                print(f"[DEBUG] Result {i}: text length = {len(text)}")
                
                # Truncate to max per chunk (character limit)
                truncated_text = text[:max_tokens_per_chunk * 4]
                estimated_tokens = len(truncated_text) // 4
                
                if token_count + estimated_tokens > max_tokens:
                    print(f"[DEBUG] Reached max tokens at result {i}")
                    break
                
                texts.append(truncated_text)
                token_count += estimated_tokens
                print(f"[DEBUG] Added result {i}, total tokens now: {token_count}")
            except Exception as e:
                print(f"[DEBUG] Error processing result {i}: {e}")
                continue
        
        context = "\n\n".join(texts)
        print(f"[DEBUG] Built context with {len(texts)} chunks, total tokens: {token_count}")
        return context
    except Exception as e:
        print(f"[ERROR] Failed to build context from results: {e}")
        return ""


def ollama_rag(
    input_path,
    query,
    collection_name,
    top_k=2,  # Reduced from 3
    qdrant_host="localhost",
    qdrant_port=6333,
    ollama_host="http://localhost:11434",
    skip_indexing=False,
):
    """
    RAG pipeline: search Qdrant for context using query_vector, send context+query to Ollama, return answer.
    
    Args:
        skip_indexing: If True, skip re-indexing (faster for repeated queries)
    """
    try:
        print(f"\n[INFO] Starting RAG pipeline for query: '{query}'")
        print(f"[INFO] Collection: {collection_name}, top_k: {top_k}")
        
        client = initialize_qdrant_client(host=qdrant_host, port=qdrant_port)
        results = search_qdrant(input_path, query, client, collection_name, top_k, skip_indexing=skip_indexing, prefer_early_pages=True)
        context = build_context(results, max_tokens=600)  # Reduced from 800
        
        if not context or len(context.strip()) == 0:
            print("[WARN] No context found. Query may not have matching documents.")
            return "No relevant documents found for your query."
        
        # Very minimal prompt to reduce token processing
        prompt = f"""Answer using ONLY this context. Be brief.

CONTEXT: {context}

QUERY: {query}

ANSWER:"""
        
        print(f"\n[DEBUG] Context length: {len(context)} chars, Query: {query}")
        print(f"[DEBUG] Prompt length: {len(prompt)} chars")
        print(f"[DEBUG] Sending request to Ollama at {ollama_host} with model {OLLAMA_MODEL}")
        
        ollama_client = Client(host=ollama_host)
        
        # Retry logic with exponential backoff
        max_retries = 2
        for attempt in range(max_retries):
            try:
                print(f"[DEBUG] Ollama call attempt {attempt + 1}/{max_retries}")
                response = ollama_client.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    stream=False,
                    options={
                        "num_predict": 200,  # Limit output tokens
                        "num_ctx": 1024,  # Limit context window
                    }
                )
                
                answer = response.get("response", "")
                if not answer:
                    print("[WARN] Ollama returned empty response")
                    return "Empty response from model."
                print("[INFO] Successfully received response from Ollama")
                return answer
            except Exception as ollama_error:
                print(f"[ERROR] Ollama call attempt {attempt + 1} failed: {ollama_error}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                    print(f"[DEBUG] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    # Final fallback
                    print("[DEBUG] All retries failed, returning fallback response")
                    return f"Summary from documents:\n\n{context[:300]}..."
            
    except Exception as e:
        print(f"[ERROR] RAG pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return "Error occurred during RAG processing."


if __name__ == "__main__":
    # Example usage
    input_path = r"input/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf"
    query = input("Enter your legal query: ")
    collection_name = "legal_example1"
    answer = ollama_rag(input_path, query, collection_name)
    print("Answer from Ollama RAG:")
    print(answer)
