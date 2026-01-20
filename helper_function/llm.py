import requests
from helper_function.qdrant_client import initialize_qdrant_client, index_chunks_to_qdrant
from qdrant_client import QdrantClient
from config import OLLAMA_URL, OLLAMA_MODEL
from sentence_transformers import SentenceTransformer

def search_qdrant(input, client: QdrantClient, collection_name: str, top_k: int = 5):
    """Search Qdrant for top_k relevant chunks using a query vector."""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        query_vector = model.encode(query).tolist()
        index_chunks_to_qdrant(input,collection_name)
        results = client.query_points(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        return results
    except Exception as e:
        print(f"[ERROR] Failed to search Qdrant: {e}")
        return []
    
def build_context(results):
    """Concatenate retrieved chunk texts for context."""
    try:
        res = "\n\n".join([point.payload.get("text", "") for point in results])
        return res
    except Exception as e:
        print(f"[ERROR] Failed to build context from results: {e}")
        return ""
    
def ollama_rag(input, query, collection_name, top_k=5, qdrant_host="localhost", qdrant_port=6333):
    """
    RAG pipeline: search Qdrant for context using query_vector, send context+query to Ollama, return answer.
    query_vector: embedding of the user query (should be generated externally)
    """
    
    client = initialize_qdrant_client(host=qdrant_host, port=qdrant_port)
    results = search_qdrant(input, client, collection_name, top_k)
    context = build_context(results)
    prompt = f"""
        ROLE: You are a legal assistant AI specialized in retrieving and citing legal document passages.
        CONTEXT: {context}
        TASK: 
        - Answer the user's query using the provided context passages.
        - Return relevant passages with:
            - Exact text passage
            - Page number(s) where the passage appears
            - Number of instances found
            - Confidence score (higher means more relevant)
        - If filter queries are provided, only use passages that match the filters (e.g., page range, deadlines).

        RULES:
        - Always cite the page number(s) for each passage.
        - If no relevant passages are found, respond with "No relevant passages found."
        - Never fabricate passages or page numbers.
        - If multiple instances are found, summarize and count them.
        - If a deadline or date filter is provided, only return passages mentioning deadlines before the specified date.
        - If a page range is provided, only return passages from those pages.
        - Respond in a clear, structured format suitable for legal review.
        """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": True
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    answer = response.json().get("response", "")
    return answer

if __name__ == "__main__":
    # Example usage
    input = r"input/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf"
    query = input("Enter your legal query: ")
    # Example query vector (should be generated using the same embedding model as used for indexing)
    # query_vector = [0.01] * 768  # Placeholder vector; replace with actual embedding
    collection_name = "legal_example1"
    answer = ollama_rag(input, query, collection_name)
    print("Answer from Ollama RAG:")
    print(answer)