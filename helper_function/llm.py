import requests
from helper_function.qdrant_client import initialize_qdrant_client, index_chunks_to_qdrant
from qdrant_client import QdrantClient
from config import OLLAMA_URL, OLLAMA_MODEL

def search_qdrant(client: QdrantClient, collection_name: str, query_vector: list, top_k: int = 5):
    """Search Qdrant for top_k relevant chunks using a query vector."""
    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k
        )
        return results
    except Exception as e:
        print(f"[ERROR] Failed to search Qdrant: {e}")
        return []