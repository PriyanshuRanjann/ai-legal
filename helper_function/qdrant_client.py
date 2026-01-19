"""import qdrant_client module."""
from qdrant_client import QdrantClient
from qdrant_client.http import models
from helper_function.embeddings import build_embedding_records

def initialize_qdrant_client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    """Initializes and returns a Qdrant client."""
    try:
        client = QdrantClient(host=host, port=port)
        print(f"[INFO] Connected to Qdrant at {host}:{port}")
        return client
    except Exception as e:
        print(f"[ERROR] Failed to connect to Qdrant: {e}")
        return None