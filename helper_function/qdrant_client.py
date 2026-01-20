from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import PointStruct , VectorParams, Distance

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

def create_qdrant_collection(
    client: QdrantClient,
    collection_name = "AI_Legal_Collection",
    vector_size = 768,
    distance = Distance.COSINE
) -> None:
    """Creates a Qdrant collection with the specified parameters."""
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,   
                distance=distance
            )
        )
        print(f"[INFO] Created Qdrant collection: {collection_name}")
    except Exception as e:
        print(f"[ERROR] Failed to create collection {collection_name}: {e}")


def format_points(records: list) -> list:
    """Converts embedding records into Qdrant PointStruct format."""
    points = []
    try:
        for record in records:
            point = PointStruct(
                id=record["id"],
                vector=record["vector"],
                payload=record["payload"]
            )
            points.append(point)
        print(f"[INFO] Formatted {len(points)} points for Qdrant.")
        return points
    except Exception as e:
        print(f"[ERROR] Failed to format points: {e}")
        return []