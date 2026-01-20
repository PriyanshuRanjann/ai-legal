from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct , VectorParams, Distance
from helper_function.embeddings import process_chunks_for_embedding

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
    collection_name = "AI_Collection",
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
    
def upsert_points(
    client: QdrantClient,
    collection_name: str,
    points: list
) -> None:
    """Inserts or updates vectors into Qdrant."""
    try:
        client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"[INFO] Upserted {len(points)} points into collection {collection_name}.")
    except Exception as e:
        print(f"[ERROR] Failed to upsert points into {collection_name}: {e}")


def index_chunks_to_qdrant(
    input,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "AI_Collection"
) -> None:
    """Full pipeline: process chunks, generate embeddings, and index into Qdrant."""
    try:
        client = initialize_qdrant_client(host=qdrant_host, port=qdrant_port)
        if not client:
            return 

        create_qdrant_collection(client, collection_name)

        records = process_chunks_for_embedding(input)
        if not records:
            print("[ERROR] No embedding records to index.")
            return

        points = format_points(records)
        if not points:
            print("[ERROR] No points formatted for Qdrant.")
            return

        upsert_points(client, collection_name, points)

    except Exception as e:
        print(f"[ERROR] Exception during indexing to Qdrant: {e}")

if __name__ == "__main__":
    input = r"input/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf"
    print(index_chunks_to_qdrant(input))
    print("Indexing to Qdrant completed.")