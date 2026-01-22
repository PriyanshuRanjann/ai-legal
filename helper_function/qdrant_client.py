import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
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
    collection_name : str,
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
        if "already exists" in str(e):
            print(f"[INFO] Collection {collection_name} already exists.")
        else:
            print(f"[ERROR] Failed to create collection {collection_name}: {e}")

def format_points(records: list) -> list:
    """Converts embedding records into Qdrant PointStruct format with UUIDs as IDs."""
    points = []
    try:
        for record in records:
            # Convert string chunk_id to UUID5 (deterministic for same input)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(record["id"])))
            point = PointStruct(
                id=point_id,
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

def index_chunks_to_qdrant(input_path, collection_name: str, document_id: str = None, qdrant_host: str = "localhost", qdrant_port: int = 6333) -> bool:
    """Full pipeline: generate embeddings, and index into Qdrant.
    
    Args:
        input_path: Path to PDF file
        collection_name: Name of Qdrant collection
        document_id: Document ID to filter chunks (speeds up processing)
        qdrant_host: Qdrant host
        qdrant_port: Qdrant port
        
    Returns:
        bool: True if indexing succeeded, False otherwise
    """
    try:
        print(f"[INFO] Initializing Qdrant client at {qdrant_host}:{qdrant_port}...")
        client = initialize_qdrant_client(host=qdrant_host, port=qdrant_port)
        if not client:
            print("[ERROR] Failed to initialize Qdrant client")
            return False

        print(f"[INFO] Creating or verifying collection: {collection_name}")
        create_qdrant_collection(client, collection_name)

        print(f"[INFO] Starting embedding generation for document (document_id={document_id})...")
        records = process_chunks_for_embedding(input_path, document_id=document_id)
        
        if not records:
            print("[ERROR] No embedding records to index.")
            return False
        
        print(f"[INFO] Generated {len(records)} embedding records")

        print(f"[INFO] Formatting {len(records)} records for Qdrant...")
        points = format_points(records)
        
        if not points:
            print("[ERROR] No points formatted for Qdrant.")
            return False
        
        print(f"[INFO] Formatted {len(points)} points successfully")

        print(f"[INFO] Upserting {len(points)} points to Qdrant collection {collection_name}...")
        upsert_points(client, collection_name, points)
        
        print(f"[SUCCESS] Successfully indexed {len(points)} points to Qdrant collection '{collection_name}'")
        return True

    except Exception as e:
        print(f"[ERROR] Exception during indexing to Qdrant: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    
    collection_name = "legal_documents"
    input_path = r"input/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf"
    index_chunks_to_qdrant(input_path, collection_name)
    print("Indexing to Qdrant completed.")