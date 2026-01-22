import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from helper_function.embeddings import process_chunks_for_embedding
from config import QDRANT_HOST, QDRANT_PORT


def initialize_qdrant_client(
    host: str = QDRANT_HOST, port: int = QDRANT_PORT
) -> QdrantClient:
    """Initializes and returns a Qdrant client."""
    try:
        client = QdrantClient(host=host, port=port)
        print(f"Connected to Qdrant at {host}:{port}")
        return client
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}")
        return None


def create_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size=768,
    distance=Distance.COSINE,
) -> None:
    """Creates a Qdrant collection with the specified parameters."""
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
        print(f"Created Qdrant collection: {collection_name}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Collection {collection_name} already exists.")
        else:
            print(f"Failed to create collection {collection_name}: {e}")


def format_points(records: list) -> list:
    """Converts embedding records into Qdrant PointStruct format with UUIDs as IDs."""
    points = []
    try:
        for record in records:
            # Convert string chunk_id to UUID5 (deterministic for same input)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(record["id"])))
            point = PointStruct(
                id=point_id, vector=record["vector"], payload=record["payload"]
            )
            points.append(point)
        print(f"Formatted {len(points)} points for Qdrant.")
        return points
    except Exception as e:
        print(f"Failed to format points: {e}")
        return []


def upsert_points(client: QdrantClient, collection_name: str, points: list) -> None:
    """Inserts or updates vectors into Qdrant."""
    try:
        client.upsert(collection_name=collection_name, points=points)
        print(f"Upserted {len(points)} points into collection {collection_name}.")
    except Exception as e:
        print(f"Failed to upsert points into {collection_name}: {e}")


def index_chunks_to_qdrant(
    input_path,
    collection_name: str,
    document_id: str = None,
    qdrant_host: str = QDRANT_HOST,
    qdrant_port: int = QDRANT_PORT,
) -> bool:
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
        print(f"Initializing Qdrant client at {qdrant_host}:{qdrant_port}...")
        client = initialize_qdrant_client(host=qdrant_host, port=qdrant_port)
        if not client:
            print("Failed to initialize Qdrant client")
            return False

        print(f"Creating or verifying collection: {collection_name}")
        create_qdrant_collection(client, collection_name)

        print(
            f"Starting embedding generation for document (document_id={document_id})..."
        )
        records = process_chunks_for_embedding(input_path, document_id=document_id)

        if not records:
            print("No embedding records to index.")
            return False

        print(f"Generated {len(records)} embedding records")

        print(f"Formatting {len(records)} records for Qdrant...")
        points = format_points(records)

        if not points:
            print("No points formatted for Qdrant.")
            return False

        print(f"Formatted {len(points)} points successfully")

        print(
            f"Upserting {len(points)} points to Qdrant collection {collection_name}..."
        )
        upsert_points(client, collection_name, points)

        print(
            f"Successfully indexed {len(points)} points to Qdrant collection '{collection_name}'"
        )
        return True

    except Exception as e:
        print(f"Exception during indexing to Qdrant: {e}")
        return False


if __name__ == "__main__":

    collection_name = "legal_documents"
    input_path = r"input/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf"
    index_chunks_to_qdrant(input_path, collection_name)
    print("Indexing to Qdrant completed.")
