import os
import json
from sentence_transformers import SentenceTransformer
from helper_function.chunker import process_parsed_pdfs_with_context_chunking


def load_json_chunks(input,directory_path: str = "output/chunks"):
    """Reads all JSON files from input/chunks and returns a list of chunk dictionaries."""
    try:
        if not os.path.exists(directory_path):
            process_parsed_pdfs_with_context_chunking(input)
    except Exception as e:
        print(f"[ERROR] Failed to process parsed PDFs: {e}")
    try:
        chunks = []
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):
                file_path = os.path.join(directory_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    chunk_data = json.load(f)
                    # If the loaded data is a list, extend; if dict, append
                    if isinstance(chunk_data, list):
                        chunks.extend(chunk_data)
                    elif isinstance(chunk_data, dict):
                        chunks.append(chunk_data)
        return chunks
    except Exception as e:
        print(f"[ERROR] Failed to load JSON chunks: {e}")
        return []
    

def validate_chunk_schema(chunk: dict) -> bool:
    """Validates that the chunk dictionary contains required fields."""
    try:
        required_fields = {"chunk_id", "text", "document_id", "page_number", "source_path", "context_prev", "context_next", "char_count"}
        missing_fields = required_fields - chunk.keys()
        if missing_fields:
            print(f"[WARN] Chunk is missing fields: {missing_fields}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Exception during chunk schema validation: {e}")
        return False


def clean_text(text: str) -> str:
    """Normalizes text by removing extra spaces and newlines."""
    try:
        if not text:
            return ""
        text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
        text = " ".join(part for part in text.split() if part)
        return text.strip()
    except Exception as e:
        print(f"[ERROR] Exception during text cleaning: {e}")
        return ""

def prepare_texts(chunks: list) -> list:
    """Extracts and returns list of texts to embed from chunk dictionaries."""
    try:
        texts = []
        for chunk in chunks:
            if validate_chunk_schema(chunk):
                cleaned_text = clean_text(chunk.get("text", ""))
                texts.append(cleaned_text)
            else:
                print(f"[WARN] Invalid chunk schema: {chunk.get('chunk_id', 'unknown')}")
        return texts
    except Exception as e:
        print(f"[ERROR] Exception during text preparation: {e}")
        return []


def load_embedding_model(model_name: str = "sentence-transformers/all-mpnet-base-v2"):
    """Loads and returns the embedding model."""
    try:
        model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")
        return model
    except ImportError as e:
        print(f"[ERROR] Failed to load embedding model: {e}")
        return None
    

def generate_embeddings(text_list: list, model) -> list:
    """Generates embeddings for a list of texts using the provided model."""
    if not model:
        print("[ERROR] Embedding model is not loaded.")
        return []
    try:
        embeddings = model.encode(text_list, show_progress_bar=True)
        print(f"[INFO] Generated {len(embeddings)} embeddings.")
        return embeddings
    except Exception as e:
        print(f"[ERROR] Failed to generate embeddings: {e}")
        return []
    

def build_embedding_records(chunks: list, embeddings: list) -> list:
    """Combines chunk metadata with embeddings into records suitable for Qdrant."""
    try:
        if len(chunks) != len(embeddings):
            print("[ERROR] Number of chunks and embeddings do not match.")
            return []
        
        records = []
        for chunk, embedding in zip(chunks, embeddings):
            record = {
                "id": chunk["chunk_id"],
                "vector": embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                "payload": {
                    "document_id": chunk["document_id"],
                    "page_number": chunk["page_number"],
                    "source_path": chunk["source_path"],
                    "text": chunk["text"],
                    "context_prev": chunk["context_prev"],
                    "context_next": chunk["context_next"],
                    "char_count": chunk["char_count"]
                }
            }
            records.append(record)
        print(f"[INFO] Built {len(records)} embedding records.")
        return records
    except Exception as e:
        print(f"[ERROR] Failed to build embedding records: {e}")
        return []


def process_chunks_for_embedding(input):
    """Full pipeline: load → clean → embed → prepare records."""
    try:
        chunks = load_json_chunks(input)
        if not chunks:
            print("[ERROR] No chunks loaded.")
            return []
    except Exception as e:
        print(f"[ERROR] Exception during chunk loading: {e}")
        return []
    try:
        texts = prepare_texts(chunks)
        if not texts:
            print("[ERROR] No valid texts to embed.")
            return []
    except Exception as e:
        print(f"[ERROR] Exception during text preparation: {e}")
        return []
    
    try:
        model = load_embedding_model()
        if not model:
            return []
    except Exception as e:
        print(f"[ERROR] Exception during model loading: {e}")
        return []
    try:
        embeddings = generate_embeddings(texts, model)
        if embeddings is None or len(embeddings) == 0:
            return []
        records = build_embedding_records(chunks, embeddings)
        return records
    except Exception as e:
        print(f"[ERROR] Exception during embedding generation or record building: {e}")
        return []

if __name__ == "__main__":
    # Example usage
    input = r"input/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf"
    embedding_records = process_chunks_for_embedding(input)
    print(f"Prepared {len(embedding_records)} records for embedding storage.")
    