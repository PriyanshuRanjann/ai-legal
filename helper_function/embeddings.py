import os
import json
from sentence_transformers import SentenceTransformer
from helper_function.chunker import process_parsed_pdfs_with_context_chunking


def load_json_chunks(input_path, document_id: str = None, directory_path: str = "output/chunks"):
    """Reads JSON chunks for a specific document or all chunks.
    
    Args:
        input_path: Path to PDF (used for processing if needed)
        document_id: If provided, load only chunks for this document
        directory_path: Directory containing chunk JSON files
    """
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
            process_parsed_pdfs_with_context_chunking(input_path)
    except Exception as e:
        print(f"[ERROR] Failed to process parsed PDFs: {e}")
    
    try:
        chunks = []
        # If document_id specified, only load matching files
        if document_id:
            matching_files = [f for f in os.listdir(directory_path) 
                            if f.endswith(".json") and document_id in f]
            print(f"[INFO] Loading {len(matching_files)} chunk files for document {document_id}")
        else:
            matching_files = [f for f in os.listdir(directory_path) if f.endswith(".json")]
        
        for filename in matching_files:
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    chunk_data = json.load(f)
                    if isinstance(chunk_data, list):
                        chunks.extend(chunk_data)
                    elif isinstance(chunk_data, dict):
                        chunks.append(chunk_data)
            except Exception as e:
                print(f"[WARN] Failed to load {filename}: {e}")
                continue
        
        print(f"[INFO] Loaded {len(chunks)} chunks")
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
        # Lazy import to avoid TensorFlow issues
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


def process_chunks_for_embedding(input_path, document_id: str = None, batch_size: int = 100):
    """Full pipeline: load → clean → embed → prepare records.
    
    Args:
        input_path: Path to PDF
        document_id: Only process chunks for this document
        batch_size: Process embeddings in batches to reduce memory usage
    """
    try:
        print(f"[INFO] Loading chunks for embedding (document_id={document_id})...")
        chunks = load_json_chunks(input_path, document_id=document_id)
        if not chunks:
            print("[WARN] No chunks loaded, attempting to generate them...")
            from helper_function.chunker import process_parsed_pdfs_with_context_chunking
            process_parsed_pdfs_with_context_chunking(input_path, doc_id_filter=document_id)
            chunks = load_json_chunks(input_path, document_id=document_id)
            
            if not chunks:
                print("[ERROR] No chunks loaded after generation attempt.")
                return []
        
        print(f"[INFO] Successfully loaded {len(chunks)} chunks for embedding")
    except Exception as e:
        print(f"[ERROR] Exception during chunk loading: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    try:
        print("[INFO] Preparing texts for embedding...")
        texts = prepare_texts(chunks)
        if not texts:
            print("[ERROR] No valid texts to embed.")
            return []
        print(f"[INFO] Prepared {len(texts)} texts from chunks")
    except Exception as e:
        print(f"[ERROR] Exception during text preparation: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    try:
        print("[INFO] Loading embedding model...")
        model = load_embedding_model()
        if not model:
            print("[ERROR] Failed to load embedding model")
            return []
    except Exception as e:
        print(f"[ERROR] Exception during model loading: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    try:
        # Process embeddings in batches to manage memory
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        print(f"[INFO] Processing {len(texts)} texts in {total_batches} batches of size {batch_size}...")
        
        for batch_idx, i in enumerate(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_num = batch_idx + 1
            print(f"[INFO] Embedding batch {batch_num}/{total_batches} ({len(batch_texts)} texts)...")
            
            try:
                batch_embeddings = generate_embeddings(batch_texts, model)
                if batch_embeddings is None or len(batch_embeddings) == 0:
                    print(f"[WARN] Batch {batch_num} returned no embeddings, skipping...")
                    continue
                all_embeddings.extend(batch_embeddings)
                print(f"[INFO] Batch {batch_num} complete - {len(all_embeddings)} embeddings total")
            except Exception as batch_error:
                print(f"[ERROR] Error processing batch {batch_num}: {batch_error}")
                continue
        
        if not all_embeddings:
            print("[ERROR] No embeddings generated")
            return []
        
        print(f"[INFO] Successfully generated {len(all_embeddings)} embeddings total")
        
        print("[INFO] Building embedding records...")
        records = build_embedding_records(chunks, all_embeddings)
        
        if not records:
            print("[ERROR] Failed to build embedding records")
            return []
        
        print(f"[INFO] Successfully built {len(records)} embedding records")
        return records
        
    except Exception as e:
        print(f"[ERROR] Exception during embedding generation or record building: {e}")
        import traceback
        traceback.print_exc()
        return []

if __name__ == "__main__":
    # Example usage
    input_path = r"input/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf"
    embedding_records = process_chunks_for_embedding(input_path)
    print(f"Prepared {len(embedding_records)} records for embedding storage.")
    