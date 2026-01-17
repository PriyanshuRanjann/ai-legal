import os
import json
from helper_function.chunker import process_parsed_pdfs_with_context_chunking

"""load_json_chunks(directory_path)
→ 
"""

def load_json_chunks(input,directory_path: str = "input/chunks"):
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
                    chunks.append(chunk_data)
        return chunks
    except Exception as e:
        print(f"[ERROR] Failed to load JSON chunks: {e}")
        return []
    

def validate_chunk_schema(chunk: dict) -> bool:
    """Validates that the chunk dictionary contains required fields."""
    required_fields = {"chunk_id", "text", "document_id", "page_number", "source_path", "context_prev", "context_next", "char_count"}
    missing_fields = required_fields - chunk.keys()
    if missing_fields:
        print(f"[WARN] Chunk is missing fields: {missing_fields}")
        return False
    return True

