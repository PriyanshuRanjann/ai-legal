import os
import json
from typing import List, Dict
from .pdf_parser import save_pdf_pages_to_json
from .schemas import ContextEnrichedChunk

INPUT_DIR = "output/parsed_pdf"
OUTPUT_DIR = "output/chunks"
MAX_CHARS_PER_CHUNK = 600
MIN_CHARS_PER_CHUNK = 200


def read_parsed_json(path: str, source_pdf: str):
    """Read parsed JSON file and return its content."""
    try:
        if not os.path.exists(path):
            save_pdf_pages_to_json(source_pdf)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Parsed JSON not found at {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read: {path}: {e}")
        return None

def write_parsed_json(path: str, data) -> None:
    """Write data to a JSON file."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Failed to write: {path}: {e}")

def split_paragraphs(text: str) -> List[str]:
    """
    Split the input text into cleaned, non-empty paragraphs by newline separators, returning the original stripped text if no paragraphs are found.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if not paragraphs:
        return [text.strip()]
    return paragraphs

def build_chunks_from_paragraphs(paragraphs: List[str]) -> List[str]:
    """
    Build text chunks from a list of paragraphs by concatenating them until each chunk reaches
    the desired length range, ensuring every chunk falls between MIN_CHARS_PER_CHUNK and
    MAX_CHARS_PER_CHUNK characters.
    Parameters:
        paragraphs (List[str]): Ordered list of paragraph strings to combine into chunks.
    Returns:
        List[str]: A list of chunk strings where each chunk is as large as possible without
        exceeding the maximum character limit, and no chunk shorter than the minimum length
        is emitted before attempting to append more text.
    """
    chunks = []
    buffer_text = ""

    for para in paragraphs:
        if len(buffer_text) + len(para) <= MAX_CHARS_PER_CHUNK:
            buffer_text += (" " + para if buffer_text else para)
        else:
            if len(buffer_text) >= MIN_CHARS_PER_CHUNK:
                chunks.append(buffer_text.strip())
                buffer_text = para
            else:
                buffer_text += " " + para

    if buffer_text.strip():
        chunks.append(buffer_text.strip())

    return chunks


def add_context_to_chunk_records(chunks: List[Dict]) -> None:
    """Mutate chunk records with neighbor context strings."""
    for i, chunk in enumerate(chunks):
        chunk["context_prev"] = chunks[i - 1]["text"] if i > 0 else ""
        chunk["context_next"] = chunks[i + 1]["text"] if i < len(chunks) - 1 else ""

def process_parsed_pdfs_with_context_chunking(
    input,
    input_dir: str = INPUT_DIR,
    output_dir: str = OUTPUT_DIR
) -> None:
    """
    Main entry point for context-enriched chunking pipeline.
    """

    try:
        os.makedirs(output_dir, exist_ok=True)

        files = sorted(f for f in os.listdir(input_dir) if f.endswith(".json"))
        print(f"[INFO] Found {len(files)} files to process.")

        pages_by_doc: Dict[str, List[Dict]] = {}
        output_buffers: Dict[str, List[Dict]] = {}

        for file_name in files:
            input_path = os.path.join(input_dir, file_name)

            try:
                data = read_parsed_json(input_path, input)
                if not data:
                    continue

                doc_id = data.get("document_id")
                if not doc_id:
                    print(f"[WARN] Missing document_id in {file_name}")
                    continue

                pages_by_doc.setdefault(doc_id, []).append({
                    "file_name": file_name,
                    "data": data
                })

            except Exception as file_error:
                print(f"[ERROR] Failed processing {file_name}: {file_error}")

        for doc_id, pages in pages_by_doc.items():
            pages.sort(key=lambda p: p["data"].get("page_number", 0))
            doc_chunks: List[Dict] = []

            for page in pages:
                data = page["data"]
                text = data.get("text", "").strip()
                if not text:
                    print(f"[WARN] Empty text in {page['file_name']}")
                    continue

                paragraphs = split_paragraphs(text)
                base_chunks = build_chunks_from_paragraphs(paragraphs)
                if not base_chunks:
                    continue

                page_chunks = output_buffers.setdefault(page["file_name"], [])

                for idx, chunk_text in enumerate(base_chunks):
                    chunk_record = {
                        "document_id": doc_id,
                        "page_number": data.get("page_number"),
                        "source_path": data.get("source_path"),
                        "chunk_id": f"{doc_id}_{data.get('page_number')}_{idx}",
                        "text": chunk_text,
                        "context_prev": "",
                        "context_next": "",
                        "char_count": len(chunk_text)
                    }
                    doc_chunks.append(chunk_record)
                    page_chunks.append(chunk_record)

            if not doc_chunks:
                continue

            add_context_to_chunk_records(doc_chunks)

        for file_name, chunks in output_buffers.items():
            validated_chunks = [
                ContextEnrichedChunk(**chunk).model_dump(mode="json")
                for chunk in chunks
            ]

            output_path = os.path.join(
                output_dir,
                file_name.replace(".json", "_chunks.json")
            )

            write_parsed_json(output_path, validated_chunks)

            print(f"[OK] Processed {file_name} → {len(validated_chunks)} chunks")


        print("[DONE] Chunking completed successfully.")

    except Exception as e:
        print(f"Chunking pipeline failed: {e}")

if __name__ == "__main__":
    input = r"input/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf"
    process_parsed_pdfs_with_context_chunking(input)