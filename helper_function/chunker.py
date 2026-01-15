import json
from .pdf_parser import save_pdf_pages_to_json

INPUT_DIR = "output/parsed_pdf"
OUTPUT_DIR = "output/chunks"
MAX_CHARS_PER_CHUNK = 600
MIN_CHARS_PER_CHUNK = 200


def read_parsed_json(path,input):
    """Read parsed JSON file and return its content."""
    try:
        save_pdf_pages_to_json(input)
        with open(path, 'r', encoding='utf-8') as f:
            print("Successfully read the parsed JSON file.")
            return json.load(f)
    except Exception as e:
        print(f"Failed to read: {path}: {e}")
        return None

