import os
import json
from .pdf_parser import save_pdf_pages_to_json

INPUT_DIR = "output/parsed_pdf"
OUTPUT_DIR = "output/chunks"
MAX_CHARS_PER_CHUNK = 600
MIN_CHARS_PER_CHUNK = 200


def read_parsed_json(path,input):
    """Read parsed JSON file and return its content."""
    try:
        if not os.path.exists(path):
            save_pdf_pages_to_json(input)
        with open(path, 'r', encoding='utf-8') as f:
            print("Successfully read the parsed JSON file.")
            return json.load(f)
    except Exception as e:
        print(f"Failed to read: {path}: {e}")
        return None

# if __name__ == "__main__":
#     input = r"input/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf"
#     output_path = os.path.join(INPUT_DIR, "the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final_page_0.json")
#     read_parsed_json(output_path,input)