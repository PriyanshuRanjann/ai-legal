import json


INPUT_DIR = "output/parsed_pdf"
OUTPUT_DIR = "output/chunks"
MAX_CHARS_PER_CHUNK = 600
MIN_CHARS_PER_CHUNK = 200

def get_parsed_json():
    pass


def read_parsed_json(path):
    """Read parsed JSON file and return its content."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to read: {path}: {e}")
        return None

