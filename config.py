import os
from dotenv import load_dotenv
# import requests
from pathlib import Path

# Load environment variables from .env file

env_path = Path.cwd() / ".env"  # Adjust the path
if env_path.exists():
    load_dotenv(env_path, override=True)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path.cwd()))

INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

if not INPUT_DIR.exists():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ollama configuration
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi:latest")

# Qdrant configuration
QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

# LLM configuration
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "4000"))

# API configuration
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
