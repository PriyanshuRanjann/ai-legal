import requests
from helper_function.qdrant_client import initialize_qdrant_client, index_chunks_to_qdrant
from qdrant_client import QdrantClient
from config import OLLAMA_URL, OLLAMA_MODEL

