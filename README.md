# ai-legal
## Setup Instructions (Linux)

- **Clone the repository**
    - `git clone https://github.com/PriyanshuRanjann/ai-legal.git`
- **Change directory to the project folder**
    - `cd ai-legal`
- **Create a Python virtual environment**
    - `python3 -m venv venv`
- **Activate the virtual environment**
    - `source venv/bin/activate`
- **Install dependencies**
    - `pip install -r requirements.txt`
- **Pull Docker qdrant image**
    - `docker pull qdrant/qdrant`
- **Run Docker container (example: for PostgreSQL)**
    - `docker run -d --name ai-legal-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant`
- **Start Ollama and pull a model (e.g., llama2)**
    - `ollama pull phi:latest`
- **Run the model with Ollama**
    - `ollama run phi:latest`
- **Run the main application**
    - `python -m uvicorn main:app --host 0.0.0.0 --port 8000`
- **Run the streamlit application**
    - `streamlit run app.py`

## Project Structure & File Descriptions

- **app.py**  
    - Streamlit web application for uploading, processing, and chatting with legal PDF documents. Handles user interface, session state, and document management.
- **main.py**  
    - Entry point for the FastAPI backend (if used). Defines API endpoints for interacting with the application programmatically.
- **config.py**  
    - Loads environment variables and sets up project-wide configuration, including input/output directories and service endpoints for Ollama and Qdrant.
- **api/endpoints.py**  
    - Contains API route definitions for serving application functionality via HTTP requests.
- **helper_function/__init__.py**  
    - Initializes the helper_function module, allowing imports from its submodules.
- **helper_function/chunker.py**  
    - Implements logic for splitting parsed PDF text into manageable chunks for further processing and embedding.
- **helper_function/embeddings.py**  
    - Handles generation of vector embeddings for text chunks, preparing them for storage and retrieval in Qdrant.
- **helper_function/llm.py**  
    - Provides functions for interacting with the Ollama language model, including retrieval-augmented generation (RAG) and search capabilities.
- **helper_function/pdf_parser.py**  
    - Parses PDF documents, extracting text and metadata for downstream processing.
- **helper_function/qdrant_client.py**  
    - Manages connection to the Qdrant vector database, including initialization and indexing of chunk embeddings.
- **helper_function/schemas.py**  
    - Defines data schemas and models used throughout the application for validation and structure.

## CHUNKING TECHNIQUE DESCRIPTION

### Chunking Strategy

- The chunking pipeline processes parsed PDF text by splitting it into paragraphs and then combining these paragraphs into context-enriched chunks.
- Each chunk is constructed to fall within a target character range (minimum 200, maximum 600 characters), ensuring that chunks are neither too short nor too long for downstream embedding and retrieval.
- Neighboring chunk text is also attached as context (`context_prev` and `context_next`) to each chunk, improving the quality of semantic search and retrieval-augmented generation.
- The resulting chunks are saved as JSON files for further processing.
    - **Splitting:** Text is first split into paragraphs using newline separators.
    - **Chunk Building:** Paragraphs are concatenated until the chunk reaches the desired length, balancing completeness and granularity.
    - **Context Enrichment:** Each chunk is annotated with the text of its immediate neighbors, providing additional context for embedding and retrieval.
    - **Output:** Chunks are validated and written to output JSON files, ready for embedding and indexing.

This strategy ensures that each chunk is semantically meaningful, context-aware, and optimized for use with language models and vector databases.

## ASSUMPTIONS AND TRADE-OFFS

- **ASSUMPTIONS:** 
    - Each PDF page contains meaningful text that can be extracted and processed.
    - The PyMuPDF library (fitz) reliably extracts text from most PDF pages.
    - The document ID can be derived from the PDF filename if not explicitly provided.
    - Normalizing text by removing extra whitespace and line breaks improves downstream chunking and embedding quality.
    - Chunks between 200 and 600 characters are optimal for embedding and retrieval tasks.

- **TRADE-OFFS:** 
    - Pages with extraction errors or empty text are skipped, which may result in loss of some content.
    - The chunking strategy prioritizes character count over semantic boundaries, so some chunks may split sentences or paragraphs.
    - Context enrichment uses only immediate neighboring chunks, which is a balance between adding useful context and avoiding excessive chunk size.
    - The pipeline does not handle images, tables, or non-textual content in PDFs.
    - All output is saved as JSON files, which is simple but may not be the most space-efficient format for large documents.
    - These choices aim to maximize reliability and downstream performance while keeping the implementation straightforward and maintainable.
