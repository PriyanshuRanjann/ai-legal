import streamlit as st
import os
from pathlib import Path
from datetime import datetime
from config import INPUT_DIR, OUTPUT_DIR
from helper_function.pdf_parser import parse_pdf
from helper_function.chunker import process_parsed_pdfs_with_context_chunking
from helper_function.embeddings import process_chunks_for_embedding
from helper_function.qdrant_client import (
    initialize_qdrant_client,
    index_chunks_to_qdrant,
)
from helper_function.llm import ollama_rag, search_qdrant


INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Page config
st.set_page_config(
    page_title="Legal PDF Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .chat-message {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
        display: flex;
        gap: 12px;
    }
    .user-message {
        background-color: #e3f2fd;
        justify-content: flex-end;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .passage-box {
        background-color: #fafafa;
        border-left: 4px solid #1f77b4;
        padding: 12px;
        margin: 8px 0;
        border-radius: 4px;
    }
    .metadata-box {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 4px;
        margin-top: 10px;
        font-size: 12px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Session state initialization
if "documents" not in st.session_state:
    st.session_state.documents = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None
if "qdrant_client" not in st.session_state:
    try:
        st.session_state.qdrant_client = initialize_qdrant_client()
    except Exception as e:
        st.error(f"Failed to connect to Qdrant: {e}")
        st.session_state.qdrant_client = None

# ======================== HELPER FUNCTIONS ========================


def save_uploaded_file(uploaded_file):
    """Save uploaded PDF to input directory."""
    try:
        file_path = INPUT_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return str(file_path)
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def process_pdf_document(file_path, doc_name):
    """Process PDF and index to Qdrant."""
    try:
        with st.spinner(f"Processing {doc_name}..."):
            # Parse PDF
            st.info("📖 Parsing PDF...")
            parse_pdf(file_path)

            # Chunk content
            st.info("✂️ Chunking content...")
            process_parsed_pdfs_with_context_chunking(file_path)

            # Generate embeddings and index
            st.info("🔍 Generating embeddings...")
            collection_name = (
                f"collection_{doc_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            index_chunks_to_qdrant(file_path, collection_name)

            return collection_name
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None


def query_document(query, collection_name, top_k=3):
    """Query document and get response with context."""
    try:
        # Get the file path from documents registry
        if collection_name not in st.session_state.documents:
            st.error("Document not found")
            return None, [], []

        if not st.session_state.qdrant_client:
            st.error("Qdrant client not connected")
            return None, [], []

        file_path = st.session_state.documents[collection_name]["file_path"]

        # Get RAG response
        answer = ollama_rag(
            input_path=file_path,
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            skip_indexing=True,
        )

        # Get search results for citations
        results = search_qdrant(
            input_path=file_path,
            query=query,
            client=st.session_state.qdrant_client,
            collection_name=collection_name,
            top_k=top_k,
            skip_indexing=True,
            prefer_early_pages=True,
        )

        passages = []
        pages_set = set()

        for point in results:
            payload = point.payload if hasattr(point, "payload") else point
            page = payload.get("page_number", 0)
            pages_set.add(page)

            passages.append(
                {
                    "text": payload.get("text", "")[:500],
                    "page": page,
                    "chunk_id": payload.get("chunk_id", "unknown"),
                    "score": point.score if hasattr(point, "score") else 0.5,
                }
            )

        return answer, passages, sorted(list(pages_set))

    except Exception as e:
        st.error(f"Error querying document: {e}")
        import traceback

        traceback.print_exc()
        return None, [], []


# ======================== SIDEBAR ========================

st.sidebar.title("📚 PDF Chat Manager")

# Document upload section
st.sidebar.subheader("📤 Upload PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file", type="pdf", key="pdf_uploader"
)

if uploaded_file:
    doc_name = uploaded_file.name.replace(".pdf", "").replace(" ", "_")

    if st.sidebar.button("📥 Process PDF", use_container_width=True):
        # Check if already processed
        if doc_name in st.session_state.documents:
            st.sidebar.warning(f"'{doc_name}' already processed")
        else:
            # Save file
            file_path = save_uploaded_file(uploaded_file)
            if file_path:
                # Process document
                collection_name = process_pdf_document(file_path, doc_name)

                if collection_name:
                    # Store metadata
                    st.session_state.documents[collection_name] = {
                        "name": doc_name,
                        "file_path": file_path,
                        "uploaded_at": datetime.now().isoformat(),
                        "collection_name": collection_name,
                    }
                    st.session_state.current_doc_id = collection_name
                    st.sidebar.success(f"✅ {doc_name} processed!")
                    st.rerun()
                else:
                    st.sidebar.error(
                        "Failed to process PDF - check error messages above"
                    )
            else:
                st.sidebar.error("Failed to save file")
else:
    st.sidebar.info("No PDF selected")  # Document selection
st.sidebar.subheader("📑 Select Document")
if st.session_state.documents:
    try:
        doc_options = {v["name"]: k for k, v in st.session_state.documents.items()}
        selected_doc_name = st.sidebar.selectbox(
            "Choose a document", list(doc_options.keys())
        )

        if selected_doc_name:
            st.session_state.current_doc_id = doc_options[selected_doc_name]

            doc_info = st.session_state.documents[st.session_state.current_doc_id]
            with st.sidebar.expander("📋 Document Info"):
                st.write(f"**Name:** {doc_info['name']}")
                st.write(f"**Uploaded:** {doc_info['uploaded_at'][:19]}")
                st.write(f"**Collection:** {doc_info['collection_name']}")
    except Exception as e:
        st.sidebar.error(f"Error selecting document: {e}")
else:
    st.sidebar.info("No documents uploaded yet")

# Settings
st.sidebar.subheader("⚙️ Settings")
top_k = st.sidebar.slider("Top K Results", min_value=1, max_value=10, value=3)

# Clear history
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()

# ======================== MAIN CONTENT ========================

col1, col2 = st.columns([0.5, 0.5])

with col1:
    st.title("📄 Legal PDF Chat")

with col2:
    if st.session_state.current_doc_id:
        st.info(
            f"📖 Current: {st.session_state.documents[st.session_state.current_doc_id]['name']}"
        )

# Chat interface
if not st.session_state.current_doc_id:
    st.warning("👈 Upload and select a PDF document from the sidebar to start chatting")
else:
    # Display chat history
    st.subheader("💬 Chat")

    chat_container = st.container()

    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(
                    f"""
                <div class="chat-message user-message">
                    <div style="flex: 1;">
                        <strong>You:</strong><br>{message['content']}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                <div class="chat-message assistant-message">
                    <div style="flex: 1;">
                        <strong>Assistant:</strong><br>{message['content']}
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Display passages if available
                if "passages" in message:
                    with st.expander(
                        f"📚 Sources ({len(message['passages'])} passages)"
                    ):
                        for i, passage in enumerate(message["passages"], 1):
                            st.markdown(
                                f"""
                            <div class="passage-box">
                                <strong>Passage {i} - Page {passage['page']}</strong> (Score: {passage['score']:.3f})<br>
                                {passage['text']}
                            </div>
                            """,
                                unsafe_allow_html=True,
                            )

                # Display metadata if available
                if "metadata" in message:
                    st.markdown(
                        f"""
                    <div class="metadata-box">
                        Pages referenced: {', '.join(map(str, message['metadata']['pages']))}
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

    # Input area
    st.divider()

    col1, col2 = st.columns([0.9, 0.1])

    with col1:
        user_input = st.text_input(
            "Ask a question about the document:",
            placeholder="What is the main topic?",
            key="chat_input",
        )

    with col2:
        submit_button = st.button("Send", use_container_width=True)

    # Process user input
    if submit_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Get assistant response
        with st.spinner("⏳ Thinking..."):
            answer, passages, pages = query_document(
                user_input, st.session_state.current_doc_id, top_k=top_k
            )

        if answer:
            # Add assistant message to history
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "passages": passages,
                    "metadata": {"pages": pages},
                }
            )

            st.rerun()
        else:
            st.error("Failed to get response")

# Footer
st.divider()
st.caption("🔧 Powered by Legal Bot")
