import requests
import time
from helper_function.qdrant_client import (
    initialize_qdrant_client,
    index_chunks_to_qdrant,
)
from ollama import Client
from config import OLLAMA_URL, OLLAMA_MODEL, QDRANT_HOST, QDRANT_PORT
from sentence_transformers import SentenceTransformer


def search_qdrant(
    input_path,
    query,
    client,
    collection_name,
    top_k=5,
    skip_indexing=False,
    prefer_early_pages=True,
):
    """Search Qdrant for top_k relevant chunks using a query vector.

    Args:
        input_path: Path to input PDF
        query: User query text
        client: Qdrant client
        collection_name: Name of Qdrant collection
        top_k: Number of top results to return
        skip_indexing: If True, skip re-indexing (faster for repeated queries)
        prefer_early_pages: If True, boost results from early pages (0-5)
    """
    try:
        # Only index if collection doesn't exist or skip_indexing is False
        if not skip_indexing:
            index_chunks_to_qdrant(input_path, collection_name)
        else:
            print("Skipping re-indexing, using existing collection")

        # Encode query to vector using SAME model as used for chunks
        model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        query_vector = model.encode(query).tolist()

        # Retrieve more results so we can filter by page
        fetch_k = top_k * 3 if prefer_early_pages else top_k
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,  # This searches the stored embeddings
            limit=fetch_k,
            with_payload=True,
            with_vectors=False,
        )

        result_points = results.points if hasattr(results, "points") else results

        # Filter and prioritize by page number if requested
        if prefer_early_pages:
            early_page_results = []
            late_page_results = []

            for point in result_points:
                page = point.payload.get("page_number", 999)
                if page <= 5:
                    early_page_results.append(point)
                else:
                    late_page_results.append(point)

            # Combine: early pages first, then late pages
            result_points = (early_page_results + late_page_results)[:top_k]
            print(
                f"Selected {len(early_page_results)} from early pages, {len([p for p in result_points if p.payload.get('page_number', 999) > 5])} from later pages"
            )
        else:
            result_points = result_points[:top_k]

        print(f"Qdrant returned {len(result_points)} results from stored embeddings")
        if result_points:
            for i, point in enumerate(result_points):
                score = point.score if hasattr(point, "score") else "N/A"
                page = point.payload.get("page_number", "?")
                text_preview = point.payload.get("text", "")[:60]
                print(
                    f"  Result {i}: page={page}, score={score}, text={text_preview}..."
                )
        return result_points
    except Exception as e:
        print(f"[ERROR] Failed to search Qdrant: {e}")
        return []


def build_context(results, max_tokens=600):
    """Concatenate retrieved chunk texts for context with token limit."""
    try:
        texts = []
        token_count = 0
        max_tokens_per_chunk = 300

        for i, point in enumerate(results):
            try:
                # Handle both direct point objects and wrapped results
                payload = point.payload if hasattr(point, "payload") else point

                if not payload:
                    continue

                # Extract text from payload dictionary
                text = (
                    payload.get("text", "")
                    if isinstance(payload, dict)
                    else getattr(payload, "text", "")
                )

                if not text:
                    continue

                # Truncate to max per chunk (character limit)
                truncated_text = text[: max_tokens_per_chunk * 4]
                estimated_tokens = len(truncated_text) // 4

                if token_count + estimated_tokens > max_tokens:
                    break

                texts.append(truncated_text)
                token_count += estimated_tokens
            except Exception as e:
                continue

        context = "\n\n".join(texts)
        return context
    except Exception as e:
        print(f"Failed to build context from results: {e}")
        return ""


def ollama_rag(
    input_path,
    query,
    collection_name,
    top_k=2,  # Reduced from 3
    qdrant_host=QDRANT_HOST,
    qdrant_port=QDRANT_PORT,
    ollama_host=OLLAMA_URL,
    skip_indexing=False,
):
    """
    RAG pipeline: search Qdrant for context using query_vector, send context+query to Ollama, return answer.

    Args:
        skip_indexing: If True, skip re-indexing (faster for repeated queries)
    """
    try:
        client = initialize_qdrant_client(host=qdrant_host, port=qdrant_port)
        results = search_qdrant(
            input_path,
            query,
            client,
            collection_name,
            top_k,
            skip_indexing=skip_indexing,
            prefer_early_pages=True,
        )
        context = build_context(results, max_tokens=600)  # Reduced from 800

        if not context or len(context.strip()) == 0:
            return "No relevant documents found for your query."

        # Very minimal prompt to reduce token processing
        prompt = f"""
        ROLE: You are a legal professional specialized in retrieving and citing legal document passages.
        TASK: Answer using ONLY this context. Be brief.
        - Return relevant passages with:
            - Exact text passage
            - Page number(s) where the passage appears
            - Number of instances found
            - Confidence score (higher means more relevant)

        CONTEXT: {context}
        QUERY: {query}

        RULES:
        - Cite exact page number(s) for every passage.
        - If none found, reply: "No relevant passages found."
        - Never fabricate passages or page numbers.
        - If page range given, search only those pages
        - Respond in a clear, structured format.

ANSWER:"""

        ollama_client = Client(host=ollama_host)

        # Retry logic with exponential backoff
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = ollama_client.generate(
                    model=OLLAMA_MODEL,
                    prompt=prompt,
                    stream=False,
                    options={
                        "num_predict": 200,
                        "num_ctx": 1024,
                    },
                )

                answer = response.get("response", "")
                if not answer:
                    return "Empty response from model."
                print("Successfully received response from Ollama")
                return answer
            except Exception as ollama_error:
                print(f"Ollama call attempt {attempt + 1} failed: {ollama_error}")
                if attempt < max_retries - 1:
                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s
                    time.sleep(wait_time)
                else:
                    return f"Summary from documents:\n\n{context[:300]}..."

    except Exception as e:
        print(f"[ERROR] RAG pipeline failed: {e}")
        return "Error occurred during RAG processing."


def ollama_rag_streaming(
    input_path,
    collection_name,
    top_k=2,
    qdrant_host=QDRANT_HOST,
    qdrant_port=QDRANT_PORT,
    ollama_host=OLLAMA_URL,
    skip_indexing=False,
):
    """
    Streaming chatbot RAG pipeline: Run until user says exit/bye/quit/etc.
    Streams responses token-by-token for real-time chatbot experience.

    Args:
        input_path: Path to input PDF
        collection_name: Name of Qdrant collection
        top_k: Number of top results to retrieve
        qdrant_host: Qdrant host
        qdrant_port: Qdrant port
        ollama_host: Ollama server URL
        skip_indexing: If True, skip re-indexing on first query
    """
    exit_keywords = {"exit", "bye", "quit", "q", "leave", "stop", "end"}

    try:
        client = initialize_qdrant_client(host=qdrant_host, port=qdrant_port)
        first_query = True

        print("\n" + "=" * 60)
        print("RAG CHATBOT - Ask questions about your documents")
        print("Type 'exit', 'bye', or 'quit' to end conversation")
        print("=" * 60 + "\n")

        while True:
            try:
                query = input("You: ").strip()

                # Check exit conditions
                if query.lower() in exit_keywords:
                    print("\nBot: Goodbye!")
                    break

                if not query:
                    print("Bot: Please enter a question.\n")
                    continue

                print("\nBot: ", end="", flush=True)

                # Retrieve context from Qdrant
                results = search_qdrant(
                    input_path,
                    query,
                    client,
                    collection_name,
                    top_k,
                    skip_indexing=(not first_query or skip_indexing),
                    prefer_early_pages=True,
                )
                first_query = False

                context = build_context(results, max_tokens=600)

                if not context or len(context.strip()) == 0:
                    print("No relevant documents found for your query.\n")
                    continue

                # Build prompt
                prompt = f"""
                        ROLE: You are a legal professional specialized in retrieving and citing legal document passages.
                        TASK: Answer using ONLY this context. Be brief.
                        - Return relevant passages with:
                            - Exact text passage
                            - Page number(s) where the passage appears
                            - Number of instances found
                            - Confidence score (higher means more relevant)

                        CONTEXT: {context}
                        QUERY: {query}

                        RULES:
                        - Cite exact page number(s) for every passage.
                        - If none found, reply: "No relevant passages found."
                        - Never fabricate passages or page numbers.
                        - If page range given, search only those pages
                        - Respond in a clear, structured format.
"""

                # Stream response from Ollama
                ollama_client = Client(host=ollama_host)

                full_response = ""
                try:
                    response_stream = ollama_client.generate(
                        model=OLLAMA_MODEL,
                        prompt=prompt,
                        stream=True,
                        options={
                            "num_predict": 200,
                            "num_ctx": 1024,
                        },
                    )

                    # Stream tokens in real-time
                    for chunk in response_stream:
                        token = chunk.get("response", "")
                        if token:
                            print(token, end="", flush=True)
                            full_response += token

                    print("\n")  # New line after response

                except Exception as stream_error:
                    print(f"Error during streaming: {stream_error}")
                    print(f"Fallback: {context[:300]}...\n")

            except KeyboardInterrupt:
                print("\n\nBot: Conversation interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError processing query: {e}")
                print("Please try again.\n")

    except Exception as e:
        print(f"Chatbot failed to initialize: {e}")


if __name__ == "__main__":
    # Example usage - streaming chatbot
    input_path = r"input/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf"
    collection_name = "legal_example1"
    ollama_rag_streaming(input_path, collection_name)
