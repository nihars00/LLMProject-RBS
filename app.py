import os
import subprocess
import sys

import streamlit as st
from dotenv import load_dotenv

from retrieval import Retriever
from generator import RAGGenerator


load_dotenv()

# --------------------------------------------------
# Phoenix Cloud tracing setup
# --------------------------------------------------
tracer = None
SpanAttributes = None

try:
    from phoenix.otel import register
    from opentelemetry import trace
    from openinference.semconv.trace import SpanAttributes

    phoenix_api_key = os.getenv("PHOENIX_API_KEY") or st.secrets.get("PHOENIX_API_KEY")
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or st.secrets.get(
        "PHOENIX_COLLECTOR_ENDPOINT"
    )

    if phoenix_api_key and phoenix_endpoint:
        os.environ["PHOENIX_API_KEY"] = phoenix_api_key
        os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = phoenix_endpoint

        register(
            project_name="rbs-student-life-assistant",
            auto_instrument=True,
            batch=True,
        )

        tracer = trace.get_tracer(__name__)

except Exception as e:
    print(f"Phoenix tracing not enabled: {e}")


# --------------------------------------------------
# Streamlit page setup
# --------------------------------------------------
st.set_page_config(page_title="RBS Student Life Assistant", page_icon="🛡️")

api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Add it to Streamlit Secrets or your local .env file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


# --------------------------------------------------
# Auto-run ingest.py if index files are missing
# --------------------------------------------------
def ensure_indexes_exist():
    output_dir = "output"

    if not os.path.exists(output_dir):
        with st.spinner("Building knowledge base indexes for the first time..."):
            result = subprocess.run(
                [sys.executable, "ingest.py"],
                capture_output=True,
                text=True,
            )

        if result.returncode != 0:
            st.error("Failed to build indexes.")
            st.code(result.stderr)
            st.stop()


ensure_indexes_exist()


# --------------------------------------------------
# Load retriever + generator
# --------------------------------------------------
@st.cache_resource
def load_system():
    retriever = Retriever()
    generator = RAGGenerator()
    return retriever, generator


# --------------------------------------------------
# App UI
# --------------------------------------------------
st.title("Student Life Assistant for Rutgers Business School 🛡️")
st.markdown(
    "Ask questions about RBS contacts, events, majors, and student life! "
    "Powered by Hybrid Retrieval (FAISS + BM25) and OpenAI."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Display previous messages
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Sources"):
                for src in msg["sources"]:
                    st.caption(
                        f"{src.get('metadata_prefix', '')}\n\n{src.get('text', '')}"
                    )


# --------------------------------------------------
# Chat input
# --------------------------------------------------
query = st.chat_input("Ask a question (e.g. 'Who is the contact for MITA?')")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    try:
        with st.spinner("Loading retrieval and generation system..."):
            retriever, generator = load_system()

    except FileNotFoundError:
        st.error("Error: Missing index files. Please run `python ingest.py` first to process documents.")
        st.stop()

    except Exception as e:
        st.error(f"System initialization failed: {e}")
        st.stop()

    try:
        with st.spinner("Searching specific knowledge base..."):
            retrieved_chunks, intent = retriever.retrieve(query, top_k=10)

    except Exception as e:
        st.error(f"Retrieval failed: {e}")
        st.stop()

    with st.chat_message("assistant"):
        if not retrieved_chunks:
            answer = "I don't have information about that in my current database."
            st.markdown(answer)

            if tracer and SpanAttributes:
                with tracer.start_as_current_span("RAG Question") as span:
                    span.set_attribute(SpanAttributes.INPUT_VALUE, query)
                    span.set_attribute("rag.intent", intent)
                    span.set_attribute("rag.num_retrieved_chunks", 0)
                    span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)

        else:
            try:
                if tracer and SpanAttributes:
                    with tracer.start_as_current_span("RAG Question") as span:
                        span.set_attribute(SpanAttributes.INPUT_VALUE, query)
                        span.set_attribute("rag.intent", intent)
                        span.set_attribute("rag.num_retrieved_chunks", len(retrieved_chunks))

                        for i, chunk in enumerate(retrieved_chunks[:5], 1):
                            span.set_attribute(
                                f"rag.retrieved_chunk_{i}",
                                chunk.get("text", "")[:1000],
                            )
                            span.set_attribute(
                                f"rag.retrieved_source_{i}",
                                chunk.get("metadata_prefix", ""),
                            )

                        with st.spinner(f"Generating answer (Router detected intent: {intent})..."):
                            answer = generator.generate_answer(query, retrieved_chunks)

                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)

                else:
                    with st.spinner(f"Generating answer (Router detected intent: {intent})..."):
                        answer = generator.generate_answer(query, retrieved_chunks)

            except Exception as e:
                answer = f"Error during answer generation: {e}"

            st.markdown(answer)

            with st.expander("View Retrieved Sources"):
                st.write(f"Retrieved chunks: {len(retrieved_chunks)}")

                for i, src in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.caption(
                        f"{src.get('metadata_prefix', '')}\n\n{src.get('text', '')}"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": retrieved_chunks}
    )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("Pipeline Info")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
    st.markdown("- **Vector DB:** FAISS (Dense)")
    st.markdown("- **Keyword:** BM25 (Sparse)")
    st.markdown("- **Reranker:** Reciprocal Rank Fusion")
    st.markdown("- **LLM:** gpt-4o-mini")
    st.markdown("- **Top-K Retrieval:** 10")
    st.markdown("- **Observability:** Phoenix Cloud")