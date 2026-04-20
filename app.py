import os

import streamlit as st
from dotenv import load_dotenv

from retrieval import Retriever
from generator import RAGGenerator

load_dotenv()

st.set_page_config(page_title="RBS Student Life Assistant", page_icon="🛡️")

api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found. Add it to Streamlit Secrets or your local .env file.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key


@st.cache_resource
def load_system():
    retriever = Retriever()
    generator = RAGGenerator()
    return retriever, generator


st.title("Student Life Assistant for Rutgers Business School 🛡️")
st.markdown(
    "Ask questions about RBS contacts, events, majors, and student life! "
    "Powered by Hybrid Retrieval (FAISS + BM25) and OpenAI."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Sources"):
                for src in msg["sources"]:
                    st.caption(f"{src.get('metadata_prefix', '')} \n\n {src.get('text', '')}")

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
        else:
            try:
                with st.spinner(f"Generating answer (Router detected intent: {intent})..."):
                    answer = generator.generate_answer(query, retrieved_chunks)
            except Exception as e:
                answer = f"Error during answer generation: {e}"

            st.markdown(answer)

            with st.expander("View Retrieved Sources"):
                st.write(f"Retrieved chunks: {len(retrieved_chunks)}")
                for i, src in enumerate(retrieved_chunks, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.caption(f"{src.get('metadata_prefix', '')} \n\n {src.get('text', '')}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": retrieved_chunks}
    )

with st.sidebar:
    st.header("Pipeline Info")
    st.markdown("- **Embeddings:** all-MiniLM-L6-v2")
    st.markdown("- **Vector DB:** FAISS (Dense)")
    st.markdown("- **Keyword:** BM25 (Sparse)")
    st.markdown("- **Reranker:** Reciprocal Rank Fusion")
    st.markdown("- **LLM:** gpt-4o-mini")
    st.markdown("- **Top-K Retrieval:** 10")