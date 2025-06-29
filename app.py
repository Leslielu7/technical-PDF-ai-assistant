# ✅ app.py (modular entry point)
import streamlit as st
import tempfile
import dotenv
import os

from utils.pdf import load_and_split_pdf
from utils.embedding import get_embedding_model
from utils.llm import get_llm
from utils.qa_chain import build_qa_chain
from langchain_community.vectorstores import FAISS

dotenv.load_dotenv(dotenv.find_dotenv(f".env.{os.getenv('ENV', 'dev')}", raise_error_if_not_found=False))

# --- UI Setup ---
st.set_page_config(page_title="Chip Spec Assistant", layout="wide")
st.title("📘 AI Assistant for Technical PDFs")

# --- Sidebar Controls ---
st.sidebar.subheader("⚙️ Model Settings")
use_openai = st.sidebar.checkbox("🔁 Use OpenAI GPT-3.5", value=False)
use_openai_embeddings = st.sidebar.checkbox("🔁 Use OpenAI Embeddings", value=True)

st.sidebar.markdown(f"**LLM:** {'OpenAI GPT-3.5 Turbo' if use_openai else 'Local (flan-t5-base)'}")
st.sidebar.markdown(f"**Embeddings:** {'OpenAI' if use_openai_embeddings else 'HuggingFace MiniLM'}")

# --- Chunk Parameters ---
chunk_size = st.sidebar.slider("📏 Chunk Size", min_value=200, max_value=1000, step=100, value=500)
chunk_overlap = st.sidebar.slider("🔁 Chunk Overlap", min_value=0, max_value=200, step=20, value=50)
max_chunks = st.sidebar.slider("🔢 Max Chunks to Use", min_value=50, max_value=1000, step=50, value=200)
k_context = st.sidebar.slider("📚 Number of Context Chunks", min_value=1, max_value=10, value=3)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PDF (e.g., chip spec)", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        with st.spinner("📄 Processing PDF..."):
            chunks = load_and_split_pdf(tmp_file.name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = chunks[:max_chunks]
    print(f"Chunk slider selected: {max_chunks}, actually loaded: {len(chunks)}")
    st.info(f"📄 Loaded {len(chunks)} chunks from the PDF")

    # --- Embedding & Vectorstore ---
    embeddings = get_embedding_model(use_openai_embeddings)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.success("✅ PDF successfully processed. Ready for questions.")

    # --- QA Chain ---
    llm = get_llm(use_openai)
    qa = build_qa_chain(llm, vectorstore)

    # --- User Query ---
    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Thinking..."):
            answer = qa.run(query)
            st.write("\n### Answer:")
            st.success(answer)

            st.write("\n### Retrieved Context:")
            docs = vectorstore.similarity_search(query, k=k_context)
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}\n")
