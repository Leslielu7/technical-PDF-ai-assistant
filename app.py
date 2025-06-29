import streamlit as st
import tempfile
import dotenv
import os

from utils.pdf import load_and_split_pdf
from utils.embedding import get_embedding_model
from utils.llm import get_llm
from utils.qa_chain import build_qa_chain
from langchain_community.vectorstores import FAISS
from utils.tokens import truncate_docs_by_tokens

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

chunk_size = st.sidebar.slider("📏 Chunk Size", 200, 1000, 500, 100)
chunk_overlap = st.sidebar.slider("🔁 Chunk Overlap", 0, 200, 50, 20)
max_chunks = st.sidebar.slider("🔢 Max Chunks to Use", 50, 1000, 200, 50)
k_context = st.sidebar.slider("📚 Number of Context Chunks", 1, 10, 3)

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PDF (e.g., chip spec)", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        with st.spinner("📄 Processing PDF..."):
            chunks = load_and_split_pdf(tmp_file.name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = chunks[:max_chunks]

    st.info(f"📄 Loaded {len(chunks)} chunks from the PDF")

    embeddings = get_embedding_model(use_openai_embeddings)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.success("✅ PDF successfully processed. Ready for questions.")

    llm = get_llm(use_openai)
    qa = build_qa_chain(llm, vectorstore)

    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Thinking..."):
            docs = vectorstore.similarity_search(query, k=k_context)
            truncated_docs = truncate_docs_by_tokens(docs, max_tokens=3000)

            if len(truncated_docs) < len(docs):
                st.warning(
                    f"⚠️ Only {len(truncated_docs)} out of {len(docs)} chunks used due to token limit. "
                    f"Some context may be omitted in the answer."
                )

            result = qa.invoke({
                "question": query,
                "context": "\n\n".join(doc.page_content for doc in truncated_docs)
            })

            answer = result["text"]  # ✅ this is guaranteed by LLMChain
            st.success(answer)


            st.write("\n### Retrieved Context:")
            for i, doc in enumerate(truncated_docs):
                st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}\n")
