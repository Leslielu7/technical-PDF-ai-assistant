# ✅ app.py (Streamlit UI + PDF QA)
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import tempfile
# import os
import dotenv; dotenv.load_dotenv()

st.set_page_config(page_title="Chip Spec Assistant", layout="wide")
st.title("📘 AI Assistant for Technical PDFs")

uploaded_file = st.file_uploader("Upload a PDF (e.g., chip spec)", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        loader = PyMuPDFLoader(tmp_file.name)
        documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # embeddings = OpenAIEmbeddings()
    # Local Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Local LLM Pipeline (text generation)
    generator = pipeline("text-generation", model="distilgpt2", max_new_tokens=100)
    llm = HuggingFacePipeline(pipeline=generator)

    # qa = RetrievalQA.from_chain_type(
    #     llm=OpenAI(temperature=0),
    #     retriever=vectorstore.as_retriever()
    # )
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
    )

    query = st.text_input("Ask a question about the document:")
    if query:
        with st.spinner("Thinking..."):
            answer = qa.run(query)
            st.write("\n### Answer:")
            st.success(answer)