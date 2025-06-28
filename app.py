# ✅ app.py (Streamlit UI + PDF QA)
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import tempfile
import os
import dotenv; dotenv.load_dotenv()

st.set_page_config(page_title="Chip Spec Assistant", layout="wide")
st.title("📘 AI Assistant for Technical PDFs")

uploaded_file = st.file_uploader("Upload a PDF (e.g., chip spec)", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        with st.spinner("📄 Processing PDF..."):
            loader = PyMuPDFLoader(tmp_file.name)
            documents = loader.load()
            print("✅ Documents loaded:", len(documents))

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    st.info(f"📄 Loaded {len(chunks)} chunks from the PDF")
    print("✅ Chunks created:", len(chunks))

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.success("✅ PDF successfully processed. Ready for questions.")

    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    generator = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200,
        truncation=True
    )
    llm = HuggingFacePipeline(pipeline=generator)

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

            st.write("\n### Retrieved Context:")
            docs = vectorstore.similarity_search(query, k=3)
            for i, doc in enumerate(docs):
                st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}\n")