# ✅ app.py (Streamlit UI + PDF QA)
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.chat_models import ChatOpenAI
import tempfile
import os
import dotenv

dotenv.load_dotenv()

# UI setup
st.set_page_config(page_title="Chip Spec Assistant", layout="wide")
st.title("📘 AI Assistant for Technical PDFs")

# Sidebar toggle: local or OpenAI
use_openai = st.sidebar.checkbox("🔁 Use OpenAI GPT-3.5", value=False)
st.caption(f"🧠 Currently using: {'OpenAI GPT-3.5 Turbo' if use_openai else 'Local: google/flan-t5-base'}")
st.sidebar.markdown(f"**Model in use:** {'🛰️ OpenAI GPT-3.5 Turbo' if use_openai else '🧩 Local (flan-t5-base)'}")

# Upload
uploaded_file = st.file_uploader("Upload a PDF (e.g., chip spec)", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        with st.spinner("📄 Processing PDF..."):
            loader = PyMuPDFLoader(tmp_file.name)
            documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    st.info(f"📄 Loaded {len(chunks)} chunks from the PDF")

    # Vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    st.success("✅ PDF successfully processed. Ready for questions.")

    # Prompt template
    custom_prompt = PromptTemplate.from_template("""
    You are an expert assistant. Use the provided context to answer the question.
    If the answer is not explicitly found in the context, say "Not found in document."

    Context:
    {context}

    Question: {question}
    Answer:
    """)

    # LLM setup
    if use_openai:
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    else:
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        local_pipeline = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            truncation=True
        )
        llm = HuggingFacePipeline(pipeline=local_pipeline)

    # QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": custom_prompt}
    )

    # Ask
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