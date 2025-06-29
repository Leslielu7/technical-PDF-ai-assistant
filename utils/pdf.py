from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter

def load_and_split_pdf(file_path, chunk_size=500, chunk_overlap=50):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)
