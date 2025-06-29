from langchain.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model(use_openai=True):
    if use_openai:
        return OpenAIEmbeddings()
    else:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
