from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_qa_chain(llm, vectorstore):
    prompt = PromptTemplate.from_template("""
    You are an expert assistant. Use the provided context to answer the question.
    If the answer is not explicitly found in the context, say "Not found in document."

    Context:
    {context}

    Question: {question}
    Answer:
    """)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
