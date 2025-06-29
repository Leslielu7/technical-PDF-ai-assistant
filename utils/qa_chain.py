from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.chains import LLMChain

def build_qa_chain(llm, vectorstore):
    prompt = PromptTemplate.from_template("""
You are an expert assistant. Use the provided context to answer the question.
If the answer is not explicitly found in the context, say "Not found in document."

Context:
{context}

Question: {question}
Answer:
""")

    chain = LLMChain(llm=llm, prompt=prompt)
    return RunnableMap({
        "question": lambda x: x["question"],
        "context": lambda x: x["context"]
    }) | chain
