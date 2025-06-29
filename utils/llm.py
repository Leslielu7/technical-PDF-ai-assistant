from langchain.chat_models import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def get_llm(use_openai=True):
    if use_openai:
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    else:
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, truncation=True)
        return HuggingFacePipeline(pipeline=pipe)
