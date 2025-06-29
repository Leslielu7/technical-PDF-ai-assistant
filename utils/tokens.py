# utils/token_utils.py
import tiktoken

def truncate_docs_by_tokens(docs, max_tokens, model_name="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(model_name)
    total_tokens = 0
    selected_docs = []

    for doc in docs:
        tokens = len(enc.encode(doc.page_content))
        if total_tokens + tokens > max_tokens:
            break
        selected_docs.append(doc)
        total_tokens += tokens

    return selected_docs
