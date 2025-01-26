from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_function():
    # model_name = "sentence-transformers/bert-base-nli-mean-tokens"
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings
