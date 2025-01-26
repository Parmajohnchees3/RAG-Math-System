import argparse
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from embedding_func import get_embedding_function
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful math expert. You should answer the question only using the context provided below.

Context:
{context}

Question: {question}

If you cannot answer using the provided context, just say that you don't know.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("You are an assistant that answers based only on the provided context."),
        HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE)
    ])

    messages = prompt.format_messages(context=context_text, question=query_text)

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        openai_api_key=api_key,
        temperature=0.7
    )

    response = llm(messages)

    sources = [doc.metadata.get("id", None) for doc, _score in results]

    formatted_response = f"Response: {response.content}\nSources: {sources}"
    print(formatted_response)
    return response.content

if __name__ == "__main__":
    main()
