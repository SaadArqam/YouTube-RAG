from src.retriever import retrieve
from src.llm import generate_answer


def query_rag(question):
    docs = retrieve(question)
    context = "\n\n".join(docs)

    answer = generate_answer(context, question)

    return {
        "answer": answer,
        "sources": docs
    }