from src.retriever import retrieve
from src.llm import generate_answer


def query_rag(question: str):
    """Top-level RAG query.

    Heuristics:
    - If the user asks for a summary, retrieve more candidates and ask the LLM to produce
      per-video summaries + an overall synthesis.
    - Otherwise use defaults.
    """
    qlow = question.lower()
    is_summary = any(w in qlow for w in ("summar", "overview", "what are", "tl;dr"))

    if is_summary:
        # retrieve broader set then re-rank and return top 20
        docs, metadatas = retrieve(question, top_k_initial=100, top_k=20)
        answer, used_context = generate_answer(docs, question, metadatas=metadatas, mode="summarize")
    else:
        docs, metadatas = retrieve(question)
        answer, used_context = generate_answer(docs, question, metadatas=metadatas, mode="qa")

    return {
        "answer": answer,
        "sources": metadatas,
        "used_context": used_context
    }