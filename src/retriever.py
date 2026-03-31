import chromadb
import numpy as np
from sentence_transformers import CrossEncoder
from src.embeddings import model as embed_model
from src.chunking import is_noisy_chunk

# chroma client
client = chromadb.Client()
collection = client.get_or_create_collection(name="youtube_rag")

# lightweight cross-encoder for re-ranking (small, works on CPU for small lists)
try:
    _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    _cross_encoder = None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return -1.0
    nom = np.dot(a, b)
    da = np.linalg.norm(a)
    db = np.linalg.norm(b)
    if da == 0 or db == 0:
        return -1.0
    return float(nom / (da * db))


def retrieve(query: str, top_k_initial: int = 50, top_k: int = 5):
    """Two-stage retrieval:
    1) vector search (Chroma) for top_k_initial
    2) re-rank candidates with a cross-encoder and return top_k

    Returns: list of documents (texts) and list of metadata dicts (same length)
    """
    query_embedding = embed_model.encode([query], convert_to_numpy=True)[0]

    # initial vector search
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k_initial,
        # 'ids' is not a valid include for Chroma; request documents, metadatas, distances, embeddings
        include=["documents", "metadatas", "distances", "embeddings"]
    )

    # unpack results (chroma returns nested lists per query)
    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    emb_lists = results.get("embeddings", [[]])[0]

    # filter noisy candidate chunks early
    candidates = []
    for doc, meta, emb in zip(docs, metadatas, emb_lists):
        if not doc:
            continue
        if is_noisy_chunk(doc):
            continue
        candidates.append((doc, meta, np.array(emb, dtype=float)))

    if not candidates:
        return [], []

    # Prepare re-ranker inputs
    texts = [c[0] for c in candidates]

    # Cross-encoder scoring (query, doc)
    pairs = [[query, t] for t in texts]
    try:
        if _cross_encoder is not None:
            scores = _cross_encoder.predict(pairs)
        else:
            raise RuntimeError("Cross-encoder not available")
    except Exception:
        # fallback to cosine similarity if cross-encoder fails
        scores = np.array([_cosine(query_embedding, c[2]) for c in candidates])

    # attach score and sort
    scored = []
    for (doc, meta, emb), score in zip(candidates, scores):
        scored.append({"doc": doc, "meta": meta, "score": float(score)})

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)

    top = scored[:top_k]

    return [t["doc"] for t in top], [t["meta"] for t in top]