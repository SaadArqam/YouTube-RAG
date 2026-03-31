import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.get_or_create_collection(name="youtube_rag")


def retrieve(query, top_k=3):
    query_embedding = model.encode([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    return results["documents"][0]