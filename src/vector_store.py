import chromadb
client = chromadb.Client()

# store metadata (video id + chunk index) to make sources traceable
collection = client.get_or_create_collection(name="youtube_rag")


def store_embeddings(chunks, embeddings, video_id):
    """Batch-add chunks + embeddings to the chroma collection with metadata.

    - chunks: list[str]
    - embeddings: numpy array or list of lists
    """
    if not chunks:
        return

    ids = [f"{video_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"video_id": video_id, "chunk_index": i} for i in range(len(chunks))]

    # ensure embeddings are plain lists for chroma
    emb_lists = [emb.tolist() if hasattr(emb, "tolist") else emb for emb in embeddings]

    collection.add(
        documents=chunks,
        embeddings=emb_lists,
        ids=ids,
        metadatas=metadatas
    )