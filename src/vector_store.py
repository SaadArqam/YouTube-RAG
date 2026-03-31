import chromadb

client = chromadb.Client()

collection = client.get_or_create_collection(name="youtube_rag")


def store_embeddings(chunks, embeddings, video_id):
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[emb.tolist()],
            ids=[f"{video_id}_{i}"]
        )