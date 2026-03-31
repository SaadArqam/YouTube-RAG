from sentence_transformers import SentenceTransformer
import numpy as np

# small, fast model for local use
model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(text_chunks, batch_size: int = 32):
    """Return numpy array of embeddings for a list of texts.

    - Uses batching and convert_to_numpy for performance on CPU.
    """
    if not text_chunks:
        return np.array([])

    embs = model.encode(
        text_chunks,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    return embs