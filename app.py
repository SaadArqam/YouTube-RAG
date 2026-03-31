import streamlit as st
import logging
from src.ingest import fetch_channel_transcripts
from src.chunking import chunk_text, is_noisy_chunk
from src.embeddings import get_embeddings
from src.vector_store import store_embeddings
from src.pipeline import query_rag

st.title("🎥 YouTube RAG")

logger = logging.getLogger(__name__)

# =========================
# 🔹 INDEXING SECTION
# =========================
channel_url = st.text_input("Enter YouTube Channel URL")

if st.button("Fetch & Index"):
    if not channel_url:
        st.warning("Please enter a YouTube channel URL")
    else:
        with st.spinner("Processing videos..."):
            data = fetch_channel_transcripts(channel_url)

            if not data:
                st.error("❌ No transcripts found.")
            else:
                total_chunks = 0

                for item in data:
                    # prefer the canonical 'text' key returned by fetch_channel_transcripts
                    source_text = item.get("text") or item.get("cleaned") or item.get("raw") or ""

                    chunks = chunk_text(source_text)

                    # drop noisy chunks early
                    good_chunks = [c for c in chunks if not is_noisy_chunk(c)]
                    logger.info("video %s: chunks=%d good=%d", item.get("video_id"), len(chunks), len(good_chunks))
                    if not good_chunks:
                        continue

                    embeddings = get_embeddings(good_chunks)
                    store_embeddings(good_chunks, embeddings, item["video_id"])

                    total_chunks += len(good_chunks)

                st.success(f"✅ Indexed {total_chunks} chunks from {len(data)} videos!")

# =========================
# 🔹 ASK SECTION
# =========================
st.divider()
st.header("Ask Questions")

question = st.text_input("Your question")

if st.button("Ask"):
    if not question:
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking... 🤖"):
            result = query_rag(question)

        st.subheader("🧠 Answer")
        st.write(result["answer"])

        st.subheader("📚 Sources (metadata)")
        for i, src in enumerate(result.get("sources", []), 1):
            # src is a metadata dict {video_id, chunk_index}
            vid = src.get("video_id")
            idx = src.get("chunk_index")
            st.write(f"{i}. video: {vid} — chunk: {idx}")

        # optional: show the context used
        st.subheader("🔎 Used Context Snippet")
        st.write(result.get("used_context", ""))