import streamlit as st
import logging

from src.ingest import fetch_channel_transcripts
from src.chunking import chunk_text, is_noisy_chunk
from src.embeddings import get_embeddings
from src.vector_store import store_embeddings
from src.pipeline import query_rag

# =====================================
# 🔥 CONFIG
# =====================================
st.set_page_config(page_title="VidMind AI", page_icon="🎥", layout="wide")

st.title("🎥 VidMind AI — YouTube RAG")

logger = logging.getLogger(__name__)

# =====================================
# 🔥 MODEL WARMUP (IMPORTANT)
# =====================================
@st.cache_resource
def warmup():
    from src.llm import load_model
    load_model()

warmup()

# =====================================
# 🔹 INDEXING SECTION
# =====================================
st.header("📥 Index YouTube Channel")

channel_url = st.text_input("Enter YouTube Channel URL")

if st.button("Fetch & Index"):
    if not channel_url:
        st.warning("⚠️ Please enter a YouTube channel URL")
    else:
        with st.spinner("🔄 Fetching transcripts and building index..."):
            data = fetch_channel_transcripts(channel_url)

            if not data:
                st.error("❌ No transcripts found. Try another channel (e.g. FreeCodeCamp).")
            else:
                total_chunks = 0
                skipped_videos = 0

                for item in data:
                    source_text = item.get("text") or ""

                    chunks = chunk_text(source_text)

                    # Filter noisy chunks
                    good_chunks = [c for c in chunks if not is_noisy_chunk(c)]

                    logger.info(
                        "video %s: total_chunks=%d, clean_chunks=%d",
                        item.get("video_id"),
                        len(chunks),
                        len(good_chunks)
                    )

                    if not good_chunks:
                        skipped_videos += 1
                        continue

                    embeddings = get_embeddings(good_chunks)
                    store_embeddings(good_chunks, embeddings, item["video_id"])

                    total_chunks += len(good_chunks)

                st.success(f"✅ Indexed {total_chunks} chunks from {len(data)} videos")
                
                if skipped_videos > 0:
                    st.info(f"ℹ️ Skipped {skipped_videos} videos due to low-quality transcripts")

# =====================================
# 🔹 ASK SECTION
# =====================================
st.divider()
st.header("💬 Ask Questions")

question = st.text_input("Ask something about the videos")

if st.button("Ask"):
    if not question:
        st.warning("⚠️ Please enter a question")
    else:
        with st.spinner("🤖 Thinking..."):
            try:
                result = query_rag(question)

                # =====================
                # ANSWER
                # =====================
                st.subheader("🧠 Answer")
                st.write(result.get("answer", "No answer generated"))

                # =====================
                # SOURCES
                # =====================
                st.subheader("📚 Sources")
                sources = result.get("sources", [])

                if not sources:
                    st.info("No sources available")
                else:
                    for i, src in enumerate(sources, 1):
                        vid = src.get("video_id", "unknown")
                        idx = src.get("chunk_index", "N/A")

                        st.write(f"{i}. 📹 Video: `{vid}` | Chunk: `{idx}`")

                # =====================
                # CONTEXT (DEBUG)
                # =====================
                with st.expander("🔎 View Used Context"):
                    st.write(result.get("used_context", ""))

            except Exception as e:
                st.error("❌ Error while generating answer")
                st.exception(e)