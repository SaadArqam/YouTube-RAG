import streamlit as st
from src.ingest import fetch_channel_transcripts
from src.chunking import chunk_text
from src.embeddings import get_embeddings
from src.vector_store import store_embeddings
from src.pipeline import query_rag

st.title("🎥 YouTube RAG")

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
                    chunks = chunk_text(item["text"])
                    embeddings = get_embeddings(chunks)
                    store_embeddings(chunks, embeddings, item["video_id"])

                    total_chunks += len(chunks)

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

        st.subheader("📚 Sources")
        for i, src in enumerate(result["sources"], 1):
            st.write(f"{i}. {src[:200]}...")