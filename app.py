import streamlit as st
from src.ingest import fetch_channel_transcripts

st.title("🎥 YouTube RAG (Step 1)")

channel_url = st.text_input("Enter YouTube Channel URL")

if st.button("Fetch Transcripts"):
    with st.spinner("Fetching transcripts..."):
        data = fetch_channel_transcripts(channel_url)

    st.success(f"Fetched {len(data)} videos")

    for item in data:
        st.subheader(f"Video ID: {item['video_id']}")
        st.write(item["text"][:500] + "...")