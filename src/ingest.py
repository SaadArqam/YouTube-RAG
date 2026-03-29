from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL


def get_video_ids(channel_url, max_videos=5):
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
    }

    video_ids = []

    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(channel_url, download=False)

        if "entries" in result:
            for entry in result["entries"][:max_videos]:
                video_ids.append(entry["id"])

    return video_ids


def get_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)

        text = " ".join([t.text for t in transcript])
        return text

    except Exception as e:
        print(f"Error fetching transcript for {video_id}: {e}")
        return None


def fetch_channel_transcripts(channel_url):
    video_ids = get_video_ids(channel_url)

    transcripts = []

    for vid in video_ids:
        text = get_transcript(vid)
        if text:
            transcripts.append({
                "video_id": vid,
                "text": text
            })

    return transcripts