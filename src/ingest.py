from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL


def get_video_ids(channel_url, max_videos=5):
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "skip_download": True,
    }

    video_ids = []   # ✅ ONLY THIS — no recursive call

    with YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(channel_url, download=False)

        if "entries" in result:
            for entry in result["entries"][:max_videos]:
                video_id = entry.get("id")
                if video_id and len(video_id) == 11:
                    video_ids.append(video_id)

    return video_ids


def get_transcript(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)

        # ✅ Convert transcript objects to plain text
        text = " ".join([t.text for t in transcript])
        return text

    except Exception:
        # ✅ Silent fail (normal for many videos)
        return None


def fetch_channel_transcripts(channel_url):
    video_ids = get_video_ids(channel_url)

    print("VIDEO IDS:", video_ids)   # 👈 ADD THIS

    transcripts = []

    for vid in video_ids:
        text = get_transcript(vid)

        if text:
            transcripts.append({
                "video_id": vid,
                "text": text
            })

    print("TRANSCRIPTS COUNT:", len(transcripts))   # 👈 ADD THIS

    return transcripts