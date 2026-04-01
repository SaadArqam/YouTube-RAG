import re
from typing import List


def clean_transcript(text: str) -> str:
    if not text:
        return ""


    text = re.sub(r"\[?\(?\d{1,2}:\d{2}(?::\d{2})?\)?\]?", " ", text)


    text = re.sub(r"\(music\)|\[music\]|\(laughs\)|\(applause\)", " ", text, flags=re.IGNORECASE)


    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:

    text = clean_transcript(text)
    sentences = split_sentences(text)

    chunks = []
    cur_chunk = []
    cur_words = 0

    for sent in sentences:
        words = sent.split()
        sent_len = len(words)


        if sent_len > chunk_size:

            if cur_chunk:
                chunks.append(" ".join(cur_chunk))

                cur_chunk = cur_chunk[-overlap:] if overlap < len(cur_chunk) else []
                cur_words = sum(len(s.split()) for s in cur_chunk)


            for i in range(0, sent_len, chunk_size):
                sub = words[i:i + chunk_size]
                chunks.append(" ".join(sub))

            continue


        if cur_words + sent_len > chunk_size and cur_chunk:
            chunks.append(" ".join(cur_chunk))

            new_start = []
            if overlap > 0:

                tail = " ".join(cur_chunk).split()[-overlap:]
                if tail:
                    new_start = [" ".join(tail)]

            cur_chunk = new_start.copy()
            cur_words = sum(len(s.split()) for s in cur_chunk)

        cur_chunk.append(sent)
        cur_words += sent_len

    if cur_chunk:
        chunks.append(" ".join(cur_chunk))

    return chunks


def is_noisy_chunk(chunk: str) -> bool:

    if not chunk:
        return True

    low = chunk.lower()
    noisy_keywords = ["sponsor", "sponsored", "ad break", "subscribe", "like this video", "buy now", "check out the link", "visit our sponsor"]


    if low.count("http") > 0 or low.count("www") > 0:
        return True

    for kw in noisy_keywords:
        if kw in low:
            return True

    return False