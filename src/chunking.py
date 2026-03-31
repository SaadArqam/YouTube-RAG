import re
from typing import List


def clean_transcript(text: str) -> str:
    """Basic cleaning of raw transcripts:
    - remove timestamps like [00:01:23] or 00:01:23
    - remove obvious non-speech markers like (music), [music]
    - collapse whitespace
    - remove common 'ad' / 'sponsor' segments markers (heuristic)
    """
    if not text:
        return ""

    # remove bracketed timestamps [00:01:23] or (00:01)
    text = re.sub(r"\[?\(?\d{1,2}:\d{2}(?::\d{2})?\)?\]?", " ", text)

    # remove common non-speech tokens
    text = re.sub(r"\(music\)|\[music\]|\(laughs\)|\(applause\)", " ", text, flags=re.IGNORECASE)


    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_sentences(text: str) -> List[str]:
    """A lightweight sentence splitter that prefers punctuation boundaries."""
    # split on sentence enders followed by space and capital letter or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """Chunk by sentence boundaries to avoid cutting sentences in half.

    - chunk_size: target words per chunk
    - overlap: words of overlap between chunks
    """
    text = clean_transcript(text)
    sentences = split_sentences(text)

    chunks = []
    cur_chunk = []
    cur_words = 0

    for sent in sentences:
        words = sent.split()
        sent_len = len(words)

        # if a single sentence is longer than chunk_size, split it by words
        if sent_len > chunk_size:
            # flush current chunk
            if cur_chunk:
                chunks.append(" ".join(cur_chunk))
                # prepare overlap
                cur_chunk = cur_chunk[-overlap:] if overlap < len(cur_chunk) else []
                cur_words = sum(len(s.split()) for s in cur_chunk)

            # break long sentence into smaller pieces
            for i in range(0, sent_len, chunk_size):
                sub = words[i:i + chunk_size]
                chunks.append(" ".join(sub))

            continue

        # if adding this sentence exceeds chunk size, flush and start new chunk with overlap
        if cur_words + sent_len > chunk_size and cur_chunk:
            chunks.append(" ".join(cur_chunk))
            # keep overlap
            new_start = []
            if overlap > 0:
                # take words from the end to serve as overlap (join then split)
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
    """Heuristic check to drop chunks that are likely ads/sponsors/outros."""
    if not chunk:
        return True

    low = chunk.lower()
    noisy_keywords = ["sponsor", "sponsored", "ad break", "subscribe", "like this video", "buy now", "check out the link", "visit our sponsor"]

    # if many URLs or short repeated tokens, treat as noisy
    if low.count("http") > 0 or low.count("www") > 0:
        return True

    for kw in noisy_keywords:
        if kw in low:
            return True

    return False