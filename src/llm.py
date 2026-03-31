from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Tuple, Optional

# Load a compact FLAN-T5 model suitable for a laptop. google/flan-t5-base is OK but
# if memory is constrained consider 'google/flan-t5-small' (faster, less capable).
MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def _build_context_text(docs: List[str], metadatas: Optional[List[dict]] = None, max_chars: int = 3000, group_by_video: bool = False) -> str:
    """Assemble context from docs with lightweight source markers.

    If group_by_video=True the function will try to include a balanced amount of context
    per distinct video id so summarization prompts get coverage for each video.
    """
    if not docs:
        return ""

    # simple non-grouped assembly (preserve order)
    if not group_by_video:
        parts = []
        total = 0
        for i, doc in enumerate(docs):
            src = ""
            if metadatas and i < len(metadatas):
                m = metadatas[i]
                vid = m.get("video_id")
                idx = m.get("chunk_index")
                if vid is not None and idx is not None:
                    src = f"[src:{vid}_{idx}] "

            piece = src + doc.strip()
            if total + len(piece) > max_chars:
                remain = max_chars - total
                if remain <= 0:
                    break
                piece = piece[:remain]
                parts.append(piece)
                break

            parts.append(piece)
            total += len(piece)

        return "\n\n".join(parts)

    # group by video id and allocate per-video budget
    video_map = {}
    for i, doc in enumerate(docs):
        vid = None
        if metadatas and i < len(metadatas):
            vid = metadatas[i].get("video_id")
        vid = vid or f"unknown_{i}"
        video_map.setdefault(vid, []).append((i, doc))

    num_videos = len(video_map)
    if num_videos == 0:
        return ""

    per_video = max(200, max_chars // num_videos)
    parts = []

    for vid, items in video_map.items():
        # header for video
        parts.append(f"[video:{vid}]")
        total = 0
        for (i, doc) in items:
            src = ""
            if metadatas and i < len(metadatas):
                idx = metadatas[i].get("chunk_index")
                if idx is not None:
                    src = f"[src:{vid}_{idx}] "

            piece = src + doc.strip()
            if total + len(piece) > per_video:
                remain = per_video - total
                if remain <= 0:
                    break
                piece = piece[:remain]
                parts.append(piece)
                break

            parts.append(piece)
            total += len(piece)

    return "\n\n".join(parts)


def generate_answer(docs: List[str], question: str, metadatas: Optional[List[dict]] = None, mode: str = "qa") -> Tuple[str, str]:
    """Given retrieved docs (list) and a question, produce an answer and a short rationale.

    - The prompt forces the model to only use the provided context and to cite sources when relevant.
    - Returns (answer, used_context_snippet)
    """
    # build context according to mode
    if mode == "summarize":
        context = _build_context_text(docs, metadatas, max_chars=6000, group_by_video=True)

        prompt = f"""
        You are an assistant that summarizes videos. Use ONLY the information in the Context. Do not
        invent facts. For each video in the Context, produce a 1-2 sentence summary and cite the
        source marker(s) used (e.g. [src:VIDEOID_3]). After listing per-video summaries, provide a
        concise overall synthesis of the main themes across the videos (2-3 sentences).

        If a video has no relevant information in the Context, write 'No relevant information provided.'

        Context:
        {context}

        Question: {question}

        Output format:
        - Per-video summaries as numbered bullets with source markers.
        - Then an 'Overall summary:' paragraph.
        """.strip()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        outputs = model.generate(
            **inputs,
            max_length=400,
            num_beams=4,
            early_stopping=True,
            do_sample=False
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer, context

    # default QA mode
    context = _build_context_text(docs, metadatas, max_chars=3500, group_by_video=False)

    prompt = f"""
    You are an assistant that answers questions using only the provided context. If the answer is
    not contained in the context, say "I don't know". Be concise and prefer a short summary of the
    most relevant points. Cite sources inline using the source markers like [src:VIDEOID_INDEX].

    Context:
    {context}

    Question: {question}

    Provide:
    1) A short answer (2-4 sentences).
    2) A one-line citation list of the source markers used, e.g. [src:abc_3], or 'No sources'.
    """.strip()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

    outputs = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        early_stopping=True,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, context