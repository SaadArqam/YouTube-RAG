import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch
from typing import List, Tuple, Optional

# =====================================
# 🔥 ENV DETECTION (IMPORTANT)
# =====================================
IS_DEPLOY = os.getenv("RENDER", "false") == "true"

# =====================================
# 🔥 MODEL LOADING
# =====================================

if IS_DEPLOY:
    # ✅ Lightweight model for Render (low memory)
    generator = pipeline(
        "text-generation",
        model="sshleifer/tiny-gpt2"
    )
else:
    # ✅ Better model for local development
    MODEL_NAME = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


# =====================================
# 🔥 CONTEXT BUILDER
# =====================================
def build_context(docs: List[str], max_chars: int = 3000) -> str:
    context = ""
    total = 0

    for doc in docs:
        if total + len(doc) > max_chars:
            break
        context += doc.strip() + "\n\n"
        total += len(doc)

    return context


# =====================================
# 🔥 MAIN GENERATE FUNCTION
# =====================================
def generate_answer(
    docs: List[str],
    question: str,
    metadatas: Optional[List[dict]] = None,
    mode: str = "qa"
) -> Tuple[str, str]:

    # Keep context small for stability
    context = build_context(docs[:3])

    prompt = f"""
You are an AI assistant.

Answer ONLY based on the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

    # =====================================
    # 🚀 DEPLOY MODE (LIGHTWEIGHT)
    # =====================================
    if IS_DEPLOY:
        response = generator(
            prompt,
            max_length=150,
            num_return_sequences=1
        )

        output = response[0]["generated_text"]

        # Clean output
        if "Answer:" in output:
            output = output.split("Answer:")[-1].strip()

        return output, context

    # =====================================
    # 💻 LOCAL MODE (BETTER QUALITY)
    # =====================================
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=200,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer, context