from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load FLAN-T5 properly (not pipeline)
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def generate_answer(context, question):
    # 🔥 Strong prompt
    prompt = f"""
    Answer the question based only on the context below.

    Context:
    {context}

    Question:
    {question}

    Give a clear answer in 3-4 sentences.
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    outputs = model.generate(
        **inputs,
        max_length=200,
        num_beams=4,
        early_stopping=True
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer