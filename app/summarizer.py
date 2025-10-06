from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

def summarize_text(text: str) -> str:
    """
    Summarize meeting transcript in bullet-point style.
    """
    if not text:
        return "No text available to summarize."

    prompt = f"summarize the meeting transcript in bullet points with actions and topics:\n{text}"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(device)
    summary_ids = model.generate(
        **inputs,
        max_length=200,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
