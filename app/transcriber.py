from faster_whisper import WhisperModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperModel("small", device=device)

def transcribe_audio(audio_path, language="en"):
    segments, _ = model.transcribe(audio_path, language=language)
    transcript = "\n".join(seg.text.strip() for seg in segments)
    return transcript
