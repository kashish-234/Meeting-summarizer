import pyttsx3

def generate_tts(text, out_path="summary.wav", voice=None, rate=150):
    engine = pyttsx3.init()
    if voice:
        engine.setProperty("voice", voice)
    engine.setProperty("rate", rate)
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    return out_path
