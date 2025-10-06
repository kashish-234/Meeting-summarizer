def create_plan(input_type):
    # Ensure text inputs also run the "transcribe" step. The orchestrator
    # implements reading text files when input_type == "text" so transcribe
    # must be enabled for text as well as audio/video.
    return {
        "steps": [
            {"tool": "transcribe", "run": input_type in ["audio", "video", "text"]},
            {"tool": "summarize", "run": True},
            {"tool": "extract", "run": True},
            {"tool": "tts", "run": True},
            {"tool": "compose_video", "run": True}
        ]
    }
