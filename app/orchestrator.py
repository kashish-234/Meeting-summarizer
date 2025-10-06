from .transcriber import transcribe_audio
from .summarizer import summarize_text
from .extractor import extract_actions_and_topics
from .tts import generate_tts
from .video_composer import compose_video
from .agent_planner import create_plan

# Prefer the LangGraph orchestrator when available to demonstrate
# agentic orchestration. If LangGraph isn't installed, fall back to the
# original procedural pipeline.
try:
    from .langgraph_orchestrator import run_pipeline_agents
except Exception:
    run_pipeline_agents = None

def run_pipeline(file_path, input_type="audio"):
    # If LangGraph orchestrator is available, prefer it to demonstrate
    # agentic orchestration. Otherwise fallback to the procedural pipeline.
    if run_pipeline_agents is not None:
        try:
            return run_pipeline_agents(file_path, file_type=input_type)
        except Exception as e:
            print(f"[WARN] LangGraph orchestrator failed, falling back: {e}")

    plan = create_plan(input_type)
    outputs = {"transcript": None, "summary": None, "insights": None, "tts": None, "video": None}

    for step in plan["steps"]:
        if not step["run"]:
            continue
        tool = step["tool"]

        if tool == "transcribe":
            if input_type in ["audio", "video"]:
                outputs["transcript"] = transcribe_audio(file_path)
            elif input_type == "text":
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        outputs["transcript"] = f.read()
                except Exception as e:
                    outputs["transcript"] = None
                    print(f"[ERROR] Could not read text file: {e}")
            print("Transcript:", outputs["transcript"])

        elif tool == "summarize":
            if outputs["transcript"]:
                outputs["summary"] = summarize_text(outputs["transcript"])
            else:
                outputs["summary"] = "No transcript available to summarize."
            print("Summary:", outputs["summary"])

        elif tool == "extract":
            if outputs["transcript"]:
                outputs["insights"] = extract_actions_and_topics(outputs["transcript"])
            else:
                # Return a stable shape expected by the UI: dict with lists
                outputs["insights"] = {"actions": [], "topics": []}
            print("Insights:", outputs["insights"])

        elif tool == "tts":
            if outputs["summary"]:
                outputs["tts"] = generate_tts(outputs["summary"])
            else:
                outputs["tts"] = None

        elif tool == "compose_video":
            if outputs["summary"] and outputs["tts"]:
                outputs["video"] = compose_video(outputs["summary"], outputs["tts"])
            else:
                outputs["video"] = None

    return outputs
