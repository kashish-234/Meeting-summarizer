"""LangGraph-powered meeting summarization orchestration.

This module connects the pipeline agents (transcribe, summarize, analyze,
text-to-speech, and video compose) using a LangGraph state machine. When the
LangGraph/LangChain stack or Gemini isn't available, it falls back to the local
deterministic pipeline so callers always receive summary, insights, audio, and
video paths.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, TypedDict, Any, Iterable, Tuple

_MISSING_DEPENDENCIES: Dict[str, str] = {}

try:
    from .transcriber import transcribe_audio
except Exception as exc:  # pragma: no cover - optional dependency
    transcribe_audio = None  # type: ignore
    _MISSING_DEPENDENCIES["transcriber"] = str(exc)

try:
    from .summarizer import summarize_text
except Exception as exc:  # pragma: no cover - optional dependency
    summarize_text = None  # type: ignore
    _MISSING_DEPENDENCIES["summarizer"] = str(exc)

try:
    from .extractor import extract_actions_and_topics
except Exception as exc:  # pragma: no cover - optional dependency
    extract_actions_and_topics = None  # type: ignore
    _MISSING_DEPENDENCIES["extractor"] = str(exc)

try:
    from .tts import generate_tts
except Exception as exc:  # pragma: no cover - optional dependency
    generate_tts = None  # type: ignore
    _MISSING_DEPENDENCIES["tts"] = str(exc)

try:
    from .video_composer import compose_video
except Exception as exc:  # pragma: no cover - optional dependency
    compose_video = None  # type: ignore
    _MISSING_DEPENDENCIES["video_composer"] = str(exc)

try:
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LANGGRAPH_AVAILABLE = False
    END = None  # type: ignore
    StateGraph = None  # type: ignore

try:
    from langchain_google_genai import GoogleGenerativeAI
except Exception:  # pragma: no cover - optional dependency
    GoogleGenerativeAI = None  # type: ignore


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
_LLM: Optional[Any] = None
if GoogleGenerativeAI is not None and GOOGLE_API_KEY:
    try:
        _LLM = GoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY)
    except Exception as exc:  # pragma: no cover - runtime configuration error
        print(f"[WARN] Failed to initialize GoogleGenerativeAI: {exc}")


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.lower() in {"1", "true", "yes", "on"}


LOG_LANGGRAPH = _env_flag("LOG_LANGGRAPH")


def _log(message: str) -> None:
    if LOG_LANGGRAPH:
        print(f"[LangGraph] {message}")


def _truncate(value: Optional[str], limit: int = 160) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _dependency_error(name: str) -> RuntimeError:
    detail = _MISSING_DEPENDENCIES.get(name)
    if detail:
        return RuntimeError(f"Dependency '{name}' is not available: {detail}")
    return RuntimeError(f"Dependency '{name}' is not available")


class PipelineState(TypedDict, total=False):
    file_path: str
    file_type: str
    transcript: Optional[str]
    summary: Optional[str]
    insights: Optional[Dict[str, Any]]
    tts: Optional[str]
    video: Optional[str]


def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _transcribe(file_path: str, file_type: str) -> str:
    lowered = (file_type or "").lower()
    extension = os.path.splitext(file_path)[1].lower()

    if lowered == "text" or extension in {".txt", ".md"}:
        return _read_text_file(file_path)

    if lowered in {"audio", "video"} or extension in {".wav", ".mp3", ".m4a", ".mp4", ".mov", ".mkv"}:
        if transcribe_audio is None:
            raise _dependency_error("transcriber")
        return transcribe_audio(file_path)

    # Default: try reading as text
    return _read_text_file(file_path)


def _fallback_pipeline(file_path: str, file_type: str) -> Dict[str, Optional[str]]:
    _log("Executing deterministic fallback pipeline")
    transcript = _transcribe(file_path, file_type)
    if summarize_text is None:
        raise _dependency_error("summarizer")
    summary = summarize_text(transcript)
    if extract_actions_and_topics is None:
        raise _dependency_error("extractor")
    insights = extract_actions_and_topics(transcript)
    if generate_tts is None:
        raise _dependency_error("tts")
    audio_path = generate_tts(summary) if summary else None
    if compose_video is None:
        raise _dependency_error("video_composer")
    video_path = compose_video(summary, audio_path) if summary and audio_path else None

    _log(
        "Fallback pipeline produced summary length="
        f"{len(summary or '')}, insights keys={list((insights or {}).keys())}"
    )

    return {
        "file": file_path,
        "transcript": transcript,
        "summary": summary,
        "insights": insights,
        "tts": audio_path,
        "video": video_path,
    }


def _invoke_llm(prompt: str) -> Optional[str]:
    if _LLM is None:
        return None
    try:
        response = _LLM.invoke(prompt)
    except Exception as exc:  # pragma: no cover - runtime failure
        print(f"[WARN] Gemini invocation failed: {exc}")
        return None

    if isinstance(response, str):
        return response.strip()

    if hasattr(response, "content"):
        content = getattr(response, "content")
        if isinstance(content, list) and content:
            part = content[0]
            if isinstance(part, dict):
                text = part.get("text") or part.get("value")
                if isinstance(text, str):
                    return text.strip()
        if isinstance(content, str):
            return content.strip()

    if hasattr(response, "text") and isinstance(response.text, str):
        return response.text.strip()

    try:
        return json.dumps(response)
    except Exception:  # pragma: no cover - fallback to string repr
        return str(response)


def _node_transcribe(state: PipelineState) -> Dict[str, Optional[str]]:
    if state.get("transcript"):
        return {}
    _log("Node 'transcribe' starting")
    transcript = _transcribe(state["file_path"], state.get("file_type", ""))
    _log(f"Node 'transcribe' produced {len(transcript)} characters")
    return {"transcript": transcript}


def _node_summarize(state: PipelineState) -> Dict[str, Optional[str]]:
    if state.get("summary") or not state.get("transcript"):
        return {}

    _log("Node 'summarize' starting")
    prompt = (
        "Summarize the following meeting transcript in 4-6 concise sentences. "
        "Highlight key decisions and action items.\n\n" + state["transcript"]
    )
    summary = _invoke_llm(prompt)
    if not summary:
        if summarize_text is None:
            raise _dependency_error("summarizer")
        summary = summarize_text(state["transcript"])
        _log("Node 'summarize' used deterministic summarizer")
    else:
        _log(f"Node 'summarize' received LLM summary preview='{_truncate(summary)}'")
    return {"summary": summary}


def _node_analyze(state: PipelineState) -> Dict[str, Optional[Dict[str, Any]]]:
    if state.get("insights") or not state.get("transcript"):
        return {}

    _log("Node 'analyze' starting")
    prompt = (
        "Extract two lists from the meeting transcript. "
        "Return ONLY JSON with keys 'actions' (list of strings) and 'topics' (list of strings).\n"
        "Transcript:\n" + state["transcript"]
    )
    raw = _invoke_llm(prompt)
    if raw:
        cleaned = raw.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        try:
            insights = json.loads(cleaned)
            if isinstance(insights, dict):
                _log(
                    "Node 'analyze' extracted insights via LLM"
                    f" (actions={len(insights.get('actions', []))},"
                    f" topics={len(insights.get('topics', []))})"
                )
                return {"insights": insights}
        except json.JSONDecodeError:
            print("[WARN] Gemini analyze response was not valid JSON; falling back to spaCy extractor")

    _log("Node 'analyze' falling back to spaCy extractor")
    if extract_actions_and_topics is None:
        raise _dependency_error("extractor")
    return {"insights": extract_actions_and_topics(state["transcript"])}


def _node_tts(state: PipelineState) -> Dict[str, Optional[str]]:
    if state.get("tts") or not state.get("summary"):
        return {}
    _log("Node 'tts' starting")
    if generate_tts is None:
        raise _dependency_error("tts")
    audio_path = generate_tts(state["summary"])
    _log(f"Node 'tts' generated audio at '{audio_path}'")
    return {"tts": audio_path}


def _node_compose(state: PipelineState) -> Dict[str, Optional[str]]:
    if state.get("video") or not state.get("summary") or not state.get("tts"):
        return {}
    _log("Node 'compose' starting")
    if compose_video is None:
        raise _dependency_error("video_composer")
    video_path = compose_video(state["summary"], state["tts"])
    _log(f"Node 'compose' generated video at '{video_path}'")
    return {"video": video_path}


_GRAPH_NODES: Tuple[str, ...] = ("transcribe", "summarize", "analyze", "tts", "compose")
_GRAPH_EDGES: Tuple[Tuple[str, str], ...] = (
    ("transcribe", "summarize"),
    ("summarize", "analyze"),
    ("analyze", "tts"),
    ("tts", "compose"),
    ("compose", "__END__"),
)

_WORKFLOW = None


def _get_workflow():
    global _WORKFLOW
    if _WORKFLOW is not None:
        return _WORKFLOW

    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError("LangGraph is not available")

    graph = StateGraph(PipelineState)
    graph.add_node("transcribe", _node_transcribe)
    graph.add_node("summarize", _node_summarize)
    graph.add_node("analyze", _node_analyze)
    graph.add_node("tts", _node_tts)
    graph.add_node("compose", _node_compose)

    for start, end in _GRAPH_EDGES:
        if end == "__END__":
            graph.add_edge(start, END)
        else:
            graph.add_edge(start, end)

    graph.set_entry_point("transcribe")
    _WORKFLOW = graph.compile()
    _log("LangGraph workflow compiled")
    return _WORKFLOW


def inspect_workflow() -> Dict[str, Any]:
    edges: Iterable[Dict[str, str]] = (
        {"from": start, "to": "END" if end == "__END__" else end}
        for start, end in _GRAPH_EDGES
    )
    return {
        "langgraph_available": LANGGRAPH_AVAILABLE,
        "workflow_compiled": _WORKFLOW is not None,
        "log_enabled": LOG_LANGGRAPH,
        "has_llm": _LLM is not None,
        "google_api_key_configured": bool(GOOGLE_API_KEY),
        "nodes": list(_GRAPH_NODES),
        "edges": list(edges),
        "fallback_enabled": True,
        "missing_dependencies": _MISSING_DEPENDENCIES,
    }


def run_pipeline_agents(file_path: str, file_type: str) -> Dict[str, Optional[str]]:
    """Run the LangGraph-orchestrated pipeline and return all outputs."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if not LANGGRAPH_AVAILABLE:
        print("[WARN] LangGraph not available; using deterministic fallback")
        return _fallback_pipeline(file_path, file_type)

    workflow = _get_workflow()

    initial_state: PipelineState = {"file_path": file_path, "file_type": file_type}
    _log(f"Invoking workflow with file='{file_path}', type='{file_type}'")
    final_state = workflow.invoke(initial_state)
    _log("Workflow completed")

    transcript = final_state.get("transcript")
    summary = final_state.get("summary")
    insights = final_state.get("insights")
    audio_path = final_state.get("tts")
    video_path = final_state.get("video")

    if not transcript or not summary:
        _log("Core outputs missing; switching to fallback pipeline")
        # If the graph failed to produce core outputs, fall back to deterministic pipeline
        return _fallback_pipeline(file_path, file_type)

    if not isinstance(insights, dict):
        _log("Insights missing or invalid; regenerating with spaCy extractor")
        if extract_actions_and_topics is None:
            raise _dependency_error("extractor")
        insights = extract_actions_and_topics(transcript)

    if not audio_path or not video_path:
        _log("Audio or video missing; regenerating final assets deterministically")
        if generate_tts is None or compose_video is None:
            raise _dependency_error("video_composer" if compose_video is None else "tts")
        audio_path = generate_tts(summary)
        video_path = compose_video(summary, audio_path) if audio_path else None

    return {
        "file": file_path,
        "transcript": transcript,
        "summary": summary,
        "insights": insights,
        "tts": audio_path,
        "video": video_path,
    }


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="LangGraph meeting summarizer orchestrator")
    parser.add_argument("file_path", nargs="?", help="Path to the meeting asset (audio/video/text)")
    parser.add_argument("file_type", nargs="?", default="auto", help="Type hint: audio, video, text")
    parser.add_argument("--inspect", action="store_true", help="Print workflow details and exit")
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable verbose LangGraph logging for this invocation (overrides LOG_LANGGRAPH env flag)",
    )

    args = parser.parse_args()

    if args.log:
        globals()["LOG_LANGGRAPH"] = True
        print("[LangGraph] Verbose logging enabled (--log)")

    if args.inspect:
        print(json.dumps(inspect_workflow(), indent=2))
        if not args.file_path:
            raise SystemExit(0)

    if not args.file_path or not args.file_type:
        parser.print_help()
        raise SystemExit(1)

    result = run_pipeline_agents(args.file_path, args.file_type)
    print(json.dumps(result, indent=2))
