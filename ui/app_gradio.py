import gradio as gr
import sys
import os
import json
from app.orchestrator import run_pipeline

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.langgraph_orchestrator import run_pipeline_agents, inspect_workflow
except Exception:
    run_pipeline_agents = None
    inspect_workflow = None


THEME = gr.themes.Soft(primary_hue="emerald", secondary_hue="cyan", neutral_hue="slate")


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body {
    font-family: 'Inter', sans-serif;
    background: radial-gradient(circle at top right, rgba(45, 212, 191, 0.15), transparent 55%),
                radial-gradient(circle at top left, rgba(59, 130, 246, 0.12), transparent 60%),
                #0f172a;
    color: #f8fafc;
}

.hero-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(14, 165, 233, 0.12));
    border: 1px solid rgba(148, 163, 184, 0.2);
    padding: 28px 32px;
    border-radius: 22px;
    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.25);
    backdrop-filter: blur(16px);
    margin-bottom: 18px;
}

.hero-card h1 {
    font-size: 2.35rem;
    margin-bottom: 12px;
    color: #e0f2fe;
}

.hero-card p {
    font-size: 1.15rem;
    color: #cbd5f5;
    line-height: 1.55;
}

.gradio-container {
    max-width: 1100px;
    margin: 0 auto;
}

label.svelte-1ipelgc {
    font-weight: 600 !important;
    letter-spacing: 0.02em;
}

.status-box textarea {
    font-size: 1.02rem !important;
    font-family: 'Inter', sans-serif !important;
}

.workflow-json {
    border-radius: 16px !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
}

.advanced-controls {
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 18px !important;
}

.run-button {
    font-size: 1.05rem !important;
    padding: 14px 18px !important;
    border-radius: 14px !important;
}

.clear-button {
    margin-left: 8px;
}

.gradio-button.secondary {
    background: rgba(15, 23, 42, 0.55) !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
}

.gradio-html .badge {
    display: inline-flex;
    align-items: center;
    padding: 6px 12px;
    border-radius: 12px;
    background: rgba(125, 211, 252, 0.15);
    border: 1px solid rgba(125, 211, 252, 0.25);
    color: #bae6fd;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.gradio-checkbox .svelte-uir3tn {
    font-size: 1.02rem !important;
}

.gradio-tabs .tab span {
    font-size: 1.05rem !important;
    font-weight: 500;
}

.output-card {
    background: rgba(15, 23, 42, 0.55);
    border-radius: 18px;
    padding: 22px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.08);
}
"""


def _initial_langgraph_status():
    if inspect_workflow is None:
        return {"error": "LangGraph orchestrator not available."}
    try:
        return inspect_workflow()
    except Exception as exc:  # pragma: no cover - inspection should not fail
        return {"error": f"Unable to inspect LangGraph: {exc}"}


def _refresh_langgraph_status():
    return _initial_langgraph_status()


def process_file(file, input_type, use_langchain=False, enable_logging=False):
    path = None
    if file is None:
        return "", "", "", None, None, "Awaiting file upload.", _initial_langgraph_status()
    try:
        path = file.name
    except Exception:
        path = file if isinstance(file, str) else None

    if path is None:
        return "", "", "", None, None, "Could not resolve uploaded file path.", _initial_langgraph_status()

    if enable_logging:
        os.environ["LOG_LANGGRAPH"] = "1"
    else:
        os.environ.pop("LOG_LANGGRAPH", None)

    status_messages = []
    workflow_state = None

    if use_langchain and run_pipeline_agents is not None:
        status_messages.append("LangGraph orchestration requested.")
        if inspect_workflow is not None:
            try:
                workflow_state = inspect_workflow()
            except Exception as exc:
                status_messages.append(f"Unable to inspect LangGraph before run: {exc}")
        try:
            result = run_pipeline_agents(path, input_type)
            status_messages.append("LangGraph agents completed successfully.")
        except Exception:
            import traceback

            traceback.print_exc()
            exc_info = sys.exc_info()[1]
            status_messages.append(
                f"LangGraph orchestration failed ({exc_info}). Falling back to deterministic pipeline."
            )
            result = run_pipeline(path, input_type=input_type)
    else:
        if use_langchain and run_pipeline_agents is None:
            status_messages.append("LangGraph orchestrator unavailable. Using deterministic pipeline.")
        result = run_pipeline(path, input_type=input_type)

    insights = result.get("insights", {}) if isinstance(result, dict) else {}
    raw_actions = insights.get("actions", []) if isinstance(insights, dict) else []
    raw_topics = insights.get("topics", []) if isinstance(insights, dict) else []

    def _format_action(action):
        if isinstance(action, str):
            return action
        if isinstance(action, dict):
            text = action.get("text") or ""
            assignees = action.get("assignees") or []
            due = action.get("due") or action.get("deadline") or []
            parts = [text]
            if assignees:
                parts.append("Assignees: " + ", ".join(map(str, assignees)))
            if due:
                due_val = due if isinstance(due, str) else ", ".join(map(str, due))
                parts.append("Due: " + due_val)
            return " \n".join(part for part in parts if part)
        return str(action)

    def _format_topic(topic):
        if isinstance(topic, str):
            return topic
        return json.dumps(topic)

    actions = "\n\n".join(_format_action(a) for a in raw_actions)
    topics = ", ".join(_format_topic(t) for t in raw_topics)

    if workflow_state is None and inspect_workflow is not None:
        try:
            workflow_state = inspect_workflow()
        except Exception as exc:
            status_messages.append(f"Unable to inspect LangGraph after run: {exc}")

    if workflow_state and workflow_state.get("missing_dependencies"):
        missing = workflow_state["missing_dependencies"]
        if missing:
            missing_text = ", ".join(f"{k}: {v}" for k, v in missing.items())
            status_messages.append(f"Missing dependencies detected: {missing_text}")

    summary = result.get("summary", "") if isinstance(result, dict) else ""
    audio = result.get("tts") if isinstance(result, dict) else None
    video = result.get("video") if isinstance(result, dict) else None
    status_text = "\n".join(status_messages) if status_messages else "Deterministic pipeline completed."

    return (
        summary,
        actions,
        topics,
        audio,
        video,
        status_text,
        workflow_state or _initial_langgraph_status(),
    )


def _clear_outputs():
    return (
        "",
        "",
        "",
        None,
        None,
        "Awaiting file upload.",
        _initial_langgraph_status(),
    )


INITIAL_LANGGRAPH_STATUS = _initial_langgraph_status()
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data.txt")
EXAMPLE_FILES = [[SAMPLE_PATH, "text"]] if os.path.exists(SAMPLE_PATH) else None


with gr.Blocks(theme=THEME, css=CUSTOM_CSS, analytics_enabled=False) as demo:
    gr.HTML(
        """
        <div class=\"hero-card\">
            <span class=\"badge\">Agentic Workflow</span>
            <h1>Meeting Summarizer Studio</h1>
            <p>
                Craft polished meeting recaps with LangGraph-driven agents orchestrating transcription,
                summarization, insight extraction, narration, and highlight video creation. Toggle advanced
                options to inspect the graph, enable verbose logging, and verify every agent executed as expected.
            </p>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=3):
            file_input = gr.File(label="Upload meeting asset (.wav, .mp4, .txt)")
            input_type = gr.Radio(["text", "audio", "video"], label="Input type", value="text")

            if EXAMPLE_FILES:
                gr.Examples(
                    examples=EXAMPLE_FILES,
                    inputs=[file_input, input_type],
                    label="Quick start examples",
                )

            with gr.Accordion("Advanced LangGraph controls", open=False, elem_classes="advanced-controls"):
                use_langgraph = gr.Checkbox(
                    label="Use LangGraph agent orchestration",
                    value=run_pipeline_agents is not None,
                )
                enable_logging = gr.Checkbox(
                    label="Enable verbose LangGraph logging",
                    value=False,
                    info="Sets LOG_LANGGRAPH=1 so node-level traces stream to the terminal.",
                )

            with gr.Row():
                run_btn = gr.Button(
                    "Generate meeting summary",
                    variant="primary",
                    elem_classes=["run-button"],
                )
                clear_btn = gr.Button(
                    "Clear",
                    variant="secondary",
                    elem_classes=["clear-button"],
                )

        with gr.Column(scale=2, min_width=380):
            workflow_out = gr.JSON(
                label="LangGraph status",
                value=INITIAL_LANGGRAPH_STATUS,
                elem_classes=["workflow-json"],
            )
            status_out = gr.Textbox(
                label="Run status",
                value="Awaiting file upload.",
                lines=4,
                interactive=False,
                elem_classes=["status-box"],
            )
            refresh_btn = gr.Button("Refresh status map", variant="secondary")

    with gr.Tabs():
        with gr.TabItem("Summary"):
            summary_out = gr.Textbox(
                label="Executive summary",
                lines=8,
                elem_classes=["output-card"],
            )
        with gr.TabItem("Action items"):
            actions_out = gr.Textbox(
                label="Action items",
                lines=8,
                elem_classes=["output-card"],
            )
        with gr.TabItem("Topics"):
            topics_out = gr.Textbox(
                label="Discussion topics",
                lines=6,
                elem_classes=["output-card"],
            )
        with gr.TabItem("Media"):
            with gr.Row():
                audio_out = gr.Audio(label="Narrated summary audio", interactive=False)
                video_out = gr.Video(label="Highlight reel", interactive=False)

    run_btn.click(
        process_file,
        inputs=[file_input, input_type, use_langgraph, enable_logging],
        outputs=[summary_out, actions_out, topics_out, audio_out, video_out, status_out, workflow_out],
        show_progress=True,
    )
    refresh_btn.click(
        _refresh_langgraph_status,
        inputs=[],
        outputs=workflow_out,
    )
    clear_btn.click(
        _clear_outputs,
        inputs=[],
        outputs=[summary_out, actions_out, topics_out, audio_out, video_out, status_out, workflow_out],
    )


if __name__ == "__main__":
    demo.queue().launch(inbrowser=True)
