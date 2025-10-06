# Meeting Summarizer with LangGraph Agents

This project transforms **multimodal meeting artifacts** — including audio, video, or raw text notes — into **structured, actionable summaries** using a **LangGraph-orchestrated multi-agent pipeline** powered by **Google Gemini 2.5 Pro**.  

The system can transcribe conversations, identify speakers, extract key decisions, generate tasks, synthesize spoken summaries via TTS, and even compose short highlight videos using MoviePy.  
A **Gradio UI** allows teams to trigger the workflow interactively and monitor each agent’s execution status through LangGraph’s real-time visualization.

---

## 🧠 How It Works — LangGraph + Agents Overview

The summarizer runs a **five-node LangGraph workflow** that mirrors the deterministic pipeline used elsewhere in the project. Each node focuses on one responsibility and hands its output to the next step:

| Agent | Description | Input → Output |
|--------|-------------|----------------|
| 🎧 **Transcribe** | Converts audio/video to text or reads text files directly. | Media/Text → Transcript |
| 🧩 **Summarize** | Generates a concise meeting recap with Gemini (falls back to local summarizer if needed). | Transcript → Summary |
| � **Insights** | Extracts action items and discussion topics via spaCy + sentence transformers (or Gemini JSON). | Transcript → `{actions, topics}` |
| � **Narrate** | Produces a spoken version of the summary using the configured TTS engine. | Summary → Audio file |
| 🎬 **Compose Video** | Renders a highlight reel with animated slides and narrated audio via MoviePy. | Summary + Audio → Video |

### 🔗 LangGraph Integration
- **Sequential graph:** Nodes execute in order with automatic skips if results already exist.
- **LLM optionality:** If Gemini is unavailable, the graph falls back to deterministic summarization and insight extraction.
- **Deterministic safety net:** Missing audio/video assets trigger regeneration using the local pipeline to guarantee complete outputs.
- **Live inspection:** The Gradio UI exposes current node status and dependency health through the `inspect_workflow()` endpoint.

#### Example Flow
Audio / Video / Text → Transcribe → Summarize → Insights → Narrate → Compose Video → Summary + Media Bundle


---

## Key Components

- **LangGraph Orchestrator**: `app/langgraph_orchestrator.py` coordinates the end-to-end agent sequence  
  (transcribe → summarize → analyze → TTS → compose).  
  The graph automatically falls back to the deterministic pipeline if any optional dependency is missing.

- **Deterministic Fallback Pipeline**: `app/orchestrator.py` mirrors the classic behavior so the UI and CLI always return usable outputs.

- **Interactive UI**: `ui/app_gradio.py` exposes toggles for LangGraph orchestration, verbose logging, and live workflow inspection.

---

## Prerequisites

| Requirement | Notes |
|--------------|-------|
| Python 3.11+ | Project developed on Python 3.12 |
| FFmpeg | Required by MoviePy for audio/video composition |
| Google Gemini API Key | Optional but recommended; set `GOOGLE_API_KEY` for LLM-enhanced steps |

Install dependencies locally:

```powershell
python -m venv venv
venv\Scripts\Activate
pip install --upgrade pip
pip install -r requirements.txt

```
Launch the Gradio UI:

```powershell
python -m ui.app_gradio
```