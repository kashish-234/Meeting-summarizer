# Meeting Summarizer with LangGraph Agents

This project transforms **multimodal meeting artifacts** — including audio, video, or raw text notes — into **structured, actionable summaries** using a **LangGraph-orchestrated multi-agent pipeline** powered by **Google Gemini 2.5 Pro**.  

The system can transcribe conversations, identify speakers, extract key decisions, generate tasks, synthesize spoken summaries via TTS, and even compose short highlight videos using MoviePy.  
A **Gradio UI** allows teams to trigger the workflow interactively and monitor each agent’s execution status through LangGraph’s real-time visualization.

---

## 🧠 How It Works — LangGraph + Agents Overview

The summarizer uses a **graph-based orchestration layer (LangGraph)** to coordinate a set of specialized AI agents.  
Each agent focuses on a single capability, while LangGraph manages the flow of data, context, and fallbacks between them.

| Agent | Description | Input → Output |
|--------|--------------|----------------|
| 🎧 **Transcription Agent** | Converts raw audio/video into text via ASR (e.g., Whisper). | Audio/Video → Text |
| 🗣️ **Speaker Identification Agent** | Detects and labels speakers using embeddings or metadata. | Transcript → Speaker-tagged text |
| 🧩 **Contextual Summarizer Agent** | Generates concise, structured meeting summaries using Gemini 2.5 Pro. | Transcript → Key insights |
| 📝 **Action Item Extraction Agent** | Identifies follow-ups, decisions, owners, and deadlines. | Summary → Task list (JSON) |
| 💬 **Sentiment & Engagement Analyzer** | Evaluates tone, sentiment, and participation balance. | Transcript → Sentiment metrics |
| 📚 **Knowledge Enrichment Agent (optional)** | Retrieves background context through RAG (retrieval-augmented generation). | Transcript → Enriched summary |
| 🕸️ **LangGraph Orchestrator (Master Agent)** | Connects and supervises all sub-agents, handling dependencies, retries, and memory. | Agents → Unified output |

### 🔗 LangGraph Integration
- **Graph-based workflow:** Each agent is a LangGraph node with explicit inputs and outputs.  
- **Memory management:** Persistent `ConversationBufferMemory` maintains context across stages.  
- **Parallel execution:** Compatible agents (e.g., sentiment and action-item extraction) run concurrently for speed.  
- **Fallback logic:** If a node fails (e.g., TTS unavailable), deterministic summarization ensures continuity.  
- **Visualization:** Gradio UI displays live LangGraph node execution and logs for transparency.

#### Example Flow
Audio / Video / Text
↓
[Transcription Agent]
↓
[Speaker ID Agent]
↓
[Summarizer Agent]
↓
[Action Item + Sentiment Agents]
↓
[TTS / Highlight Composer]
↓
Structured Report + Media Output


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