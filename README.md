# Psychologically Adaptive AI Voice Assistant

A real-time conversational AI that listens to your voice, detects your emotional state across multiple modalities, and responds with psychologically adaptive language and voice — adjusting tone, pace, and style based on how your emotions are trending over the conversation.

> For full technical details, see [documentation.md](documentation.md).

---

## What It Does

- **Transcribes** speech with OpenAI Whisper (English only)
- **Detects emotions** from text (20 classes), voice (4 classes), and audio events (22 categories)
- **Fuses** signals into a 4-dimensional psychological state: Valence, Arousal, Stress, Clarity
- **Tracks emotional trajectories** — adapts based on *where* emotions are heading, not just current values
- **Stores memories** in Qdrant (episodic + semantic) for contextually aware long-term responses
- **Generates responses** via GPT-4o-mini informed by emotional state and memory
- **Speaks back** using Azure Neural TTS with dynamically chosen voice style and prosody
- **Saves sessions** with a ChatGPT-like sidebar (create, switch, delete sessions)

---

## Prerequisites

- Python 3.11+
- [Modal](https://modal.com) account with CLI installed
- OpenAI API key
- Azure Cognitive Services Speech key + region
- Qdrant Cloud cluster (or self-hosted Qdrant)

---

## Setup

### 1. Clone and install Modal

```bash
git clone <repo-url>
cd "Emotion psychologist"
pip install modal
modal setup        # authenticates your Modal account
```

### 2. Create Modal secrets

```bash
# API keys for OpenAI and Azure
modal secret create emotion-env \
  OPENAI_API_KEY="sk-..." \
  AZURE_SPEECH_KEY="..." \
  AZURE_SPEECH_REGION="eastus"

# Qdrant connection (Cloud example — no QDRANT_PORT needed for HTTPS URLs)
modal secret create qdrant-credentials \
  QDRANT_URL="https://your-cluster-id.region.cloud.qdrant.io" \
  QDRANT_API_KEY="..."

# Self-hosted Qdrant — include port
modal secret create qdrant-credentials \
  QDRANT_URL="http://your-host" \
  QDRANT_PORT="6333" \
  QDRANT_API_KEY="optional"
```

### 3. Deploy

```bash
modal deploy realtime_conversational_ai.py
```

First deployment builds all Docker layers and downloads ~2.2 GB of models. This takes ~3–5 minutes. Subsequent code-only redeployments take ~10 seconds.

---

## Using the App

After deployment, Modal prints your app URL:
```
https://<your-username>--conversational-ai-realtime-fastapi-app.modal.run
```

1. Open that URL in a browser
2. Select a voice from the dropdown
3. Click **Start** and speak — the system records until you click **Stop**
4. Wait ~2–3 seconds for the response
5. The AI replies in audio; all model outputs and emotional states are shown in the debug panels below

### Session Management

- Click **New Chat** in the left sidebar to start a fresh session
- Previous sessions appear in the sidebar — click to switch
- Click the trash icon on any session to delete it

---

## Verify Deployment

```bash
# Check health
curl https://<your-app>.modal.run/health
# → {"status": "healthy"}

# Check Qdrant connectivity
curl https://<your-app>.modal.run/readiness
# → {"status": "ready", "memory": "available"}

# Stream live logs
modal logs realtime_conversational_ai -f
```

---

## Local Development (optional)

To run with live reloading instead of a fixed deployment:

```bash
modal serve realtime_conversational_ai.py
```

Modal provides a temporary URL that hot-reloads on file changes.

---

## Project Structure

```
.
├── realtime_conversational_ai.py   # Main Modal app — WebSocket, API, frontend
├── services.py                     # ASR, Text Emotion, SER, AST models
├── fusion.py                       # Psychological state fusion
├── emotional_trends.py             # Trend tracking, EMA, mode switching
├── conversation_state.py           # Dialogue state, intent, coherence
├── memory_manager.py               # Topic detection, retrieval policy
├── memory_store.py                 # Qdrant vector store (episodic + semantic)
├── memory_orchestrator.py          # Memory worthiness, storage, extraction
├── llm_client.py                   # GPT-4o-mini client, prompt construction
├── tts_azure.py                    # Azure Neural TTS, prosody controller
└── documentation.md                # Full technical reference
```

---

## Troubleshooting

**Qdrant not connecting**
Check that `QDRANT_URL` is the full HTTPS URL and `QDRANT_API_KEY` is correct. For cloud URLs, do not set `QDRANT_PORT`. The system degrades gracefully — conversations still work without long-term memory.

**WebSocket closes immediately**
Check browser console for the close code: `1008` = client-side issue (bad JSON, rate limit), `1011` = server-side issue (see Modal logs).

**Cold start takes a long time**
Only the first deploy or a dependency change triggers a full rebuild. Code-only changes redeploy in ~10 seconds.

For all other issues, see the [Troubleshooting section in documentation.md](documentation.md#14-troubleshooting) or check logs:
```bash
modal logs realtime_conversational_ai -f
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Hosting | Modal (serverless GPU — A10G) |
| ASR | OpenAI Whisper API |
| Text Emotion | RoBERTa GoEmotions (HuggingFace) |
| Speech Emotion | Wav2Vec2 SUPERB (HuggingFace) |
| Audio Events | AudioSet AST (HuggingFace) |
| LLM | GPT-4o-mini |
| TTS | Azure Neural TTS |
| Long-term Memory | Qdrant + sentence-transformers |
| Web Framework | FastAPI + WebSockets |
| Audio Processing | ffmpeg, Librosa |

---

## License

Research and educational use.
