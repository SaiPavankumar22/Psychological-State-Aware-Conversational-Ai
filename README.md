# Psychologically Adaptive AI Voice Assistant

A real-time conversational AI that listens to your voice, detects your emotional state across multiple modalities, and responds with psychologically adaptive language and voice — adjusting tone, pace, and style based on how your emotions are trending over the conversation.

> For full technical details, see [documentation.md](documentation.md).

---

## What It Does

- **Transcribes** speech with OpenAI Whisper (local, English)
- **Detects emotions** from text (28 classes via RoBERTa), voice (4 classes via Wav2Vec2), and audio events (22 categories via AST)
- **Fuses** all signals into a 4-dimensional psychological state: Valence, Arousal, Stress, Clarity
- **Tracks emotional trajectories** — adapts based on *where* emotions are heading, not just current values
- **Stores long-term memories** in Qdrant (episodic + semantic) for contextually aware responses
- **Generates responses** via LLM (Nebius Gemma 3 27B / OpenAI GPT-4o-mini fallback) informed by emotional state and memory
- **Speaks back** using Azure Neural TTS with dynamically chosen voice style, prosody, and volume
- **Streams responses** — text appears immediately while audio synthesizes in the background
- **Saves sessions** with a ChatGPT-like sidebar (create, switch, delete, search, export)

---

## Features

### Emotion-Adaptive Voice
The TTS controller adjusts six parameters in real-time:

- **Style** — empathetic, cheerful, serious, friendly, etc. (validated per-voice)
- **Style degree** — how strongly the style is applied
- **Rate** — slower for stress, faster for engagement
- **Pitch** — lower for empathy, higher for excitement
- **Volume** — softer for de-escalation, louder for celebration
- **Hysteresis gates** — prevent rapid style flip-flopping

### Long-Term Memory
- **Episodic memories** — conversation episodes with emotional context
- **Semantic memories** — extracted user facts, preferences, and behaviors
- **Reinforcement** — repeated similar experiences strengthen existing memories
- **Event-driven retrieval** — topic shifts, emotional shifts, explicit recall triggers
- **Graceful degradation** — conversations work without Qdrant; memory is optional

### Streaming LLM → TTS
The WebSocket protocol sends four progressive messages per turn:
1. `thinking` — processing started
2. `transcript` — user's speech transcribed
3. `response_text` — AI response text (displayed immediately)
4. `audio_ready` — TTS audio (played as soon as ready)

---

## Prerequisites

- Python 3.11+
- [Modal](https://modal.com) account with CLI installed
- OpenAI API key (for LLM fallback)
- Nebius API key (for primary LLM)
- Azure Cognitive Services Speech key + region (for TTS)
- Qdrant Cloud cluster or self-hosted Qdrant (for long-term memory)

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
# API keys for LLM providers
modal secret create emotion-env \
  OPENAI_API_KEY="sk-..." \
  NEBIUS_API_KEY="..." \
  AZURE_TTS_KEY="..." \
  AZURE_TTS_REGION="centralindia" \
  WHISPER_MODEL="small"

# Qdrant connection (Cloud — no QDRANT_PORT needed for HTTPS)
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
modal deploy backend/main.py
```

First deployment builds all Docker layers, pre-downloads ~2.2 GB of models, and sets up the environment. This takes ~3–5 minutes. Subsequent code-only redeployments take ~10 seconds because models are baked into Docker image layers.

```mermaid
graph TB
    L1["Layer 1: Base System<br/>Debian + git + ffmpeg + sox"]
    L2["Layer 2: Core Dependencies<br/>FastAPI + WebSockets + tenacity"]
    L3["Layer 3: ML/AI<br/>torch + transformers + sentence-transformers"]
    L4["Layer 4: Audio & API<br/>whisper + azure-tts + qdrant-client + librosa"]
    L5["Layer 4.5: Pre-downloaded Models<br/>Whisper small + RoBERTa + AST + SER + MiniLM"]
    L6["Layer 5: Application Code<br/>main.py + services/ + frontend/"]

    L1 --> L2 --> L3 --> L4 --> L5 --> L6

    style L6 fill:#8b5cfc,color:#fff
    style L5 fill:#6366f1,color:#fff
    style L4 fill:#4f46e5,color:#fff
    style L3 fill:#4338ca,color:#fff
    style L2 fill:#3730a3,color:#fff
    style L1 fill:#312e81,color:#fff
```

> 💡 Code changes only rebuild Layer 5 (~10s). Dependency changes rebuild from Layer 3+ (~3-5 min).

---

## Using the App

After deployment, Modal prints your app URL:
```
https://<your-username>--conversational-ai-realtime-fastapi-app.modal.run
```

1. Open that URL in a browser
2. Select a voice from the pill selector
3. Click **Start** and speak — the system records until you click **Stop**
4. Watch the transcript appear, then the AI response streams in word-by-word
5. Audio plays automatically once synthesized
6. Emotional state updates in real-time with circular progress indicators

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **Space** | Start / Stop recording |
| **Escape** | Cancel recording |

### Session Management

- Click **New Chat** (pen icon) in the sidebar to start a fresh session
- Previous sessions appear in the sidebar — click to switch
- Use the **search bar** to filter conversations by title
- Click the **✕** on any session to delete it
- Click **Export** to download all session data as JSON

### Debug Panels

Click **Debug** in the sidebar footer to show/hide the debug panels. Click the panel headers to expand/collapse:

- **Model Outputs** — ASR transcript, text emotion scores, speech emotion, audio events, acoustic features, fusion output
- **Memory & State** — session info, topic tracking, emotional trends with sparkline, dialogue state, long-term memory stats

---

## Local Development (optional)

To run with live reloading instead of a fixed deployment:

```bash
modal serve backend/main.py
```

Modal provides a temporary URL that hot-reloads on file changes.

---

## Project Structure

```
.
├── frontend/
│   ├── index.html              # HTML structure
│   ├── styles.css              # All CSS (design tokens, glassmorphism, responsive)
│   └── app.js                  # All JavaScript (WebSocket, sessions, UI updates)
├── backend/
│   ├── main.py                 # Modal app — WebSocket, API, session management
│   ├── conversation_state.py   # Dialogue state, intent detection, coherence
│   ├── emotional_trends.py     # Trend tracking, EMA, mode switching
│   ├── memory_manager.py       # Topic detection, retrieval policy
│   ├── memory_store.py         # Qdrant vector store with graceful degradation
│   ├── memory_orchestrator.py  # Memory worthiness, storage, extraction
│   └── services/
│       ├── services.py         # Whisper ASR, Text Emotion, SER, AST
│       ├── fusion.py           # Psychological state fusion engine
│       ├── llm_client.py       # Nebius primary + OpenAI fallback
│       └── tts_azure.py        # Azure Neural TTS, prosody controller
├── documentation.md            # Full technical reference
└── README.md
```

---

## Architecture

### Deployment Architecture

```mermaid
graph LR
    subgraph User
        BROWSER["🌐 Browser<br/>HTML + CSS + JS"]
        MIC["🎤 Microphone"]
        SPK["🔊 Speakers"]
    end

    subgraph Modal["Modal Cloud (A10G GPU)"]
        FASTAPI["FastAPI Server<br/>WebSocket + REST"]
        DOCKER["Docker Container<br/>5-layer cached image"]
        GPU["GPU Models<br/>Whisper + RoBERTa<br/>Wav2Vec2 + AST"]
    end

    subgraph External
        NEBIUS_["Nebius API<br/>Gemma 3 27B"]
        OPENAI_["OpenAI API<br/>GPT-4o-mini fallback"]
        AZURE_["Azure TTS<br/>Neural Voice"]
        QDRANT_["Qdrant Cloud<br/>Vector Store"]
    end

    MIC --> BROWSER
    BROWSER <-->|WebSocket| FASTAPI
    FASTAPI --> DOCKER --> GPU
    FASTAPI <--> NEBIUS_
    FASTAPI <--> OPENAI_
    FASTAPI <--> AZURE_
    FASTAPI <--> QDRANT_
    FASTAPI --> BROWSER
    BROWSER --> SPK
```

### System Overview

```mermaid
graph TB
    subgraph Frontend ["🖥️ Frontend (Browser)"]
        REC[Microphone Recording]
        WS[WebSocket Client]
        UI[UI Renderer]
        AUDIO[Audio Player]
    end

    subgraph Modal ["☁️ Modal (A10G GPU)"]
        subgraph Pipeline
            ASR[Whisper ASR]
            TEXTEMO[RoBERTa 28-class]
            SER[Wav2Vec2 SER]
            AST_[AudioSet AST]
            ACOUSTIC[Librosa Features]
            FUSION[Fusion Engine]
            TRENDS[Emotion Trends]
            LLM[Nebius / OpenAI]
            TTS[Azure Neural TTS]
        end
        subgraph Memory
            MEMORCH[Memory Orchestrator]
            QDRANT[(Qdrant Vector Store)]
        end
        SESSION[Session Manager]
    end

    subgraph APIs ["🔑 External APIs"]
        NEBIUS[Nebius API]
        OPENAI[OpenAI API]
        AZURE[Azure TTS API]
    end

    REC --> WS
    WS --> ASR
    WS --> SER
    WS --> AST_
    ASR --> TEXTEMO
    ASR --> ACOUSTIC
    TEXTEMO --> FUSION
    SER --> FUSION
    AST_ --> FUSION
    ACOUSTIC --> FUSION
    FUSION --> TRENDS
    TRENDS --> LLM
    LLM --> TTS
    LLM --> NEBIUS
    LLM --> OPENAI
    TTS --> AZURE
    TRENDS --> MEMORCH
    MEMORCH --> QDRANT
    TTS --> WS
    LLM --> WS
    WS --> UI
    WS --> AUDIO
    SESSION --> WS
```

### Fusion Engine

Multimodal signals are fused into four psychological dimensions:

```mermaid
graph LR
    subgraph Inputs
        TEXT["📝 Text Emotion<br/>(RoBERTa 28-class)"]
        SER_["🎙️ Speech Emotion<br/>(Wav2Vec2 4-class)"]
        AST__["🔊 Audio Events<br/>(AST 22-class)"]
        ACOS["🎵 Acoustic Features<br/>(Librosa)"]
    end

    subgraph Fusion["🧠 Psychological Fusion"]
        direction TB
        V["Valence<br/>[-1, +1]<br/>Negative ↔ Positive"]
        A["Arousal<br/>[0, 1]<br/>Calm ↔ Activated"]
        S["Stress<br/>[0, 1]<br/>Relaxed ↔ Stressed"]
        C["Clarity<br/>[0, 1]<br/>Confused ↔ Clear"]
    end

    subgraph Weights
        WV["Valence: Text 60%<br/>SER 20% | AST 20%"]
        WA["Arousal: SER 60%<br/>Text 20% | AST 20%"]
        WS_["Stress: Text 50%<br/>SER 30% | AST 20%"]
        WC["Clarity: Text 80%<br/>SER 10% | AST 10%"]
    end

    TEXT --> WV --> V
    SER_ --> WA --> A
    AST__ --> WS_ --> S
    ACOS --> C
    TEXT --> WS_ --> S
    TEXT --> WC --> C
    SER_ --> WV --> V
    AST__ --> WV --> V
```

| Dimension | Text | SER | AST | Meaning |
|-----------|------|-----|-----|---------|
| **Valence** | 60% | 20% | 20% | Negative ↔ Positive |
| **Arousal** | 20% | 60% | 20% | Calm ↔ Activated |
| **Stress** | 50% | 30% | 20% | Relaxed ↔ Stressed |
| **Clarity** | 80% | 10% | 10% | Confused ↔ Clear |

### Emotional Trend Tracking

```mermaid
graph TB
    subgraph Inputs
        INSTANT["Instant State<br/>(from Fusion)"]
    end

    subgraph Buffers["Circular Buffer (8 samples)"]
        CB["Time-weighted values<br/>+ exponential decay"]
    end

    subgraph Signals["Three Trend Signals"]
        INST["⚡ Instant Trend<br/>(last → current)<br/>Most responsive"]
        EMA_["📈 EMA Trend<br/>(α = 0.4)<br/>Smoothed"]
        REG["📊 Regression Trend<br/>(time-weighted linear)<br/>Most stable"]
    end

    subgraph Blend["Adaptive Blend"]
        EARLY["Early (n<4):<br/>70% Instant + 30% EMA"]
        MID["Mid (n<6):<br/>40% Instant + 40% EMA<br/>+ 20% Regression"]
        LATE["Late (n≥6):<br/>20% Instant + 40% EMA<br/>+ 40% Regression"]
    end

    subgraph Output
        MODE["Mode Decision<br/>Instant vs Trend<br/>(hysteresis gated)"]
        ADAPTIVE["Adaptive State<br/>→ TTS Controller<br/>→ LLM Prompt"]
    end

    INSTANT --> CB
    CB --> INST
    CB --> EMA_
    CB --> REG
    INST --> EARLY
    EMA_ --> EARLY
    INST --> MID
    EMA_ --> MID
    REG --> MID
    INST --> LATE
    EMA_ --> LATE
    REG --> LATE
    EARLY --> MODE
    MID --> MODE
    LATE --> MODE
    MODE --> ADAPTIVE
```

- **Hysteresis gates** prevent rapid mode switching (1.5s minimum dwell time)
- **JND thresholds** — rate changes <2%, pitch <1%, degree <0.05 are ignored (inaudible)
- **EMA smoothing** (α=0.75) on all prosody parameters prevents jarring voice changes

### Memory System

```mermaid
graph TB
    subgraph Storage
        USER["User Input + AI Response"]
        WORTH["Memory Worthiness<br/>Score = 0.3×Emotion<br/>+ 0.3×Topic Shift<br/>+ 0.2×Self-Reference<br/>+ 0.2×Novelty"]
        DECIDE{"Score > 0.45?<br/>Store memory?"}
        SIMILAR{"Similar memory<br/>exists? (≥0.85 sim)"}
        REINFORCE["🔄 Reinforce<br/>+0.1 importance<br/>+1 count"]
        STORE_E["💾 Store Episodic<br/>(conversation + emotion)"]
        SEM_EXTRACTION["🔬 Semantic Extraction<br/>(every 5 turns)<br/>LLM extracts facts"]
        STORE_S["💾 Store Semantic<br/>(preference / fact / behavior)"]
    end

    subgraph Retrieval
        TRIGGER{"Retrieval Trigger?<br/>• Explicit recall<br/>• Session start<br/>• Topic shift<br/>• Emotional shift<br/>• Rate limit (15s)"}
        SEARCH["Qdrant Vector Search<br/>+ Composite Scoring"]
        COMPOSE["Format for LLM<br/>Semantic + Episodic<br/>≤ 400 tokens"]
        INJECT["Inject into<br/>LLM Prompt"]
    end

    USER --> WORTH --> DECIDE
    DECIDE -->|Yes| SIMILAR
    SIMILAR -->|Yes| REINFORCE
    SIMILAR -->|No| STORE_E
    DECIDE -->|No| TRIGGER
    STORE_E --> TRIGGER
    USER --> SEM_EXTRACTION --> STORE_S
    STORE_S --> TRIGGER
    TRIGGER --> SEARCH --> COMPOSE --> INJECT
```

**Composite relevance score:**
```
score = 0.50 × semantic_similarity
      + 0.20 × importance_score
      + 0.20 × reinforcement_ratio
      + 0.10 × recency_decay (0.95^age_days)
```

### Interaction Mode Classification

```mermaid
graph TD
    START["Psychological State<br/>(v, a, s, cl, trends)"] --> Q1{"High stress<br/>AND v < -0.35?"}
    Q1 -->|Yes| DE["🔴 De-escalation<br/>calm style, soft, slow"]
    Q1 -->|No| Q2{"Trend toward<br/>distress?<br/>(s↑ + v↓)"}
    Q2 -->|Yes| DE
    Q2 -->|No| Q3{"Needs support?<br/>(s > 0.35 OR v < -0.15)"}
    Q3 -->|Yes| SUP["🟡 Support<br/>empathetic style, soft"]
    Q3 -->|No| Q4{"Low clarity?<br/>(cl < 0.40)"}
    Q4 -->|Yes| CLA["🔵 Clarification<br/>assistant style, clear"]
    Q4 -->|No| Q5{"Joyful + energized?<br/>(v > 0.3, a > 0.55)"}
    Q5 -->|Yes| CEL["🟢 Celebration<br/>cheerful style, loud"]
    Q5 -->|No| Q6{"Curious/engaged?<br/>(a > 0.35, v > 0.10)"}
    Q6 -->|Yes| ENG["🟢 Engaged<br/>friendly style, medium"]
    Q6 -->|No| NEU["⚪ Neutral<br/>assistant style, default"]
```

Each mode maps to a specific Azure TTS style, prosody profile, and volume level. The TTS controller further adjusts parameters based on utterance context (question vs empathy vs explanation).

### TTS Voice Adaptation

```mermaid
graph TB
    subgraph Input
        AS["Adaptive State<br/>(valence, arousal,<br/>stress, clarity, mode)"]
        RT["Response Text<br/>(for context detection)"]
        VN["Voice Name<br/>(per-voice validation)"]
    end

    subgraph Controller["TTS Controller (per-session)"]
        GATES["Hysteresis Gates<br/>stress | arousal<br/>valence | support"]
        MODE_["Interaction Mode<br/>deescalation | support<br/>clarification | celebration<br/>engaged | neutral"]
        CTX["Utterance Context<br/>question | empathy<br/>explanation | exclamation<br/>neutral"]
        STYLE["Style Selection<br/>(per-mode × context matrix)<br/>+ per-voice validation"]
        PROSODY["Prosody Calculation<br/>rate | pitch | degree<br/>+ EMA smoothing (α=0.75)<br/>+ JND thresholding"]
    end

    subgraph Output
        SSML["SSML Parameters<br/>style, styledegree,<br/>rate, pitch, volume"]
        SYNTH["Azure TTS Synthesis<br/>per-segment SSML<br/>+ <break> tags"]
    end

    AS --> GATES --> MODE_
    RT --> CTX
    MODE_ --> STYLE
    CTX --> STYLE
    AS --> PROSODY
    STYLE --> SSML
    PROSODY --> SSML
    VN --> STYLE
    SSML --> SYNTH
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Hosting | Modal (serverless GPU — A10G) |
| ASR | Local Whisper (`openai-whisper`) |
| Text Emotion | RoBERTa GoEmotions — 28 classes (HuggingFace) |
| Speech Emotion | Wav2Vec2 SUPERB — 4 classes (HuggingFace) |
| Audio Events | AudioSet AST — 22 categories (HuggingFace) |
| LLM | Nebius Gemma 3 27B (OpenAI GPT-4o-mini fallback) |
| TTS | Azure Neural TTS with adaptive prosody |
| Long-term Memory | Qdrant + sentence-transformers (MiniLM-L6-v2) |
| Web Framework | FastAPI + WebSockets |
| Audio Processing | ffmpeg, Librosa |
| Frontend | Vanilla JS, CSS Custom Properties, Glassmorphism UI |

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full license text.
