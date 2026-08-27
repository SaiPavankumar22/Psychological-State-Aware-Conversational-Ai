# Documentation — Psychologically Adaptive AI Voice Assistant

Complete technical reference for the system architecture, components, configuration, API, and internals.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Components](#3-components)
   - [services.py](#31-servicespy--multimodal-emotion-detection)
   - [fusion.py](#32-fusionpy--psychological-state-fusion)
   - [emotional_trends.py](#33-emotional_trendspy--trend-tracking)
   - [conversation_state.py](#34-conversation_statepy--dialogue-management)
   - [memory_manager.py](#35-memory_managerpy--event-driven-retrieval)
   - [memory_store.py](#36-memory_storepy--qdrant-vector-store)
   - [memory_orchestrator.py](#37-memory_orchestratorpy--memory-orchestration)
   - [llm_client.py](#38-llm_clientpy--llm-integration)
   - [tts_azure.py](#39-tts_azurepy--voice-synthesis)
   - [realtime_conversational_ai.py](#310-realtime_conversational_aipy--modal-application)
4. [Data Flow — Per Turn](#4-data-flow--per-turn)
5. [WebSocket Protocol](#5-websocket-protocol)
6. [REST API Reference](#6-rest-api-reference)
7. [Configuration Reference](#7-configuration-reference)
8. [Hybrid Memory System](#8-hybrid-memory-system)
9. [Session Management](#9-session-management)
10. [Dynamic Acoustic Features](#10-dynamic-acoustic-features)
11. [Frontend UI](#11-frontend-ui)
12. [Infrastructure & Deployment](#12-infrastructure--deployment)
13. [Performance Characteristics](#13-performance-characteristics)
14. [Troubleshooting](#14-troubleshooting)
15. [Extending the System](#15-extending-the-system)
16. [Academic Foundation](#16-academic-foundation)

---

## 1. System Overview

This system is a **research-grade, real-time conversational AI** that:

- Transcribes user speech (OpenAI Whisper)
- Detects emotions from **28 text emotions** (RoBERTa), voice (Wav2Vec2), and audio events (AudioSet AST)
- Sanitizes Whisper transcripts (repetition collapse, Unicode cleanup)
- Extracts acoustic features dynamically from each audio clip
- Fuses signals into a 4-dimensional psychological state (Valence, Arousal, Stress, Clarity)
- Tracks emotional **trajectories** over time — not just instant values
- Stores and retrieves long-term memories via Qdrant vector database with auto-reconnect
- Generates a psychologically adaptive LLM response (GPT-4o-mini, streaming)
- Synthesizes speech with an emotionally adaptive voice (Azure Neural TTS) with per-voice style validation
- Falls back to browser SpeechSynthesis when Azure TTS fails
- Streams responses in 3 steps: thinking → text → audio (cuts perceived latency by 2-4s)
- Maintains full session history (ChatGPT-like sidebar with search, export)
- Features a premium glassmorphism UI with emotion sparklines and keyboard shortcuts

The key innovation is **trend-based adaptation**: the system responds to where emotions are heading, not just where they are right now.

---

## 2. Architecture

```
USER SPEAKS (Browser)
       │
       ▼
WebM Audio → FFmpeg → 16kHz Mono WAV
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
ASR + Text Emotion                 Audio Emotion
  • Whisper API (ASR)                • Wav2Vec2 (SER)
  • RoBERTa GoEmotions               • AudioSet AST
  Output: transcript,                Output: ser_dict,
          20 text emotions                   ast events
       │                                 │
       └──────────────┬──────────────────┘
                      ▼
          + Acoustic Feature Extraction
            (speech_rate, pause_duration, jitter)
                      │
                      ▼
           Psychological Fusion (fusion.py)
           → Instant State: {valence, arousal, stress, clarity}
                      │
                      ▼
           Emotional State Tracker (emotional_trends.py)
           → Adaptive State + Trends + Mode + Confidence
                      │
                      ▼
        ┌─────────────┴──────────────┐
        ▼                            ▼
  Topic Detection              Conversation State
  (memory_manager.py)          (conversation_state.py)
  → topic, confidence           → 5-turn buffer, intents
        │                            │
        └─────────────┬──────────────┘
                      ▼
          Event-Driven Memory Retrieval
          (memory_manager.py + memory_orchestrator.py)
          → memory_context (≤ 400 tokens, if triggered)
                      │
                      ▼
              LLM — GPT-4o-mini (llm_client.py)
              → llm_reply
                      │
                      ├──── Memory Storage (background)
                      │     (memory_orchestrator.py → Qdrant)
                      ▼
              TTS Controller (tts_azure.py)
              → style, styledegree, rate, pitch, volume
                      │
                      ▼
              Azure Neural TTS
              → WAV audio bytes
                      │
                      ▼
         WebSocket response to browser
         (transcript, reply, audio, all states)
```

---

## 3. Components

### 3.1 `services.py` — Multimodal Emotion Detection

Houses three services loaded once at container startup.

#### `ASRTextService`

**ASR:**
- Model: `openai/whisper-1` (API call)
- Language locked to English (`language="en"`)
- Returns: `transcript`, `asr_latency_ms`

**Text Emotion Classification:**
- Model: `SamLowe/roberta-base-go_emotions` (HuggingFace Transformers)
- Device: CUDA → CPU fallback
- Returns: **28 emotion labels** with probabilities (all emotions from the GoEmotions taxonomy)

| Category | Emotions |
|----------|----------|
| Positive | joy, optimism, love, gratitude, relief, excitement, pride, amusement, admiration, approval |
| Negative | sadness, grief, fear, anger, nervousness, disgust, remorse, disappointment, embarrassment, annoyance, disapproval |
| Cognitive | confusion, realization, surprise |
| Social | curiosity, caring, desire |
| Neutral | neutral |

**Acoustic Feature Extraction** (`extract_acoustic_features`):
- Computed dynamically from the actual audio file per turn
- See [Section 10](#10-dynamic-acoustic-features) for details

#### `AudioEmotionService`

**SER (Speech Emotion Recognition):**
- Model: `superb/wav2vec2-large-superb-er`
- Device: CUDA
- Returns: 4-class probabilities — `{angry, happy, neutral, sad}`

**AST (Audio Event Detection):**
- Model: `MIT/ast-finetuned-audioset-10-10-0.4593`
- 22 tracked event categories:

| Valence | Events |
|---------|--------|
| Positive | laughter, giggle, cheering, applause |
| Negative | crying, sobbing, whimpering, wailing |
| High arousal | screaming, shouting, yelling |
| Other | speech, music, silence, noise, … |

---

### 3.2 `fusion.py` — Psychological State Fusion

Combines multimodal signals into a single psychological state.

#### Fusion Weights

| Dimension | Text | SER | AST |
|-----------|------|-----|-----|
| Valence | 60% | 20% | 20% |
| Arousal | 20% | 60% | 20% |

#### Four Dimensions

**Valence** `[-1, 1]` — emotional positivity
- Positive: joy, love, gratitude, excitement, pride, amusement, admiration, laughter, cheering
- Negative: sadness, anger, fear, annoyance, disapproval, crying, grief

**Arousal** `[0, 1]` — energy / activation
- High: screaming, excitement, fear, shouting
- Low: calm, neutral, soft speech

**Clarity** `[0, 1]` — communication clarity
```
clarity = 0.4 × semantic  +  0.3 × acoustic  +  0.3 × fluency

where:
  semantic  = 1 - confusion_score
  acoustic  = (rate_dev / 3.0)^1.5   (soft curve, not harsh linear)
  fluency   = 1 / (1 + pause_duration)

Note: rate_dev = abs(speech_rate - 4.0) / 3.0
  6 WPS → ~0.5 clarity (was 0.0 with old linear formula)
  Normal speech (2-6 WPS) stays high clarity
```

**Stress** `[0, 1]` — psychological stress
```
stress = 0.4 × quadrant  +  0.3 × jitter  +  0.3 × environment

where:
  quadrant    = f(arousal, negative_valence)
  jitter      = pitch instability score
  environment = crying / screaming event probabilities
```

#### Axis Maps

Internal mappings that translate model outputs to emotional dimensions:

- `TEXT_AXIS_MAP`: 20 text emotions → `{valence_delta, arousal_delta}`
- `SER_AXIS_MAP`: 4 SER classes → `{valence_delta, arousal_delta}`
- `AST_AXIS_MAP`: 22 audio events → `{valence_delta, arousal_delta}`

---

### 3.3 `emotional_trends.py` — Trend Tracking

The core novelty of the system — tracking where emotions are **going**, not just where they are.

#### `TrendConfig` Hyperparameters

```python
window_size            = 8      # Circular buffer capacity
ema_alpha              = 0.4    # EMA smoothing (0.4 = responsive)
time_decay_half_life   = 45.0   # Seconds; weights recent values more
derivative_max         = 0.8    # Maximum trend magnitude (clamp)
stability_threshold    = 0.12   # Variance below this = "stable"
min_samples_for_trend  = 2      # Minimum buffer entries before trends
```

**Mode Switching Hysteresis:**
- `instant` mode: confidence < 0.3
- `trend` mode: confidence > 0.6
- Minimum dwell time: 2.0s (was 1.5s — prevents flip-flopping)

#### `BoundedEmotionalDimension`

Per-dimension tracker:
- Circular buffer (8 values, FIFO, O(1))
- Hard bounds `[-1, 1]` enforced after every update
- EMA re-clamped after every step

**Three Trend Signals:**

| Signal | Formula | Best for |
|--------|---------|----------|
| Instant | `current - previous` | Turns 1–3, immediate reaction |
| EMA | `current_ema - previous_ema` | Turns 4–6, medium smoothing |
| Regression | `weighted_linear_slope(values, timestamps)` | Turns 7+, most stable |

**Adaptive Blending:**

| Stage | Instant | EMA | Regression |
|-------|---------|-----|------------|
| Early (< 4 samples) | 70% | 30% | 0% |
| Mid (4–6 samples) | 40% | 40% | 20% |
| Late (6+ samples) | 20% | 40% | 40% |

#### `EmotionalStateTracker`

Manages all four dimensions together.

**Hysteresis Mode Switching:**
- `instant` mode: confidence < 0.3
- `trend` mode: confidence > 0.6
- Minimum dwell time: 1.5s (prevents flip-flopping)

**Confidence Estimation:**
```
confidence = sample_confidence × stability_confidence

sample_confidence    = min(buffer_size / window_size, 1.0)
stability_confidence = 1.0 - normalized_variance
```

**Major Shift Detection:**
- Threshold: `|trend| > 0.25` on any dimension
- Used to trigger long-term memory retrieval

**Output:**
```json
{
  "valence": 0.34,
  "arousal": 0.67,
  "stress": 0.45,
  "clarity": 0.82,
  "mode": "trend",
  "confidence": 0.78,
  "trends": {
    "valence_trend": -0.12,
    "valence_instant_trend": -0.18,
    "valence_ema_trend": -0.10,
    "valence_regression_trend": -0.08,
    "arousal_trend": 0.04,
    "stress_trend": 0.21,
    "clarity_trend": -0.06
  },
  "stability": {
    "valence_stable": true,
    "arousal_stable": true,
    "stress_stable": false,
    "clarity_stable": true
  }
}
```

---

### 3.4 `conversation_state.py` — Dialogue Management

#### `detect_simple_intent()`

Rule-based intent classifier:

| Intent | Triggers |
|--------|----------|
| `greeting` | "hello", "hi", "hey", "good morning" |
| `question` | ends with "?" or starts with question words |
| `affirmation` | "yes", "yeah", "sure", "okay" |
| `negation` | "no", "nope", "don't", "disagree" |
| `statement` | default (declarative) |

#### `DialogueTurn` (dataclass)

```python
timestamp:          float
user_utterance:     str
system_response:    str
user_intent:        str    # auto-detected
topic:              str    # from TopicDetector
psychological_state: dict  # instant fusion state
```

#### `AbstractDialogueState`

Formal state vector:
- `primary_topic` — current stable topic
- `topic_confidence` — `[0, 1]`
- `recent_intents` — rolling buffer (max 3)
- `coherence_score` — conversation quality `[0, 1]`
- `turn_count` — total turns

**Coherence formula:**
```
coherence = 0.6 × topic_consistency + 0.4 × intent_diversity

topic_consistency = 1.0 - (unique_topics / total_topics)
intent_diversity  = unique_intents / total_intents
```

Interpretation: `0.9+` = highly coherent, `< 0.5` = fragmented

#### `ConversationState`

- **Buffer**: 5 turns (FIFO, O(1) space)
- **LLM context**: all 5 turns (up to 10 utterances) formatted as `Turn N - User: … / Turn N - You: …`
- **Serialization**: `to_dict()` / `from_dict()` for Modal Dict persistence

---

### 3.5 `memory_manager.py` — Event-Driven Retrieval

#### `TopicDetector`

Keyword-based topic detection with stability requirements.

**Topics:** `work`, `personal`, `health`, `relationships`, `general`

**Update logic:**
1. Same topic → raise confidence, reset candidate
2. Different topic detected → start tracking as candidate
3. Candidate confirmed N consecutive turns → switch topic
4. High-confidence topics require more evidence to switch away from (hysteresis band)

**Config defaults:**
```python
confidence_threshold = 0.7    # Min confidence to accept switch
stability_window     = 2      # Consecutive matches required
hysteresis_band      = 0.15   # Extra difficulty switching from high-confidence topics
```

#### `MemoryRetrievalPolicy`

**Triggers** (checked in priority order):

| Priority | Trigger | Condition |
|----------|---------|-----------|
| 1 (highest) | Explicit recall | "remember", "you said", "last time", "earlier" |
| 2 | Topic shift | New topic confirmed with confidence > 0.7 |
| 3 | Emotional shift | `|trend|` > 0.3 on any dimension |

**Rate limiting:**
- Minimum interval: 30s between retrievals
- Explicit recall bypasses the interval

Returns: `(should_retrieve: bool, reason: str)`

---

### 3.6 `memory_store.py` — Qdrant Vector Store

Low-level Qdrant interface. Lazy-initialized — no network calls at import time.

#### Collections

| Collection | Purpose | Vector size | Distance |
|------------|---------|------------|---------|
| `episodic_memory` | Conversation episodes with emotional context | 384 | Cosine |
| `semantic_memory` | User preferences, facts, behaviours | 384 | Cosine |

#### Payload Indexes (auto-created)

| Collection | Field | Index type |
|------------|-------|------------|
| `episodic_memory` | `user_id` | KEYWORD |
| `episodic_memory` | `importance_score` | FLOAT |
| `semantic_memory` | `user_id` | KEYWORD |
| `semantic_memory` | `confidence` | FLOAT |

#### Embedding Model

`sentence-transformers/all-MiniLM-L6-v2` — 384-dimensional embeddings, loaded once per container.

#### Composite Retrieval Score

```
score = 0.50 × semantic_similarity
      + 0.20 × importance_score
      + 0.20 × log_reinforcement_factor
      + 0.10 × recency_decay

where:
  log_reinforcement_factor = 1.0 - 1.0 / (1.0 + reinforcement_count)
  recency_decay            = 0.95 ^ age_in_days
```

#### Graceful Degradation

- Connection failure → auto-reconnect via `_reconnect()` first
- If reconnect succeeds → immediate recovery, no fallback needed
- If reconnect fails → `_fallback = True`, 60-second cooldown before next retry
- Tenacity retry: 3 attempts, exponential backoff (2–10s)
- Error logging unwraps `RetryError` to reveal root cause

#### Key Methods

| Method | Description |
|--------|-------------|
| `store_episodic_memory()` | Store a conversation episode |
| `store_semantic_memory()` | Store a user fact/preference |
| `retrieve_episodic_memories()` | Semantic search + composite re-rank (min_importance: 0.35) |
| `retrieve_semantic_memories()` | Semantic search + composite re-rank (min_importance: 0.35) |
| `find_similar_memory()` | Find existing memory above similarity threshold |
| `reinforce_memory()` | Increment reinforcement_count, boost importance |
| `get_memory_stats()` | Per-user counts for both collections |
| `is_available()` | Connectivity check with auto-reconnect |
| `_reconnect()` | Tear down dead client, re-initialize fresh |

---

### 3.7 `memory_orchestrator.py` — Memory Orchestration

High-level decision layer above `QdrantMemoryStore`.

#### `MemoryDecision` (dataclass)

```python
should_store:     bool
importance_score: float
reason:           str
```

#### `detect_memory_worthiness()`

Calculates importance from 4 signals:

```
importance = 0.3 × emotional_intensity
           + 0.3 × topic_shift_strength
           + 0.2 × user_self_reference
           + 0.1 × novelty_score

Store only if importance > 0.6
```

- `emotional_intensity`: magnitude of current emotional state
- `topic_shift_strength`: confidence of detected topic change
- `user_self_reference`: presence of "I", "my", "me", "myself" etc.
- `novelty_score`: combined length score (60%) + word diversity score (40%) — not just word count

**Importance Threshold:** `0.45` (lowered from 0.6 — moderate emotional content or self-disclosure alone can now trigger storage)

#### `store_episodic_memory_with_reinforcement()`

1. Check for similar existing memory (threshold: 0.85 cosine similarity)
2. If found → `reinforce_memory()` (increment count, boost importance)
3. If not found → `store_episodic_memory()` with `reinforcement_count = 1`

#### `extract_semantic_memories()`

Every 5 turns, calls a lightweight LLM prompt to extract stable user facts from recent conversation turns, stores each fact as a separate semantic memory entry.

**Cooldown:** 60 seconds wall-clock (reduced from 300s — extraction now fires reliably in normal-paced conversations).

Uses `extract_semantic_facts()` from `llm_client.py`.

#### `retrieve_relevant_memories()`

Retrieves top-3 episodic + top-3 semantic memories, applies combined scoring, returns at most 6 results.

#### `format_memory_for_llm()`

Formats retrieved memories into a text block injected before the LLM system prompt. Hard limit: 400 tokens.

---

### 3.8 `llm_client.py` — LLM Integration

#### `psychological_llm_response()`

**Model:** `gpt-4o-mini`

**System Prompt Structure:**
1. Role definition — explicitly NOT a therapist
2. English-only instruction
3. Optional memory injection block (`USER LONG-TERM MEMORY:`, `RELEVANT PAST EVENTS:`)
4. Conversation context (5-turn buffer, formatted)
5. Current emotional state with trend descriptions
6. Trend direction guidance ("Stress is increasing", "Mood is declining", etc.)

**Parameters:** `temperature=0.7`, `max_tokens=500`

**Transcript Sanitization:**
Before reaching the LLM, Whisper transcripts are cleaned by `_sanitize_transcript()`:
- Strips invisible Unicode (zero-width spaces, replacement characters)
- Removes control characters
- Collapses repeating phrases (2-3 word phrases repeated 3+ times)
- Collapses excessive whitespace
- Truncates transcripts over 500 chars

#### `extract_semantic_facts()`

Lightweight call used by `MemoryOrchestrator` for periodic semantic extraction:

```python
model        = "gpt-4o-mini"
temperature  = 0.3
max_tokens   = 150
system       = "Extract stable user facts and preferences. One per line."
```

---

### 3.9 `tts_azure.py` — Voice Synthesis

#### `TTSController` (stateful per session)

**Hysteresis Gates:**

| Dimension | Lower threshold | Upper threshold | Dwell time |
|-----------|----------------|----------------|------------|
| Stress | 0.35 | 0.65 | 4s |
| Arousal | 0.25 | 0.55 | 4s |
| Valence | -0.30 | 0.30 | 4s |

**Voice Style Selection** (evaluated top to bottom):

| Priority | Condition | Style |
|----------|-----------|-------|
| 1 | high stress AND valence < -0.2 | `calm` |
| 2 | valence_trend < -0.15 AND stress > 0.4 | `empathetic` |
| 3 | positive AND high arousal AND clarity > 0.6 | `cheerful` |
| 4 | valence_trend > 0.15 AND arousal > 0.5 | `excited` |
| 5 | clarity < 0.4 OR stress > 0.6 | `newscast-casual` |
| 6 | arousal < 0.3 AND stress < 0.4 | `calm` |
| Default | — | `friendly` |

Style history requires **2 consecutive suggestions** before switching (prevents flickering).

**Prosody Parameters:**

| Parameter | Range | Trend influence |
|-----------|-------|----------------|
| `styledegree` | 0.6 – 2.0 | Amplified if stress/arousal rising |
| `rate` | -30% to +25% | Slower if stressed; faster if de-escalating |
| `pitch` | -15% to +15% | Higher for arousal, lower for declining valence |
| `volume` | soft / medium | Soft if stressed or stress rising |

All parameters are exponentially smoothed (`α = 0.75`, increased from 0.6 for more responsive adaptation).

**Per-Voice Style Validation:**
- Each voice has a mapped set of supported styles
- If a selected style isn't supported by the voice, it falls back through a priority chain
- Example: `calm` → `empathetic` → `assistant` → `friendly`
- Prevents Azure silently ignoring unsupported styles

**Widened Prosody Bounds:**
- Rate: -40% to +35% (was -30% to +25%)
- Pitch: -25% to +25% (was -15% to +15%)
- Style degree floor: 0.3 (was 0.5)
- JND thresholds lowered: rate 2%, pitch 1%, degree 0.05
- Hysteresis dwell times reduced: 2-2.5s (was 4-5s)

**State Persistence:**
- `to_dict()` / `from_dict()` serialize all smoothed prosody state
- Session restore preserves continuous voice adaptation
- Prevents jarring voice resets mid-conversation

**Available Voices:**

| Voice | Description |
|-------|-------------|
| `en-IN-KavyaNeural` | Default — Indian female, expressive |
| `en-IN-AnanyaNeural` | Indian female, expressive |
| `en-IN-AashiNeural` | Indian female, calm |
| `en-US-AvaMultilingualNeural` | Female, expressive |
| `en-US-AndrewMultilingualNeural` | Male, warm |
| `en-US-EmmaMultilingualNeural` | Female, clear |
| `en-US-BrianMultilingualNeural` | Male, friendly |

**SSML Output:**
```xml
<speak>
  <voice name="{voice_name}">
    <mstts:express-as style="{style}" styledegree="{degree}">
      <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">
        {text}
      </prosody>
    </mstts:express-as>
  </voice>
</speak>
```

---

### 3.10 `realtime_conversational_ai.py` — Modal Application

#### Modal Configuration

```python
app_name      = "conversational-ai-realtime"
base_image    = debian_slim (Python 3.11)
gpu           = "A10G"
timeout       = 3600s
scaledown     = 600s (10 minutes idle)
secrets       = ["emotion-env", "qdrant-credentials"]
```

#### Docker Image Layers (6-layer caching strategy)

```
Layer 1    System packages (ffmpeg, sox, build-essential)    ~rarely changes
Layer 2    PyTorch 2.6.0 + CUDA 12.1                        ~rarely changes
Layer 3    ML packages (transformers, librosa, azure-cognitiveservices-speech …)
Layer 4    Python utilities (fastapi, uvicorn, openai …)
Layer 4.5  Pre-downloaded ML models (Whisper, RoBERTa, AST, SER, SentenceTransformer)
Layer 5    Local source files (backend + frontend)           ~changes most often
```

Code-only changes rebuild only Layer 5 (~8–10s vs 45–60s for full rebuild).

**Model Pre-downloading:** All 6 ML models are pre-downloaded during image build and cached in `HF_HOME`/`TRANSFORMERS_CACHE` directories. Cold start drops from ~30-60s to ~5-10s.

**Frontend Serving:** CSS and JS files are read into memory at startup and embedded inline in the HTML response. This eliminates Modal container filesystem issues.

#### Persistent Storage

```python
modal.Dict.from_name("conversation-sessions")  # session state + TTS controller state
modal.Dict.from_name("session-metadata")        # session list/metadata
```

Stores per session: `ConversationState`, `EmotionalStateTracker`, `TTSController` (serialized), `turn_history`.

#### Modal Async Usage

Session metadata uses Modal's async interfaces to avoid `AsyncUsageWarning`:
```python
await session_metadata.put.aio(session_id, metadata)    # not sync assignment
await session_metadata.delete.aio(session_id)            # not sync deletion
```

---

## 4. Data Flow — Per Turn

```
1.  Receive base64-encoded WebM from client via WebSocket
2.  Decode → write to /tmp as .webm
3.  ffmpeg: .webm → 16kHz mono PCM WAV
4.  PARALLEL (asyncio.to_thread):
    a. ASR + Text Emotion  (Whisper API + RoBERTa)
    b. Audio Emotion        (Wav2Vec2 SER + AudioSet AST)
5.  extract_acoustic_features(wav_path, transcript)
    → speech_rate, pause_duration, jitter
6.  fusion.fuse(text=..., ser=..., ast=..., features=...)
    → instant_psychological_state
7.  emotional_tracker.update(instant_state)
    → adaptive_state, trends, mode, confidence
8.  topic_detector.update(transcript)
    → topic, topic_confidence
9.  emotional_tracker.detect_major_shift()
    → (emotional_shift_detected, shift_dimension)
10. retrieval_policy.check(transcript, topic_shift, emotional_shift)
    → (should_retrieve, retrieval_reason)
11. IF should_retrieve:
      memory_orchestrator.retrieve_relevant_memories()
      → memory_context (formatted, ≤400 tokens)
12. conv_state.get_llm_context()
    → llm_context (5-turn buffer formatted)
13. psychological_llm_response(transcript, adaptive_state, llm_context, memory_context)
    → llm_reply
14. tts_controller.compute(adaptive_state)
    → tts_params
15. synthesize_azure_tts(llm_reply, tts_params, voice_name)
    → wav_audio_bytes
16. BACKGROUND (asyncio.create_task):
    memory_orchestrator.detect_memory_worthiness(...)
    → IF worthy: store_episodic_memory_with_reinforcement()
    → IF turn % 5 == 0: extract_semantic_memories()
17. BACKGROUND (fire-and-forget via asyncio.create_task):
    → save_session(session_id, session)   → Modal Dict (non-blocking)
18. Return JSON payload over WebSocket

**Streaming Pipeline (3-step):**
The WebSocket now sends 4 messages per turn:
1. `{"status": "thinking"}` — immediately when processing starts
2. `{"status": "transcript", ...}` — after ASR completes
3. `{"status": "response_text", ...}` — after LLM (text shown immediately)
4. `{"status": "audio_ready", ...}` — after TTS (audio played concurrently)

**Parallel Execution:**
- TTS synthesis + memory storage run concurrently via `asyncio.gather()`
- Session save is fire-and-forget after response is sent
```

---

## 5. WebSocket Protocol

**Endpoint:** `wss://{app-url}/ws/conversation`

**Client → Server:**
```json
{
  "audio":      "<base64-encoded WebM>",
  "session_id": "session_abc123",
  "voice_name": "en-US-DragonV2.1Neural"
}
```

**Server → Client (success):**

The WebSocket now sends **4 messages** per turn in sequence:

**Message 1 — Thinking:**
```json
{ "status": "thinking" }
```

**Message 2 — Transcript:**
```json
{
  "status": "transcript",
  "transcript": "I've been feeling overwhelmed lately.",
  "model_outputs": { ... },
  "adaptive_state": { ... },
  "turn_count": 4
}
```

**Message 3 — Response Text:**
```json
{
  "status": "response_text",
  "llm_reply": "That sounds really difficult...",
  "tts_params": { "style": "empathetic", ... },
  "memory_view": { ... },
  "turn_history": [ ... ]
}
```

**Message 4 — Audio Ready:**
```json
{
  "status": "audio_ready",
  "tts_audio": "<base64-encoded WAV>"
}
```

**Frontend Behavior:**
- Text is streamed word-by-word as Message 3 arrives
- Audio plays concurrently (doesn't wait for text to finish)
- If Azure TTS fails, browser SpeechSynthesis fallback is used
- Markdown asterisks are stripped from displayed text

**Error close codes:**
- `1008` — client error (malformed JSON, missing fields, rate limited)
- `1011` — server error (processing failed, timeout) — safe to retry

---

## 6. REST API Reference

### `GET /health`
Basic liveness check (always returns 200).
```json
{ "status": "healthy" }
```

### `GET /readiness`
Checks Qdrant availability.
```json
{ "status": "ready", "memory": "available" }
{ "status": "ready", "memory": "unavailable" }
```

### `GET /liveness`
Returns 500 if the process is dying; used by Modal for auto-restart.
```json
{ "status": "alive" }
```

### `GET /api/sessions`
All sessions for the sidebar.
```json
{
  "sessions": [
    {
      "session_id": "session_abc123",
      "title": "I've been feeling overwhelmed...",
      "turn_count": 7,
      "created_at": "2026-02-05T10:30:00",
      "last_updated": "2026-02-05T10:45:00"
    }
  ]
}
```

### `POST /api/sessions/new`
Create a new session.
```json
{ "session_id": "session_xyz789" }
```

### `DELETE /api/sessions/{session_id}`
Delete a session and its metadata.
```json
{ "success": true, "message": "Session session_abc123 deleted" }
```

---

## 7. Configuration Reference

All defaults are in-code. To change a value, edit the file and redeploy.

### `emotional_trends.py` — `TrendConfig`

```python
window_size            = 8       # Circular buffer capacity per dimension
ema_alpha              = 0.4     # EMA factor (higher = more reactive)
time_decay_half_life   = 45.0    # Seconds; halves weight of older samples
derivative_max         = 0.8     # Clamp on trend magnitude
stability_threshold    = 0.12    # Variance below this → stable
min_samples_for_trend  = 2       # Buffer entries needed before computing trends
```

### `conversation_state.py` — `ConversationState`

```python
max_recent_turns        = 5     # FIFO buffer size
summary_update_interval = 5     # Turns between summary updates
```

### `memory_manager.py` — `TopicDetector`

```python
confidence_threshold = 0.7   # Minimum confidence to accept a topic switch
stability_window     = 2     # Consecutive confirmations required
hysteresis_band      = 0.15  # Extra difficulty switching from high-confidence topics
```

### `memory_manager.py` — `MemoryRetrievalPolicy`

```python
min_retrieval_interval     = 30.0  # Seconds between automatic retrievals
topic_shift_threshold      = 0.7   # Minimum topic confidence to trigger
emotional_shift_threshold  = 0.3   # Minimum trend magnitude to trigger
```

### `tts_azure.py` — `TTSController`

```python
smoothing_alpha = 0.6          # Exponential smoothing for prosody params
stress_gate     = (0.35, 0.65) # Hysteresis band
arousal_gate    = (0.25, 0.55)
valence_gate    = (-0.3, 0.3)
min_dwell_time  = 4.0          # Seconds before style can switch
```

### `fusion.py` — Fusion Weights

```python
valence: { text: 0.6, ser: 0.2, ast: 0.2 }
arousal: { text: 0.2, ser: 0.6, ast: 0.2 }
clarity: { semantic: 0.4, acoustic: 0.3, fluency: 0.3 }
stress:  { quadrant: 0.4, jitter: 0.3, environment: 0.3 }
```

### `realtime_conversational_ai.py` — Rate Limiting

```python
rate_limit_window = 2.0  # Minimum seconds between requests per session
```

---

## 8. Hybrid Memory System

Two-layer memory: short-term (in-memory buffer) + long-term (Qdrant vector database).

### Short-Term Memory

- 5-turn FIFO buffer in `ConversationState`
- Persisted to Modal Dict after every turn
- Sent directly to LLM as formatted context

### Long-Term Memory (Qdrant)

**Episodic Memory** (`episodic_memory` collection)

Stores conversation episodes with emotional context.

| Payload field | Type | Description |
|---------------|------|-------------|
| `user_id` | keyword | Session ID (used as user identifier) |
| `session_id` | keyword | Session ID |
| `memory_type` | keyword | Always `"episodic"` |
| `content` | text | User + AI exchange |
| `topic` | keyword | Detected topic |
| `valence` | float | Emotional valence at time of storage |
| `stress` | float | Stress level at time of storage |
| `importance_score` | float | Importance [0, 1] |
| `reinforcement_count` | integer | Times this memory was reinforced |
| `created_at` | datetime | ISO timestamp |
| `updated_at` | datetime | ISO timestamp |
| `metadata` | object | Turn number, mode, etc. |

**Semantic Memory** (`semantic_memory` collection)

Stores user facts, preferences, and behaviour patterns.

| Payload field | Type | Description |
|---------------|------|-------------|
| `user_id` | keyword | Session ID |
| `memory_type` | keyword | `"preference"`, `"fact"`, or `"behavior"` |
| `content` | text | The extracted fact |
| `importance_score` | float | Importance [0, 1] |
| `reinforcement_count` | integer | Times reinforced |
| `confidence` | float | Extraction confidence [0, 1] |
| `created_at` | datetime | ISO timestamp |
| `updated_at` | datetime | ISO timestamp |

**Memory Safety Rules**
- Never store every message
- Never retrieve all memories
- Always filter by `user_id`
- LLM injection hard-capped at 400 tokens
- At most 6 memories retrieved per turn

---

## 9. Session Management

Modelled after ChatGPT's sidebar pattern.

**Session ID:** Generated client-side (`crypto.randomUUID()`), persisted in `localStorage`.

**Session lifecycle:**
1. Browser generates ID on first visit (or loads from `localStorage`)
2. First message to `/ws/conversation` creates session entry in Modal Dict
3. Sidebar calls `GET /api/sessions` to populate history
4. User can create a new session (`POST /api/sessions/new`)
5. User can delete a session (`DELETE /api/sessions/{id}`)

**Session metadata** stored in `session-metadata` Modal Dict:
```python
{
  "session_id":   str,
  "title":        str,   # First user utterance (truncated to 50 chars)
  "turn_count":   int,
  "created_at":   str,   # ISO datetime
  "last_updated": str    # ISO datetime
}
```

**Turn history** (`turn_history` list in session data):
Populated per turn; drives the Turn-wise Psychological Log table in the frontend.

---

## 10. Dynamic Acoustic Features

Extracted from the actual audio file every turn via `extract_acoustic_features()` in `services.py`.

| Feature | Method | Range | Notes |
|---------|--------|-------|-------|
| `speech_rate` | `word_count / duration` | WPS | Typical: 2–5 WPS |
| `pause_duration` | Librosa RMS energy silence detection | seconds | Minimum pause: 0.3s |
| `jitter` | Librosa YIN (F0) — `std(pitch) / mean(pitch)` | 0–1 | Typical: 0.01–0.15 |
| `word_count` | `len(transcript.split())` | int | — |
| `audio_duration` | `librosa.get_duration()` | seconds | — |

All five values are fed into `fusion.fuse()` and displayed in the frontend.

---

## 11. Frontend UI

Single-page application served by FastAPI, communicating via WebSocket.

### File Structure

```
frontend/
├── index.html    (~250 lines)  Clean HTML structure
├── styles.css    (~540 lines)  CSS with custom properties (glassmorphism)
└── app.js        (~750 lines)  All JavaScript logic
```

CSS and JS are embedded inline in the HTML response at startup for reliability.

### Layout

```
┌───────────────────────────────────────────────────────────────┐
│  Sidebar (260px)            │  Main Content (flex: 1)         │
│  • Brand: MindVoice         │  • Title: MindVoice (gradient)  │
│  • New Chat button          │  • Voice selector (pill)        │
│  • Search bar               │  • Turn counter + mode badge    │
│  • Session list             │  • Mic ring (animated states)   │
│    - active indicator (█)   │    - 9-bar waveform during rec  │
│    - delete on hover        │  • Start / Stop buttons (pill)  │
│    - turn count + timestamp │  • Thinking indicator (amber)   │
│  • Footer: Export + Debug   │  • Transcript card (user)       │
│                             │  • Response card (AI, streaming)│
│                             │  • Emotion circles (4 SVG rings)│
│                             │  • Sparkline (valence/stress)   │
│                             │  • Debug Panels (2-col, collaps)│
│                             │  Left: Model Outputs             │
│                             │  Right: Memory & State           │
└───────────────────────────────────────────────────────────────┘
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Start / stop recording |
| Escape | Cancel recording |

### Model Outputs Panel (left debug panel)

- **ASR**: transcript, latency
- **Text Emotion**: top 5 emotions with probability bars
- **SER**: 4-class probabilities with bars
- **AST**: top 5 audio events with bars
- **Acoustic Features**: speech rate (WPS), pause duration, jitter, word count, audio duration
- **Fusion**: instant state values for all 4 dimensions

### Memory & State Panel (right debug panel)

- **Session Info**: session ID, turn count
- **Topic Tracking**: current topic, confidence
- **Emotional Trends**: mode, confidence, 4 dimensions with trend arrows (↗️ ↘️ →)
- **Recent Turns**: 5-turn buffer (user + AI for each)
- **Dialogue State**: recent intents, coherence score
- **Long-Term Memory**: episodic count, semantic count, last retrieval reason

### Turn-wise Psychological Log (research table)

Academic-format table updated after each turn. Columns:

`Turn | Mode | Confidence | Valence | Arousal | Stress | Clarity | Valence Trend | Arousal Trend | Stress Trend | Clarity Trend`

- Numerical values rounded to 2 decimal places
- Trend values prefixed with `+` or `−`
- **Copy Table** button exports as TSV (paste-ready for Word / Excel / LaTeX)

---

## 12. Infrastructure & Deployment

### Prerequisites

- Python 3.11+
- Modal account — [modal.com](https://modal.com)
- OpenAI API key (Whisper + GPT-4o-mini)
- Azure Speech Services key + region
- Qdrant — Cloud or self-hosted

### Modal Secrets Required

**Secret name: `emotion-env`**
```
OPENAI_API_KEY       = sk-...
AZURE_SPEECH_KEY     = ...
AZURE_SPEECH_REGION  = eastus
```

**Secret name: `qdrant-credentials`**
```
QDRANT_URL     = https://your-cluster.qdrant.io   (Cloud URL, already includes port)
QDRANT_API_KEY = ...

# For self-hosted only, also set:
QDRANT_PORT    = 6333
```

> **Note for Cloud:** Do NOT set `QDRANT_PORT` for cloud URLs — the port is already embedded in the HTTPS URL.

### GPU Requirements

- Minimum: T4 (8 GB VRAM)
- Recommended: A10G (24 GB VRAM)
- Total model VRAM: ~2.1 GB (RoBERTa 500 MB + Wav2Vec2 1.2 GB + AST 380 MB)

### Model Sizes

| Model | Size |
|-------|------|
| RoBERTa GoEmotions | ~500 MB |
| Wav2Vec2 SER | ~1.2 GB |
| AudioSet AST | ~380 MB |
| all-MiniLM-L6-v2 | ~90 MB |
| Whisper, GPT-4o-mini | API (no local storage) |
| **Total on-disk** | **~2.2 GB** |

### Latency Breakdown (typical, A10G)

| Stage | Time |
|-------|------|
| Audio decode (ffmpeg) | ~50ms |
| ASR (Whisper API) | 500–800ms |
| Text Emotion (RoBERTa, GPU) | 50–100ms |
| Audio Emotion (Wav2Vec2 + AST, GPU) | 300–500ms |
| Acoustic features (Librosa, CPU) | ~30ms |
| Fusion + trend update | < 5ms |
| LLM (GPT-4o-mini) | 800–1500ms |
| Azure TTS | 300–600ms |
| **Total (typical)** | **~2–3.5s** |

---

## 13. Performance Characteristics

### Space complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| Emotional buffer | O(1) | Fixed 8-turn circular buffer |
| Conversation buffer | O(1) | Fixed 5-turn FIFO |
| LLM context | O(1) | Max 10 utterances |
| Turn history | O(n) | Grows per session |
| Qdrant | O(n) | Bounded by `user_id` filter, importance threshold |

### Cost efficiency

- `gpt-4o-mini` (cheapest capable model, max 500 output tokens)
- Memory retrieval rate-limited (max once per 30s)
- Long-term memory storage gated by importance score (> 0.6)
- Modal GPU autoscaling: scaledown after 10 minutes idle

### Rate limiting

- Per-session: 1 request per 2 seconds
- Memory retrieval: 1 per 30 seconds (explicit recall bypasses this)

---

## 14. Troubleshooting

### Qdrant connection fails at startup

**Symptom in logs:**
```
❌ Qdrant init failed (RetryError). Root cause: ResponseHandlingException: [Errno 104] Connection reset by peer
```

**Checklist:**
1. Is `QDRANT_URL` set to the full HTTPS cloud URL?
2. Is `QDRANT_API_KEY` correct? (Check Qdrant Cloud dashboard)
3. For cloud URLs, is `QDRANT_PORT` absent from secrets? (The system auto-strips `:6333` from HTTPS URLs)
4. Can Modal's outbound network reach Qdrant? (Check firewall/VPN)

**Expected fallback:** System continues in degraded mode (short-term memory only). No crash. Auto-reconnect will attempt recovery on next operation.

```bash
modal logs realtime_conversational_ai -f | grep -E "Qdrant|memory"
```

### Response truncated mid-sentence

**Cause:** `max_tokens` too low (was 150, now 500)
**Fix:** Ensure `max_tokens=500` in `llm_client.py` `psychological_llm_response()`

### CSS/JS not loading in frontend

**Cause:** Modal container not serving `/static/` route reliably
**Fix:** CSS and JS are now embedded inline in the HTML response at startup. If still broken, ensure `styles.css` and `app.js` exist in the `frontend/` directory.

---

### WebSocket immediately closes

**Symptom:** Connection drops with code `1008`

**Causes:**
- Malformed JSON from client
- `audio` field missing or not base64-encoded
- `session_id` missing
- Rate limit exceeded (< 2s since last request)

**Symptom:** Connection drops with code `1011`

**Causes:**
- ffmpeg conversion failed (corrupt audio)
- ASR returned empty transcript
- GPT-4o-mini or Azure TTS timed out

---

### Cold start takes 45–60 seconds

This is expected on the **first** deployment or after dependency changes (Layers 1–4 rebuild).

For code-only changes (no new packages), only Layer 5 rebuilds (~8–10s).

---

### Coherence score stuck at 1.00

**Cause:** Conversation has only 1 unique topic and 1 intent pattern so far. This is normal in early turns.

---

### Memory retrieval never triggers

**Checklist:**
1. Has Qdrant initialized successfully? (check `/readiness`)
2. Is the conversation long enough? (needs at least a topic shift or emotional shift > 0.3)
3. Has 30 seconds passed since last retrieval?
4. Are there stored memories for this `user_id`? (check `get_memory_stats`)

---

## 15. Extending the System

### Add a new topic

Edit `memory_manager.py`:
```python
self.topic_keywords = {
    "work":      [...],
    "your_topic": ["keyword1", "keyword2", ...]
}
```

### Add a new TTS voice

Edit `tts_azure.py`:
```python
AVAILABLE_VOICES = [
    ...,
    "en-US-YourNewVoice"
]
```

### Adjust fusion weights

Edit `fusion.py`:
```python
self.weights = {
    "valence": {"text": 0.6, "ser": 0.2, "ast": 0.2},
    "arousal": {"text": 0.2, "ser": 0.6, "ast": 0.2}
}
```

### Change memory importance threshold

Edit `memory_orchestrator.py`:
```python
IMPORTANCE_THRESHOLD = 0.45   # Lower = store more, higher = store less
```

### Change retrieval rate limit

Edit `memory_manager.py`:
```python
MemoryRetrievalPolicy(
    min_retrieval_interval = 30.0   # Change to 15.0 for more frequent retrieval
)
```

---

## 16. Academic Foundation

### Psychological Model

**Circumplex Model of Affect** — 2D space of Valence × Arousal, extended here to 4D with Stress and Clarity.

**Emotional Trajectory Theory** — This system tracks not just current state but direction of change, enabling preemptive rather than reactive adaptation.

### Signal Processing Techniques

| Technique | Used for |
|-----------|----------|
| Exponential Moving Average (EMA) | Smoothing raw emotional signals |
| Time-weighted linear regression | Stable trend estimation |
| Schmitt trigger (hysteresis) | Mode switching, style selection |
| Circular buffer (FIFO) | Bounded memory management |
| Exponential decay weighting | Recency weighting in regression |

### Memory Architecture

**Short-term working memory:** Bounded circular buffer (cognitive science: ~7±2 items)

**Long-term episodic memory:** Vector similarity retrieval (Tulving 1972 — episodic vs. semantic memory distinction)

**Reinforcement:** Repeated exposure strengthens memory traces — mirroring psychological repetition effect

### Dialogue Management

- Formal state vectors (not string summaries)
- Coherence metric (topic consistency × intent diversity)
- Event-driven retrieval (avoids flooding LLM context)

---

## File Summary

| File | Lines | Responsibility |
|------|-------|----------------|
| `realtime_conversational_ai.py` | ~1600 | Modal app, WebSocket, FastAPI, pipeline orchestration, streaming pipeline |
| `services.py` | ~200 | ASR, Text Emotion, SER, AST, acoustic feature extraction |
| `fusion.py` | ~350 | Multimodal fusion, psychological state computation |
| `emotional_trends.py` | ~520 | Trend tracking, circular buffers, EMA, confidence, mode switching |
| `conversation_state.py` | ~285 | Dialogue state, intent detection, coherence, buffer management |
| `memory_manager.py` | ~385 | Topic detection, event-driven retrieval policy, rate limiting |
| `memory_store.py` | ~705 | Qdrant client, collections, embeddings, retrieval, reinforcement |
| `memory_orchestrator.py` | ~405 | Memory worthiness, storage decisions, semantic extraction |
| `llm_client.py` | ~140 | GPT-4o-mini client, prompt construction, semantic fact extraction |
| `tts_azure.py` | ~445 | Azure Neural TTS, prosody control, voice style selection, state persistence |
| `frontend/index.html` | ~250 | Clean HTML structure (CSS/JS embedded at startup) |
| `frontend/styles.css` | ~540 | Premium glassmorphism UI with CSS custom properties |
| `frontend/app.js` | ~750 | WebSocket streaming, session management, UI interactions |

---

*Documentation last updated: August 2026*
