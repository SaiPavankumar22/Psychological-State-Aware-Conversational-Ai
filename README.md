# 🎙️ Psychologically Adaptive AI Voice Assistant

A **research-grade, real-time conversational AI system** that adapts its voice, tone, and responses based on the user's emotional state and conversation trajectory.

---

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [Technical Specifications](#technical-specifications)
- [Installation & Deployment](#installation--deployment)
- [Features](#features)
- [Configuration](#configuration)
- [Monitoring & Debugging](#monitoring--debugging)
- [API Reference](#api-reference)

---

## 🎯 Overview

This system combines **multimodal emotion detection**, **psychological state fusion**, and **trend-based adaptation** to create a conversational AI that:

- **Listens** to user's speech and detects emotions from voice, text, and audio events
- **Understands** emotional trajectories over time (not just instant reactions)
- **Responds** with psychologically appropriate language and emotionally adaptive voice
- **Remembers** conversation context with bounded, efficient memory management
- **Adapts** voice style, speed, pitch, and volume based on emotional trends

### Key Innovation

Unlike traditional chatbots that react to instant emotional spikes, this system tracks **emotional trajectories** and adapts based on **where emotions are heading**, creating more natural and empathetic interactions.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER SPEAKS                              │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────────┐
│                    AUDIO PROCESSING                              │
│  WebM (Browser) → FFmpeg → WAV (16kHz Mono)                     │
└──────────────────────┬──────────────────────────────────────────┘
                       ↓
         ┌─────────────┴─────────────┐
         ↓                           ↓
┌──────────────────┐      ┌──────────────────────┐
│  ASR + TEXT      │      │   AUDIO EMOTION      │
│  EMOTION         │      │   (SER + AST)        │
│                  │      │                      │
│ • Whisper API    │      │ • Wav2Vec2 (SER)     │
│ • RoBERTa Go     │      │ • AudioSet AST       │
│   Emotions       │      │                      │
│                  │      │ Output:              │
│ Output:          │      │ • 4-class SER        │
│ • Transcript     │      │ • Audio events       │
│ • 20 emotions    │      │ • Latencies          │
└────────┬─────────┘      └──────────┬───────────┘
         │                           │
         └─────────────┬─────────────┘
                       ↓
         ┌─────────────────────────────┐
         │   PSYCHOLOGICAL FUSION       │
         │                              │
         │  Combines:                   │
         │  • Text emotions (60%)       │
         │  • SER (20%)                 │
         │  • AST (20%)                 │
         │  • Prosodic features         │
         │                              │
         │  Output:                     │
         │  • Valence [-1, 1]          │
         │  • Arousal [0, 1]           │
         │  • Stress [0, 1]            │
         │  • Clarity [0, 1]           │
         └──────────────┬──────────────┘
                        ↓
         ┌──────────────────────────────┐
         │  EMOTIONAL STATE TRACKER     │
         │                              │
         │  • 8-turn circular buffer    │
         │  • EMA smoothing (α=0.4)     │
         │  • Time-weighted trends      │
         │  • 3 trend signals:          │
         │    - Instant (responsive)    │
         │    - EMA (smooth)            │
         │    - Regression (stable)     │
         │  • Adaptive blending         │
         │  • Hysteresis mode switch    │
         │                              │
         │  Output:                     │
         │  • Mode: instant/trend       │
         │  • Adaptive state            │
         │  • Trend vectors             │
         │  • Confidence score          │
         └──────────────┬──────────────┘
                        ↓
    ┌──────────────────┴───────────────────┐
    ↓                                      ↓
┌────────────────────┐        ┌─────────────────────┐
│  MEMORY MANAGER    │        │  CONVERSATION STATE │
│                    │        │                     │
│ • Topic detection  │        │ • 5-turn buffer     │
│ • Event-driven     │        │ • Intent tracking   │
│   retrieval        │        │ • Coherence score   │
│ • Rate limiting    │        │ • Topic state       │
└─────────┬──────────┘        └──────────┬──────────┘
          │                              │
          └──────────────┬───────────────┘
                         ↓
              ┌────────────────────┐
              │   LLM (GPT-4o-mini)│
              │                    │
              │ • Trend-aware      │
              │ • Context-aware    │
              │ • English-only     │
              │ • Max 150 tokens   │
              │                    │
              │ Output: Text reply │
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │  TTS CONTROLLER    │
              │                    │
              │ • Trend-based      │
              │ • Hysteresis gates │
              │ • Style selection  │
              │ • Prosody control  │
              │ • Smoothing (α=0.6)│
              │                    │
              │ Styles:            │
              │ • calm             │
              │ • empathetic       │
              │ • cheerful         │
              │ • excited          │
              │ • friendly         │
              └─────────┬──────────┘
                        ↓
              ┌────────────────────┐
              │  AZURE NEURAL TTS  │
              │                    │
              │ • SSML generation  │
              │ • 8 voice options  │
              │ • Emotional styles │
              │ • Prosody control  │
              │                    │
              │ Output: WAV audio  │
              └─────────┬──────────┘
                        ↓
         ┌──────────────────────────┐
         │   FRONTEND (Browser)     │
         │                          │
         │ • WebSocket connection   │
         │ • Audio playback         │
         │ • Voice selector         │
         │ • Model outputs view     │
         │ • Memory view            │
         └──────────────────────────┘
```

---

## 🧩 Core Components

### 1. **`services.py`** — Multimodal Emotion Detection

#### **ASRTextService**
- **ASR (Automatic Speech Recognition)**
  - Model: OpenAI Whisper API (`whisper-1`)
  - Language: **Strict English only** (`language="en"`)
  - Output: Transcript + latency
  
- **Text Emotion Classification**
  - Model: `SamLowe/roberta-base-go_emotions`
  - Framework: Hugging Face Transformers
  - Device: GPU (CUDA) or CPU fallback
  - Output: 20 psychological emotions with probabilities
    - Positive: joy, optimism, love, gratitude, relief
    - Negative: sadness, grief, fear, anger, nervousness, disgust, remorse, disappointment, embarrassment
    - Cognitive: confusion, realization, surprise
    - Social: curiosity, caring, desire
    - Neutral: neutral

#### **AudioEmotionService**
- **SER (Speech Emotion Recognition)**
  - Model: `superb/wav2vec2-large-superb-er`
  - Output: 4-class probabilities [angry, happy, neutral, sad]
  - Device: GPU (CUDA)
  
- **AST (Audio Event Detection)**
  - Model: `MIT/ast-finetuned-audioset-10-10-0.4593`
  - Target Events: 22 categories
    - Positive: laughter, giggle, cheering, applause
    - Negative: crying, sobbing, whimpering, wailing
    - High-arousal: screaming, shouting, yelling
  - Output: Event probabilities for detected sounds

---

### 2. **`fusion.py`** — Psychological State Fusion

#### **PsychologicalFusion**

Combines multimodal signals into unified psychological state using **weighted fusion**:

**Fusion Weights:**
- **Valence**: Text (60%), SER (20%), AST (20%)
- **Arousal**: SER (60%), Text (20%), AST (20%)

**Computed Dimensions:**

1. **Valence** [-1, 1]: Emotional positivity/negativity
   - Positive: joy, love, gratitude, laughter
   - Negative: sadness, anger, fear, crying

2. **Arousal** [0, 1]: Energy/activation level
   - High: screaming, excitement, fear
   - Low: calm, neutral, muted speech

3. **Clarity** [0, 1]: Communication clarity
   - Components:
     - Semantic (1 - confusion score)
     - Acoustic (speech rate proximity to baseline 4.0 WPS)
     - Fluency (inverse of pause duration)
   - Formula: `0.4×semantic + 0.3×acoustic + 0.3×fluency`

4. **Stress** [0, 1]: Psychological stress level
   - Components:
     - Emotional quadrant (arousal + negative valence)
     - Acoustic jitter
     - Environmental sounds (crying, screaming)
   - Formula: `0.4×quadrant + 0.3×jitter + 0.3×environment`

**Axis Maps:**
- `AST_AXIS_MAP`: Audio events → emotional dimensions
- `TEXT_AXIS_MAP`: Text emotions → emotional dimensions
- `SER_AXIS_MAP`: Speech emotions → emotional dimensions

---

### 3. **`emotional_trends.py`** — Trend-Based State Tracking

#### **TrendConfig** (Hyperparameters)
```python
window_size: 8              # Circular buffer size
ema_alpha: 0.4             # EMA smoothing (0.4 = responsive)
time_decay_half_life: 45s  # Exponential time decay
derivative_max: 0.8        # Maximum trend magnitude
stability_threshold: 0.12  # Variance threshold
min_samples_for_trend: 2   # Minimum samples for trends
```

#### **BoundedEmotionalDimension**

**Per-dimension tracker** with:
- **Circular buffer** (8 values max, FIFO)
- **Strict bounds** [-1, 1] with no drift
- **EMA tracking** with re-clamping
- **Time-weighted statistics**

**Three Trend Signals:**

1. **Instant Trend** (Most Responsive)
   ```python
   instant_trend = current_value - previous_value
   ```
   Use: Early turns (1-3), immediate reactions

2. **EMA Trend** (Medium Smoothing)
   ```python
   ema_trend = current_ema - previous_ema
   ```
   Use: Mid conversation (4-6)

3. **Regression Trend** (Most Stable)
   ```python
   slope = weighted_linear_regression(values, timestamps)
   ```
   Use: Late conversation (7+)

**Adaptive Blending:**
- Early (< 4 samples): 70% instant, 30% EMA
- Mid (4-6 samples): 40% instant, 40% EMA, 20% regression
- Late (6+ samples): 20% instant, 40% EMA, 40% regression

#### **EmotionalStateTracker**

Manages all 4 dimensions (valence, arousal, stress, clarity) with:

- **Hysteresis-controlled mode switching**
  - Instant mode: confidence < 0.3
  - Trend mode: confidence > 0.6
  - Min dwell time: 1.5s (prevents flip-flopping)

- **Confidence estimation**
  - Based on: sample count + variance
  - Formula: `sample_confidence × stability_confidence`

- **Major shift detection**
  - Threshold: |trend| > 0.25
  - Triggers: Memory retrieval
  - Returns: (has_shift, dimension_name)

**Output:**
```python
{
  "valence": 0.34,
  "arousal": 0.67,
  "stress": 0.45,
  "clarity": 0.82,
  "mode": "trend",
  "confidence": 0.78,
  "trends": {
    "valence_trend": -0.12,      # Adaptive blend
    "valence_instant_trend": -0.18,
    "valence_ema_trend": -0.10,
    "valence_regression_trend": -0.08,
    ...
  },
  "stability": {
    "valence_stable": True,
    ...
  }
}
```

---

### 4. **`conversation_state.py`** — Dialogue Management

#### **Intent Detection**

Rule-based classifier (`detect_simple_intent`):
- **greeting**: "hello", "hi", "hey", "good morning"
- **question**: Ends with "?" or starts with question words
- **affirmation**: "yes", "yeah", "sure", "okay"
- **negation**: "no", "nope", "don't", "disagree"
- **statement**: Default (declarative sentences)

#### **DialogueTurn** (Structured Turn)
```python
@dataclass
class DialogueTurn:
    timestamp: float
    user_utterance: str
    system_response: str
    user_intent: str           # Auto-detected
    topic: str                 # From TopicDetector
    psychological_state: dict  # Instant state
```

#### **AbstractDialogueState** (Formal State Vector)
- `primary_topic`: Current stable topic
- `topic_confidence`: Confidence score [0, 1]
- `recent_intents`: Rolling buffer (max 3)
- `coherence_score`: Conversation quality [0, 1]
- `turn_count`: Total turns

#### **ConversationState**

**Bounded memory management:**
- **Buffer size**: 5 turns (FIFO)
- **LLM context**: All 5 turns sent (up to 10 utterances)
- **Guarantees**: No unbounded growth, O(1) space

**Coherence Calculation:**
```python
coherence = 0.6 × topic_consistency + 0.4 × intent_diversity

topic_consistency = 1.0 - (unique_topics / total_topics)
intent_diversity = unique_intents / total_intents
```

**Interpretation:**
- 0.9-1.0: Highly coherent (single topic, logical flow)
- 0.7-0.9: Good (few topic shifts)
- 0.5-0.7: Moderate (diverse topics)
- <0.5: Fragmented conversation

**Serialization:**
- `to_dict()`: Save to Modal Dict / Redis
- `from_dict()`: Restore from storage

---

### 5. **`memory_manager.py`** — Event-Driven Retrieval

#### **TopicDetector**

**Keyword-based detection** (upgradeable to embeddings):
- **Topics**: work, personal, health, relationships, general
- **Confidence scoring**: Keyword overlap + density
- **Hysteresis**: Requires 2 consecutive matches
- **Confidence bands**: Harder to switch from high-confidence topics

**Update logic:**
1. Same topic → update confidence, reset candidate
2. New candidate → start tracking
3. Consecutive match → increment count
4. Meets requirements → switch topic

#### **MemoryRetrievalPolicy**

**Event-driven retrieval** (NOT every turn):

**Triggers:**
1. **Explicit recall** (highest priority)
   - Keywords: "remember", "you said", "last time", "earlier"
   
2. **Topic shift** (confidence > 0.7)
   - New topic detected with high confidence
   
3. **Emotional shift** (magnitude > 0.3)
   - Major trend change in any dimension

**Rate limiting:**
- Min interval: 30 seconds between retrievals
- Explicit recall bypasses rate limit

**Returns:** `(should_retrieve: bool, reason: str)`

---

### 6. **`llm_client.py`** — Psychologically-Aware LLM

#### **psychological_llm_response()**

**Model:** GPT-4o-mini (low latency, cost-effective)

**System Prompt Structure:**
```
1. Role definition (not a therapist)
2. English-only enforcement
3. Conversation context:
   - Current topic
   - Turn count
   - All 5 recent turns (formatted)
4. Emotional state (trend-based):
   - Valence trend (very positive → very negative)
   - Energy level (high → low)
   - Clarity (very clear → confused)
   - Stress level (high → low)
5. Trend directions:
   - "Stress is increasing"
   - "Mood is declining"
   - etc.
```

**Parameters:**
- Temperature: 0.7 (balanced creativity)
- Max tokens: 150 (low latency)

**Context Formatting:**
```
Turn 1 - User: Hello, how are you?
Turn 1 - You: Hello! I'm here to help...
Turn 2 - User: I'm feeling stressed.
Turn 2 - You: I understand. Stress can...
...
```

---

### 7. **`tts_azure.py`** — Trend-Based Voice Synthesis

#### **TTSController** (Stateful)

**Hysteresis Gates:**
- Stress: [0.35, 0.65], dwell 4s
- Arousal: [0.25, 0.55], dwell 4s
- Valence: [-0.3, 0.3], dwell 4s

**Style Selection** (Multi-dimensional, prioritized):

```python
Priority 1: De-escalation
  Condition: high_stress AND valence < -0.2
  Style: "calm"
  
Priority 2: Empathetic
  Condition: valence_trend < -0.15 AND stress > 0.4
  Style: "empathetic"
  
Priority 3: Cheerful
  Condition: positive AND high_arousal AND clarity > 0.6
  Style: "cheerful"
  
Priority 4: Excited
  Condition: valence_trend > 0.15 AND arousal > 0.5
  Style: "excited"
  
Priority 5: Clarity mode
  Condition: clarity < 0.4 OR stress > 0.6
  Style: "newscast-casual"
  
Priority 6: Calm
  Condition: arousal < 0.3 AND stress < 0.4
  Style: "calm"
  
Default: "friendly"
```

**Prosody Parameters:**

1. **Style Degree** [0.6, 2.0]
   - Instant: `1.0 + stress×0.5 + arousal×0.3`
   - Trend: Amplifies if stress/arousal INCREASING
   - Smoothing: α=0.6

2. **Rate** [-30%, +25%]
   - Slower if stressed or unclear
   - Significantly slower if stress RISING
   - Faster if stress DECREASING

3. **Pitch** [-15%, +15%]
   - Higher for arousal, lower for stress
   - Amplifies if arousal RISING
   - Lowers if valence DECLINING

4. **Volume** {soft, medium}
   - Soft if: high_stress OR (stress_trend > 0.15 AND stress > 0.5)
   - Medium: otherwise

**Smoothing:** Exponential (α=0.6) prevents abrupt changes

**Style History:** Requires 2 consecutive suggestions before switching

#### **synthesize_azure_tts()**

**SSML Generation:**
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

**Supported Voices:**
- `en-US-DragonV2.1Neural` (default)
- `en-US-AvaMultilingualNeural`
- `en-US-AndrewMultilingualNeural`
- `en-US-EmmaMultilingualNeural`
- `en-US-BrianMultilingualNeural`
- `en-IN-KavyaNeural`
- `en-IN-AnanyaNeural`
- `en-IN-AashiNeural`

---

### 8. **`realtime_conversational_ai.py`** — Modal Application

#### **Infrastructure**

**Modal Configuration:**
- App: `conversational-ai-realtime`
- Image: Debian Slim (Python 3.11)
- GPU: A10G
- Timeout: 3600s (1 hour)
- Scaledown: 600s (10 minutes)
- Secrets: `emotion-env` (API keys)

**Dependencies:**
```
- ffmpeg, sox (audio processing)
- PyTorch 2.6.0 + CUDA 12.1
- Transformers, Accelerate
- OpenAI, Azure Speech SDK
- Librosa, SoundFile
- FastAPI, WebSockets
```

**Persistent Storage:**
- `modal.Dict.from_name("conversation-sessions")`
- Stores: ConversationState, EmotionalStateTracker
- Per-session persistence across container restarts

#### **Pipeline Flow**

**Per-turn processing:**

```python
1. Decode base64 audio → WAV conversion (ffmpeg)
2. PARALLEL:
   - ASR + Text Emotion (Whisper + RoBERTa)
   - Audio Emotion (SER + AST)
3. Fusion → Instant psychological state
4. Update emotional tracker → Adaptive state
5. Topic detection → Update dialogue state
6. Emotional shift detection
7. Event-driven memory retrieval (if triggered)
8. Get LLM context (5-turn buffer)
9. LLM response (GPT-4o-mini)
10. TTS controller → Compute voice parameters
11. Azure TTS synthesis → WAV audio
12. Save session state → Modal Dict
13. Return: transcript, reply, audio, states, model outputs
```

**WebSocket Protocol:**

**Client → Server:**
```json
{
  "audio": "base64_encoded_webm",
  "session_id": "session_abc123...",
  "voice_name": "en-US-DragonV2.1Neural"
}
```

**Server → Client:**
```json
{
  "transcript": "Hello, how are you?",
  "llm_reply": "I'm doing well, thank you!",
  "tts_audio": "base64_encoded_wav",
  "turn_count": 7,
  "emotional_mode": "trend",
  "instant_state": {...},
  "adaptive_state": {...},
  "tts_params": {...},
  "model_outputs": {...},
  "memory_view": {...}
}
```

#### **Frontend Interface**

**Session Sidebar (ChatGPT-like):**
- 🆕 **New Chat** button: Create fresh session
- 📚 **Session List**: All conversations sorted by recency
  - Session title (auto-generated from first message)
  - Last updated timestamp
  - Turn count
  - Active session highlighting
- 🗑️ **Delete button**: Remove unwanted sessions
- 📱 **Responsive**: Adapts to mobile screens (top bar on small devices)

**Main UI:**
- Voice selector dropdown (8 voices)
- Animated microphone ring (listening/processing/speaking states)
- Start/Stop buttons
- Turn counter + mode badge (Instant/Trend)
- Transcript display
- Response display
- 4-dimension state visualization

**Debug Panels** (2-column grid):

**Left Panel: Model Outputs**
- 🎤 ASR (Whisper): Transcript, latency
- 💭 Text Emotion: Top 5 emotions with bars
- 🎵 SER: 4-class probabilities with bars
- 🔊 AST: Top 5 audio events with bars
- 🎵 **Acoustic Features (Dynamic)**: Speech rate (WPS), pause duration, jitter, word count, audio duration
- 🎯 Fusion: Instant state (4 dimensions)

**Right Panel: Memory & State**
- 📋 Session Info: Session ID, turn count
- 🎯 Topic Tracking: Current topic, confidence
- 📈 Emotional Trends: Mode, confidence, 4 dimensions with trend arrows
- 💬 Recent Turns: 5-turn buffer with full context
- 🎭 Dialogue State: Recent intents, coherence score

**Trend Indicators:**
- ↗️ (red): Increasing (trend > 0.05)
- ↘️ (green): Decreasing (trend < -0.05)
- → (yellow): Stable (|trend| ≤ 0.05)

---

## 🔧 Technical Specifications

### **Audio Pipeline**
- Input: WebM (Opus codec, browser recording)
- Processing: 16 kHz mono WAV
- Conversion: ffmpeg
- Format: PCM 16-bit

### **Dynamic Acoustic Feature Extraction**

**Speech Rate:**
- Metric: Words per second (WPS)
- Calculation: `word_count / audio_duration`
- Typical range: 2-5 WPS (conversational)
- Uses: Actual transcript and audio file duration

**Pause Duration:**
- Detection: RMS energy threshold (bottom 20% = silence)
- Minimum pause: 0.3 seconds (ignore short gaps)
- Calculation: Average of all detected pauses
- Uses: Librosa RMS energy analysis

**Jitter (Voice Stability):**
- Definition: Pitch variability/instability
- Method: YIN algorithm for F0 extraction
- Calculation: `std(pitch) / mean(pitch)` (normalized)
- Range: [0, 1] (0 = very stable, 1 = very unstable)
- Typical: 0.01-0.15 for normal speech
- Uses: Librosa YIN pitch tracker

**Why Dynamic?**
- Reflects actual user vocal behavior (fast/slow speech, long pauses, voice tremor)
- Improves emotional estimation accuracy
- Captures stress, anxiety, hesitation, excitement in real-time
- No hardcoded assumptions about user's speaking style

### **GPU Requirements**
- NVIDIA GPU with CUDA 12.1 support
- VRAM: ~8GB (for all models)
- Recommended: A10G, T4, or better

### **API Dependencies**
- **OpenAI**: Whisper (ASR), GPT-4o-mini (LLM)
- **Azure**: Cognitive Services Speech (TTS)

### **Model Sizes**
- Whisper: API (no local storage)
- RoBERTa Go Emotions: ~500MB
- Wav2Vec2 SER: ~1.2GB
- AudioSet AST: ~380MB
- **Total**: ~2.1GB models

### **Latency Breakdown** (Typical)
```
ASR (Whisper API):     500-800ms
Text Emotion (RoBERTa): 50-100ms
Audio Emotion (GPU):    300-500ms
Fusion:                 <5ms
LLM (GPT-4o-mini):     800-1500ms
TTS (Azure):           300-600ms
─────────────────────────────────
Total:                 ~2-3.5s
```

### **Memory Footprint**
- Per-session state: ~5-10 KB
- Model memory (GPU): ~2.1 GB
- Circular buffers: O(1) bounded
- LLM context: O(1) bounded (max 5 turns × 2 utterances)

---

## 🚀 Installation & Deployment

### **1. Prerequisites**
```bash
- Python 3.11
- Modal account (https://modal.com)
- OpenAI API key
- Azure Speech Services key
- NVIDIA GPU (for local testing) or Modal GPU (production)
```

### **2. Environment Variables**

Create `.env` file:
```bash
OPENAI_API_KEY=sk-...
AZURE_TTS_KEY=...
AZURE_TTS_REGION=centralindia
```

### **3. Modal Secret Setup**
```bash
modal secret create emotion-env \
  OPENAI_API_KEY=sk-... \
  AZURE_TTS_KEY=... \
  AZURE_TTS_REGION=centralindia
```

### **4. Deploy to Modal**
```bash
# Deploy
modal deploy realtime_conversational_ai.py

# Run locally (with GPU)
modal serve realtime_conversational_ai.py
```

### **5. Access Application**
```
https://{your-username}--conversational-ai-realtime-fastapi-app.modal.run
```

---

## ✨ Features

### **1. Multimodal Emotion Detection**
- Text-based: 20 psychological emotions
- Voice-based: 4 speech emotions (angry, happy, neutral, sad)
- Audio events: 22 categories (laughter, crying, screaming, etc.)
- **Prosodic Features (DYNAMIC)**: 
  - Speech rate: Computed from transcript word count / audio duration
  - Pause duration: Detected from audio silence (threshold: 0.3s)
  - Jitter: Voice pitch instability (normalized std of F0)
  - All features extracted in real-time from actual audio (no hardcoded values)

### **2. Psychological State Fusion**
- Weighted fusion of 3 modalities
- 4-dimensional state space (valence, arousal, stress, clarity)
- Bounded values (no drift)

### **3. Emotional Trend Tracking**
- 8-turn circular buffer per dimension
- 3 trend signals (instant, EMA, regression)
- Adaptive blending based on conversation stage
- Confidence estimation
- Hysteresis-controlled mode switching

### **4. Conversation Memory**
- 5-turn rolling buffer (FIFO)
- Structured dialogue state (not strings)
- Auto-intent detection
- Coherence tracking
- Persistent storage (Modal Dict)

### **5. Topic Detection**
- Keyword-based (5 categories)
- Confidence scoring
- Hysteresis (prevents flip-flopping)
- Stability requirements (2 consecutive matches)

### **6. Event-Driven Memory Retrieval**
- NOT called every turn
- Triggers: explicit recall, topic shift, emotional shift
- Rate limiting (min 30s interval)
- Cost-efficient

### **7. Trend-Based TTS Adaptation**
- 6 voice styles (calm, empathetic, cheerful, excited, newscast-casual, friendly)
- Trend-aware prosody (responds to emotional trajectories)
- Hysteresis gates (prevents flickering)
- Exponential smoothing (α=0.6)
- Style history tracking

### **8. Dynamic Voice Selection**
- 8 voice options (US English + Indian English)
- User-selectable in frontend
- Default: Dragon V2.1 Neural

### **9. Session Management (ChatGPT-like)**
- **Create new sessions**: Start fresh conversations with one click
- **Access old sessions**: Browse conversation history in sidebar
- **Delete sessions**: Remove unwanted conversations
- **Session metadata**: Auto-generated titles, timestamps, turn counts
- **Persistent storage**: All sessions saved to Modal Dict
- **Sidebar UI**: Intuitive history panel with session list
- **Active session highlighting**: Visual indicator for current conversation
- **Mobile responsive**: Sidebar adapts to small screens

### **9. Real-Time Monitoring**
- Model outputs view (ASR, Text, SER, AST, Fusion)
- Memory state view (session, topic, trends, buffer, dialogue)
- Trend indicators (↗️↘️→)
- Scrollable debug panels

---

## ⚙️ Configuration

### **Emotional Trends** (`emotional_trends.py`)

```python
TrendConfig(
    window_size=8,              # Buffer size
    ema_alpha=0.4,             # Smoothing factor
    time_decay_half_life=45.0, # Time weighting
    derivative_max=0.8,        # Max trend magnitude
    stability_threshold=0.12,  # Variance threshold
    min_samples_for_trend=2    # Min samples
)
```

### **Conversation State** (`conversation_state.py`)

```python
ConversationState(
    session_id="...",
    max_recent_turns=5,        # Buffer size
    summary_update_interval=5  # Summary every N turns
)
```

### **Topic Detection** (`memory_manager.py`)

```python
TopicDetector(
    confidence_threshold=0.7,   # Min confidence for switch
    stability_window=2,         # Consecutive matches needed
    hysteresis_band=0.15       # Switching difficulty
)
```

### **Memory Retrieval** (`memory_manager.py`)

```python
MemoryRetrievalPolicy(
    min_retrieval_interval=30.0,      # Seconds
    topic_shift_threshold=0.7,         # Min confidence
    emotional_shift_threshold=0.3      # Min magnitude
)
```

### **TTS Controller** (`tts_azure.py`)

```python
TTSController(
    smoothing_alpha=0.6,              # Prosody smoothing
    stress_gate=(0.35, 0.65),         # Hysteresis band
    arousal_gate=(0.25, 0.55),
    valence_gate=(-0.3, 0.3),
    min_dwell_time=4.0                # Seconds
)
```

### **Fusion Weights** (`fusion.py`)

```python
Valence: text=0.6, ser=0.2, ast=0.2
Arousal: ser=0.6, text=0.2, ast=0.2
Clarity: semantic=0.4, acoustic=0.3, fluency=0.3
Stress: quadrant=0.4, jitter=0.3, environment=0.3
```

---

## 🔍 Monitoring & Debugging

### **Frontend Monitoring**

**Model Outputs Panel:**
- Real-time view of all model predictions
- Visual probability bars
- Latency metrics
- Top-N filtering (reduces clutter)

**Memory & State Panel:**
- Session persistence tracking
- Topic detection confidence
- Emotional trend visualization
- 5-turn conversation buffer
- Intent patterns
- Coherence metrics

**Console Logging** (Backend):
```
🔥 Turn 7 - 145832 bytes
📝 Transcript: I'm feeling really stressed today.
🎭 Instant State: valence=-0.42, stress=0.78
🧠 Mode: trend, Confidence: 0.81
📊 Adaptive State: valence=-0.38, stress=0.71
📌 Topic shift: personal (confidence: 0.85)
⚡ Emotional shift detected in stress: 0.34
🗄️ Memory retrieval triggered: emotional_shift
🤖 Generating response...
💬 Response: I hear you. Persistent stress can be challenging...
🎵 TTS: style=empathetic, degree=1.45, rate=-18%, pitch=-6%
```

---

## 📡 API Reference

### **Modal Method: `process_conversation_turn`**

**Input:**
```python
{
  "audio_data": str,        # Base64-encoded WebM
  "session_id": str,        # Unique session identifier
  "voice_name": str         # Azure TTS voice (default: Dragon V2.1)
}
```

**Output:**
```python
{
  "transcript": str,
  "llm_reply": str,
  "tts_audio": str,         # Base64-encoded WAV
  "turn_count": int,
  "emotional_mode": str,    # "instant" or "trend"
  "instant_state": {
    "valence": float,
    "arousal": float,
    "stress": float,
    "clarity": float
  },
  "adaptive_state": {
    "valence": float,
    "arousal": float,
    "stress": float,
    "clarity": float,
    "mode": str,
    "confidence": float
  },
  "tts_params": {
    "style": str,
    "styledegree": float,
    "rate": str,
    "pitch": str,
    "volume": str
  },
  "model_outputs": {
    "asr": {...},
    "text_emotion": {...},
    "audio_analysis": {
      "ser": {...},
      "ast": {...}
    },
    "acoustic_features": {
      "speech_rate": float,        # Words per second
      "pause_duration": float,     # Average pause length (seconds)
      "jitter": float,             # Pitch instability [0-1]
      "word_count": int,
      "audio_duration": float
    },
    "fusion": {...}
  },
  "memory_view": {
    "session_id": str,
    "dialogue_state": {...},
    "recent_turns": [...],
    "emotional_trends": {...},
    "topic_info": {...}
  }
}
```

### **WebSocket Endpoint**

**URL:** `/ws/conversation`

**Protocol:**
1. Client connects
2. Client sends JSON message (audio + session_id + voice_name)
3. Server processes (2-3.5s)
4. Server sends JSON response
5. Connection closes

**Audio Format:**
- Input: WebM (Opus, 16kHz, mono)
- Output: WAV (PCM 16-bit, 16kHz, mono)

### **REST API Endpoints**

#### **GET `/api/sessions`**
Get all sessions for history sidebar.

**Response:**
```json
{
  "sessions": [
    {
      "session_id": "session_abc123",
      "title": "I'm feeling stressed today...",
      "turn_count": 7,
      "created_at": "2026-02-05T10:30:00",
      "last_updated": "2026-02-05T10:45:00"
    }
  ]
}
```

#### **POST `/api/sessions/new`**
Create a new session.

**Response:**
```json
{
  "session_id": "session_xyz789"
}
```

#### **DELETE `/api/sessions/{session_id}`**
Delete a session.

**Response:**
```json
{
  "success": true,
  "message": "Session session_abc123 deleted"
}
```

---

## 🎯 Use Cases

### **1. Stress Detection & De-escalation**
```
User (stressed): "I can't handle this anymore!"
├─ Instant: valence=-0.8, stress=0.9
├─ Trend: stress↗️ (rising)
├─ Style: "calm"
├─ Prosody: slow, soft, low pitch
└─ Response: Calm, supportive language
```

### **2. Emotional Support During Mood Decline**
```
Turn 1: User neutral → AI friendly
Turn 2: User slightly sad → AI supportive
Turn 3: User very sad → Trend detected (valence↘️)
├─ Style switches: friendly → empathetic
├─ Prosody: slower, softer
└─ Response: More empathetic phrasing
```

### **3. Excitement Amplification**
```
Turn 1-3: User happy, energetic
Turn 4: Trend detected (valence↗️, arousal↗️)
├─ Style switches: friendly → excited
├─ Prosody: faster, higher pitch
└─ Response: Matches user's energy
```

### **4. Confusion Handling**
```
User: "Wait, what? I don't understand..."
├─ Instant: clarity=0.3
├─ Style: "newscast-casual"
├─ Prosody: slower, clearer articulation
└─ Response: Simpler, more structured language
```

---

## 🔬 Research-Grade Features

### **1. Formal Bounds**
- All values strictly bounded (no drift)
- Re-clamping after every operation
- Circular buffers (O(1) space)

### **2. Hysteresis Control**
- Mode switching (instant ↔ trend)
- Topic detection
- TTS parameter updates
- Prevents oscillation

### **3. Time-Weighted Aggregation**
- Exponential decay (half-life: 45s)
- Recent values weighted more
- Robust to outliers

### **4. Confidence Estimation**
- Sample confidence (buffer fullness)
- Stability confidence (variance-based)
- Combined metric for mode switching

### **5. Event-Driven Architecture**
- Memory retrieval only when needed
- Rate limiting (prevents spam)
- Cost-efficient (fewer LLM calls)

### **6. Multi-Scale Trend Detection**
- Instant (most responsive)
- EMA (medium smoothing)
- Regression (most stable)
- Adaptive blending

### **7. Perceptually Significant Changes**
- Aggressive smoothing (α=0.6)
- Wide parameter bounds
- Amplified trend responses
- Dwell times prevent flickering

---

## 📊 Performance Characteristics

### **Scalability**
- **State size**: O(1) per session
- **LLM context**: O(1) bounded (max 10 utterances)
- **Buffer growth**: O(1) (FIFO circular)
- **Memory retrieval**: Event-driven, not O(n)

### **Cost Efficiency**
- GPT-4o-mini (cheaper than GPT-4)
- Max 150 tokens per response
- Memory retrieval rate-limited (max 1 per 30s)
- Modal GPU autoscaling (10min scaledown)

### **Latency Optimization**
- Parallel ASR + Audio analysis
- GPU acceleration (Transformers)
- Async/await throughout
- Bounded computations

### **Robustness**
- Strict numerical bounds
- Error handling at every stage
- Graceful degradation
- Session persistence (survives restarts)

---

## 🎓 Academic Foundation

### **Psychological Models**

**Circumplex Model of Affect:**
- 2D: Valence × Arousal
- Extended: + Stress + Clarity
- Weighted multimodal fusion

**Emotional Trajectories:**
- Not just current state, but direction
- Trend-based adaptation
- Hysteresis prevents over-reaction

### **Signal Processing**

**Time-Series Analysis:**
- Exponential Moving Average (EMA)
- Time-weighted linear regression
- Exponential decay weighting
- Circular buffer management

**Control Theory:**
- Schmitt trigger (hysteresis)
- Bounded integrators (no windup)
- Rate limiting
- Minimum dwell times

### **Dialogue Management**

**Structured State:**
- Not string-based summaries
- Formal state vectors
- Intent tracking
- Coherence metrics

**Memory Retrieval:**
- Event-driven (not every turn)
- Confidence-gated
- Rate-limited
- Multi-trigger (topic, emotion, explicit)

---

## 🔐 Privacy & Ethics

### **Non-Therapeutic Disclaimer**
- System is **NOT a therapist**
- Does **NOT diagnose** mental health conditions
- Does **NOT provide medical advice**
- Designed for conversational assistance only

### **Data Storage**
- Sessions stored in Modal Dict (ephemeral)
- No long-term user data retention
- Session IDs are random, non-identifying
- Audio processed in-memory, immediately deleted

### **Transparency**
- All model outputs visible in frontend
- User can see exactly what system detects
- Emotional state calculations shown
- Full pipeline transparency

---

## 🛠️ Extending the System

### **Add New Topics**
Edit `memory_manager.py`:
```python
self.topic_keywords = {
    "work": [...],
    "your_topic": ["keyword1", "keyword2", ...]
}
```

### **Add New Voice Styles**
Edit `tts_azure.py`:
```python
AVAILABLE_VOICES.append("en-US-YourVoice")
```

Update `_select_style()` logic.

### **Adjust Fusion Weights**
Edit `fusion.py`:
```python
self.weights = {
    "valence": {"text": 0.6, "ser": 0.2, "ast": 0.2},
    "arousal": {"text": 0.2, "ser": 0.6, "ast": 0.2}
}
```

### **Add Vector Database for Long-Term Memory**
Integrate in `realtime_conversational_ai.py`:
```python
if should_retrieve:
    # Query Pinecone/Weaviate/Qdrant
    memories = vector_db.query(
        embedding=embed(transcript),
        top_k=3
    )
    # Add to LLM context
```

---

## 📈 Future Enhancements

### **Short-Term**
- [ ] Intent classification via LLM (replace rule-based)
- [ ] Semantic topic detection (embeddings)
- [ ] Vector database integration
- [ ] Multi-language support
- [ ] Voice activity detection (VAD)

### **Long-Term**
- [ ] Personality modeling
- [ ] Long-term user profiles
- [ ] Multi-speaker diarization
- [ ] Video emotion detection (facial expressions)
- [ ] Adaptive learning (fine-tuning on user data)

---

## 📝 File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `realtime_conversational_ai.py` | 1466 | Modal app, WebSocket, frontend UI, pipeline orchestration |
| `services.py` | 198 | ASR (Whisper), Text Emotion (RoBERTa), SER (Wav2Vec2), AST (AudioSet) |
| `fusion.py` | 167 | Multimodal fusion, psychological state computation, axis maps |
| `emotional_trends.py` | 516 | Trend tracking, circular buffers, EMA, confidence estimation, mode switching |
| `conversation_state.py` | 281 | Dialogue management, intent detection, coherence, buffer management |
| `memory_manager.py` | 266 | Topic detection, event-driven retrieval, hysteresis, rate limiting |
| `llm_client.py` | 136 | OpenAI GPT-4o-mini client, context formatting, trend descriptions |
| `tts_azure.py` | 444 | Azure TTS, hysteresis controller, trend-based prosody, style selection |

**Total:** ~3,674 lines of research-grade Python

---

## 🙏 Acknowledgments

### **Models Used**
- **OpenAI Whisper** (ASR)
- **SamLowe/roberta-base-go_emotions** (Text Emotion)
- **superb/wav2vec2-large-superb-er** (SER)
- **MIT/ast-finetuned-audioset-10-10-0.4593** (AST)
- **OpenAI GPT-4o-mini** (LLM)
- **Azure Neural TTS** (Voice Synthesis)

### **Frameworks**
- Modal (Serverless GPU)
- FastAPI (Web framework)
- PyTorch (Deep learning)
- Hugging Face Transformers (Model hub)

---

## 📄 License

This project is for research and educational purposes.

---

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue in the repository.

---

**Built with ❤️ for research in emotionally adaptive conversational AI.**
