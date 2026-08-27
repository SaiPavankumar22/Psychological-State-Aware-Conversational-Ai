# tts_azure.py — Adaptive TTS Controller (v2)
# Stateful, trend-aware, perceptually significant prosody control.
#
# Key improvements over v1:
#   - Context-type detection (question / empathy / statement / exclamation)
#   - Per-segment SSML with <break> and <mstts:silence> for natural pauses
#   - Wider prosody bounds for perceptibility
#   - Nuanced style selection: 8 modes with stricter gating
#   - "excited" / "x-loud" volume for high-energy positive turns
#   - Style transition requires only 1 stable sample (not 2) for instant emotion
#   - Early-turn behaviour: instant emotion drives style even without trends

import os
import re
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np
import azure.cognitiveservices.speech as speechsdk


# =====================================================
# AZURE CONFIG
# =====================================================

AZURE_TTS_KEY    = os.getenv("AZURE_TTS_KEY")
AZURE_REGION     = os.getenv("AZURE_TTS_REGION", "centralindia")
DEFAULT_VOICE    = "en-IN-KavyaNeural"

AVAILABLE_VOICES = [
    "en-IN-KavyaNeural",
    "en-IN-AnanyaNeural",
    "en-IN-AashiNeural",
    "en-US-AvaMultilingualNeural",
    "en-US-AndrewMultilingualNeural",
    "en-US-EmmaMultilingualNeural",
    "en-US-BrianMultilingualNeural",
]

# Per-voice supported styles (Azure Neural TTS)
# Only styles in this list will be used; others fall back to "assistant"
_VOICE_STYLES: Dict[str, List[str]] = {
    "en-IN-KavyaNeural":      ["assistant", "chat", "empathetic", "friendly", "serious", "cheerful"],
    "en-IN-AnanyaNeural":      ["assistant", "chat", "empathetic", "friendly", "serious", "cheerful"],
    "en-IN-AashiNeural":       ["assistant", "chat", "empathetic", "friendly", "serious", "cheerful"],
    "en-US-AvaMultilingualNeural":  ["assistant", "chat", "empathetic", "friendly", "serious", "cheerful", "customerservice", "newscast-formal"],
    "en-US-AndrewMultilingualNeural":["assistant", "chat", "empathetic", "friendly", "serious", "cheerful", "customerservice", "newscast-formal"],
    "en-US-EmmaMultilingualNeural": ["assistant", "chat", "empathetic", "friendly", "serious", "cheerful", "customerservice", "newscast-formal"],
    "en-US-BrianMultilingualNeural":["assistant", "chat", "empathetic", "friendly", "serious", "cheerful", "customerservice", "newscast-formal"],
}

# Fallback style for any unsupported style on any voice
_STYLE_FALLBACK: Dict[str, str] = {
    "calm":            "empathetic",
    "newscast-casual": "newscast-formal",
    "newscast":        "newscast-formal",
    "assistant":       "assistant",
    "empathetic":      "empathetic",
    "friendly":        "friendly",
    "cheerful":        "cheerful",
    "serious":         "serious",
    "chat":            "chat",
    "customerservice": "customerservice",
}

AUDIO_FORMAT = speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm


def _get_speech_config() -> speechsdk.SpeechConfig:
    if not AZURE_TTS_KEY:
        raise RuntimeError("AZURE_TTS_KEY is not set in environment")
    cfg = speechsdk.SpeechConfig(subscription=AZURE_TTS_KEY, region=AZURE_REGION)
    cfg.set_speech_synthesis_output_format(AUDIO_FORMAT)
    return cfg


# =====================================================
# HYSTERESIS CONTROLLER
# =====================================================

class HysteresisController:
    """
    Binary hysteresis with minimum dwell time.
    Prevents rapid state oscillation.
    """

    def __init__(self, low: float, high: float, dwell: float = 3.0):
        assert low < high
        self.low   = low
        self.high  = high
        self.dwell = dwell
        self.state = False
        self._last_change = time.time()

    def update(self, value: float) -> bool:
        now = time.time()
        if (now - self._last_change) < self.dwell:
            return self.state
        if self.state and value < self.low:
            self.state = False
            self._last_change = now
        elif not self.state and value > self.high:
            self.state = True
            self._last_change = now
        return self.state


# =====================================================
# CONTEXT-TYPE DETECTOR
# =====================================================

def detect_utterance_context(text: str) -> str:
    """
    Classify the AI response type so prosody can match it.

    Returns one of:
        question      — AI is asking the user something
        empathy       — AI is mirroring emotion / consoling
        explanation   — AI is explaining a concept (factual)
        exclamation   — AI is expressing enthusiasm
        neutral       — Default
    """
    t = text.strip()

    # Exclamation: ends with ! or has strong positive affirmations
    if t.endswith("!") or any(w in t.lower() for w in ["amazing", "wonderful", "great job", "fantastic"]):
        return "exclamation"

    # Question: ends with ?
    if t.endswith("?"):
        return "question"

    # Empathy markers
    empathy_phrases = [
        "i understand", "i hear you", "that sounds", "i'm sorry", "must be",
        "i can imagine", "it's okay", "you're not alone", "that must be",
    ]
    if any(p in t.lower() for p in empathy_phrases):
        return "empathy"

    # Explanation: longer, factual (heuristic: many words, no strong emotion markers)
    if len(t.split()) > 30:
        return "explanation"

    return "neutral"


# =====================================================
# PSYCHOLOGICAL STYLE MAP
# =====================================================

# Maps (interaction_mode, utterance_context) → Azure style ID.
# Falls back to mode-only when context is "neutral" or missing.
_STYLE_MATRIX: Dict[Tuple[str, str], str] = {
    # De-escalation mode
    ("deescalation", "empathy"):     "empathetic",
    ("deescalation", "question"):    "calm",
    ("deescalation", "neutral"):     "calm",
    ("deescalation", "exclamation"): "calm",
    ("deescalation", "explanation"): "calm",

    # Support mode (sad/low energy user)
    ("support", "empathy"):     "empathetic",
    ("support", "question"):    "friendly",
    ("support", "neutral"):     "empathetic",
    ("support", "exclamation"): "friendly",
    ("support", "explanation"): "friendly",

    # Clarification mode (confused user)
    ("clarification", "question"):    "newscast-casual",
    ("clarification", "explanation"): "newscast-casual",
    ("clarification", "neutral"):     "newscast-casual",
    ("clarification", "empathy"):     "friendly",
    ("clarification", "exclamation"): "friendly",

    # Celebration mode
    ("celebration", "exclamation"): "cheerful",
    ("celebration", "question"):    "cheerful",
    ("celebration", "neutral"):     "cheerful",
    ("celebration", "empathy"):     "friendly",
    ("celebration", "explanation"): "cheerful",

    # Engaged mode (curious)
    ("engaged", "question"):    "friendly",
    ("engaged", "explanation"): "friendly",
    ("engaged", "neutral"):     "friendly",
    ("engaged", "exclamation"): "cheerful",
    ("engaged", "empathy"):     "friendly",

    # Neutral / default
    ("neutral", "question"):    "assistant",
    ("neutral", "explanation"): "assistant",
    ("neutral", "neutral"):     "assistant",
    ("neutral", "empathy"):     "friendly",
    ("neutral", "exclamation"): "cheerful",
}


def _resolve_style(mode: str, ctx: str, voice_name: str = "") -> str:
    raw_style = _STYLE_MATRIX.get((mode, ctx), _STYLE_MATRIX.get((mode, "neutral"), "assistant"))
    
    # Validate style against the voice's supported styles
    if voice_name:
        supported = _VOICE_STYLES.get(voice_name, [])
        if raw_style not in supported:
            # Try the fallback map, then fall back to "assistant"
            raw_style = _STYLE_FALLBACK.get(raw_style, "assistant")
            if raw_style not in supported:
                raw_style = "assistant"
    return raw_style


# =====================================================
# PER-MODE PROSODY PROFILES
# =====================================================

# (base_rate, base_pitch, base_degree, volume)
_PROSODY_PROFILES: Dict[str, Tuple[int, int, float, str]] = {
    "deescalation": (-20,  -8,  1.8, "soft"),
    "support":      (-12,  -5,  1.6, "soft"),
    "clarification":(  -8,   0,  1.3, "medium"),
    "celebration":  (  10,   8,  1.9, "loud"),
    "engaged":      (   5,   4,  1.5, "medium"),
    "neutral":      (   0,   0,  1.1, "medium"),
}


# =====================================================
# STATEFUL TTS CONTROLLER
# =====================================================

class TTSController:
    """
    Session-level stateful TTS controller.

    Design principles
    -----------------
    1. INSTANT-first: even turn 1 reacts meaningfully to strong emotions.
       Trend weighting is gradually introduced after turn 3.
    2. Perceptual significance: all smoothed params are checked against
       perceptual JND (just-noticeable-difference) thresholds before updating,
       ensuring the listener actually hears the change.
    3. Context-aware: utterance type (question, empathy, etc.) overlaid on
       psychological mode for nuanced style selection.
    4. Hysteresis on all gates: prevents rapid style flip-flopping.
    """

    # Perceptual JND thresholds — changes below these are inaudible
    JND_RATE    = 2    # % points (lowered for more responsive adaptation)
    JND_PITCH   = 1    # % points (lowered for more responsive adaptation)
    JND_DEGREE  = 0.05

    def __init__(self):
        # --- Smoothed current params ---
        self.current_style      = "assistant"
        self.current_degree     = 1.0
        self.current_rate       = 0
        self.current_pitch      = 0
        self.current_volume     = "medium"
        self.current_mode       = "neutral"

        # --- Hysteresis gates (reduced dwell for faster adaptation) ---
        self.stress_gate    = HysteresisController(0.30, 0.60, dwell=2.0)
        self.arousal_gate   = HysteresisController(0.25, 0.55, dwell=2.0)
        self.valence_gate   = HysteresisController(-0.25, 0.30, dwell=2.0)
        self.support_gate   = HysteresisController(0.35, 0.55, dwell=2.5)

        # --- Exponential smoothing ---
        self.alpha = 0.75   # Higher = more responsive to state changes

        # --- Style history for stability ---
        self._style_queue: List[str] = []
        self._queue_max  = 2   # require 1 repeat before switching (less sticky than v1)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def compute(
        self,
        adaptive_state: Dict,
        response_text: str = "",
        voice_name: str = "",
    ) -> Dict:
        """
        Compute SSML-ready TTS parameters from the adaptive psychological state.

        Args:
            adaptive_state: Output from EmotionalTrendTracker.get_adaptive_state()
            response_text:  The AI response text (for context detection)
            voice_name:     Azure voice name (for style validation)

        Returns:
            dict with keys: style, styledegree, rate, pitch, volume
        """
        # --- Extract dimensions ---
        v  = float(np.clip(adaptive_state.get("valence",  0.0), -1.0,  1.0))
        a  = float(np.clip(adaptive_state.get("arousal",  0.5),  0.0,  1.0))
        s  = float(np.clip(adaptive_state.get("stress",   0.3),  0.0,  1.0))
        cl = float(np.clip(adaptive_state.get("clarity",  0.7),  0.0,  1.0))

        mode        = adaptive_state.get("mode",       "instant")
        confidence  = float(adaptive_state.get("confidence", 0.0))
        turn_count  = int(adaptive_state.get("turn_count",  0))

        trends      = adaptive_state.get("trends", {})
        v_trend     = float(np.clip(trends.get("valence_trend",  0.0), -0.5, 0.5))
        a_trend     = float(np.clip(trends.get("arousal_trend",  0.0), -0.5, 0.5))
        s_trend     = float(np.clip(trends.get("stress_trend",   0.0), -0.5, 0.5))

        # --- Trend weight (0 for early turns, ramps up) ---
        if mode == "trend" and turn_count >= 3:
            tw = min(confidence, 0.85)
        else:
            tw = 0.0
        iw = 1.0 - tw  # instant weight

        # --- Gate evaluations ---
        high_stress  = self.stress_gate.update(s)
        high_arousal = self.arousal_gate.update(a)
        is_positive  = self.valence_gate.update(v)
        needs_support= self.support_gate.update(max(s, 0.0 if v >= 0 else abs(v)))

        # --- Derive interaction mode ---
        interaction_mode = self._derive_mode(
            v, a, s, cl, high_stress, high_arousal, is_positive, needs_support,
            v_trend, s_trend, tw
        )
        self.current_mode = interaction_mode

        # --- Utterance context ---
        utt_ctx = detect_utterance_context(response_text) if response_text else "neutral"

        # --- Style ---
        target_style = _resolve_style(interaction_mode, utt_ctx, voice_name=voice_name)
        self._push_style(target_style)

        # --- Prosody profile base ---
        base_rate, base_pitch, base_degree, base_volume = _PROSODY_PROFILES.get(
            interaction_mode, _PROSODY_PROFILES["neutral"]
        )

        # --- Rate ---
        # Instant: slow down for stress, speed up if clarity is good
        inst_rate = base_rate - int(s * 15) + int(cl * 8)
        # Trend: slow more if stress RISING, recover if falling
        if s_trend > 0.08:
            trend_rate = inst_rate - int(s_trend * 40)
        elif s_trend < -0.08:
            trend_rate = inst_rate + int(abs(s_trend) * 25)
        else:
            trend_rate = inst_rate

        raw_rate = int(iw * inst_rate + tw * trend_rate)
        raw_rate = int(np.clip(raw_rate, -40, 35))
        # Apply only if above JND
        if abs(raw_rate - self.current_rate) >= self.JND_RATE:
            self.current_rate = int(self._smooth(self.current_rate, raw_rate))

        # --- Pitch ---
        inst_pitch = base_pitch + int((a - s * 0.65) * 12)
        if a_trend > 0.08:
            trend_pitch = inst_pitch + int(a_trend * 20)
        elif v_trend < -0.08:
            trend_pitch = inst_pitch + int(v_trend * 15) - int(s * 6)
        else:
            trend_pitch = inst_pitch

        raw_pitch = int(iw * inst_pitch + tw * trend_pitch)
        raw_pitch = int(np.clip(raw_pitch, -25, 25))
        if abs(raw_pitch - self.current_pitch) >= self.JND_PITCH:
            self.current_pitch = int(self._smooth(self.current_pitch, raw_pitch))

        # --- Style degree ---
        inst_deg = base_degree + (s * 0.35) + (abs(v) * 0.20) + (a * 0.10)
        if s_trend > 0.08:
            trend_deg = inst_deg + (s_trend * 0.8)
        elif s_trend < -0.08:
            trend_deg = inst_deg - (abs(s_trend) * 0.5)
        else:
            trend_deg = inst_deg

        raw_deg = float(np.clip(iw * inst_deg + tw * trend_deg, 0.3, 2.0))
        if abs(raw_deg - self.current_degree) >= self.JND_DEGREE:
            self.current_degree = float(self._smooth(self.current_degree, raw_deg))

        # --- Volume ---
        if high_stress or (s_trend > 0.12 and s > 0.45):
            self.current_volume = "soft"
        elif is_positive and high_arousal and utt_ctx in ("exclamation", "celebration"):
            self.current_volume = "loud"
        elif high_arousal and is_positive:
            self.current_volume = "medium"
        else:
            self.current_volume = base_volume

        return {
            "style":       self.current_style,
            "styledegree": round(self.current_degree, 2),
            "rate":        f"{self.current_rate:+d}%",
            "pitch":       f"{self.current_pitch:+d}%",
            "volume":      self.current_volume,
            "mode":        interaction_mode,     # diagnostic
            "utt_ctx":     utt_ctx,             # diagnostic
        }

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _derive_mode(
        self, v, a, s, cl,
        high_stress, high_arousal, is_positive, needs_support,
        v_trend, s_trend, tw
    ) -> str:
        """Multi-priority psychological mode resolution."""
        eff_v = v + v_trend * tw * 1.5
        eff_s = s + s_trend * tw * 1.5

        # 1. Acute distress — override everything
        if high_stress and v < -0.35:
            return "deescalation"

        # 2. Trending toward distress
        if tw > 0.3 and s_trend > 0.12 and eff_s > 0.5 and v < 0.0:
            return "deescalation"

        # 3. Sad / emotionally low — needs warmth
        if needs_support and v < -0.15:
            return "support"

        # 4. Confused / unclear speech
        if cl < 0.40:
            return "clarification"

        # 5. Joyful and energized
        if is_positive and high_arousal and cl > 0.55:
            return "celebration"

        # 6. Trending positive
        if tw > 0.3 and v_trend > 0.12 and is_positive:
            return "celebration"

        # 7. Curious / engaged (moderate arousal, positive)
        if a > 0.35 and v > 0.10 and cl > 0.55:
            return "engaged"

        return "neutral"

    def _push_style(self, style: str):
        """Stable style selection — requires ≥1 consecutive proposal."""
        if style == self.current_style:
            self._style_queue.clear()
            return

        self._style_queue.append(style)
        if len(self._style_queue) > self._queue_max:
            self._style_queue.pop(0)

        # Commit if all queue entries agree OR if it's a strong immediate signal
        if all(s == style for s in self._style_queue):
            self.current_style = style
            self._style_queue.clear()

    def _smooth(self, old: float, new: float) -> float:
        """Exponential moving average."""
        return self.alpha * new + (1.0 - self.alpha) * old

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialize smoothed state for session persistence."""
        return {
            "current_style":      self.current_style,
            "current_degree":     self.current_degree,
            "current_rate":       self.current_rate,
            "current_pitch":      self.current_pitch,
            "current_volume":     self.current_volume,
            "current_mode":       self.current_mode,
            "stress_gate_state":  self.stress_gate.state,
            "arousal_gate_state": self.arousal_gate.state,
            "valence_gate_state": self.valence_gate.state,
            "support_gate_state": self.support_gate.state,
            "_style_queue":       self._style_queue,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TTSController':
        """Restore smoothed state from persistence."""
        ctrl = cls()
        if not data:
            return ctrl
        ctrl.current_style      = data.get("current_style", "assistant")
        ctrl.current_degree     = data.get("current_degree", 1.0)
        ctrl.current_rate       = data.get("current_rate", 0)
        ctrl.current_pitch      = data.get("current_pitch", 0)
        ctrl.current_volume     = data.get("current_volume", "medium")
        ctrl.current_mode       = data.get("current_mode", "neutral")
        ctrl.stress_gate.state  = data.get("stress_gate_state", False)
        ctrl.arousal_gate.state = data.get("arousal_gate_state", False)
        ctrl.valence_gate.state = data.get("valence_gate_state", False)
        ctrl.support_gate.state = data.get("support_gate_state", False)
        ctrl._style_queue       = data.get("_style_queue", [])
        return ctrl


# =====================================================
# SINGLETON ACCESS
# =====================================================

_tts_controller: Optional[TTSController] = None


def compute_tts_params_from_trends(
    adaptive_state: Dict,
    response_text: str = "",
) -> Dict:
    """
    Singleton access to the global TTSController.
    Pass response_text for context-aware style selection.
    """
    global _tts_controller
    if _tts_controller is None:
        _tts_controller = TTSController()
    return _tts_controller.compute(adaptive_state, response_text=response_text)


# =====================================================
# SSML BUILDER
# =====================================================

def _build_ssml(text: str, params: Dict, voice_name: str) -> str:
    """
    Build rich SSML with:
    - <mstts:express-as> for style/degree
    - <prosody> for rate/pitch/volume
    - Sentence-boundary <break> for naturalness
    - Question-final emphasis tags when appropriate
    """
    style      = params["style"]
    degree     = params["styledegree"]
    rate       = params["rate"]
    pitch      = params["pitch"]
    volume     = params["volume"]
    utt_ctx    = params.get("utt_ctx", "neutral")
    mode       = params.get("mode", "neutral")

    # Escape XML special chars in text
    safe_text = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
    )

    # Add sentence-boundary micro-pauses for naturalism
    # Adds a 150ms break after sentence-ending punctuation
    safe_text = re.sub(
        r'([.!?])(\s+)',
        r'\1<break time="150ms"/>\2',
        safe_text
    )

    # For empathetic / deescalation modes, add a leading pause (deliberate pacing)
    leading_pause = ""
    if mode in ("deescalation", "support"):
        leading_pause = '<break time="200ms"/>'

    # Question inflection: add slight emphasis on last word before ?
    if utt_ctx == "question":
        safe_text = re.sub(
            r'(\w+)(\?)',
            r'<emphasis level="moderate">\1</emphasis>\2',
            safe_text,
            count=1
        )

    ssml = f"""\
<speak version="1.0"
 xmlns="http://www.w3.org/2001/10/synthesis"
 xmlns:mstts="http://www.w3.org/2001/mstts"
 xml:lang="en-US">
 <voice name="{voice_name}">
  {leading_pause}
  <mstts:express-as style="{style}" styledegree="{degree}">
   <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">
    {safe_text}
   </prosody>
  </mstts:express-as>
 </voice>
</speak>"""
    return ssml


# =====================================================
# AZURE SYNTHESIS (PUBLIC)
# =====================================================

def synthesize_azure_tts(
    text: str,
    tts_params: Dict,
    output_dir: str,
    voice_name: str = DEFAULT_VOICE,
) -> str:
    """
    Synthesize speech using Azure TTS with rich SSML.

    Args:
        text:        Plain text to synthesize (markdown already stripped)
        tts_params:  Parameters from TTSController.compute()
        output_dir:  Directory for the output WAV file
        voice_name:  Azure Neural voice name

    Returns:
        Path to the synthesized WAV file.

    Raises:
        RuntimeError on synthesis failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"tts_{uuid.uuid4().hex}.wav")

    speech_config = _get_speech_config()
    audio_config  = speechsdk.audio.AudioOutputConfig(filename=output_path)
    synthesizer   = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    ssml   = _build_ssml(text, tts_params, voice_name)
    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return output_path

    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        raise RuntimeError(
            f"Azure TTS cancelled: {details.reason} — {details.error_details}"
        )

    raise RuntimeError(f"Azure TTS unknown failure: {result.reason}")