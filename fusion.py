# fusion.py — Psychological Fusion Engine (v2)
# Computes a rich psychological state from multimodal signals.
# Drives adaptive TTS voice parameters via azure_tts_input().

from typing import Dict


# =====================================================
# AXIS MAPS — weighted emotional→dimension contributions
# =====================================================

AST_AXIS_MAP = {
    # Positive vocalizations
    "laughter":         {"valence": 0.80, "arousal": 0.55},
    "giggle":           {"valence": 0.65, "arousal": 0.40},
    "baby laughter":    {"valence": 0.70, "arousal": 0.40},
    "belly laugh":      {"valence": 0.85, "arousal": 0.65},
    "chuckle":          {"valence": 0.55, "arousal": 0.30},
    "cheering":         {"valence": 0.90, "arousal": 0.80},
    "applause":         {"valence": 0.65, "arousal": 0.60},
    "whoop":            {"valence": 0.75, "arousal": 0.65},

    # Distress vocalizations
    "crying":           {"stress": 0.90, "valence": -0.80, "arousal": 0.50},
    "sobbing":          {"stress": 1.00, "valence": -1.00, "arousal": 0.60},
    "baby cry":         {"stress": 0.80, "valence": -0.60, "arousal": 0.55},
    "whimper":          {"stress": 0.70, "valence": -0.50, "arousal": 0.35},
    "groan":            {"stress": 0.60, "valence": -0.40, "arousal": 0.30},
    "wail":             {"stress": 0.90, "valence": -0.90, "arousal": 0.70},
    "moan":             {"stress": 0.50, "valence": -0.35, "arousal": 0.25},
    "sigh":             {"stress": 0.35, "valence": -0.20, "arousal": -0.25},

    # High-arousal vocalizations
    "screaming":        {"arousal": 1.00, "stress": 0.90, "valence": -0.50},
    "shout":            {"arousal": 0.80, "stress": 0.65, "valence": -0.30},
    "yell":             {"arousal": 0.80, "stress": 0.60, "valence": -0.30},
    "children shouting":{"arousal": 0.70, "stress": 0.45},
    "battle cry":       {"arousal": 0.90, "stress": 0.60, "valence": 0.10},
    "grunt":            {"arousal": 0.50, "stress": 0.40},
}


TEXT_AXIS_MAP = {
    # Positive emotions
    "joy":              {"valence": 0.85, "arousal": 0.45},
    "optimism":         {"valence": 0.65, "arousal": 0.25},
    "love":             {"valence": 0.80, "arousal": 0.20},
    "gratitude":        {"valence": 0.70, "arousal": 0.20},
    "relief":           {"valence": 0.50, "stress": -0.50, "arousal": -0.10},
    "pride":            {"valence": 0.70, "arousal": 0.35},
    "excitement":       {"valence": 0.60, "arousal": 0.75},
    "amusement":        {"valence": 0.65, "arousal": 0.45},
    "admiration":       {"valence": 0.60, "arousal": 0.15},

    # Negative emotions
    "sadness":          {"valence": -0.75, "stress": 0.55, "arousal": -0.30},
    "grief":            {"valence": -1.00, "stress": 0.90, "arousal": -0.20},
    "fear":             {"arousal": 0.85, "stress": 0.90, "valence": -0.60},
    "anger":            {"arousal": 0.90, "stress": 0.80, "valence": -0.70},
    "nervousness":      {"arousal": 0.65, "stress": 0.75, "valence": -0.30},
    "disgust":          {"valence": -0.65, "stress": 0.45, "arousal": 0.20},
    "remorse":          {"valence": -0.55, "stress": 0.60, "arousal": -0.20},
    "disappointment":   {"valence": -0.65, "stress": 0.50, "arousal": -0.10},
    "embarrassment":    {"valence": -0.45, "stress": 0.55, "arousal": 0.30},
    "annoyance":        {"valence": -0.50, "stress": 0.55, "arousal": 0.45},
    "desperation":      {"valence": -0.80, "stress": 0.90, "arousal": 0.70},

    # Cognitive/meta
    "confusion":        {"clarity": -0.90, "arousal": 0.25},
    "realization":      {"clarity": 0.65, "arousal": 0.40},
    "surprise":         {"arousal": 0.65, "clarity": 0.20},
    "curiosity":        {"arousal": 0.40, "clarity": 0.25, "valence": 0.15},
    "desire":           {"arousal": 0.45, "valence": 0.25},
    "caring":           {"valence": 0.35, "arousal": 0.10},
    "approval":         {"valence": 0.50, "arousal": 0.15},
    "disapproval":      {"valence": -0.45, "arousal": 0.30},

    "neutral":          {},
}


SER_AXIS_MAP = {
    "angry":    {"arousal": 0.90, "stress": 0.80, "valence": -0.60},
    "happy":    {"valence": 0.65, "arousal": 0.50},
    "sad":      {"valence": -0.65, "stress": 0.60, "arousal": -0.25},
    "neutral":  {},
}


# =====================================================
# UTILITIES
# =====================================================

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =====================================================
# FUSION ENGINE
# =====================================================

class PsychologicalFusion:
    """
    Multimodal psychological fusion.
    
    Fuses text emotion, speech emotion recognition (SER), and audio scene
    tagging (AST) into a unified psychological state vector:
      - valence:  [-1, 1]  negative ↔ positive affect
      - arousal:  [0, 1]   calm ↔ activated
      - stress:   [0, 1]   relaxed ↔ stressed
      - clarity:  [0, 1]   confused ↔ clear
    
    Weights reflect modality reliability:
      - Valence: text-dominant (60%) — semantics carry emotional meaning
      - Arousal: SER-dominant  (60%) — vocal acoustics encode activation level
    """

    def __init__(self):
        # Fusion weights per dimension per modality
        self.weights = {
            "valence": {"text": 0.60, "ser": 0.20, "ast": 0.20},
            "arousal": {"text": 0.20, "ser": 0.60, "ast": 0.20},
            "stress":  {"text": 0.50, "ser": 0.30, "ast": 0.20},
            "clarity": {"text": 0.80, "ser": 0.10, "ast": 0.10},
        }

        # Speech rate baseline (syllables per second at normal pace)
        self.baseline_rate = 4.0
        # Max pause considered (seconds)
        self.max_pause = 2.0

    def _sum_axis(
        self, signal: Dict[str, float], axis_map: Dict, axis: str
    ) -> float:
        """Weighted sum of axis contributions from a signal dict."""
        val = 0.0
        for label, prob in signal.items():
            if label in axis_map and axis in axis_map[label]:
                val += prob * axis_map[label][axis]
        return val

    def compute_clarity(
        self,
        confusion_prob: float,
        speech_rate: float,
        pause_duration: float,
    ) -> float:
        """
        Clarity = f(semantic confusion, speech rate deviation, fluency).
        
        Returns [0, 1] — 1 = crystal clear, 0 = highly confused/disrupted.
        """
        semantic   = 1.0 - clamp(confusion_prob, 0.0, 1.0)
        rate_dev   = abs(speech_rate - self.baseline_rate) / 2.0
        acoustic   = clamp(1.0 - rate_dev, 0.0, 1.0)
        fluency    = clamp(1.0 - pause_duration / self.max_pause, 0.0, 1.0)
        return clamp(0.40 * semantic + 0.30 * acoustic + 0.30 * fluency, 0.0, 1.0)

    def compute_stress(
        self,
        valence: float,
        arousal: float,
        jitter: float,
        ast: Dict[str, float],
    ) -> float:
        """
        Stress = f(valence, arousal, jitter, acoustic scene).
        
        Quadrant model: high arousal + low valence → high stress.
        Jitter (vocal tremor) and distress scenes amplify this.
        
        Returns [0, 1].
        """
        # Distress signals from acoustic scene
        distress_labels = {"crying", "sobbing", "whimper", "groan", "wail", "screaming", "shout", "yell"}
        env_distress = max((ast.get(lbl, 0.0) for lbl in distress_labels), default=0.0)

        # Quadrant activation: stress peaks at high arousal + negative valence
        valence_penalty = (1.0 - valence) / 2.0  # maps [-1,1] → [1,0]
        quad = 0.5 * clamp(arousal, 0.0, 1.0) + 0.5 * clamp(valence_penalty, 0.0, 1.0)

        stress = 0.40 * quad + 0.30 * clamp(jitter * 5.0, 0.0, 1.0) + 0.30 * env_distress
        return clamp(stress, 0.0, 1.0)

    def fuse(
        self,
        text: Dict[str, float],
        ser: Dict[str, float],
        ast: Dict[str, float],
        features: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Fuse all modalities into a psychological state dict.
        
        Args:
            text:     Text emotion probabilities  {label: prob}
            ser:      SER emotion probabilities   {label: prob}
            ast:      AST scene probabilities     {label: prob}
            features: Acoustic features           {speech_rate, pause_duration, jitter, ...}
        
        Returns:
            state: {valence, arousal, stress, clarity}
        """
        state: Dict[str, float] = {}

        # --- Valence ---
        t_v = self._sum_axis(text, TEXT_AXIS_MAP, "valence")
        s_v = self._sum_axis(ser,  SER_AXIS_MAP,  "valence")
        a_v = self._sum_axis(ast,  AST_AXIS_MAP,  "valence")
        w_v = self.weights["valence"]
        state["valence"] = clamp(
            w_v["text"] * t_v + w_v["ser"] * s_v + w_v["ast"] * a_v,
            -1.0, 1.0
        )

        # --- Arousal ---
        t_a = self._sum_axis(text, TEXT_AXIS_MAP, "arousal")
        s_a = self._sum_axis(ser,  SER_AXIS_MAP,  "arousal")
        a_a = self._sum_axis(ast,  AST_AXIS_MAP,  "arousal")
        w_a = self.weights["arousal"]
        state["arousal"] = clamp(
            w_a["text"] * t_a + w_a["ser"] * s_a + w_a["ast"] * a_a,
            0.0, 1.0
        )

        # --- Clarity (text-dominant + acoustic) ---
        confusion_prob = text.get("confusion", 0.0)
        state["clarity"] = self.compute_clarity(
            confusion_prob=confusion_prob,
            speech_rate=features.get("speech_rate", self.baseline_rate),
            pause_duration=features.get("pause_duration", 0.0),
        )

        # --- Stress (derived from valence, arousal, jitter, AST) ---
        state["stress"] = self.compute_stress(
            valence=state["valence"],
            arousal=state["arousal"],
            jitter=features.get("jitter", 0.0),
            ast=ast,
        )

        return state


# =====================================================
# INTERACTION MODE CLASSIFIER
# =====================================================

def interaction_mode(state: Dict[str, float]) -> str:
    """
    Classify the conversational interaction mode from psychological state.
    
    Modes:
      deescalation  — user is acutely distressed, needs calming
      support       — user is sad/stressed, needs empathetic warmth
      clarification — user is confused or incoherent, needs clear delivery
      celebration   — user is happy and energized, match the energy
      engaged       — user is curious/interested, enthusiastic but measured
      neutral       — default balanced mode
    """
    v  = state.get("valence",  0.0)
    a  = state.get("arousal",  0.5)
    s  = state.get("stress",   0.3)
    cl = state.get("clarity",  0.7)

    # Acute distress: screaming/crying scenario
    if s > 0.75 and v < -0.40:
        return "deescalation"

    # Sad / stressed (but not acute): empathetic warmth
    if v < -0.30 and s > 0.45:
        return "support"

    # User confused or speech incoherent
    if cl < 0.40:
        return "clarification"

    # Positive + energized: celebrate with them
    if v > 0.45 and a > 0.60:
        return "celebration"

    # Curious / engaged: warm enthusiasm
    if a > 0.40 and v > 0.15 and cl > 0.60:
        return "engaged"

    return "neutral"


# =====================================================
# AZURE TTS PARAMETER MAPPING
# =====================================================

# Maps voice styles to Azure TTS style IDs.
# These are validated Azure Neural voice style names.
_STYLE_MAP = {
    "deescalation": "calm",
    "support":      "empathetic",
    "clarification":"newscast-casual",
    "celebration":  "cheerful",
    "engaged":      "friendly",
    "neutral":      "assistant",
}

# Per-mode prosody adjustments
_PROSODY_MAP = {
    #            rate  pitch  styledeg  volume
    "deescalation": (-18,  -6,   1.8,  "soft"),
    "support":      (-10,  -4,   1.6,  "soft"),
    "clarification":( -5,   0,   1.2,  "medium"),
    "celebration":  (  8,   6,   1.8,  "medium"),
    "engaged":      (  4,   3,   1.5,  "medium"),
    "neutral":      (  0,   0,   1.0,  "medium"),
}


def azure_tts_input(state: Dict[str, float]) -> Dict:
    """
    Convert a psychological state dict into Azure TTS SSML parameters.
    
    This is the FUSION-LEVEL mapping (used without TTSController history).
    For session-level smoothed parameters, use TTSController in tts_azure.py.
    
    Returns dict with keys: style, styledegree, rate, pitch, volume
    """
    mode = interaction_mode(state)
    style = _STYLE_MAP[mode]
    base_rate, base_pitch, base_degree, base_volume = _PROSODY_MAP[mode]

    s  = state.get("stress",  0.3)
    a  = state.get("arousal", 0.5)
    v  = state.get("valence", 0.0)
    cl = state.get("clarity", 0.7)

    # Fine-tune rate: slow down more when stress or confusion rises
    rate_adj   = base_rate   - int(s * 10) + int(cl * 5)
    pitch_adj  = base_pitch  + int((a - s * 0.6) * 8)
    degree_adj = base_degree + (s * 0.30) + (abs(v) * 0.20)

    return {
        "style":       style,
        "styledegree": round(clamp(degree_adj, 0.6, 2.0), 2),
        "rate":        f"{int(clamp(rate_adj,  -25, 20)):+d}%",
        "pitch":       f"{int(clamp(pitch_adj, -12, 12)):+d}%",
        "volume":      base_volume,
        "mode":        mode,   # pass-through for logging/debugging
    }