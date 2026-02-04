#fusion.py

# =====================================================
# AXIS MAPS (FINAL)
# =====================================================

AST_AXIS_MAP = {
    "laughter": {"valence": 0.7, "arousal": 0.5},
    "giggle": {"valence": 0.6, "arousal": 0.4},
    "baby laughter": {"valence": 0.7, "arousal": 0.4},
    "belly laugh": {"valence": 0.8, "arousal": 0.6},
    "chuckle": {"valence": 0.5, "arousal": 0.3},
    "cheering": {"valence": 0.9, "arousal": 0.7},
    "applause": {"valence": 0.6, "arousal": 0.6},
    "whoop": {"valence": 0.7, "arousal": 0.6},

    "crying": {"stress": 0.9, "valence": -0.8},
    "sobbing": {"stress": 1.0, "valence": -1.0},
    "baby cry": {"stress": 0.8, "valence": -0.6},
    "whimper": {"stress": 0.7, "valence": -0.5},
    "groan": {"stress": 0.6, "valence": -0.4},
    "wail": {"stress": 0.9, "valence": -0.9},
    "moan": {"stress": 0.5, "valence": -0.3},
    "sigh": {"stress": 0.4, "arousal": -0.2},

    "screaming": {"arousal": 1.0, "stress": 0.9},
    "shout": {"arousal": 0.8, "stress": 0.6},
    "yell": {"arousal": 0.8, "stress": 0.6},
    "children shouting": {"arousal": 0.7, "stress": 0.5},
    "battle cry": {"arousal": 0.9, "stress": 0.6},
    "grunt": {"arousal": 0.5, "stress": 0.4},
}


TEXT_AXIS_MAP = {
    "joy": {"valence": 0.8, "arousal": 0.3},
    "optimism": {"valence": 0.6},
    "love": {"valence": 0.7},
    "gratitude": {"valence": 0.6},
    "relief": {"valence": 0.4, "stress": -0.4},

    "sadness": {"valence": -0.7, "stress": 0.6},
    "grief": {"valence": -1.0, "stress": 0.9},
    "fear": {"arousal": 0.8, "stress": 0.9},
    "anger": {"arousal": 0.9, "stress": 0.8},
    "nervousness": {"arousal": 0.6, "stress": 0.7},
    "disgust": {"valence": -0.6, "stress": 0.4},
    "remorse": {"valence": -0.5, "stress": 0.6},
    "disappointment": {"valence": -0.6, "stress": 0.5},
    "embarrassment": {"valence": -0.4, "stress": 0.5},

    "confusion": {"clarity": -0.9},
    "realization": {"clarity": 0.6},
    "surprise": {"arousal": 0.6},

    "curiosity": {"arousal": 0.3, "clarity": 0.2},
    "desire": {"arousal": 0.4, "valence": 0.2},
    "caring": {"valence": 0.3},

    "neutral": {}
}


SER_AXIS_MAP = {
    "angry": {"arousal": 0.9, "stress": 0.8},
    "happy": {"valence": 0.6, "arousal": 0.4},
    "sad": {"valence": -0.6, "stress": 0.6},
    "neutral": {}
}


# =====================================================
# FUSION ENGINE
# =====================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class PsychologicalFusion:
    def __init__(self):
        self.weights = {
            "valence": {"text": 0.6, "ser": 0.2, "ast": 0.2},
            "arousal": {"text": 0.2, "ser": 0.6, "ast": 0.2},
        }
        self.baseline_rate = 4.0
        self.max_pause = 2.0

    def _sum_axis(self, signal, axis_map, axis):
        val = 0.0
        for label, prob in signal.items():
            if label in axis_map and axis in axis_map[label]:
                val += prob * axis_map[label][axis]
        return val

    def compute_clarity(self, confusion, speech_rate, pause):
        semantic = 1.0 - confusion
        rate_penalty = abs(speech_rate - self.baseline_rate) / 2.0
        acoustic = clamp(1.0 - rate_penalty, 0, 1)
        fluency = clamp(1.0 - pause / self.max_pause, 0, 1)
        return 0.4 * semantic + 0.3 * acoustic + 0.3 * fluency

    def compute_stress(self, valence, arousal, jitter, ast):
        env = max(ast.values()) if ast else 0.0
        quad = 0.5 * arousal + 0.5 * (1 - valence)
        stress = 0.4 * quad + 0.3 * jitter + 0.3 * env
        return clamp(stress, 0, 1)

    def fuse(self, text, ser, ast, features):
        state = {}

        for axis in ["valence", "arousal"]:
            t = self._sum_axis(text, TEXT_AXIS_MAP, axis)
            s = self._sum_axis(ser, SER_AXIS_MAP, axis)
            a = self._sum_axis(ast, AST_AXIS_MAP, axis)

            w = self.weights[axis]
            state[axis] = w["text"] * t + w["ser"] * s + w["ast"] * a

        state["clarity"] = self.compute_clarity(
            text.get("confusion", 0.0),
            features.get("speech_rate", 4.0),
            features.get("pause_duration", 0.0),
        )

        state["stress"] = self.compute_stress(
            state["valence"],
            state["arousal"],
            features.get("jitter", 0.0),
            ast,
        )

        return state


# =====================================================
# AZURE TTS CONVERSION
# =====================================================

def interaction_mode(state):
    if state["stress"] > 0.7 and state["valence"] < -0.3:
        return "deescalation"
    if state["clarity"] < 0.4:
        return "clarification"
    if state["valence"] > 0.3 and state["arousal"] > 0.4 and state["clarity"] > 0.7:
        return "flow"
    return "neutral"


def azure_tts_input(state):
    mode = interaction_mode(state)

    style = {
        "deescalation": "calm",
        "clarification": "newscast-casual",
        "flow": "cheerful",
        "neutral": "assistant",
    }[mode]

    return {
        "style": style,
        "styledegree": round(clamp(1 + 0.5 * state["stress"], 0.8, 1.5), 2),
        "rate": f"{int(clamp((state['clarity'] - 1) * 0.3, -0.2, 0.15) * 100)}%",
        "pitch": f"{int(clamp((state['arousal'] - state['stress']) * 5, -5, 5))}%",
        "volume": "soft" if state["stress"] > 0.6 else "medium",
    }
