# tts_azure.py — REFACTORED (Trend-Based, Perceptually Responsive)

import os
import time
import uuid
from typing import Dict, Optional

import numpy as np
import azure.cognitiveservices.speech as speechsdk


# =====================================================
# AZURE CONFIG
# =====================================================

AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY")
AZURE_REGION = os.getenv("AZURE_TTS_REGION", "centralindia")
DEFAULT_VOICE_NAME = "en-IN-KavyaNeural"

# Available voices for frontend selection
AVAILABLE_VOICES = [
    "en-IN-KavyaNeural",
    "en-IN-AnanyaNeural",
    "en-IN-AashiNeural",
    "en-US-AvaMultilingualNeural",
    "en-US-AndrewMultilingualNeural",
    "en-US-EmmaMultilingualNeural",
    "en-US-BrianMultilingualNeural",
]

AUDIO_FORMAT = speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm


def _get_speech_config() -> speechsdk.SpeechConfig:
    if not AZURE_TTS_KEY:
        raise RuntimeError("AZURE_TTS_KEY is not set")

    cfg = speechsdk.SpeechConfig(
        subscription=AZURE_TTS_KEY,
        region=AZURE_REGION,
    )
    cfg.set_speech_synthesis_output_format(AUDIO_FORMAT)
    return cfg


# =====================================================
# HYSTERESIS CONTROLLER (Enhanced)
# =====================================================

class HysteresisController:
    """
    Binary hysteresis controller with dwell time.
    Prevents rapid oscillation between states.
    """

    def __init__(self, low: float, high: float, min_dwell_time: float = 3.0):
        assert low < high
        self.low = low
        self.high = high
        self.state = False
        self.last_transition_time = time.time()
        self.min_dwell_time = min_dwell_time

    def update(self, value: float) -> bool:
        """Update with minimum dwell time to prevent rapid switching"""
        now = time.time()
        time_in_state = now - self.last_transition_time
        
        # Require minimum dwell time before allowing state change
        if time_in_state < self.min_dwell_time:
            return self.state
        
        if self.state:
            if value < self.low:
                self.state = False
                self.last_transition_time = now
        else:
            if value > self.high:
                self.state = True
                self.last_transition_time = now
        
        return self.state


# =====================================================
# STATEFUL TTS CONTROLLER (TREND-BASED)
# =====================================================

class TTSController:
    """
    Research-grade TTS controller with TREND-BASED prosody.
    
    Key Features:
    - Early turns: Reactive (instant values + text sentiment)
    - Later turns: Trend-driven (emotional trajectory)
    - Smooth transitions via exponential smoothing
    - Perceptually significant changes (avoid subtle shifts)
    - Multi-dimensional style selection
    """

    def __init__(self):
        # Current TTS parameters (smoothed)
        self.current_style = "friendly"
        self.current_styledegree = 1.0
        self.current_rate = 0
        self.current_pitch = 0
        self.current_volume = "medium"

        # Hysteresis gates (with longer dwell time)
        self.stress_gate = HysteresisController(0.35, 0.65, min_dwell_time=4.0)
        self.arousal_gate = HysteresisController(0.25, 0.55, min_dwell_time=4.0)
        self.valence_gate = HysteresisController(-0.3, 0.3, min_dwell_time=4.0)  # Negative/Positive
        
        # Smoothing parameters (AGGRESSIVE for perceptibility)
        self.smoothing_alpha = 0.6  # Higher = more responsive to changes
        
        # Style transition management
        self.style_history = []
        self.max_style_history = 3
        
        # Baseline for relative changes
        self.baseline_params = {
            "styledegree": 1.0,
            "rate": 0,
            "pitch": 0,
            "volume": "medium"
        }

    def compute(self, adaptive_state: Dict) -> Dict:
        """
        Compute TTS parameters based on emotional trends.
        
        CRITICAL: Uses trends when available, falls back to instant values.
        """
        # Extract base emotional dimensions
        valence = np.clip(adaptive_state.get("valence", 0), -1, 1)
        arousal = np.clip(adaptive_state.get("arousal", 0), 0, 1)
        stress = np.clip(adaptive_state.get("stress", 0), 0, 1)
        clarity = np.clip(adaptive_state.get("clarity", 0), 0, 1)

        # Get mode and confidence
        mode = adaptive_state.get("mode", "instant")
        confidence = adaptive_state.get("confidence", 0.0)
        turn_count = adaptive_state.get("turn_count", 0)

        # Extract trends (bounded)
        trends = adaptive_state.get("trends", {})
        valence_trend = np.clip(trends.get("valence_trend", 0), -0.5, 0.5)
        arousal_trend = np.clip(trends.get("arousal_trend", 0), -0.5, 0.5)
        stress_trend = np.clip(trends.get("stress_trend", 0), -0.5, 0.5)
        clarity_trend = np.clip(trends.get("clarity_trend", 0), -0.5, 0.5)

        # Update hysteresis gates
        high_stress = self.stress_gate.update(stress)
        high_arousal = self.arousal_gate.update(arousal)
        is_positive = self.valence_gate.update(valence)  # True if valence > 0.3

        # ===================================================================
        # ADAPTIVE STRATEGY: Early turns use instant, later use trends
        # ===================================================================
        
        # Blend factor: 0.0 = all instant, 1.0 = all trend
        if mode == "trend" and turn_count >= 3:
            trend_weight = min(confidence, 0.9)  # Max 90% trend
        else:
            trend_weight = 0.0  # Pure instant for early turns
        
        instant_weight = 1.0 - trend_weight
        
        # ===================================================================
        # STYLE SELECTION (Multi-Dimensional)
        # ===================================================================
        
        style = self._select_style(
            valence=valence,
            arousal=arousal,
            stress=stress,
            clarity=clarity,
            high_stress=high_stress,
            high_arousal=high_arousal,
            is_positive=is_positive,
            valence_trend=valence_trend,
            stress_trend=stress_trend,
            trend_weight=trend_weight
        )
        
        # ===================================================================
        # STYLE DEGREE (Amplify emotional intensity)
        # ===================================================================
        
        # Instant component: Based on current stress/arousal
        instant_degree = 1.0 + (stress * 0.5) + (arousal * 0.3)
        
        # Trend component: Amplify if emotions are INCREASING
        if stress_trend > 0.1:
            trend_degree = 1.0 + (stress_trend * 3.0) + (stress * 0.3)
        elif stress_trend < -0.1:
            trend_degree = 1.0 - (abs(stress_trend) * 1.5)
        else:
            trend_degree = 1.0 + (stress * 0.3)
        
        # Blend instant and trend
        target_degree = (instant_weight * instant_degree + 
                        trend_weight * trend_degree)
        
        # Aggressive bounds for perceptibility
        target_degree = np.clip(target_degree, 0.6, 2.0)
        
        # Smooth update
        self.current_styledegree = self._smooth(
            self.current_styledegree, target_degree
        )

        # ===================================================================
        # RATE (Speech speed - inversely related to stress/clarity issues)
        # ===================================================================
        
        # Instant component: Slower if stressed or unclear
        instant_rate = int(-stress * 25 + clarity * 15)
        
        # Trend component: Slow down if stress INCREASING, speed up if DECREASING
        if stress_trend > 0.1:
            trend_rate = int(-stress_trend * 50 - stress * 20)  # Slow down significantly
        elif stress_trend < -0.1:
            trend_rate = int(abs(stress_trend) * 30)  # Speed up moderately
        else:
            trend_rate = int(-stress * 20 + clarity * 10)
        
        # Blend instant and trend
        target_rate = int(instant_weight * instant_rate + 
                         trend_weight * trend_rate)
        
        # Perceptible bounds
        target_rate = np.clip(target_rate, -30, 25)
        
        # Smooth update
        self.current_rate = int(self._smooth(self.current_rate, target_rate))

        # ===================================================================
        # PITCH (Emotional expressiveness)
        # ===================================================================
        
        # Instant component: Higher pitch for arousal, lower for stress
        instant_pitch = int((arousal - stress * 0.7) * 10)
        
        # Trend component: Amplify changes
        if arousal_trend > 0.1:
            # Getting more excited -> raise pitch significantly
            trend_pitch = int((arousal + arousal_trend * 2.0) * 12)
        elif valence_trend < -0.1:
            # Getting sadder -> lower pitch
            trend_pitch = int((valence_trend * 15) - stress * 8)
        else:
            trend_pitch = int((arousal - stress * 0.5) * 8)
        
        # Blend instant and trend
        target_pitch = int(instant_weight * instant_pitch + 
                          trend_weight * trend_pitch)
        
        # Perceptible bounds
        target_pitch = np.clip(target_pitch, -15, 15)
        
        # Smooth update
        self.current_pitch = int(self._smooth(self.current_pitch, target_pitch))

        # ===================================================================
        # VOLUME (Stress-responsive)
        # ===================================================================
        
        # Volume changes based on stress level and trend
        if high_stress or (stress_trend > 0.15 and stress > 0.5):
            self.current_volume = "soft"
        elif high_arousal and is_positive:
            self.current_volume = "medium"  # Normal for energetic positive
        else:
            self.current_volume = "medium"
        
        # Update style with history tracking
        self._update_style(style)

        return self._current_params()

    def _select_style(
        self, 
        valence: float, 
        arousal: float, 
        stress: float, 
        clarity: float,
        high_stress: bool,
        high_arousal: bool,
        is_positive: bool,
        valence_trend: float,
        stress_trend: float,
        trend_weight: float
    ) -> str:
        """
        Select voice style based on multi-dimensional emotional state.
        
        Priority (highest to lowest):
        1. De-escalation (high stress + negative)
        2. Empathetic (trending negative + stressed)
        3. Cheerful (positive + energetic)
        4. Excited (trending positive + arousal increasing)
        5. Calm (low arousal + neutral/positive)
        6. Friendly (default)
        """
        
        # CRITICAL: Use trends for more nuanced style selection
        effective_valence = valence + valence_trend * trend_weight * 2.0
        effective_stress = stress + stress_trend * trend_weight * 2.0
        
        # Priority 1: De-escalation (immediate response to high stress)
        if high_stress and valence < -0.2:
            return "calm"
        
        # Priority 2: Empathetic (user getting sadder/more stressed)
        if trend_weight > 0.4 and valence_trend < -0.15 and effective_stress > 0.4:
            return "empathetic"
        
        # Priority 3: Cheerful (positive and energetic)
        if is_positive and high_arousal and clarity > 0.6:
            return "cheerful"
        
        # Priority 4: Excited (trending more positive/aroused)
        if trend_weight > 0.4 and valence_trend > 0.15 and arousal > 0.5:
            return "excited"
        
        # Priority 5: Clarity issues (confused user)
        if clarity < 0.4 or (trend_weight > 0.4 and effective_stress > 0.6):
            return "newscast-casual"  # Clear, articulate
        
        # Priority 6: Calm (low energy, stable)
        if arousal < 0.3 and effective_stress < 0.4 and abs(valence) < 0.3:
            return "calm"
        
        # Default: Friendly
        return "friendly"

    def _update_style(self, new_style: str):
        """Update style with history tracking for smoother transitions"""
        if new_style != self.current_style:
            self.style_history.append(self.current_style)
            if len(self.style_history) > self.max_style_history:
                self.style_history.pop(0)
            
            # Only update if consistently suggested
            if self.style_history.count(new_style) >= 2 or len(self.style_history) < 2:
                self.current_style = new_style
        else:
            # Reset history if style is stable
            self.style_history = []

    def _smooth(self, old: float, new: float) -> float:
        """
        Exponential smoothing with AGGRESSIVE alpha for perceptibility.
        
        Higher alpha = more responsive to new values
        """
        return self.smoothing_alpha * new + (1 - self.smoothing_alpha) * old

    def _current_params(self) -> Dict:
        """Return current TTS parameters as dict"""
        return {
            "style": self.current_style,
            "styledegree": round(self.current_styledegree, 2),
            "rate": f"{int(self.current_rate):+d}%",
            "pitch": f"{int(self.current_pitch):+d}%",
            "volume": self.current_volume,
        }


# =====================================================
# SINGLETON ACCESS
# =====================================================

_tts_controller: Optional[TTSController] = None


def compute_tts_params_from_trends(adaptive_state: Dict) -> Dict:
    """Singleton access to TTS controller"""
    global _tts_controller
    if _tts_controller is None:
        _tts_controller = TTSController()
    return _tts_controller.compute(adaptive_state)


# =====================================================
# AZURE SYNTHESIS
# =====================================================

def synthesize_azure_tts(
    text: str, 
    tts_params: Dict, 
    output_dir: str,
    voice_name: str = DEFAULT_VOICE_NAME
) -> str:
    """
    Synthesize speech using Azure TTS with SSML parameters.
    
    Args:
        text: Text to synthesize
        tts_params: TTS parameters (style, rate, pitch, etc.)
        output_dir: Directory to save audio
        voice_name: Voice to use (default: en-US-DragonV2.1Neural)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"tts_{uuid.uuid4().hex}.wav")

    speech_config = _get_speech_config()
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    ssml = f"""
<speak version="1.0"
 xmlns="http://www.w3.org/2001/10/synthesis"
 xmlns:mstts="http://www.w3.org/2001/mstts"
 xml:lang="en-US">
 <voice name="{voice_name}">
  <mstts:express-as style="{tts_params['style']}"
   styledegree="{tts_params['styledegree']}">
   <prosody rate="{tts_params['rate']}"
    pitch="{tts_params['pitch']}"
    volume="{tts_params['volume']}">
    {text}
   </prosody>
  </mstts:express-as>
 </voice>
</speak>
"""

    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return output_file

    if result.reason == speechsdk.ResultReason.Canceled:
        raise RuntimeError(result.cancellation_details.error_details)

    raise RuntimeError("Unknown Azure TTS failure")