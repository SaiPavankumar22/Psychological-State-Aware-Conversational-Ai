# services.py (updated for better async support)

import os
import time
import torch
import librosa
import whisper

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

# ===============================
# LABEL DEFINITIONS
# ===============================

SER_LABELS = ["angry", "happy", "neutral", "sad"]

PSYCHOLOGICAL_EMOTIONS = [
    # Negative
    "anger", "fear", "sadness", "grief", "remorse",
    "disgust", "nervousness", "disappointment", "embarrassment",
    "annoyance", "disapproval",
    # Cognitive
    "confusion", "realization", "surprise",
    # Positive
    "joy", "optimism", "love", "gratitude", "relief",
    "excitement", "pride", "amusement", "admiration", "approval",
    # Social
    "curiosity", "caring", "desire",
    # Default
    "neutral",
]

TARGET_LABELS = [
    "laughter", "giggle", "baby laughter", "belly laugh",
    "chuckle", "cheering", "applause", "whoop",
    "crying", "sobbing", "baby cry", "whimper",
    "groan", "wail", "moan", "sigh",
    "screaming", "shout", "yell", "children shouting",
    "battle cry", "grunt",
]


# ===============================
# ACOUSTIC FEATURE EXTRACTION
# ===============================

def extract_acoustic_features(audio_path: str, transcript: str) -> dict:
    """
    Extract dynamic acoustic features from audio.
    
    Features:
    - speech_rate: Words per second
    - pause_duration: Average pause length (silence > 0.3s)
    - jitter: Voice pitch instability (std of pitch)
    
    Args:
        audio_path: Path to WAV file
        transcript: Transcribed text
        
    Returns:
        dict with speech_rate, pause_duration, jitter
    """
    import numpy as np
    import librosa
    
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(audio) / sr
    
    # ========== 1. SPEECH RATE ==========
    words = transcript.strip().split()
    word_count = len(words)
    
    if duration > 0 and word_count > 0:
        speech_rate = word_count / duration
    else:
        speech_rate = 3.6  # Default fallback
    
    # ========== 2. PAUSE DURATION ==========
    # Detect silence using RMS energy threshold
    frame_length = 2048
    hop_length = 512
    
    # Compute RMS energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]
    
    # Threshold for silence (adaptive)
    silence_threshold = np.percentile(rms, 20)  # Bottom 20% is silence
    
    # Find silent frames
    is_silent = rms < silence_threshold
    
    # Compute pause durations (contiguous silent frames)
    pauses = []
    current_pause_frames = 0
    
    for silent in is_silent:
        if silent:
            current_pause_frames += 1
        else:
            if current_pause_frames > 0:
                # Convert frames to seconds
                pause_duration_sec = (current_pause_frames * hop_length) / sr
                # Only count pauses > 0.3s
                if pause_duration_sec > 0.3:
                    pauses.append(pause_duration_sec)
                current_pause_frames = 0
    
    # Average pause duration
    if len(pauses) > 0:
        avg_pause_duration = np.mean(pauses)
    else:
        avg_pause_duration = 0.0
    
    # ========== 3. JITTER (Pitch Instability) ==========
    # Extract pitch using YIN algorithm
    f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz('C2'),  # ~65 Hz
        fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
        sr=sr
    )
    
    # Remove unvoiced frames (f0 == fmin means unvoiced)
    voiced_f0 = f0[f0 > librosa.note_to_hz('C2')]
    
    if len(voiced_f0) > 1:
        # Jitter = normalized std of pitch
        mean_pitch = np.mean(voiced_f0)
        if mean_pitch > 0:
            jitter = np.std(voiced_f0) / mean_pitch
        else:
            jitter = 0.1
    else:
        jitter = 0.1  # Default fallback
    
    # Normalize jitter to [0, 1] range (typical jitter is 0.01-0.15)
    jitter = np.clip(jitter, 0.0, 0.3) / 0.3
    
    return {
        "speech_rate": round(speech_rate, 2),
        "pause_duration": round(avg_pause_duration, 3),
        "jitter": round(jitter, 3),
        "audio_duration": round(duration, 2),
        "word_count": word_count
    }


# ===============================
# ASR (Local Whisper) + TEXT EMOTION
# ===============================

class ASRTextService:
    """
    ASR via local openai-whisper (no OpenAI API) + Text Emotion via RoBERTa.

    Model size is controlled by WHISPER_MODEL env var (default: "small").
    Common options: tiny, base, small, medium, large-v3
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model_name = os.getenv("WHISPER_MODEL", "small")

        print(f"🚀 Loading Whisper ASR ({self.whisper_model_name}) on {self.device}")
        print(f"🚀 Loading Text Emotion model on {self.device}")

        self.whisper_model = whisper.load_model(
            self.whisper_model_name,
            device=self.device,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "SamLowe/roberta-base-go_emotions"
        )
        self.text_model = (
            AutoModelForSequenceClassification.from_pretrained(
                "SamLowe/roberta-base-go_emotions"
            )
            .to(self.device)
            .eval()
        )

        print("✅ ASR + Text Emotion models loaded")

    def _transcribe_whisper(self, audio_path: str) -> str:
        """Transcribe a WAV file with local Whisper."""
        result = self.whisper_model.transcribe(
            audio_path,
            language="en",
            fp16=(self.device == "cuda"),
            verbose=False,
        )
        return (result.get("text") or "").strip()

    def run(self, audio_path: str) -> dict:
        start = time.perf_counter()

        # ---------- ASR (local Whisper) ----------
        transcript = self._transcribe_whisper(audio_path)

        # ---------- Text Emotion ----------
        inputs = self.tokenizer(
            transcript,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            logits = self.text_model(**inputs).logits[0]

        probs = torch.sigmoid(logits)
        id2label = self.text_model.config.id2label

        text_emotion = {
            id2label[i]: float(probs[i])
            for i in range(len(id2label))
            if id2label[i] in PSYCHOLOGICAL_EMOTIONS
        }

        latency = round(time.perf_counter() - start, 3)

        return {
            "transcript": transcript,
            "text_emotion": text_emotion,
            "latency": latency,
        }


# ===============================
# AUDIO EVENTS (AST) + SPEECH EMOTION (SER)
# ===============================

class AudioEmotionService:
    """
    Audio Event Detection (AST) + Speech Emotion Recognition (SER)
    Runs on GPU
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"🚀 Loading AST + SER models on {self.device}")

        # ---------- AST ----------
        self.ast_extractor = AutoFeatureExtractor.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.ast_model = (
            AutoModelForAudioClassification.from_pretrained(
                "MIT/ast-finetuned-audioset-10-10-0.4593"
            )
            .to(self.device)
            .eval()
        )

        # ---------- SER ----------
        self.ser_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-large-superb-er"
        )
        self.ser_model = (
            Wav2Vec2ForSequenceClassification.from_pretrained(
                "superb/wav2vec2-large-superb-er"
            )
            .to(self.device)
            .eval()
        )
        
        print("✅ AST + SER models loaded")

    def run(self, audio_path: str) -> dict:
        start = time.perf_counter()

        audio, _ = librosa.load(audio_path, sr=16000, mono=True)

        # ---------- AST ----------
        ast_inputs = self.ast_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            ast_logits = self.ast_model(**ast_inputs).logits[0]

        ast_probs = torch.softmax(ast_logits, dim=-1)
        ast_labels = self.ast_model.config.id2label

        ast = {}
        for idx, prob in enumerate(ast_probs):
            label = ast_labels[idx].lower()
            for target in TARGET_LABELS:
                if target in label:
                    ast[target] = max(ast.get(target, 0.0), float(prob))

        # ---------- SER ----------
        ser_inputs = self.ser_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            ser_logits = self.ser_model(**ser_inputs).logits[0]

        ser_probs = torch.softmax(ser_logits, dim=-1).tolist()

        latency = round(time.perf_counter() - start, 3)

        return {
            "ast": ast,
            "ser": ser_probs,
            "latency": latency,
        }