# services.py (updated for better async support)

import os
import time
import torch
import librosa
import openai

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
    "anger", "fear", "sadness", "grief", "remorse",
    "disgust", "nervousness", "disappointment", "embarrassment",
    "confusion", "realization", "surprise",
    "joy", "optimism", "love", "gratitude", "relief",
    "curiosity", "caring", "desire", "neutral",
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
# ASR (OpenAI Whisper) + TEXT EMOTION
# ===============================

class ASRTextService:
    """
    ASR via OpenAI Whisper API + Text Emotion via RoBERTa
    """

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")

        self.client = openai.OpenAI(api_key=api_key)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("🚀 Initializing Whisper ASR (API)")
        print(f"🚀 Loading Text Emotion model on {self.device}")

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

    def run(self, audio_path: str) -> dict:
        start = time.perf_counter()

        # ---------- ASR (Whisper API) ----------
        with open(audio_path, "rb") as audio_file:
            transcript_resp = self.client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language="en",  # Strict English only
                response_format="verbose_json",
            )

        transcript = transcript_resp.text.strip()

        # ---------- Text Emotion ----------
        inputs = self.tokenizer(
            transcript, 
            return_tensors="pt",
            truncation=True,
            max_length=512
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