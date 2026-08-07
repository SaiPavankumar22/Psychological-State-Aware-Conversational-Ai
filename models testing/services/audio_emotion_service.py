import modal
from services.image import image

app = modal.App("emotion-system")

@app.cls(gpu="A10G", image=image, timeout=600)
class AudioEmotionService:
    @modal.enter()
    def load_models(self):
        from transformers import (
            AutoFeatureExtractor,
            AutoModelForAudioClassification,
            Wav2Vec2FeatureExtractor,
            Wav2Vec2ForSequenceClassification,
        )

        self.device = "cuda"

        # AST
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

        # SER
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

    @modal.method()
    def run(self, audio_path: str):
        import time
        import torch
        import librosa

        start = time.perf_counter()

        audio, _ = librosa.load(audio_path, sr=16000, mono=True)

        ast_inputs = self.ast_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            ast_logits = self.ast_model(**ast_inputs).logits[0]

        ser_inputs = self.ser_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            ser_logits = self.ser_model(**ser_inputs).logits[0]

        return {
            "ast": torch.softmax(ast_logits, dim=-1).tolist(),
            "ser": torch.softmax(ser_logits, dim=-1).tolist(),
            "latency": round(time.perf_counter() - start, 3),
        }
