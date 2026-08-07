import modal
from services.image import image
app = modal.App("emotion-system")

@app.cls(gpu="A100", image=image, timeout=900)
class ASRTextService:
    @modal.enter()
    def load_models(self):
        import torch
        from nemo.collections.speechlm2.models import SALM
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.device = "cuda"

        # ASR
        self.asr_model = (
            SALM.from_pretrained("nvidia/canary-qwen-2.5b")
            .to(self.device)
            .eval()
        )

        # Text Emotion
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

    @modal.method()
    def run(self, audio_path: str):
        import time
        import torch
        import librosa

        start = time.perf_counter()

        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
        audio_tensor = torch.tensor(audio).unsqueeze(0).to(self.device)
        audio_lens = torch.tensor([audio_tensor.shape[1]]).to(self.device)

        output_ids = self.asr_model.generate(
            prompts=[[{
                "role": "user",
                "content": f"Transcribe the following: {self.asr_model.audio_locator_tag}",
            }]],
            audios=audio_tensor,
            audio_lens=audio_lens,
            max_new_tokens=256,
        )

        transcript = self.asr_model.tokenizer.ids_to_text(
            output_ids[0].cpu()
        )

        inputs = self.tokenizer(
            transcript, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.text_model(**inputs).logits[0]

        return {
            "transcript": transcript,
            "text_emotion": torch.sigmoid(logits).tolist(),
            "latency": round(time.perf_counter() - start, 3),
        }
