import torch
import librosa
import json
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# -------------------------------
# Configuration
# -------------------------------
AUDIO_PATH = "/content/T0001G0003S00001.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SER_MODEL_ID = "superb/wav2vec2-large-superb-er"

# -------------------------------
# Load SER model
# -------------------------------
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(SER_MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(SER_MODEL_ID)
model.to(DEVICE).eval()

# -------------------------------
# Load audio
# -------------------------------
audio, sr = librosa.load(AUDIO_PATH, sr=16000, mono=True)

inputs = feature_extractor(
    audio,
    sampling_rate=16000,
    padding=True,
    return_tensors="pt"
)
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

# -------------------------------
# Inference
# -------------------------------
with torch.no_grad():
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

pred_id = torch.argmax(probs).item()
emotion_label = model.config.id2label[pred_id]

emotion_scores = {
    model.config.id2label[i]: round(probs[i].item(), 4)
    for i in range(len(probs))
}

# -------------------------------
# Output
# -------------------------------
output = {
    "emotion": emotion_label,
    "emotion_scores": emotion_scores
}

print(json.dumps(output, indent=2))
