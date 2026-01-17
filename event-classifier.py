import os
os.environ["TORCHCODEC_DISABLE"] = "1"

import torch
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

model.eval()  # inference mode
device = torch.device("cpu")
model.to(device)

# Label mapping
id2label = model.config.id2label


TARGET_LABELS = [
    # Speech
    #"speech",
    #"conversation",
    #"narration",
    #"monologue",
    #"male speech",
    #"female speech",
    #"child speech",
    #"babbling",
    #"whispering",

    # Positive emotion
    "laughter",
    "giggle",
    "baby laughter",
    "belly laugh",
    "chuckle",
    "cheering",
    "applause",
    "whoop",

    # Distress / sadness
    "crying",
    "sobbing",
    "baby cry",
    "whimper",
    "groan",
    "wail",
    "moan",
    "sigh",
    "gasp",
    "pant",
    "sniff",
    "wheeze",

    # Anger / high arousal
    "screaming",
    "shout",
    "yell",
    "children shouting",
    "battle cry",
    "grunt",

    # Neutral / fatigue
    #"snoring",
    #"cough",
    #"sneeze",
    #"hiccup",
    #"throat clearing",
    #"humming",

    # Optional behavior
    #"clapping",
    #"finger snapping",
    #"hands",
    #"writing",
    #"typing"
]


def get_audio_events(audio_path, top_k=10):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Feature extraction
    inputs = extractor(
        audio,
        sampling_rate=sr,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert to probabilities
    probs = torch.softmax(logits, dim=-1)[0]

    # Top-K predictions
    top_probs, top_indices = torch.topk(probs, top_k)

    relevant_events = {}

    for prob, idx in zip(top_probs, top_indices):
        label = id2label[idx.item()]
        score = prob.item()

        if any(t in label.lower() for t in TARGET_LABELS):
            relevant_events[label] = round(score, 3)

    return relevant_events

print(get_audio_events(r"D:\All\Emotion psychologist\kids-laugh-45357.mp3"))