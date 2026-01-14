import os
os.environ["TORCHCODEC_DISABLE"] = "1"

import librosa
from transformers import pipeline

event_classifier = pipeline(
    "audio-classification",
    model="MIT/ast-finetuned-audioset-10-10-0.4593",
    device=-1  # CPU
)

def get_audio_events(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    results = event_classifier(
        {"array": audio, "sampling_rate": sr},
        top_k=10
    )

    target_labels = [
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

    relevant_events = {}
    for r in results:
        if any(t in r["label"].lower() for t in target_labels):
            relevant_events[r["label"]] = round(r["score"], 3)

    return relevant_events


print(get_audio_events("D:\All\Emotion psychologist\kids-laugh-45357.mp3"))
