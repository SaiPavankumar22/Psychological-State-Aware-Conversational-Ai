import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "SamLowe/roberta-base-go_emotions"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.eval()
device = torch.device("cpu") 
model.to(device)

# Label mapping
id2label = model.config.id2label

def get_text_emotions(text, top_k=5):
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]

    # GoEmotions is MULTI-LABEL → use sigmoid
    probs = torch.sigmoid(logits)

    # Top-K emotions
    top_probs, top_indices = torch.topk(probs, top_k)

    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "label": id2label[idx.item()],
            "score": round(prob.item(), 3)
        })

    return results


texts = [
    "I am so angry and frustrated with everything.",
    "I feel very sad and hopeless today.",
    "Wow, I did not expect that at all!",
    "I feel calm and relaxed.",
    "I am scared about what will happen next."
]

for text in texts:
    emotions = get_text_emotions(text)

    print(f"\nText: {text}")
    print("Top emotions:")
    for e in emotions:
        print(f"  {e['label']}: {e['score']}")
