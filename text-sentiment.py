from transformers import pipeline

# Load GoEmotions pipeline
pipe = pipeline(
    task="text-classification",
    model="SamLowe/roberta-base-go_emotions",
    return_all_scores=True
)

# Test sentences
texts = [
    "I am so angry and frustrated with everything.",
    "I feel very sad and hopeless today.",
    "Wow, I did not expect that at all!",
    "I feel calm and relaxed.",
    "I am scared about what will happen next."
]

# Run inference
for text in texts:
    results = pipe(text)[0]

    # Sort emotions by confidence
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    print(f"\nText: {text}")
    print("Top emotions:")
    for r in results[:5]:
        print(f"  {r['label']}: {round(r['score'], 3)}")
