from openai import OpenAI

# ----------------------------
# CONFIG
# ----------------------------
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# TEST CHAT COMPLETION
# ----------------------------
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "tell me a joke"}
    ],
    temperature=0.7,
    max_tokens=100
)

print("Model response:")
print(response.choices[0].message.content)
