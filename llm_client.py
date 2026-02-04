import os

from openai import OpenAI


_client = None

def _get_openai_client():
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _client = OpenAI(api_key=api_key)
    return _client


def psychological_llm_response(user_text: str, psychological_state: dict) -> str:
    """
    Generates a psychologically-aware response without diagnosis.
    """

    system_prompt = f"""
You are a psychologically-aware conversational assistant.

You are NOT a therapist.
You do NOT diagnose mental health conditions.
You do NOT give medical advice.

Your role:
- Be emotionally appropriate
- Adjust tone based on user's psychological state
- Be calm if stressed
- Be clear if confused
- Be engaging if positive
- Be neutral if neutral

User psychological state (do not mention this explicitly):
{psychological_state}
"""

    client = _get_openai_client()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.6,
        max_tokens=200,
    )

    return response.choices[0].message.content.strip()
