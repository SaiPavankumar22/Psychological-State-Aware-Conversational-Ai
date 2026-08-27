# services/llm_client.py

import os
import logging
from typing import Dict, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

# =====================================================
# CLIENT SINGLETONS
# =====================================================

_nebius_client: Optional[OpenAI] = None
_openai_client: Optional[OpenAI] = None

NEBIUS_MODEL = "google/gemma-3-27b-it"
OPENAI_FALLBACK_MODEL = "gpt-4o-mini"


def _get_nebius_client() -> OpenAI:
    global _nebius_client
    if _nebius_client is None:
        api_key = os.getenv("NEBIUS_API_KEY")
        if not api_key:
            raise RuntimeError("NEBIUS_API_KEY not set")
        _nebius_client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=api_key,
        )
    return _nebius_client


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _chat_with_fallback(
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 150,
) -> str:
    """
    Try Nebius (Gemma 3 27B) first; fall back to OpenAI GPT-4o-mini on any error.
    Returns the response text.
    """
    # --- Primary: Nebius ---
    try:
        client = _get_nebius_client()
        response = client.chat.completions.create(
            model=NEBIUS_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.debug(f"✅ Nebius response received ({NEBIUS_MODEL})")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"⚠️ Nebius call failed ({type(e).__name__}: {e}) — falling back to OpenAI")

    # --- Fallback: OpenAI ---
    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_FALLBACK_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info(f"✅ OpenAI fallback response received ({OPENAI_FALLBACK_MODEL})")
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"❌ OpenAI fallback also failed: {e}", exc_info=True)
        raise RuntimeError(f"Both Nebius and OpenAI LLM calls failed. Last error: {e}")


# =====================================================
# PUBLIC API
# =====================================================

def psychological_llm_response(
    user_text: str,
    adaptive_state: Dict,
    conversation_context: Dict,
    memory_context: Optional[str] = None,
) -> str:
    """
    Generates a psychologically-aware response with conversation continuity and
    long-term memory. Uses Nebius (Gemma 3 27B) with OpenAI (GPT-4o-mini) fallback.
    """
    memory_section = ""
    if memory_context and memory_context.strip():
        memory_section = f"\n{memory_context}\n"

    system_prompt = f"""You are a psychologically-aware conversational assistant.

CRITICAL CONSTRAINTS:
- You MUST respond ONLY in English. Do NOT use any other language under any circumstances.
- Do NOT use markdown formatting. No asterisks (*), hashes (#), underscores (_), backticks (`), or brackets.
- Write in plain natural conversation, as if speaking aloud. The user will hear your response as speech.

You are NOT a therapist. You do NOT diagnose. You do NOT give medical advice.

Your role:
- Respond exclusively in English
- Be emotionally appropriate based on user's psychological trends
- Maintain conversation continuity naturally
- Adjust tone based on emotional trajectory, not instant spikes
- Be calm if user shows sustained stress
- Be clear if user shows confusion trends
- Be engaging if user shows positive trajectory
- Use long-term memory to personalize responses when relevant

{memory_section}Current conversation context:
Topic: {conversation_context.get('dialogue_state', {}).get('primary_topic', 'general')}
Turn count: {conversation_context.get('turn_count', 0)}

Recent exchange:
{_format_recent_context(conversation_context.get('recent_context', []))}

User's emotional state (trend-based, {adaptive_state.get('mode', 'instant')} mode):
- Emotion trend: {_describe_valence(adaptive_state.get('valence', 0))}
- Energy level: {_describe_arousal(adaptive_state.get('arousal', 0))}
- Clarity: {_describe_clarity(adaptive_state.get('clarity', 0))}
- Stress level: {_describe_stress(adaptive_state.get('stress', 0))}

{_format_trends(adaptive_state.get('trends', {}))}

Respond naturally to continue the conversation. Remember: plain text only, no formatting symbols."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    return _chat_with_fallback(messages, temperature=0.7, max_tokens=500)


def extract_semantic_facts(prompt: str) -> str:
    """
    Lightweight LLM call for semantic memory extraction.
    Uses Nebius with OpenAI fallback.
    """
    messages = [
        {
            "role": "system",
            "content": "You extract stable user facts and preferences from conversations. Return short declarative statements only, one per line.",
        },
        {"role": "user", "content": prompt},
    ]
    return _chat_with_fallback(messages, temperature=0.3, max_tokens=150)


# =====================================================
# PROMPT HELPERS
# =====================================================

def _format_recent_context(recent: list) -> str:
    if not recent:
        return "No recent context"
    formatted = []
    for idx, turn in enumerate(recent, 1):
        formatted.append(f"Turn {idx} - User: {turn.get('user_summary', 'N/A')}")
        formatted.append(f"Turn {idx} - You: {turn.get('system_summary', 'N/A')}")
    return "\n".join(formatted)


def _format_trends(trends: Dict) -> str:
    if not trends:
        return ""
    parts = []
    if trends.get('stress_trend', 0) > 0.1:
        parts.append("Stress is increasing")
    elif trends.get('stress_trend', 0) < -0.1:
        parts.append("Stress is decreasing")
    if trends.get('valence_trend', 0) > 0.1:
        parts.append("Mood is improving")
    elif trends.get('valence_trend', 0) < -0.1:
        parts.append("Mood is declining")
    return f"Trends: {', '.join(parts)}" if parts else ""


def _describe_valence(v: float) -> str:
    if v > 0.5: return "Very positive"
    if v > 0.2: return "Positive"
    if v > -0.2: return "Neutral"
    if v > -0.5: return "Negative"
    return "Very negative"


def _describe_arousal(a: float) -> str:
    if a > 0.7: return "High energy"
    if a > 0.4: return "Moderate energy"
    return "Low energy"


def _describe_clarity(c: float) -> str:
    if c > 0.7: return "Very clear"
    if c > 0.4: return "Somewhat clear"
    return "Confused"


def _describe_stress(s: float) -> str:
    if s > 0.7: return "High stress"
    if s > 0.4: return "Moderate stress"
    return "Low stress"
