# llm_client.py (UPDATED)

import os
from typing import Dict, Optional
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


def psychological_llm_response(
    user_text: str,
    adaptive_state: Dict,
    conversation_context: Dict,
    memory_context: Optional[str] = None,
) -> str:
    """
    Generates a psychologically-aware response with conversation continuity and long-term memory.
    
    Args:
        user_text: Current user input
        adaptive_state: Trend-based emotional state
        conversation_context: Minimal context (summary + recent turns)
        memory_context: Optional long-term memory block (episodic + semantic)
    """
    
    # Build memory section (if provided)
    memory_section = ""
    if memory_context and memory_context.strip():
        memory_section = f"\n{memory_context}\n"
    
    # Build minimal, constant-size prompt
    system_prompt = f"""You are a psychologically-aware conversational assistant.

CRITICAL: You MUST respond ONLY in English. Do NOT use any other language under any circumstances.

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

Respond naturally to continue the conversation."""

    client = _get_openai_client()

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Faster model for lower latency
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.7,
        max_tokens=150,  # Limit response length for lower latency
    )

    return response.choices[0].message.content.strip()


def extract_semantic_facts(prompt: str) -> str:
    """
    Lightweight LLM call for semantic memory extraction.
    Used by memory orchestrator to extract user facts/preferences.
    """
    client = _get_openai_client()
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You extract stable user facts and preferences from conversations. Return short declarative statements only, one per line."
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # Lower temperature for factual extraction
        max_tokens=150
    )
    
    return response.choices[0].message.content.strip()


def _format_recent_context(recent: list) -> str:
    """Format recent turns concisely (all turns in buffer)"""
    if not recent:
        return "No recent context"
    
    formatted = []
    for idx, turn in enumerate(recent, 1):  # All turns in buffer (max 5)
        formatted.append(f"Turn {idx} - User: {turn.get('user_summary', 'N/A')}")
        formatted.append(f"Turn {idx} - You: {turn.get('system_summary', 'N/A')}")
    
    return "\n".join(formatted)


def _format_trends(trends: Dict) -> str:
    """Format emotional trends for context"""
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