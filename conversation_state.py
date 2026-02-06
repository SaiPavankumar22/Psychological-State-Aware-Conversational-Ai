# conversation_state.py - RESEARCH GRADE

import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np

# ========================================
# STRUCTURED DIALOGUE STATE
# ========================================

@dataclass
class DialogueTurn:
    """Single turn with structured representation"""
    timestamp: float
    user_utterance: str
    system_response: str
    user_intent: str = "unknown"  # Abstracted intent
    topic: str = "general"  # Current topic
    psychological_state: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)
    
    def get_compact_representation(self) -> Dict:
        """Compact representation for context (no full text)"""
        return {
            "intent": self.user_intent,
            "topic": self.topic,
            "user_summary": self.user_utterance[:30] + "..." if len(self.user_utterance) > 30 else self.user_utterance,
            "system_summary": self.system_response[:30] + "..." if len(self.system_response) > 30 else self.system_response,
            "timestamp": self.timestamp,
        }


@dataclass
class AbstractDialogueState:
    """
    Structured, bounded dialogue state representation.
    Not a string, but a formal state vector.
    """
    # Primary topic (most recent stable topic)
    primary_topic: str = "general"
    topic_confidence: float = 0.0
    
    # Intent tracking (last N intents)
    recent_intents: List[str] = field(default_factory=list)
    max_intent_history: int = 7
    
    # Coherence score (derived from turn similarity)
    coherence_score: float = 1.0
    
    # Turn count
    turn_count: int = 0
    
    def update_topic(self, new_topic: str, confidence: float):
        """Update primary topic with confidence gating"""
        # Only update if confidence is high enough
        if confidence > 0.7:
            self.primary_topic = new_topic
            self.topic_confidence = confidence
        elif confidence > self.topic_confidence:
            # Gradual transition
            self.topic_confidence = confidence
    
    def add_intent(self, intent: str):
        """Add intent to rolling buffer"""
        self.recent_intents.append(intent)
        if len(self.recent_intents) > self.max_intent_history:
            self.recent_intents.pop(0)
    
    def get_dominant_intent(self) -> Optional[str]:
        """Get most frequent recent intent"""
        if not self.recent_intents:
            return None
        
        from collections import Counter
        counts = Counter(self.recent_intents)
        return counts.most_common(1)[0][0]
    
    def to_dict(self) -> Dict:
        return {
            "primary_topic": self.primary_topic,
            "topic_confidence": self.topic_confidence,
            "recent_intents": self.recent_intents,
            "coherence_score": self.coherence_score,
            "turn_count": self.turn_count,
        }


# ========================================
# CONVERSATION STATE
# ========================================

class ConversationState:
    """
    Research-grade conversation state with:
    - Structured dialogue state (not strings)
    - Bounded rolling buffer
    - Coherence tracking
    - Formal guarantees on context size
    """
    
    def __init__(
        self,
        session_id: str,
        max_recent_turns: int = 3,
        summary_update_interval: int = 5
    ):
        self.session_id = session_id
        self.created_at = time.time()
        
        # Bounded rolling buffer (strict size limit)
        self.recent_turns: List[DialogueTurn] = []
        self.max_recent_turns = max_recent_turns
        
        # Structured dialogue state (replaces string summary)
        self.dialogue_state = AbstractDialogueState()
        
        # Summary update policy
        self.summary_update_interval = summary_update_interval
    
    def add_turn(
        self,
        user_text: str,
        system_response: str,
        psychological_state: Dict[str, float],
        user_intent: str = "unknown",
        topic: str = "general"
    ):
        """
        Add turn with strict buffer management.
        
        Guarantees:
        - Buffer size never exceeds max_recent_turns
        - Oldest turn always removed first (FIFO)
        """
        turn = DialogueTurn(
            timestamp=time.time(),
            user_utterance=user_text,
            system_response=system_response,
            user_intent=user_intent,
            topic=topic,
            psychological_state=psychological_state
        )
        
        # FIFO buffer management
        self.recent_turns.append(turn)
        if len(self.recent_turns) > self.max_recent_turns:
            self.recent_turns.pop(0)
        
        # Update structured dialogue state
        self.dialogue_state.add_intent(user_intent)
        self.dialogue_state.turn_count += 1
    
    def should_update_summary(self) -> bool:
        """Check if dialogue state needs updating"""
        return self.dialogue_state.turn_count % self.summary_update_interval == 0
    
    def get_context_for_llm(self) -> Dict:
        """
        Get minimal, bounded context for LLM.
        
        Guarantees:
        - Fixed maximum size (max_recent_turns * 2 utterances)
        - No unbounded growth
        - Structured representation
        """
        # Compact recent turns (fixed size)
        recent_context = [
            turn.get_compact_representation()
            for turn in self.recent_turns[-2:]  # Last 2 turns only
        ]
        
        return {
            "dialogue_state": self.dialogue_state.to_dict(),
            "recent_context": recent_context,
            "turn_count": self.dialogue_state.turn_count,
        }
    
    def to_dict(self) -> Dict:
        """Serialize for storage"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "recent_turns": [turn.to_dict() for turn in self.recent_turns],
            "dialogue_state": self.dialogue_state.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ConversationState':
        """Deserialize from storage"""
        state = cls(session_id=data["session_id"])
        state.created_at = data["created_at"]
        
        state.recent_turns = [
            DialogueTurn(**turn) for turn in data["recent_turns"]
        ]
        
        ds_data = data["dialogue_state"]
        state.dialogue_state = AbstractDialogueState(
            primary_topic=ds_data["primary_topic"],
            topic_confidence=ds_data["topic_confidence"],
            recent_intents=ds_data["recent_intents"],
            coherence_score=ds_data["coherence_score"],
            turn_count=ds_data["turn_count"],
        )
        
        return state