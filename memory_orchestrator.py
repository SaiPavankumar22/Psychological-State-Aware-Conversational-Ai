# memory_orchestrator.py
"""
Memory Orchestrator Layer
Manages when and how to store/retrieve long-term memories.
Implements importance scoring, reinforcement, and semantic extraction.
"""

import time
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from memory_store import QdrantMemoryStore


@dataclass
class MemoryDecision:
    """Decision on whether to store a memory"""
    should_store: bool
    importance_score: float
    reason: str


class MemoryOrchestrator:
    """
    Orchestrates long-term memory operations:
    - Decides what to store
    - Manages reinforcement
    - Extracts semantic memories
    - Retrieves relevant memories
    """
    
    def __init__(self):
        """Initialize orchestrator with memory store"""
        self.memory_store = QdrantMemoryStore()
        self.last_semantic_extraction = {}  # user_id -> timestamp
        self.semantic_extraction_interval = 5  # Extract every 5 turns
    
    def detect_memory_worthiness(
        self,
        user_text: str,
        system_response: str,
        emotional_state: Dict[str, float],
        topic: str,
        topic_confidence: float,
        previous_topic: Optional[str] = None
    ) -> MemoryDecision:
        """
        Determine if this turn is worth storing as episodic memory.
        
        Importance formula:
        importance = 0.3 * emotional_intensity
                   + 0.3 * topic_shift_strength
                   + 0.2 * user_self_reference
                   + 0.2 * novelty_score
        
        Store only if importance > 0.6
        """
        
        # 1. Emotional Intensity (0-1)
        valence = abs(emotional_state.get("valence", 0.0))
        arousal = emotional_state.get("arousal", 0.5)
        stress = emotional_state.get("stress", 0.5)
        
        # High emotion = valence extreme OR high arousal OR high stress
        emotional_intensity = max(
            valence,  # Strong positive or negative
            arousal,  # High activation
            stress    # High stress
        )
        
        # 2. Topic Shift Strength (0-1)
        topic_shift_strength = 0.0
        if previous_topic and previous_topic != topic:
            # Significant topic shift with high confidence
            topic_shift_strength = topic_confidence
        
        # 3. User Self-Reference (0-1)
        # Detect personal pronouns and self-disclosure
        self_reference_patterns = [
            r'\bI\b', r'\bme\b', r'\bmy\b', r'\bmine\b',
            r'\bmyself\b', r'\bI\'m\b', r'\bI\'ve\b', r'\bI\'ll\b'
        ]
        
        user_lower = user_text.lower()
        self_ref_count = sum(
            len(re.findall(pattern, user_lower, re.IGNORECASE))
            for pattern in self_reference_patterns
        )
        
        # Normalize (cap at 5 mentions)
        user_self_reference = min(1.0, self_ref_count / 5.0)
        
        # 4. Novelty Score (0-1)
        # Heuristic: longer messages = more information
        # Normalize by typical length (50 words)
        word_count = len(user_text.split())
        novelty_score = min(1.0, word_count / 50.0)
        
        # Calculate importance
        importance = (
            0.3 * emotional_intensity +
            0.3 * topic_shift_strength +
            0.2 * user_self_reference +
            0.2 * novelty_score
        )
        
        # Decision threshold
        should_store = importance > 0.6
        
        # Reason
        if should_store:
            reasons = []
            if emotional_intensity > 0.7:
                reasons.append("high emotion")
            if topic_shift_strength > 0.7:
                reasons.append("topic shift")
            if user_self_reference > 0.5:
                reasons.append("self-disclosure")
            if novelty_score > 0.5:
                reasons.append("detailed message")
            
            reason = f"Store: {', '.join(reasons)}"
        else:
            reason = "Below importance threshold"
        
        return MemoryDecision(
            should_store=should_store,
            importance_score=importance,
            reason=reason
        )
    
    def store_episodic_memory_with_reinforcement(
        self,
        user_id: str,
        session_id: str,
        user_text: str,
        system_response: str,
        emotional_state: Dict[str, float],
        topic: str,
        importance_score: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store episodic memory or reinforce if similar exists.
        
        Returns:
            memory_id (new or reinforced)
        """
        # Create memory content (combine user + AI for context)
        content = f"User: {user_text}\nAI: {system_response}"
        
        # Check for similar memory
        similar = self.memory_store.find_similar_memory(
            collection_name=self.memory_store.EPISODIC_COLLECTION,
            user_id=user_id,
            content=content,
            similarity_threshold=0.85
        )
        
        if similar:
            # Reinforce existing memory
            memory_id = similar["id"]
            self.memory_store.reinforce_memory(
                collection_name=self.memory_store.EPISODIC_COLLECTION,
                memory_id=memory_id,
                importance_boost=0.1
            )
            print(f"🔄 Reinforced similar memory (similarity: {similar['similarity']:.2f})")
        else:
            # Store new memory
            memory_id = self.memory_store.store_episodic_memory(
                user_id=user_id,
                session_id=session_id,
                content=content,
                topic=topic,
                valence=emotional_state.get("valence", 0.0),
                stress=emotional_state.get("stress", 0.5),
                importance_score=importance_score,
                metadata=metadata
            )
        
        return memory_id
    
    def extract_semantic_memories(
        self,
        user_id: str,
        recent_turns: List[Dict],
        llm_extract_function: Optional[callable] = None
    ) -> List[str]:
        """
        Extract stable semantic memories from recent conversation.
        
        Called every N turns to consolidate facts/preferences.
        
        Args:
            recent_turns: List of recent dialogue turns
            llm_extract_function: Function to call LLM for extraction
        
        Returns:
            List of extracted memory contents
        """
        # Check if extraction is due
        current_time = time.time()
        last_extraction = self.last_semantic_extraction.get(user_id, 0)
        
        if current_time - last_extraction < 300:  # Min 5 minutes between extractions
            return []
        
        # Build context from recent turns
        context = "\n".join([
            f"User: {turn.get('user', '')}\nAI: {turn.get('ai', '')}"
            for turn in recent_turns[-5:]  # Last 5 turns
        ])
        
        # Extract using LLM (if provided)
        extracted_facts = []
        
        if llm_extract_function:
            try:
                # Call LLM to extract facts
                prompt = f"""Extract stable long-term user facts or preferences from this conversation.
Return short declarative statements only.
Ignore temporary context or current emotions.

Conversation:
{context}

Extract 1-3 key facts about the user (preferences, behaviors, goals):"""
                
                facts_text = llm_extract_function(prompt)
                
                # Parse facts (one per line)
                extracted_facts = [
                    line.strip()
                    for line in facts_text.split('\n')
                    if line.strip() and len(line.strip()) > 10
                ]
                
            except Exception as e:
                print(f"❌ LLM extraction failed: {e}")
        
        # Store extracted facts as semantic memories
        stored_ids = []
        for fact in extracted_facts:
            # Determine memory type (heuristic)
            fact_lower = fact.lower()
            if any(word in fact_lower for word in ['prefer', 'like', 'love', 'hate', 'enjoy']):
                memory_type = "preference"
            elif any(word in fact_lower for word in ['always', 'usually', 'often', 'never']):
                memory_type = "behavior"
            else:
                memory_type = "fact"
            
            # Check for similar memory
            similar = self.memory_store.find_similar_memory(
                collection_name=self.memory_store.SEMANTIC_COLLECTION,
                user_id=user_id,
                content=fact,
                similarity_threshold=0.85
            )
            
            if similar:
                # Reinforce
                self.memory_store.reinforce_memory(
                    collection_name=self.memory_store.SEMANTIC_COLLECTION,
                    memory_id=similar["id"],
                    importance_boost=0.15
                )
                print(f"🔄 Reinforced semantic: {fact[:50]}...")
            else:
                # Store new
                memory_id = self.memory_store.store_semantic_memory(
                    user_id=user_id,
                    content=fact,
                    memory_type=memory_type,
                    importance_score=0.8,  # High importance for extracted facts
                    confidence=0.9
                )
                stored_ids.append(memory_id)
                print(f"💾 New semantic: {fact[:50]}...")
        
        # Update last extraction time
        self.last_semantic_extraction[user_id] = current_time
        
        return extracted_facts
    
    def retrieve_relevant_memories(
        self,
        user_id: str,
        current_query: str,
        max_memories: int = 6
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve relevant episodic and semantic memories.
        
        Returns:
            {
                "episodic": [...],
                "semantic": [...]
            }
        """
        # Retrieve top 3 episodic
        episodic = self.memory_store.retrieve_episodic_memories(
            user_id=user_id,
            query=current_query,
            limit=3,
            min_importance=0.6
        )
        
        # Retrieve top 3 semantic
        semantic = self.memory_store.retrieve_semantic_memories(
            user_id=user_id,
            query=current_query,
            limit=3,
            min_confidence=0.7
        )
        
        # Limit total to max_memories
        total = episodic + semantic
        if len(total) > max_memories:
            # Re-rank all by score
            total.sort(key=lambda x: x["score"], reverse=True)
            total = total[:max_memories]
            
            # Separate again
            episodic = [m for m in total if "topic" in m]
            semantic = [m for m in total if "memory_type" in m]
        
        return {
            "episodic": episodic,
            "semantic": semantic
        }
    
    def format_memory_for_llm(
        self,
        memories: Dict[str, List[Dict]],
        max_tokens: int = 400
    ) -> str:
        """
        Format memories into concise text block for LLM injection.
        
        Returns:
            Formatted memory string (under max_tokens)
        """
        lines = []
        
        # Semantic memories (preferences/facts)
        if memories["semantic"]:
            lines.append("USER LONG-TERM MEMORY:")
            for mem in memories["semantic"][:3]:
                lines.append(f"- {mem['content']}")
            lines.append("")
        
        # Episodic memories (past events)
        if memories["episodic"]:
            lines.append("RELEVANT PAST EVENTS:")
            for mem in memories["episodic"][:3]:
                # Shorten content
                content = mem["content"].split("\n")[0]  # User utterance only
                if len(content) > 80:
                    content = content[:77] + "..."
                
                lines.append(f"- {content} (topic: {mem['topic']})")
            lines.append("")
        
        formatted = "\n".join(lines)
        
        # Token estimation (rough: ~4 chars per token)
        estimated_tokens = len(formatted) / 4
        
        if estimated_tokens > max_tokens:
            # Truncate
            char_limit = int(max_tokens * 4)
            formatted = formatted[:char_limit] + "..."
        
        return formatted.strip()
    
    def should_extract_semantic_memory(
        self,
        user_id: str,
        turn_count: int
    ) -> bool:
        """Check if semantic extraction should run this turn"""
        return turn_count > 0 and turn_count % self.semantic_extraction_interval == 0
    
    def get_memory_stats(self, user_id: str) -> Dict:
        """Get memory statistics for debugging"""
        return self.memory_store.get_memory_stats(user_id)
