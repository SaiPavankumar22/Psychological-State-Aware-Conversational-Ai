# memory_manager.py - RESEARCH GRADE

from typing import Dict, List, Optional, Tuple
import numpy as np
import time

# ========================================
# TOPIC DETECTION WITH HYSTERESIS
# ========================================

class TopicDetector:
    """
    Research-grade topic detection with:
    - Confidence scoring
    - Hysteresis (prevents flip-flopping)
    - Temporal stability requirements
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        stability_window: int = 2,
        hysteresis_band: float = 0.15
    ):
        """
        Args:
            confidence_threshold: Minimum confidence for topic change
            stability_window: Number of consecutive matches required
            hysteresis_band: Confidence difference required to switch back
        """
        self.confidence_threshold = confidence_threshold
        self.stability_window = stability_window
        self.hysteresis_band = hysteresis_band
        
        # State tracking
        self.current_topic = "general"
        self.topic_confidence = 1.0
        self.candidate_topic: Optional[str] = None
        self.candidate_count = 0
        
        # Simple keyword-based topics (can be replaced with embeddings)
        self.topic_keywords = {
            "work": ["work", "job", "career", "office", "meeting", "colleague", "boss", "project"],
            "personal": ["feel", "emotion", "stress", "anxious", "happy", "sad", "worried", "upset"],
            "health": ["health", "sick", "doctor", "medical", "pain", "tired", "sleep"],
            "relationships": ["friend", "family", "partner", "relationship", "love", "argument"],
            "general": []
        }
    
    def _compute_topic_confidence(self, text: str, topic: str) -> float:
        """
        Compute confidence score for topic match.
        
        In production: use sentence embeddings + cosine similarity
        Here: simple keyword overlap with normalization
        """
        if topic not in self.topic_keywords:
            return 0.0
        
        keywords = self.topic_keywords[topic]
        if len(keywords) == 0:
            return 0.5  # Default for "general"
        
        text_lower = text.lower()
        words = text_lower.split()
        
        # Count keyword matches
        matches = sum(1 for kw in keywords if kw in text_lower)
        
        # Normalize by number of keywords and text length
        keyword_score = matches / len(keywords) if keywords else 0
        density_score = matches / max(len(words), 1)
        
        # Combined score
        confidence = (keyword_score + density_score) / 2
        
        return min(confidence, 1.0)
    
    def detect_topic(self, text: str) -> Tuple[str, float]:
        """
        Detect topic with confidence scoring.
        
        Returns:
            (topic, confidence)
        """
        # Compute confidence for all topics
        confidences = {
            topic: self._compute_topic_confidence(text, topic)
            for topic in self.topic_keywords.keys()
        }
        
        # Get best match
        best_topic = max(confidences, key=confidences.get)
        best_confidence = confidences[best_topic]
        
        return best_topic, best_confidence
    
    def update(self, text: str) -> Tuple[bool, str, float]:
        """
        Update topic with hysteresis and stability requirements.
        
        Returns:
            (has_changed, current_topic, confidence)
        """
        detected_topic, confidence = self.detect_topic(text)
        
        # Case 1: Same topic, update confidence
        if detected_topic == self.current_topic:
            self.topic_confidence = max(
                self.topic_confidence,
                confidence
            )
            self.candidate_topic = None
            self.candidate_count = 0
            return False, self.current_topic, self.topic_confidence
        
        # Case 2: New candidate topic
        if detected_topic != self.candidate_topic:
            # Reset candidate tracking
            self.candidate_topic = detected_topic
            self.candidate_count = 1
            return False, self.current_topic, self.topic_confidence
        
        # Case 3: Same candidate, increment count
        self.candidate_count += 1
        
        # Case 4: Check if candidate meets stability + confidence requirements
        if self.candidate_count >= self.stability_window:
            # Hysteresis: require higher confidence to switch
            required_confidence = self.confidence_threshold
            if confidence > self.topic_confidence:
                # Switching to higher-confidence topic: easier
                required_confidence = self.confidence_threshold - self.hysteresis_band
            else:
                # Switching to lower-confidence topic: harder
                required_confidence = self.confidence_threshold + self.hysteresis_band
            
            if confidence >= required_confidence:
                # Topic change confirmed
                self.current_topic = detected_topic
                self.topic_confidence = confidence
                self.candidate_topic = None
                self.candidate_count = 0
                return True, self.current_topic, self.topic_confidence
        
        # No change yet
        return False, self.current_topic, self.topic_confidence


# ========================================
# MEMORY RETRIEVAL POLICY
# ========================================

class MemoryRetrievalPolicy:
    """
    Event-driven memory retrieval with rate limiting.
    NOT called every turn.
    """
    
    def __init__(
        self,
        min_retrieval_interval: float = 30.0,  # seconds
        topic_shift_threshold: float = 0.7,
        emotional_shift_threshold: float = 0.3
    ):
        self.min_retrieval_interval = min_retrieval_interval
        self.topic_shift_threshold = topic_shift_threshold
        self.emotional_shift_threshold = emotional_shift_threshold
        
        self.last_retrieval_time = 0.0
    
    def should_retrieve(
        self,
        topic_changed: bool,
        topic_confidence: float,
        emotional_shift_magnitude: float,
        has_explicit_recall: bool,
        current_time: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Decide if memory retrieval is needed.
        
        Returns:
            (should_retrieve, reason)
        """
        current_time = current_time or time.time()
        
        # Rate limiting: don't retrieve too frequently
        time_since_last = current_time - self.last_retrieval_time
        if time_since_last < self.min_retrieval_interval and not has_explicit_recall:
            return False, "rate_limited"
        
        # Explicit recall request (highest priority)
        if has_explicit_recall:
            self.last_retrieval_time = current_time
            return True, "explicit_recall"
        
        # Topic shift with high confidence
        if topic_changed and topic_confidence >= self.topic_shift_threshold:
            self.last_retrieval_time = current_time
            return True, "topic_shift"
        
        # Major emotional shift
        if emotional_shift_magnitude >= self.emotional_shift_threshold:
            self.last_retrieval_time = current_time
            return True, "emotional_shift"
        
        return False, "none"
    
    @staticmethod
    def detect_explicit_recall(user_query: str) -> bool:
        """Detect explicit memory recall requests"""
        recall_keywords = [
            "remember", "recall", "earlier", "before", 
            "you said", "we talked", "last time", "previously"
        ]
        return any(kw in user_query.lower() for kw in recall_keywords)


# ========================================
# MEMORY MANAGER
# ========================================

class MemoryManager:
    """
    Research-grade memory management with:
    - Principled topic detection
    - Event-driven retrieval
    - Rate limiting
    - Confidence scoring
    """
    
    def __init__(self):
        self.topic_detector = TopicDetector()
        self.retrieval_policy = MemoryRetrievalPolicy()
    
    def update_topic(self, user_text: str) -> Tuple[bool, str, float]:
        """
        Update topic state.
        
        Returns:
            (topic_changed, current_topic, confidence)
        """
        return self.topic_detector.update(user_text)
    
    def should_retrieve_memory(
        self,
        user_query: str,
        topic_changed: bool,
        topic_confidence: float,
        emotional_shift_magnitude: float
    ) -> Tuple[bool, str]:
        """
        Decide if memory retrieval is needed.
        
        Returns:
            (should_retrieve, reason)
        """
        has_explicit_recall = self.retrieval_policy.detect_explicit_recall(user_query)
        
        return self.retrieval_policy.should_retrieve(
            topic_changed=topic_changed,
            topic_confidence=topic_confidence,
            emotional_shift_magnitude=emotional_shift_magnitude,
            has_explicit_recall=has_explicit_recall
        )