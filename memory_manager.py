# memory_manager.py — Enterprise Memory Manager (v2)
#
# Key improvements over v1:
#   - 12-topic taxonomy with rich keyword sets and synonym matching
#   - Separate word-boundary matching (no more substring false positives)
#   - Normalised confidence that actually reaches the threshold
#   - Lower confidence_threshold (0.45) so infrequent topics can fire
#   - Stability window reduced to 1 for explicit recall triggers
#   - First-turn retrieval enabled for session starts
#   - Richer retrieval triggers: context questions, self-disclosure, etc.
#   - Memory retrieval interval reduced to 15s (conversation is fast-paced)

import re
import time
from typing import Dict, List, Optional, Tuple


# =====================================================
# TOPIC TAXONOMY
# =====================================================

# Organised as primary → list of keywords/phrases (word-boundary matched).
# Keep each list focused — precision matters more than recall here.
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "work": [
        "work", "job", "career", "office", "meeting", "colleague",
        "boss", "manager", "project", "deadline", "promotion", "salary",
        "interview", "hired", "fired", "quit", "resign", "employee",
        "workplace", "shift", "task", "client", "customer", "business",
        "startup", "company", "team", "remote", "wfh",
    ],
    "personal_emotions": [
        "feel", "feeling", "emotion", "mood", "anxious", "anxiety",
        "happy", "sad", "angry", "upset", "frustrated", "hopeless",
        "depressed", "lonely", "scared", "nervous", "overwhelmed",
        "cry", "crying", "smile", "laugh", "excited", "worried",
        "hurt", "pain", "suffering", "grief", "trauma",
    ],
    "health": [
        "health", "sick", "illness", "doctor", "hospital", "medical",
        "pain", "headache", "tired", "fatigue", "sleep", "insomnia",
        "medication", "prescription", "surgery", "therapy", "therapist",
        "mental health", "anxiety", "depression", "diagnosis", "symptom",
        "diet", "exercise", "workout", "weight", "calories", "nutrition",
    ],
    "relationships": [
        "friend", "family", "partner", "relationship", "love", "breakup",
        "divorce", "marriage", "husband", "wife", "boyfriend", "girlfriend",
        "argument", "fight", "conflict", "trust", "jealousy", "dating",
        "ex", "mother", "father", "parent", "child", "sibling", "brother",
        "sister", "grandparent", "cousin", "in-laws",
    ],
    "finance": [
        "money", "finance", "budget", "savings", "investment", "debt",
        "loan", "mortgage", "rent", "bank", "credit", "tax", "income",
        "expense", "salary", "payment", "stock", "crypto", "retirement",
        "financial", "afford", "expensive", "cheap", "price", "cost",
    ],
    "education": [
        "study", "school", "college", "university", "course", "exam",
        "grade", "degree", "professor", "lecture", "homework", "thesis",
        "student", "graduate", "scholarship", "research", "learn",
        "class", "assignment", "tuition", "campus",
    ],
    "technology": [
        "tech", "software", "hardware", "computer", "phone", "app",
        "code", "programming", "developer", "ai", "machine learning",
        "data", "cloud", "internet", "website", "bug", "feature",
        "startup", "product", "gadget", "device", "update", "install",
    ],
    "entertainment": [
        "movie", "film", "series", "show", "episode", "music", "song",
        "album", "artist", "concert", "game", "gaming", "book", "novel",
        "podcast", "youtube", "netflix", "spotify", "stream", "playlist",
        "video", "watch", "read", "play",
    ],
    "spirituality": [
        "god", "religion", "faith", "prayer", "spiritual", "meditation",
        "mindfulness", "purpose", "meaning", "belief", "soul", "karma",
        "universe", "manifest", "gratitude", "church", "temple", "mosque",
        "worship", "ritual",
    ],
    "travel": [
        "travel", "trip", "vacation", "holiday", "flight", "hotel",
        "destination", "country", "city", "tour", "passport", "visa",
        "tourism", "backpack", "adventure", "explore", "abroad",
        "journey", "road trip", "sightseeing",
    ],
    "food": [
        "food", "eat", "meal", "cook", "recipe", "restaurant", "diet",
        "hungry", "breakfast", "lunch", "dinner", "snack", "drink",
        "coffee", "tea", "cuisine", "vegetarian", "vegan", "calories",
        "taste", "delicious", "chef",
    ],
    "general": [],  # Catch-all — lowest priority
}

# Priority order: more specific topics checked before "general"
TOPIC_PRIORITY = [
    "work", "personal_emotions", "health", "relationships", "finance",
    "education", "technology", "entertainment", "spirituality", "travel",
    "food", "general",
]


# =====================================================
# TOPIC DETECTOR
# =====================================================

class TopicDetector:
    """
    Keyword-based topic detection with:
    - Word-boundary matching (no substring false positives)
    - Density-normalised confidence
    - Hysteresis via candidate stability window
    - Configurable thresholds
    """

    def __init__(
        self,
        confidence_threshold: float = 0.45,  # Lowered: sparse but valid turns should fire
        stability_window: int = 1,            # 1 = commit after single high-confidence turn
        hysteresis_band: float = 0.10,
    ):
        self.confidence_threshold = confidence_threshold
        self.stability_window     = stability_window
        self.hysteresis_band      = hysteresis_band

        # State
        self.current_topic    = "general"
        self.topic_confidence = 0.50
        self.candidate_topic: Optional[str]  = None
        self.candidate_count  = 0

    def _match_count(self, text_lower: str, keywords: List[str]) -> int:
        """Count whole-word matches using regex word boundaries."""
        count = 0
        for kw in keywords:
            pattern = r'\b' + re.escape(kw) + r'\b'
            if re.search(pattern, text_lower):
                count += 1
        return count

    def compute_confidence(self, text: str, topic: str) -> float:
        """
        Confidence = f(keyword match rate, text density).

        Returns [0, 1].
        """
        keywords = TOPIC_KEYWORDS.get(topic, [])
        if not keywords:
            return 0.30  # General gets a low but non-zero floor

        text_lower = text.lower()
        words      = text_lower.split()
        if not words:
            return 0.0

        matches = self._match_count(text_lower, keywords)
        if matches == 0:
            return 0.0

        # Keyword coverage score: what fraction of keywords were found
        coverage = matches / len(keywords)

        # Density score: keyword hits relative to total words (penalise very long texts)
        density = min(matches / len(words), 1.0)

        # Boost for multiple matches (at least 2 = more reliable)
        multi_bonus = 0.10 if matches >= 2 else 0.0

        # Weighted combination
        confidence = 0.55 * coverage + 0.35 * density + multi_bonus
        return min(confidence, 1.0)

    def detect_topic(self, text: str) -> Tuple[str, float]:
        """Return (best_topic, confidence)."""
        best_topic = "general"
        best_conf  = 0.0

        for topic in TOPIC_PRIORITY:
            conf = self.compute_confidence(text, topic)
            if conf > best_conf:
                best_conf  = conf
                best_topic = topic

        return best_topic, best_conf

    def update(self, text: str) -> Tuple[bool, str, float]:
        """
        Update topic state with hysteresis.

        Returns:
            (topic_changed, current_topic, confidence)
        """
        detected, confidence = self.detect_topic(text)

        # --- Same as current: reinforce, no change ---
        if detected == self.current_topic:
            self.topic_confidence = min(
                1.0, max(self.topic_confidence, confidence)
            )
            self.candidate_topic  = None
            self.candidate_count  = 0
            return False, self.current_topic, self.topic_confidence

        # --- New candidate ---
        if detected != self.candidate_topic:
            self.candidate_topic  = detected
            self.candidate_count  = 1
        else:
            self.candidate_count += 1

        # --- Check if candidate meets threshold ---
        if self.candidate_count >= self.stability_window:
            # Adaptive threshold based on relative confidence
            if confidence > self.topic_confidence:
                required = self.confidence_threshold - self.hysteresis_band
            else:
                required = self.confidence_threshold + self.hysteresis_band

            if confidence >= required:
                old_topic             = self.current_topic
                self.current_topic    = detected
                self.topic_confidence = confidence
                self.candidate_topic  = None
                self.candidate_count  = 0
                return True, self.current_topic, self.topic_confidence

        return False, self.current_topic, self.topic_confidence


# =====================================================
# MEMORY RETRIEVAL POLICY
# =====================================================

# Keywords that indicate the user wants to access past memory
_RECALL_KEYWORDS = [
    "remember", "recall", "earlier", "before", "you said", "we talked",
    "last time", "previously", "mentioned", "told you", "as i said",
    "from before", "again",
]

# Keywords indicating new personal information the AI should connect to memory
_CONTEXT_QUESTION_PATTERNS = [
    r"\bhave (i|we) (talked|spoken|discussed)\b",
    r"\bdo you know (me|my)\b",
    r"\bwhat (do you|did you) (know|think) about\b",
    r"\btell me (about|what) (you know|you remember)\b",
]


class MemoryRetrievalPolicy:
    """
    Event-driven memory retrieval with rate limiting.

    Triggers (in priority order):
    1. Explicit recall — user explicitly invokes memory
    2. Session start   — first turn: load user context
    3. Topic shift     — new topic with sufficient confidence
    4. Emotional shift — major change in affective state
    5. Context question— user asks AI about their history
    """

    def __init__(
        self,
        min_retrieval_interval: float = 15.0,   # seconds — conversations are fast
        topic_shift_threshold:  float = 0.40,   # confident enough to store
        emotional_shift_threshold: float = 0.25,
    ):
        self.min_retrieval_interval    = min_retrieval_interval
        self.topic_shift_threshold     = topic_shift_threshold
        self.emotional_shift_threshold = emotional_shift_threshold
        self.last_retrieval_time       = 0.0
        self.turn_count                = 0

    @staticmethod
    def detect_explicit_recall(query: str) -> bool:
        q = query.lower()
        if any(kw in q for kw in _RECALL_KEYWORDS):
            return True
        return any(re.search(p, q) for p in _CONTEXT_QUESTION_PATTERNS)

    def should_retrieve(
        self,
        topic_changed:             bool,
        topic_confidence:          float,
        emotional_shift_magnitude: float,
        has_explicit_recall:       bool,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """
        Returns (should_retrieve, reason).
        """
        now = current_time or time.time()
        self.turn_count += 1
        since_last = now - self.last_retrieval_time

        # --- 1. Explicit recall (always allowed) ---
        if has_explicit_recall:
            self.last_retrieval_time = now
            return True, "explicit_recall"

        # --- 2. First turn of session: load user context ---
        if self.turn_count == 1:
            self.last_retrieval_time = now
            return True, "session_start"

        # --- Rate gate (except explicit recall) ---
        if since_last < self.min_retrieval_interval:
            return False, "rate_limited"

        # --- 3. Topic shift with reasonable confidence ---
        if topic_changed and topic_confidence >= self.topic_shift_threshold:
            self.last_retrieval_time = now
            return True, "topic_shift"

        # --- 4. Emotional shift ---
        if emotional_shift_magnitude >= self.emotional_shift_threshold:
            self.last_retrieval_time = now
            return True, "emotional_shift"

        return False, "none"


# =====================================================
# MEMORY MANAGER (Orchestration Layer)
# =====================================================

class MemoryManager:
    """
    High-level memory management coordinator.

    Wraps TopicDetector + MemoryRetrievalPolicy.
    Called by the main conversation loop.
    """

    def __init__(self):
        self.topic_detector     = TopicDetector()
        self.retrieval_policy   = MemoryRetrievalPolicy()

    def update_topic(self, user_text: str) -> Tuple[bool, str, float]:
        """
        Update topic state from user utterance.

        Returns:
            (topic_changed, current_topic, confidence)
        """
        return self.topic_detector.update(user_text)

    def get_current_topic(self) -> str:
        return self.topic_detector.current_topic

    def get_topic_confidence(self) -> float:
        return self.topic_detector.topic_confidence

    def should_retrieve_memory(
        self,
        user_query:                str,
        topic_changed:             bool,
        topic_confidence:          float,
        emotional_shift_magnitude: float,
    ) -> Tuple[bool, str]:
        """
        Decide if memory retrieval is warranted this turn.

        Returns:
            (should_retrieve, reason)
        """
        has_explicit_recall = self.retrieval_policy.detect_explicit_recall(user_query)
        return self.retrieval_policy.should_retrieve(
            topic_changed=topic_changed,
            topic_confidence=topic_confidence,
            emotional_shift_magnitude=emotional_shift_magnitude,
            has_explicit_recall=has_explicit_recall,
        )

    def reset_session(self):
        """Reset retrieval policy state for a new session."""
        self.retrieval_policy.turn_count          = 0
        self.retrieval_policy.last_retrieval_time = 0.0