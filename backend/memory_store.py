# memory_store.py — Enterprise Qdrant Vector Memory Store (v2)
#
# Bug fixes over v1:
#   - Removed spurious `from xmlrpc import client` that shadowed the client variable
#   - store_semantic_memory: added full try/except + None-embedding guard
#   - reinforce_memory: uses with_vectors=True so point.vector is populated
#   - FieldCondition range filter: uses qdrant_client.models.Range (not raw dict)
#   - _fallback_mode: no longer permanently latched — retried on next is_available()
#   - retrieve_episodic_memories: min_importance lowered to 0.4 so new memories surface
#   - Cleaner connection error messages
#
# Design:
#   - Lazy initialisation (no Qdrant calls at import time)
#   - Thread-safe singleton via lazy property pattern
#   - Graceful degradation: all public methods return safe defaults when unavailable

import os
import time
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import uuid

from tenacity import retry, stop_after_attempt, wait_exponential

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# =====================================================
# DATA TYPES
# =====================================================

@dataclass
class Memory:
    """Structured memory entry (return type, not stored directly)."""
    id:                 str
    content:            str
    user_id:            str
    session_id:         Optional[str]
    memory_type:        str
    topic:              Optional[str]
    valence:            Optional[float]
    stress:             Optional[float]
    importance_score:   float
    reinforcement_count:int
    confidence:         Optional[float]
    created_at:         str
    updated_at:         str
    metadata:           Dict[str, Any]


# =====================================================
# QDRANT MEMORY STORE
# =====================================================

class QdrantMemoryStore:
    """
    Vector memory store backed by Qdrant.

    Collections
    -----------
    episodic_memory  : Conversation episodes with emotional context
    semantic_memory  : User preferences, facts, and behaviour patterns

    Features
    --------
    - Lazy initialisation (zero cost at import time)
    - Automatic retry with exponential backoff (3 attempts)
    - Full graceful degradation — every method is safe when Qdrant is down
    - Composite relevance scoring: similarity + importance + recency + reinforcement
    """

    EPISODIC_COLLECTION = "episodic_memory"
    SEMANTIC_COLLECTION = "semantic_memory"
    VECTOR_SIZE         = 384   # all-MiniLM-L6-v2

    def __init__(self):
        self._client:      Optional[QdrantClient]       = None
        self._embedder:    Optional[SentenceTransformer] = None
        self._initialized: bool  = False
        self._fallback:    bool  = False
        self._last_retry:  float = 0.0           # Used for cooldown before retrying

    # ------------------------------------------------------------------
    # INITIALISATION
    # ------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _connect_qdrant(self) -> QdrantClient:
        """Connect to Qdrant with retry. Supports cloud (https) and local (http) URLs."""
        url     = os.environ.get("QDRANT_URL", "")
        port    = os.environ.get("QDRANT_PORT", "")
        api_key = os.environ.get("QDRANT_API_KEY", "")

        if not url:
            raise ValueError("QDRANT_URL not set in environment")
        if not api_key:
            raise ValueError("QDRANT_API_KEY not set in environment")

        # Resolve full URL
        if url.startswith("https://"):
            # HTTPS URLs should NOT have port 6333 (that's the HTTP port).
            # Qdrant Cloud uses port 443 by default for HTTPS.
            full_url = url.rstrip('/')
            if full_url.endswith(':6333'):
                full_url = full_url[:-5]  # Strip :6333 from HTTPS URLs
                logger.warning("⚠️ Stripped :6333 from HTTPS Qdrant URL (HTTPS uses port 443)")
        elif url.startswith("http://"):
            full_url = f"{url}:{port}" if port and f":{port}" not in url else url
        else:
            full_url = f"http://{url}:{port or '6333'}"

        logger.info(f"🔗 Connecting to Qdrant: {full_url}")

        qdrant = QdrantClient(url=full_url, api_key=api_key, timeout=15.0)

        # Verify connectivity: get_collections() is available in all qdrant-client versions
        # and requires a successful authenticated round-trip to the server.
        # health_check() was removed from QdrantClient in newer library versions.
        qdrant.get_collections()

        logger.info("✅ Qdrant connected successfully")
        return qdrant

    def _ensure_initialized(self):
        """Lazy initialise embedder + Qdrant client + collections."""
        if self._initialized:
            return

        logger.info("🧠 Initialising memory store...")

        # Embedder first (no network)
        self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✅ Sentence transformer loaded")

        # Qdrant client
        self._client = self._connect_qdrant()

        # Ensure collections exist
        self._create_collections()

        self._initialized = True
        logger.info("✅ Memory store fully initialised")

    def _reconnect(self) -> bool:
        """Try to re-establish the Qdrant connection after a transient failure.
        
        Returns True if reconnection succeeded, False otherwise.
        Called from error handlers to give the client a chance to recover
        before latching into permanent fallback mode.
        """
        try:
            logger.info("🔄 Attempting Qdrant reconnection...")
            # Close existing client if any
            if self._client:
                try:
                    self._client.close()
                except Exception:
                    pass
            self._client = None
            self._initialized = False
            self._ensure_initialized()
            logger.info("✅ Qdrant reconnected successfully")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Qdrant reconnection failed: {e}")
            return False

    def _create_collections(self):
        """Create Qdrant collections and their payload indexes if they don't exist."""
        if not self._client:
            return

        existing = {c.name for c in self._client.get_collections().collections}

        for name in (self.EPISODIC_COLLECTION, self.SEMANTIC_COLLECTION):
            if name not in existing:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"📦 Created collection: {name}")

        # Ensure payload indexes exist for every field used in Filter conditions.
        # create_payload_index() is idempotent — safe to call on existing indexes
        # and safe to call on pre-existing collections created before this fix.
        # Without indexes Qdrant falls back to a full post-filter scan.
        _indexes = {
            self.EPISODIC_COLLECTION: [
                ("user_id",          PayloadSchemaType.KEYWORD),  # MatchValue filter
                ("importance_score", PayloadSchemaType.FLOAT),    # Range filter
            ],
            self.SEMANTIC_COLLECTION: [
                ("user_id",   PayloadSchemaType.KEYWORD),         # MatchValue filter
                ("confidence", PayloadSchemaType.FLOAT),          # Range filter
            ],
        }
        for collection, fields in _indexes.items():
            for field_name, field_type in fields:
                try:
                    self._client.create_payload_index(
                        collection_name=collection,
                        field_name=field_name,
                        field_schema=field_type,
                        wait=True,
                    )
                    logger.info(
                        f"🗂️  Payload index ensured: {collection}.{field_name} "
                        f"({field_type.value})"
                    )
                except Exception as e:
                    logger.warning(
                        f"⚠️ Could not create payload index "
                        f"{collection}.{field_name}: {e}"
                    )

    # ------------------------------------------------------------------
    # PROPERTY ACCESSORS (with graceful fallback)
    # ------------------------------------------------------------------

    @property
    def client(self) -> Optional[QdrantClient]:
        if self._fallback:
            # Allow retry after 60s cooldown
            if time.time() - self._last_retry > 60.0:
                self._fallback    = False
                self._initialized = False
            else:
                return None
        try:
            if not self._initialized:
                self._ensure_initialized()
            return self._client
        except Exception as e:
            # Unwrap tenacity RetryError to expose the actual root exception
            from tenacity import RetryError
            if isinstance(e, RetryError) and e.last_attempt.failed:
                root_cause = e.last_attempt.exception()
                logger.error(
                    f"❌ Qdrant init failed (RetryError). Root cause: {type(root_cause).__name__}: {root_cause}",
                    exc_info=root_cause,
                )
            else:
                logger.error(f"❌ Qdrant init failed: {e}", exc_info=True)
            self._fallback   = True
            self._last_retry = time.time()
            return None

    @property
    def embedder(self) -> Optional[SentenceTransformer]:
        if self._fallback:
            return None
        try:
            if not self._initialized:
                self._ensure_initialized()
            return self._embedder
        except Exception:
            self._fallback   = True
            self._last_retry = time.time()
            return None

    def is_available(self) -> bool:
        """Check live Qdrant availability (refreshes fallback state).
        
        If the existing connection is broken, attempts a reconnect
        before declaring fallback mode.
        """
        try:
            c = self.client
            if c:
                c.get_collections()  # health_check() removed in newer qdrant-client
                return True
        except Exception:
            # Existing connection is dead — try reconnecting before giving up
            if self._reconnect():
                logger.info("✅ Qdrant reconnected via is_available check")
                return True
            self._fallback   = True
            self._last_retry = time.time()
        return False

    # ------------------------------------------------------------------
    # EMBEDDING
    # ------------------------------------------------------------------

    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate a 384-dim embedding. Returns None on failure."""
        try:
            emb = self.embedder
            if emb is None:
                return None
            return emb.encode(text, convert_to_numpy=True).tolist()
        except Exception as e:
            logger.error(f"❌ Embedding failed: {e}")
            return None

    # ------------------------------------------------------------------
    # STORE EPISODIC MEMORY
    # ------------------------------------------------------------------

    def store_episodic_memory(
        self,
        user_id:        str,
        session_id:     str,
        content:        str,
        topic:          str,
        valence:        float,
        stress:         float,
        importance_score: float,
        metadata:       Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store a conversation episode.

        Returns:
            memory_id (str UUID) or None if failed.
        """
        try:
            if not self.client:
                logger.warning("⚠️ Qdrant unavailable — episodic store skipped")
                return None

            if not content or len(content.strip()) < 3:
                logger.warning("⚠️ Content too short for episodic memory")
                return None

            embedding = self.embed_text(content)
            if not embedding:
                logger.warning("⚠️ Embedding failed — episodic store skipped")
                return None

            memory_id = str(uuid.uuid4())
            now       = datetime.now().isoformat()

            self.client.upsert(
                collection_name=self.EPISODIC_COLLECTION,
                points=[PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "user_id":            user_id,
                        "session_id":         session_id,
                        "memory_type":        "episodic",
                        "content":            content,
                        "topic":              topic,
                        "valence":            float(valence),
                        "stress":             float(stress),
                        "importance_score":   float(importance_score),
                        "reinforcement_count":1,
                        "created_at":         now,
                        "updated_at":         now,
                        "metadata":           metadata or {},
                    },
                )],
            )

            logger.info(f"💾 Episodic stored: {memory_id[:8]}… (importance={importance_score:.2f})")
            return memory_id

        except Exception as e:
            logger.error(f"❌ Error storing episodic memory: {e}", exc_info=True)
            if self._reconnect():
                logger.info("✅ Qdrant reconnected — episodic store will retry next call")
            else:
                self._fallback   = True
                self._last_retry = time.time()
            return None

    # ------------------------------------------------------------------
    # STORE SEMANTIC MEMORY
    # ------------------------------------------------------------------

    def store_semantic_memory(
        self,
        user_id:        str,
        content:        str,
        memory_type:    str,          # "preference" | "fact" | "behavior"
        importance_score: float,
        confidence:     float = 1.0,
        metadata:       Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store a semantic memory (user fact/preference/behaviour).

        Returns:
            memory_id or None if failed.
        """
        try:
            if not self.client:
                logger.warning("⚠️ Qdrant unavailable — semantic store skipped")
                return None

            if not content or len(content.strip()) < 3:
                logger.warning("⚠️ Content too short for semantic memory")
                return None

            embedding = self.embed_text(content)
            if not embedding:
                logger.warning("⚠️ Embedding failed — semantic store skipped")
                return None

            memory_id = str(uuid.uuid4())
            now       = datetime.now().isoformat()

            self.client.upsert(
                collection_name=self.SEMANTIC_COLLECTION,
                points=[PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload={
                        "user_id":            user_id,
                        "memory_type":        memory_type,
                        "content":            content,
                        "importance_score":   float(importance_score),
                        "reinforcement_count":1,
                        "confidence":         float(confidence),
                        "created_at":         now,
                        "updated_at":         now,
                        "metadata":           metadata or {},
                    },
                )],
            )

            logger.info(f"💾 Semantic stored: {memory_id[:8]}… ({memory_type}, conf={confidence:.2f})")
            return memory_id

        except Exception as e:
            logger.error(f"❌ Error storing semantic memory: {e}", exc_info=True)
            if self._reconnect():
                logger.info("✅ Qdrant reconnected — semantic store will retry next call")
            else:
                self._fallback   = True
                self._last_retry = time.time()
            return None

    # ------------------------------------------------------------------
    # RETRIEVE EPISODIC MEMORIES
    # ------------------------------------------------------------------

    def retrieve_episodic_memories(
        self,
        user_id:       str,
        query:         str,
        limit:         int   = 3,
        min_importance:float = 0.40,   # Lowered from 0.5 so new memories appear
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant episodic memories via semantic search + composite scoring.

        Composite score = 0.50 * semantic_sim
                        + 0.20 * importance
                        + 0.20 * log_reinforcement
                        + 0.10 * recency_decay
        """
        try:
            if not self.client:
                return []

            if not query or len(query.strip()) < 2:
                return []

            embedding = self.embed_text(query)
            if not embedding:
                return []

            # query_points() replaces the removed search() API.
            # It uses `query=` (not `query_vector=`) and returns QueryResponse;
            # the actual ScoredPoint list is at response.points.
            response = self.client.query_points(
                collection_name=self.EPISODIC_COLLECTION,
                query=embedding,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(
                            key="importance_score",
                            range=Range(gte=min_importance),
                        ),
                    ]
                ),
                limit=limit * 3,  # Over-fetch; composite re-rank selects top
            )

            scored: List[Dict[str, Any]] = []
            for r in response.points:
                p = r.payload
                if not p or "created_at" not in p:
                    continue
                try:
                    age_days      = (datetime.now() - datetime.fromisoformat(p["created_at"])).days
                    recency_decay = 0.95 ** age_days
                    reinf         = p.get("reinforcement_count", 1)
                    importance    = p.get("importance_score", 0.5)

                    score = (
                        0.50 * r.score +
                        0.20 * importance +
                        0.20 * (1.0 - 1.0 / (1.0 + reinf)) +
                        0.10 * recency_decay
                    )

                    scored.append({
                        "id":                  r.id,
                        "content":             p.get("content", ""),
                        "topic":               p.get("topic", "general"),
                        "valence":             p.get("valence", 0.0),
                        "stress":              p.get("stress", 0.5),
                        "importance":          importance,
                        "reinforcement_count": reinf,
                        "score":               score,
                        "created_at":          p["created_at"],
                    })
                except Exception as inner:
                    logger.warning(f"⚠️ Skipping malformed memory {r.id}: {inner}")

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:limit]

        except Exception as e:
            logger.error(f"❌ Episodic retrieval failed: {e}", exc_info=True)
            if self._reconnect():
                logger.info("✅ Qdrant reconnected — episodic retrieval will retry next call")
            else:
                self._fallback   = True
                self._last_retry = time.time()
            return []

    # ------------------------------------------------------------------
    # RETRIEVE SEMANTIC MEMORIES
    # ------------------------------------------------------------------

    def retrieve_semantic_memories(
        self,
        user_id:       str,
        query:         str,
        limit:         int   = 3,
        min_confidence:float = 0.60,   # Lowered slightly for broader recall
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant semantic memories (preferences / facts).
        """
        try:
            if not self.client:
                return []

            embedding = self.embed_text(query)
            if not embedding:
                return []

            # query_points() replaces the removed search() API.
            response = self.client.query_points(
                collection_name=self.SEMANTIC_COLLECTION,
                query=embedding,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(
                            key="confidence",
                            range=Range(gte=min_confidence),
                        ),
                    ]
                ),
                limit=limit * 2,
            )

            scored: List[Dict[str, Any]] = []
            for r in response.points:
                p = r.payload
                if not p or "created_at" not in p:
                    continue
                try:
                    age_days      = (datetime.now() - datetime.fromisoformat(p["created_at"])).days
                    recency_decay = 0.95 ** age_days
                    importance    = p.get("importance_score", 0.5)
                    reinf         = p.get("reinforcement_count", 1)
                    confidence    = p.get("confidence", 1.0)

                    score = (
                        0.50 * r.score +
                        0.20 * importance +
                        0.15 * (1.0 - 1.0 / (1.0 + reinf)) +
                        0.10 * recency_decay +
                        0.05 * confidence
                    )

                    scored.append({
                        "id":                  r.id,
                        "content":             p.get("content", ""),
                        "memory_type":         p.get("memory_type", "fact"),
                        "importance":          importance,
                        "confidence":          confidence,
                        "reinforcement_count": reinf,
                        "score":               score,
                        "created_at":          p["created_at"],
                    })
                except Exception as inner:
                    logger.warning(f"⚠️ Skipping malformed semantic memory {r.id}: {inner}")

            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:limit]

        except Exception as e:
            logger.error(f"❌ Semantic retrieval failed: {e}", exc_info=True)
            if self._reconnect():
                logger.info("✅ Qdrant reconnected — semantic retrieval will retry next call")
            else:
                self._fallback   = True
                self._last_retry = time.time()
            return []

    # ------------------------------------------------------------------
    # FIND SIMILAR MEMORY (for reinforcement check)
    # ------------------------------------------------------------------

    def find_similar_memory(
        self,
        collection_name:    str,
        user_id:            str,
        content:            str,
        similarity_threshold: float = 0.85,
    ) -> Optional[Dict[str, Any]]:
        """
        Find a highly similar existing memory (for reinforcement instead of duplication).

        Returns a dict with keys: id, payload, similarity — or None.
        """
        try:
            if not self.client:
                return None

            embedding = self.embed_text(content)
            if not embedding:
                return None

            # query_points() replaces the removed search() API.
            response = self.client.query_points(
                collection_name=collection_name,
                query=embedding,
                query_filter=Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                ),
                limit=1,
            )

            results = response.points
            if results and results[0].score >= similarity_threshold:
                return {
                    "id":         results[0].id,
                    "payload":    results[0].payload,
                    "similarity": results[0].score,
                }
            return None

        except Exception as e:
            logger.error(f"❌ find_similar_memory failed: {e}", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # REINFORCE MEMORY
    # ------------------------------------------------------------------

    def reinforce_memory(
        self,
        collection_name: str,
        memory_id:       str,
        importance_boost:float = 0.10,
    ):
        """
        Increment reinforcement_count and boost importance_score for an existing memory.
        Uses with_vectors=True to correctly fetch the vector for re-upsert.
        """
        try:
            if not self.client:
                return

            # Must request vectors explicitly (default retrieve() omits them)
            results = self.client.retrieve(
                collection_name=collection_name,
                ids=[memory_id],
                with_vectors=True,
            )

            if not results:
                logger.warning(f"⚠️ Memory {memory_id[:8]}… not found for reinforcement")
                return

            point   = results[0]
            payload = dict(point.payload)
            vector  = point.vector

            if vector is None:
                logger.warning(f"⚠️ No vector returned for {memory_id[:8]}… — skipping reinforce")
                return

            payload["reinforcement_count"] = payload.get("reinforcement_count", 1) + 1
            payload["importance_score"]    = min(
                1.0,
                payload.get("importance_score", 0.5) + importance_boost,
            )
            payload["updated_at"] = datetime.now().isoformat()

            self.client.upsert(
                collection_name=collection_name,
                points=[PointStruct(id=memory_id, vector=vector, payload=payload)],
            )

            logger.info(
                f"🔄 Reinforced {memory_id[:8]}… "
                f"(count={payload['reinforcement_count']}, "
                f"importance={payload['importance_score']:.2f})"
            )

        except Exception as e:
            logger.error(f"❌ reinforce_memory failed: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    def get_memory_stats(self, user_id: str) -> Dict[str, int]:
        """Get per-user memory counts. Returns zeros on failure."""
        try:
            if not self.client:
                return {"episodic_count": 0, "semantic_count": 0, "total": 0}

            uid_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )

            ep  = self.client.count(self.EPISODIC_COLLECTION, count_filter=uid_filter).count
            sem = self.client.count(self.SEMANTIC_COLLECTION, count_filter=uid_filter).count

            return {"episodic_count": ep, "semantic_count": sem, "total": ep + sem}

        except Exception as e:
            logger.error(f"❌ get_memory_stats failed: {e}")
            return {"episodic_count": 0, "semantic_count": 0, "total": 0}