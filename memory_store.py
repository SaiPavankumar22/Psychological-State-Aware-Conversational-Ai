# memory_store.py
"""
Enterprise-grade vector memory store using Qdrant.
Manages episodic and semantic long-term memory with reinforcement learning.
"""

import os
import time
import asyncio
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import uuid
from xmlrpc import client
from tenacity import retry, stop_after_attempt, wait_exponential

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """Memory entry structure"""
    id: str
    content: str
    embedding: List[float]
    user_id: str
    session_id: Optional[str]
    memory_type: str
    topic: Optional[str]
    valence: Optional[float]
    stress: Optional[float]
    importance_score: float
    reinforcement_count: int
    confidence: Optional[float]
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]


class QdrantMemoryStore:
    """
    Vector memory store using Qdrant for long-term memory.
    
    Collections:
    - episodic_memory: Conversational episodes with emotional context
    - semantic_memory: User preferences, facts, behaviors
    
    Features:
    - Thread-safe initialization with lock
    - Automatic retry with exponential backoff (3 attempts)
    - Connection resilience & fallback modes
    """
    
    EPISODIC_COLLECTION = "episodic_memory"
    SEMANTIC_COLLECTION = "semantic_memory"
    VECTOR_SIZE = 384  # all-MiniLM-L6-v2 dimension
    
    def __init__(self):
        """Initialize Qdrant client and embedding model (lazy loading)"""
        self._client: Optional[QdrantClient] = None
        self._embedder: Optional[SentenceTransformer] = None
        self._initialized = False
        self._fallback_mode = False  # Use fallback if Qdrant unavailable
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _connect_qdrant(self):
        """Connect to Qdrant with retry logic (handles both local and cloud URLs)"""
        qdrant_url = os.environ.get("QDRANT_URL")
        qdrant_port = os.environ.get("QDRANT_PORT")
        api_key = os.environ.get("QDRANT_API_KEY")
        
        if not api_key:
            raise ValueError("QDRANT_API_KEY not found in environment")
        if not qdrant_url:
            raise ValueError("QDRANT_URL not found in environment")
        
        # Handle both cloud URLs (https://...) and local URLs (http://localhost)
        if qdrant_url.startswith("https://"):
            # Cloud URL - use as-is (port already included in domain)
            full_url = qdrant_url
            logger.debug("🔗 Using Qdrant Cloud URL (HTTPS)")
        elif qdrant_url.startswith("http://"):
            # Local URL - append port if provided and not already in URL
            if qdrant_port and f":{qdrant_port}" not in qdrant_url:
                full_url = f"{qdrant_url}:{qdrant_port}"
            else:
                full_url = qdrant_url
            logger.debug("🔗 Using Qdrant Local URL (HTTP)")
        else:
            # Assume it's just hostname, needs protocol
            if qdrant_port:
                full_url = f"http://{qdrant_url}:{qdrant_port}"
            else:
                full_url = f"http://{qdrant_url}:6333"  # Default Qdrant port
            logger.debug("🔗 Constructed Qdrant URL from hostname")
        
        logger.info(f"🔗 Connecting to Qdrant: {full_url}")
        
        try:
            # Create client with timeout
            client = QdrantClient(
                url=full_url,
                api_key=api_key,
                timeout=15.0  # Increased timeout for retries
            )
            
            # Test connection with health check
            health = client.health_check()
            if not health:
                raise RuntimeError("Qdrant health check failed")
            
            logger.info("✅ Qdrant connection successful")
            return client
            
        except Exception as e:
            logger.error(f"❌ Qdrant connection failed: {type(e).__name__}: {e}")
            raise
        return client
    
    def _ensure_initialized(self):
        """Lazy initialization of client and embedder"""
        if self._initialized:
            return
        
        try:
            # Initialize embedder first (cheaper than Qdrant)
            logger.info("🧠 Loading sentence-transformers model...")
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✅ Embedder loaded")
            
            # Initialize Qdrant client with retry
            self._client = self._connect_qdrant()
            
            # Ensure collections exist
            self._create_collections()
            
            self._initialized = True
            logger.info("✅ Memory store fully initialized")
            
        except Exception as e:
            logger.error(f"❌ Memory store initialization failed: {e}")
            self._fallback_mode = True
            logger.warning("⚠️ Entering fallback mode: Memory functionality disabled")
            raise
    
    @property
    def client(self) -> Optional[QdrantClient]:
        """Get Qdrant client (initializes if needed, returns None in fallback)"""
        try:
            if not self._initialized and not self._fallback_mode:
                self._ensure_initialized()
            return self._client
        except Exception:
            self._fallback_mode = True
            return None
    
    @property
    def embedder(self) -> Optional[SentenceTransformer]:
        """Get embedder (initializes if needed)"""
        try:
            if not self._initialized and not self._fallback_mode:
                self._ensure_initialized()
            return self._embedder
        except Exception:
            self._fallback_mode = True
            return None
    
    def is_available(self) -> bool:
        """Check if Qdrant is available"""
        if self._fallback_mode:
            return False
        try:
            if self.client:
                self.client.health_check()
                return True
        except Exception:
            self._fallback_mode = True
        return False
    
    def _create_collections(self):
        """Create Qdrant collections if they don't exist"""
        try:
            if not self.client:
                logger.warning("⚠️ Qdrant unavailable, skipping collection creation")
                return
            
            # Check if episodic_memory exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            # Create episodic_memory collection
            if self.EPISODIC_COLLECTION not in collection_names:
                logger.info(f"📦 Creating collection: {self.EPISODIC_COLLECTION}")
                self.client.create_collection(
                    collection_name=self.EPISODIC_COLLECTION,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created {self.EPISODIC_COLLECTION}")
            
            # Create semantic_memory collection
            if self.SEMANTIC_COLLECTION not in collection_names:
                logger.info(f"📦 Creating collection: {self.SEMANTIC_COLLECTION}")
                self.client.create_collection(
                    collection_name=self.SEMANTIC_COLLECTION,
                    vectors_config=VectorParams(
                        size=self.VECTOR_SIZE,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"✅ Created {self.SEMANTIC_COLLECTION}")
        
        except Exception as e:
            logger.error(f"❌ Error creating collections: {e}")
            self._fallback_mode = True
            # Continue without throwing - let system work in degraded mode
    
    def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text"""
        try:
            if not self.embedder:
                logger.warning("⚠️ Embedder unavailable, returning None")
                return None
            embedding = self.embedder.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"❌ Error embedding text: {e}")
            return None
    
    def store_episodic_memory(
        self,
        user_id: str,
        session_id: str,
        content: str,
        topic: str,
        valence: float,
        stress: float,
        importance_score: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Store an episodic memory (conversation turn).
        Returns: memory_id (UUID of stored memory) or None if failed
        """
        try:
            if not self.client:
                logger.warning("⚠️ Qdrant unavailable, cannot store episodic memory")
                return None
            
            if not content or len(content.strip()) < 2:
                logger.warning("❌ Invalid content for episodic memory")
                return None
            
            memory_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Generate embedding
            embedding = self.embed_text(content)
            if not embedding:
                logger.warning("❌ Failed to generate embedding for episodic memory")
                return None
            
            # Create point
            point = PointStruct(
                id=memory_id,
                vector=embedding,
                payload={
                    "user_id": user_id,
                    "session_id": session_id,
                    "memory_type": "episodic",
                    "content": content,
                    "topic": topic,
                    "valence": float(valence),
                    "stress": float(stress),
                    "importance_score": float(importance_score),
                    "reinforcement_count": 1,
                    "created_at": timestamp,
                    "updated_at": timestamp,
                    "metadata": metadata or {}
                }
            )
            
            # Insert
            self.client.upsert(
                collection_name=self.EPISODIC_COLLECTION,
                points=[point]
            )
            
            logger.info(f"💾 Stored episodic memory: {memory_id[:8]}... (importance: {importance_score:.2f})")
            return memory_id
        
        except Exception as e:
            logger.error(f"❌ Error storing episodic memory: {e}")
            self._fallback_mode = True
            return None
    
    def store_semantic_memory(
        self,
        user_id: str,
        content: str,
        memory_type: str,  # "preference", "fact", "behavior"
        importance_score: float,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a semantic memory (user preference/fact).
        
        Returns:
            memory_id: UUID of stored memory
        """

        if not self.client:
            logger.warning("⚠️ Qdrant unavailable, cannot store semantic memory")
            return None

        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Generate embedding
        embedding = self.embed_text(content)
        
        # Create point
        point = PointStruct(
            id=memory_id,
            vector=embedding,
            payload={
                "user_id": user_id,
                "memory_type": memory_type,
                "content": content,
                "importance_score": importance_score,
                "reinforcement_count": 1,
                "confidence": confidence,
                "created_at": timestamp,
                "updated_at": timestamp,
                "metadata": metadata or {}
            }
        )
        
        # Insert
        self.client.upsert(
            collection_name=self.SEMANTIC_COLLECTION,
            points=[point]
        )
        
        print(f"💾 Stored semantic memory: {memory_id[:8]}... ({memory_type}, confidence: {confidence:.2f})")
        return memory_id
    
    def retrieve_episodic_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 3,
        min_importance: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant episodic memories (gracefully fails on Qdrant unavailable).
        
        Returns: List of scored memories with content and metadata
        """
        try:
            if not self.client:
                logger.warning("⚠️ Qdrant unavailable, returning empty episodic memories")
                return []
            
            if not query or len(query.strip()) < 2:
                logger.warning("⚠️ Invalid query for episodic retrieval")
                return []
            
            # Generate query embedding
            query_embedding = self.embed_text(query)
            if not query_embedding:
                logger.warning("⚠️ Failed to generate query embedding")
                return []
            
            # Search with filter
            results = self.client.search(
                collection_name=self.EPISODIC_COLLECTION,
                query_vector=query_embedding,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        ),
                        FieldCondition(
                            key="importance_score",
                            range={"gte": min_importance}
                        )
                    ]
                ),
                limit=limit * 2  # Over-fetch for scoring
            )
            
            # Score and rank
            scored_memories = []
            for result in results:
                payload = result.payload
                
                # Validate payload structure
                if not payload or "created_at" not in payload:
                    logger.warning(f"⚠️ Invalid payload for memory {result.id}, skipping")
                    continue
                
                try:
                    semantic_sim = result.score
                    importance = payload.get("importance_score", 0.5)
                    reinforcement = payload.get("reinforcement_count", 1)
                    
                    # Recency decay (exponential)
                    created_at = datetime.fromisoformat(payload["created_at"])
                    age_days = (datetime.now() - created_at).days
                    recency_decay = 0.95 ** age_days  # Decay 5% per day
                    
                    # Composite score
                    score = (
                        0.5 * semantic_sim +
                        0.2 * importance +
                        0.2 * (1.0 - 1.0 / (1.0 + reinforcement)) +  # log-like
                        0.1 * recency_decay
                    )
                    
                    scored_memories.append({
                        "id": result.id,
                        "content": payload.get("content", ""),
                        "topic": payload.get("topic", "unknown"),
                        "valence": payload.get("valence", 0.0),
                        "stress": payload.get("stress", 0.5),
                        "importance": importance,
                        "reinforcement_count": reinforcement,
                        "score": score,
                        "created_at": payload["created_at"]
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Error scoring memory {result.id}: {e}")
                    continue
            
            # Sort by composite score
            scored_memories.sort(key=lambda x: x["score"], reverse=True)
            
            return scored_memories[:limit]
        
        except Exception as e:
            logger.error(f"❌ Error retrieving episodic memories: {e}")
            self._fallback_mode = True
            return []
    
    def retrieve_semantic_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 3,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant semantic memories (preferences/facts).
        """
        if not self.client:
            logger.warning("⚠️ Qdrant unavailable, returning empty semantic memories")
            return []

        # Generate query embedding
        query_embedding = self.embed_text(query)
        
        # Search with filter
        results = self.client.search(
            collection_name=self.SEMANTIC_COLLECTION,
            query_vector=query_embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    ),
                    FieldCondition(
                        key="confidence",
                        range={
                            "gte": min_confidence
                        }
                    )
                ]
            ),
            limit=limit * 2
        )
        
        # Score and rank
        scored_memories = []
        for result in results:
            payload = result.payload
            
            semantic_sim = result.score
            importance = payload.get("importance_score", 0.5)
            reinforcement = payload.get("reinforcement_count", 1)
            confidence = payload.get("confidence", 1.0)
            
            # Recency decay
            created_at = datetime.fromisoformat(payload["created_at"])
            age_days = (datetime.now() - created_at).days
            recency_decay = 0.95 ** age_days
            
            # Composite score
            score = (
                0.5 * semantic_sim +
                0.2 * importance +
                0.15 * (1.0 - 1.0 / (1.0 + reinforcement)) +
                0.1 * recency_decay +
                0.05 * confidence
            )
            
            scored_memories.append({
                "id": result.id,
                "content": payload["content"],
                "memory_type": payload.get("memory_type", "fact"),
                "importance": importance,
                "confidence": confidence,
                "reinforcement_count": reinforcement,
                "score": score,
                "created_at": payload["created_at"]
            })
        
        scored_memories.sort(key=lambda x: x["score"], reverse=True)
        
        return scored_memories[:limit]
    
    def find_similar_memory(
        self,
        collection_name: str,
        user_id: str,
        content: str,
        similarity_threshold: float = 0.85
    ) -> Optional[Dict[str, Any]]:
        """
        Check if similar memory exists (for reinforcement).
        
        Returns:
            Memory dict if found, None otherwise
        """
        if not self.client:
            logger.warning("⚠️ Qdrant unavailable, cannot find similar memory")
            return None

        query_embedding = self.embed_text(content)
        
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id)
                    )
                ]
            ),
            limit=1
        )
        
        if results and results[0].score >= similarity_threshold:
            return {
                "id": results[0].id,
                "payload": results[0].payload,
                "similarity": results[0].score
            }
        
        return None
    
    def reinforce_memory(
        self,
        collection_name: str,
        memory_id: str,
        importance_boost: float = 0.1
    ):
        """Reinforce existing memory (increase count and importance)"""
        try:
            if not self.client:
                logger.warning("⚠️ Qdrant unavailable, cannot reinforce memory")
                return
            
            # Get current memory
            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[memory_id]
            )
            
            if not result:
                logger.warning(f"⚠️ Memory {memory_id} not found")
                return
            
            point = result[0]
            payload = point.payload
            
            # Update fields
            payload["reinforcement_count"] = payload.get("reinforcement_count", 1) + 1
            payload["importance_score"] = min(
                1.0,
                payload.get("importance_score", 0.5) + importance_boost
            )
            payload["updated_at"] = datetime.now().isoformat()
            
            # Upsert
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=memory_id,
                        vector=point.vector,
                        payload=payload
                    )
                ]
            )
            
            logger.info(f"🔄 Reinforced memory {memory_id[:8]}... (count: {payload['reinforcement_count']})")
        
        except Exception as e:
            logger.error(f"❌ Error reinforcing memory: {e}")
            self._fallback_mode = True
    
    def get_memory_stats(self, user_id: str) -> Dict[str, int]:
        """Get memory statistics for a user"""
        try:
            if not self.client:
                logger.warning("⚠️ Qdrant unavailable, returning zero stats")
                return {"episodic_count": 0, "semantic_count": 0, "total": 0}
            
            # Count episodic memories
            episodic_count = self.client.count(
                collection_name=self.EPISODIC_COLLECTION,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                )
            )
            
            # Count semantic memories
            semantic_count = self.client.count(
                collection_name=self.SEMANTIC_COLLECTION,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="user_id",
                            match=MatchValue(value=user_id)
                        )
                    ]
                )
            )
            
            return {
                "episodic_count": episodic_count.count,
                "semantic_count": semantic_count.count,
                "total": episodic_count.count + semantic_count.count
            }
        except Exception as e:
            logger.error(f"❌ Error getting stats: {e}")
            self._fallback_mode = True
            return {"episodic_count": 0, "semantic_count": 0, "total": 0}
