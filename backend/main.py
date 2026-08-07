# backend/main.py (PRODUCTION HARDENED)

import asyncio
import tempfile
import os
import json
import base64
import logging
import traceback
import subprocess
from typing import Dict, TYPE_CHECKING
import modal
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# =====================================================
# LOGGING SETUP
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from conversation_state import ConversationState

# =====================================================
# SESSION LOCKING (Thread-safe state management)
# =====================================================
session_locks: Dict[str, asyncio.Lock] = {}

# =====================================================
# MODAL APP & IMAGE (Optimized for cold start)
# =====================================================
app = modal.App("conversational-ai-realtime")

# Layer 1: Base system packages (rarely changes)
image = modal.Image.debian_slim(python_version="3.11").apt_install(
    "git", "ffmpeg", "sox"
)

# Layer 2: Core dependencies (infrequently changes)
image = image.pip_install(
    "fastapi[standard]",
    "websockets",
    "tenacity>=8.0.0",
    "pydantic>=2.0",
)

# Layer 3: ML/AI dependencies (infrequently changes)
image = image.pip_install(
    "numpy",
    "torch==2.6.0",
    "torchaudio==2.6.0",
    "transformers",
    "accelerate",
    "sentence-transformers",
    extra_index_url="https://download.pytorch.org/whl/cu121",
)

# Layer 4: Audio & API dependencies
image = image.pip_install(
    "openai",
    "openai-whisper",
    "azure-cognitiveservices-speech",
    "librosa",
    "soundfile",
    "qdrant-client",
)

# Layer 5: Local files (changes frequently - should be last!)
# Backend core files land in /root/
# Backend service files land in /root/services/ (as a package)
image = (
    image
    .add_local_file("backend/services/__init__.py", remote_path="/root/services/__init__.py")
    .add_local_file("backend/services/fusion.py", remote_path="/root/services/fusion.py")
    .add_local_file("backend/services/services.py", remote_path="/root/services/services.py")
    .add_local_file("backend/services/tts_azure.py", remote_path="/root/services/tts_azure.py")
    .add_local_file("backend/services/llm_client.py", remote_path="/root/services/llm_client.py")
    .add_local_file("backend/emotional_trends.py", remote_path="/root/emotional_trends.py")
    .add_local_file("backend/conversation_state.py", remote_path="/root/conversation_state.py")
    .add_local_file("backend/memory_manager.py", remote_path="/root/memory_manager.py")
    .add_local_file("backend/memory_store.py", remote_path="/root/memory_store.py")
    .add_local_file("backend/memory_orchestrator.py", remote_path="/root/memory_orchestrator.py")
    .add_local_file("frontend/index.html", remote_path="/root/frontend/index.html")
)

# =====================================================
# PERSISTENT SESSION STORAGE (Modal Dict with locking)
# =====================================================
persistent_sessions = modal.Dict.from_name("conversation-sessions", create_if_missing=True)
session_metadata = modal.Dict.from_name("session-metadata", create_if_missing=True)

def get_session_lock(session_id: str) -> asyncio.Lock:
    """Get or create asyncio lock for session (thread-safe)"""
    if session_id not in session_locks:
        session_locks[session_id] = asyncio.Lock()
    return session_locks[session_id]

def get_session(session_id: str) -> Dict:
    """Get or create session state with persistence (NOT thread-safe for Modal - see conversation_endpoint)"""
    try:
        session_data = persistent_sessions[session_id]
        logger.info(f"📂 Loaded existing session: {session_id}")
        
        from conversation_state import ConversationState
        from emotional_trends import EmotionalStateTracker
        from services.tts_azure import TTSController
        
        return {
            "conversation": ConversationState.from_dict(session_data["conversation"]),
            "emotional_tracker": EmotionalStateTracker.from_dict(session_data["emotional_tracker"]),
            "tts_controller": TTSController(),
        }
    except KeyError:
        logger.info(f"✨ Creating new session: {session_id}")
        from conversation_state import ConversationState
        from emotional_trends import EmotionalStateTracker
        from services.tts_azure import TTSController
        
        return {
            "conversation": ConversationState(session_id=session_id),
            "emotional_tracker": EmotionalStateTracker(),
            "tts_controller": TTSController(),
        }
    except Exception as e:
        logger.error(f"❌ Error loading session {session_id}: {e}")
        return get_session(session_id + "_fallback")

def save_session(session_id: str, session: Dict):
    """Save session state to persistent storage"""
    try:
        persistent_sessions[session_id] = {
            "conversation": session["conversation"].to_dict(),
            "emotional_tracker": session["emotional_tracker"].to_dict(),
        }
        logger.debug(f"💾 Saved session: {session_id}")
    except Exception as e:
        logger.error(f"❌ Error saving session {session_id}: {e}")

def update_session_metadata(session_id: str, conversation_state: 'ConversationState', transcript: str = ""):
    """Update or create session metadata for listing/history"""
    import time
    from datetime import datetime
    
    try:
        metadata = session_metadata.get(session_id, {})
    except:
        metadata = {}
    
    if not metadata.get("title") and transcript:
        title = transcript[:50].strip()
        if len(transcript) > 50:
            title += "..."
        metadata["title"] = title if title else "New Conversation"
    elif not metadata.get("title"):
        metadata["title"] = "New Conversation"
    
    metadata.update({
        "session_id": session_id,
        "turn_count": conversation_state.dialogue_state.turn_count,
        "last_updated": datetime.now().isoformat(),
        "created_at": metadata.get("created_at", datetime.now().isoformat()),
    })
    
    session_metadata[session_id] = metadata

def delete_session(session_id: str):
    """Delete a session and its metadata"""
    try:
        if session_id in persistent_sessions:
            del persistent_sessions[session_id]
        if session_id in session_metadata:
            del session_metadata[session_id]
        return True
    except Exception as e:
        logger.error(f"❌ Error deleting session {session_id}: {e}", exc_info=True)
        return False

async def list_sessions() -> list:
    """List all sessions sorted by last updated (uses Modal async Dict interface)"""
    try:
        sessions = []
        async for session_id, metadata in session_metadata.aio.items():
            sessions.append(metadata)
        
        sessions.sort(key=lambda x: x.get("last_updated", ""), reverse=True)
        return sessions
    except Exception as e:
        logger.error(f"❌ Error listing sessions: {e}", exc_info=True)
        return []

# =====================================================
# AUDIO UTILITIES
# =====================================================

def webm_to_wav(webm_bytes, output_path: str) -> str:
    """Convert WebM audio to WAV using ffmpeg"""
    import subprocess
    
    webm_temp = output_path.replace('.wav', '.webm')
    
    with open(webm_temp, 'wb') as f:
        f.write(webm_bytes)
    
    cmd = [
        'ffmpeg', '-y', '-i', webm_temp,
        '-ar', '16000',
        '-ac', '1',
        '-f', 'wav',
        output_path
    ]
    
    subprocess.run(cmd, capture_output=True, check=True)
    os.remove(webm_temp)
    
    return output_path

# =====================================================
# MARKDOWN CLEANING (for TTS)
# =====================================================

def _clean_markdown_for_tts(text: str) -> str:
    """
    Remove markdown formatting from text before sending to TTS.
    Azure TTS reads markdown symbols literally, so we clean them
    to ensure natural speech output.
    """
    import re
    
    if not text:
        return text
    
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'__(.+?)__', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'_(.+?)_', r'\1', text)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'^[\s]*[-*+]\s+', ' ', text, flags=re.MULTILINE)
    text = re.sub(r'\*{3,}(.+?)\*{3,}', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

# =====================================================
# MODAL CLASS - GPU BACKEND
# =====================================================

@app.cls(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[
        modal.Secret.from_name("emotion-env"),
        modal.Secret.from_name("qdrant-credentials"),
    ],
    scaledown_window=600,
)
class ConversationalAI:
    
    @modal.enter()
    def load_models(self):
        """Load all models on container startup"""
        logger.info("🚀 Loading models...")
        
        from services.services import ASRTextService, AudioEmotionService
        from services.fusion import PsychologicalFusion
        from memory_manager import MemoryManager
        from memory_orchestrator import MemoryOrchestrator
        
        self.asr_service = ASRTextService()
        self.audio_service = AudioEmotionService()
        self.fusion = PsychologicalFusion()
        self.memory_mgr = MemoryManager()
        self.memory_orchestrator = MemoryOrchestrator()
        
        logger.info("✅ All models loaded successfully")
        logger.info("🧠 Memory orchestrator initialized")
    
    @modal.method()
    async def process_conversation_turn(
        self, 
        audio_data: str,
        session_id: str,
        voice_name: str = "en-IN-KavyaNeural"
    ) -> dict:
        """Process a conversation turn with full state management"""
        import asyncio
        from services.llm_client import psychological_llm_response
        from services.tts_azure import synthesize_azure_tts
        from services.services import SER_LABELS
        
        should_retrieve = False
        retrieval_reason = "not_triggered"
        emotional_shift_detected = False
        shift_dimension = None
        topic_confidence = 0.0
        memory_context = ""
        
        session = get_session(session_id)
        conv_state = session["conversation"]
        emotional_tracker = session["emotional_tracker"]
        tts_controller = session["tts_controller"]
        
        audio_bytes = base64.b64decode(audio_data)
        current_turn = conv_state.dialogue_state.turn_count + 1
        logger.info(f"🔥 Session: {session_id[:12]}... | Turn {current_turn} | {len(audio_bytes)} bytes")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wav_path = tmp_file.name
        
        try:
            webm_to_wav(audio_bytes, wav_path)
            
            # === PARALLEL ANALYSIS ===
            asr_task = asyncio.to_thread(self.asr_service.run, wav_path)
            audio_task = asyncio.to_thread(self.audio_service.run, wav_path)
            
            asr_result, audio_result = await asyncio.gather(asr_task, audio_task)
            
            transcript = asr_result.get('transcript', '').strip()
            
            if not transcript or len(transcript) < 2:
                logger.warning(f"⚠️ Empty/invalid transcript, using fallback")
                transcript = "[User provided empty or very short input]"
            
            logger.info(f"📝 Transcript: {transcript}")
            
            text_emotion = asr_result.get('text_emotion', {})
            if not isinstance(text_emotion, dict):
                logger.warning("⚠️ Invalid text_emotion format, using empty dict")
                text_emotion = {}
            
            # === EXTRACT ACOUSTIC FEATURES ===
            from services.services import extract_acoustic_features
            acoustic_features = extract_acoustic_features(wav_path, transcript)
            logger.info(f"🎵 Acoustic: rate={acoustic_features['speech_rate']:.2f} WPS, "
                  f"pause={acoustic_features['pause_duration']:.2f}s, "
                  f"jitter={acoustic_features['jitter']:.3f}")
            
            # === FUSION ===
            ser_dict = dict(zip(SER_LABELS, audio_result.get("ser", [0, 0, 0, 0])))
            
            instant_psychological_state = self.fusion.fuse(
                text=text_emotion,
                ser=ser_dict,
                ast=audio_result.get("ast", {}),
                features=acoustic_features,
            )
            
            logger.debug(f"🎭 Instant State: valence={instant_psychological_state['valence']:.2f}, "
                  f"stress={instant_psychological_state['stress']:.2f}")
            
            # === UPDATE EMOTIONAL TRENDS ===
            emotional_tracker.update(instant_psychological_state)
            adaptive_state = emotional_tracker.get_adaptive_state()
            
            logger.debug(f"🧠 Mode: {adaptive_state['mode']}, Confidence: {adaptive_state.get('confidence', 0):.2f}")
            logger.debug(f"📊 Adaptive State: valence={adaptive_state['valence']:.2f}, "
                  f"stress={adaptive_state['stress']:.2f}")
            
            # === MEMORY & TOPIC DETECTION ===
            topic_shifted, new_topic, topic_confidence = self.memory_mgr.update_topic(transcript)
            
            if topic_shifted:
                conv_state.dialogue_state.update_topic(new_topic, topic_confidence)
                logger.info(f"📌 Topic shift: {new_topic} (confidence: {topic_confidence:.2f})")
            
            emotional_shift_detected, shift_dimension = emotional_tracker.detect_major_shift()
            emotional_shift_magnitude = (
                abs(emotional_tracker.dimensions[shift_dimension].get_normalized_trend()) 
                if shift_dimension else 0.0
            )
            
            if emotional_shift_detected:
                logger.info(f"⚡ Emotional shift detected in {shift_dimension}: {emotional_shift_magnitude:.2f}")
            
            # === EVENT-DRIVEN MEMORY RETRIEVAL ===
            should_retrieve, retrieval_reason = self.memory_mgr.should_retrieve_memory(
                user_query=transcript,
                topic_changed=topic_shifted,
                topic_confidence=topic_confidence,
                emotional_shift_magnitude=emotional_shift_magnitude
            )
            
            # === LONG-TERM MEMORY RETRIEVAL ===
            user_id = session_id
            
            if should_retrieve:
                logger.info(f"🗄️ Memory retrieval triggered: {retrieval_reason}")
                try:
                    memories = self.memory_orchestrator.retrieve_relevant_memories(
                        user_id=user_id,
                        current_query=transcript,
                        max_memories=6
                    )
                    
                    memory_context = self.memory_orchestrator.format_memory_for_llm(
                        memories,
                        max_tokens=400
                    )
                    
                    if memory_context:
                        logger.info(f"💡 Retrieved {len(memories['episodic'])} episodic + {len(memories['semantic'])} semantic memories")
                except Exception as e:
                    logger.error(f"⚠️ Memory retrieval failed: {e}", exc_info=True)
                    memory_context = ""
            
            # === GET CONTEXT FOR LLM ===
            llm_context = conv_state.get_context_for_llm()
            
            # === LLM RESPONSE ===
            logger.info("🤖 Generating response...")
            llm_reply = await asyncio.to_thread(
                psychological_llm_response,
                transcript,
                adaptive_state,
                llm_context,
                memory_context
            )
            
            llm_reply = (llm_reply or "").strip()
            
            if not llm_reply or len(llm_reply) < 3:
                logger.warning("⚠️ Invalid LLM response, using fallback")
                llm_reply = "I'm having trouble responding right now. Could you try that again?"
            
            llm_reply_for_tts = _clean_markdown_for_tts(llm_reply)
            
            if len(llm_reply) > 800:
                logger.warning(f"ℹ️ Long LLM response ({len(llm_reply)} chars) - may take longer to synthesize")
            
            logger.info(f"💬 Response: {llm_reply[:100]}...")
            
            # === COMPUTE TTS PARAMS ===
            tts_params = tts_controller.compute(adaptive_state)
            logger.debug(f"🎵 TTS: style={tts_params['style']}, degree={tts_params['styledegree']}, "
                  f"rate={tts_params['rate']}, pitch={tts_params['pitch']}")

            # === GENERATE TTS ===
            tts_audio_bytes = None
            try:
                logger.info("🔊 Synthesizing audio...")
                tts_audio_path = await asyncio.to_thread(
                    synthesize_azure_tts,
                    llm_reply_for_tts,
                    tts_params,
                    "/tmp",
                    voice_name
                )
                
                if not os.path.exists(tts_audio_path):
                    raise FileNotFoundError(f"TTS output file not created: {tts_audio_path}")
                
                file_size = os.path.getsize(tts_audio_path)
                if file_size < 100:
                    raise ValueError(f"TTS audio too small: {file_size} bytes")
                
                with open(tts_audio_path, "rb") as f:
                    tts_audio_bytes = f.read()
                
                if not tts_audio_bytes or len(tts_audio_bytes) < 100:
                    raise ValueError(f"TTS audio read invalid: {len(tts_audio_bytes)} bytes")
                
                logger.info(f"✅ Audio synthesized ({len(tts_audio_bytes)} bytes)")
                os.unlink(tts_audio_path)
                
            except Exception as e:
                logger.error(f"❌ TTS synthesis failed: {e}")
                tts_audio_bytes = None
            
            # === UPDATE CONVERSATION STATE ===
            conv_state.add_turn(
                transcript, 
                llm_reply, 
                instant_psychological_state,
                user_intent=None,
                topic=conv_state.dialogue_state.primary_topic
            )
            
            # === LONG-TERM MEMORY STORAGE ===
            try:
                previous_topic = conv_state.dialogue_state.primary_topic if current_turn > 1 else None
                
                memory_decision = self.memory_orchestrator.detect_memory_worthiness(
                    user_text=transcript,
                    system_response=llm_reply,
                    emotional_state=instant_psychological_state,
                    topic=conv_state.dialogue_state.primary_topic,
                    topic_confidence=topic_confidence,
                    previous_topic=previous_topic
                )
                
                if memory_decision.should_store:
                    logger.info(f"💾 {memory_decision.reason} (importance: {memory_decision.importance_score:.2f})")
                    
                    self.memory_orchestrator.store_episodic_memory_with_reinforcement(
                        user_id=user_id,
                        session_id=session_id,
                        user_text=transcript,
                        system_response=llm_reply,
                        emotional_state=instant_psychological_state,
                        topic=conv_state.dialogue_state.primary_topic,
                        importance_score=memory_decision.importance_score,
                        metadata={
                            "turn": current_turn,
                            "mode": adaptive_state['mode']
                        }
                    )
                else:
                    logger.debug(f"⏭️ {memory_decision.reason}")
                
                if self.memory_orchestrator.should_extract_semantic_memory(user_id, current_turn):
                    logger.info("🔬 Extracting semantic memories...")
                    
                    from services.llm_client import extract_semantic_facts
                    
                    recent_turns_data = [
                        {
                            "user": turn.user_utterance,
                            "ai": turn.system_response
                        }
                        for turn in conv_state.recent_turns
                    ]
                    
                    extracted = await asyncio.to_thread(
                        self.memory_orchestrator.extract_semantic_memories,
                        user_id,
                        recent_turns_data,
                        extract_semantic_facts
                    )
                    
                    if extracted:
                        logger.info(f"✅ Extracted {len(extracted)} semantic memories")
            
            except Exception as e:
                logger.error(f"❌ Memory storage failed: {e}", exc_info=True)
            
            # === SAVE SESSION STATE ===
            save_session(session_id, session)
            
            # === UPDATE SESSION METADATA ===
            update_session_metadata(session_id, conv_state, transcript)
            
            # === PREPARE AUDIO ===
            tts_audio_b64 = ""
            if tts_audio_bytes:
                try:
                    tts_audio_b64 = base64.b64encode(tts_audio_bytes).decode('utf-8')
                    logger.debug(f"✅ Audio encoded ({len(tts_audio_b64)} chars)")
                except Exception as e:
                    logger.error(f"❌ Audio encoding failed: {e}")
                    tts_audio_b64 = ""
            else:
                logger.warning("⚠️ No audio available - text-only response")
            
            # === RETURN RESPONSE ===
            return {
                "transcript": transcript,
                "llm_reply": llm_reply,
                "tts_audio": tts_audio_b64,
                "turn_count": conv_state.dialogue_state.turn_count,
                "emotional_mode": adaptive_state['mode'],
                "instant_state": instant_psychological_state,
                "adaptive_state": {
                    k: v for k, v in adaptive_state.items() 
                    if k not in ['trends', 'stability']
                },
                "tts_params": tts_params,
                "model_outputs": {
                    "asr": {
                        "transcript": transcript,
                        "latency_ms": asr_result['latency'] * 1000,
                    },
                    "text_emotion": {
                        "top_emotions": dict(sorted(
                            asr_result['text_emotion'].items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:5]),
                        "all_emotions": asr_result['text_emotion'],
                    },
                    "audio_analysis": {
                        "ser": {
                            "angry": ser_dict.get("angry", 0),
                            "happy": ser_dict.get("happy", 0),
                            "neutral": ser_dict.get("neutral", 0),
                            "sad": ser_dict.get("sad", 0),
                        },
                        "ast": dict(sorted(
                            audio_result["ast"].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:5]) if audio_result["ast"] else {},
                        "latency_ms": audio_result['latency'] * 1000,
                    },
                    "acoustic_features": acoustic_features,
                    "fusion": {
                        "instant_state": instant_psychological_state,
                        "features_used": acoustic_features
                    },
                },
                "memory_view": {
                    "session_id": session_id,
                    "dialogue_state": conv_state.dialogue_state.to_dict(),
                    "recent_turns": [
                        {
                            "user": turn.user_utterance[:50] + "..." if len(turn.user_utterance) > 50 else turn.user_utterance,
                            "ai": turn.system_response[:50] + "..." if len(turn.system_response) > 50 else turn.system_response,
                            "topic": turn.topic,
                        }
                        for turn in conv_state.recent_turns
                    ],
                    "emotional_trends": {
                        "mode": adaptive_state['mode'],
                        "confidence": adaptive_state.get('confidence', 0),
                        "valence": {
                            "current": adaptive_state['valence'],
                            "trend": adaptive_state.get('trends', {}).get('valence_trend', 0),
                        },
                        "arousal": {
                            "current": adaptive_state['arousal'],
                            "trend": adaptive_state.get('trends', {}).get('arousal_trend', 0),
                        },
                        "stress": {
                            "current": adaptive_state['stress'],
                            "trend": adaptive_state.get('trends', {}).get('stress_trend', 0),
                        },
                        "clarity": {
                            "current": adaptive_state['clarity'],
                            "trend": adaptive_state.get('trends', {}).get('clarity_trend', 0),
                        },
                    },
                    "topic_info": {
                        "current_topic": conv_state.dialogue_state.primary_topic,
                        "confidence": conv_state.dialogue_state.topic_confidence,
                    },
                    "long_term_memory": {
                        "stats": self.memory_orchestrator.get_memory_stats(user_id),
                        "last_retrieval": retrieval_reason if should_retrieve else "N/A"
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            raise
            
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
    
    @modal.method()
    def test_memory_pipeline(self) -> dict:
        """
        Test function to verify memory system integration.
        """
        logger.info("🧪 Running memory pipeline tests...")
        results = {}
        
        try:
            logger.info("📝 Test 1: Storing episodic memory")
            memory_id1 = self.memory_orchestrator.memory_store.store_episodic_memory(
                user_id="test_user",
                session_id="test_session",
                content="User: I'm working on a startup called SmartForge\nAI: That's exciting! What does SmartForge do?",
                topic="startup",
                valence=0.6,
                stress=0.4,
                importance_score=0.8
            )
            results["insert_episodic"] = {"status": "✅ PASS", "memory_id": memory_id1[:12] if memory_id1 else "None"}
            
            logger.info("📝 Test 2: Retrieving episodic memory")
            retrieved_episodic = self.memory_orchestrator.memory_store.retrieve_episodic_memories(
                user_id="test_user",
                query="Tell me about my startup",
                limit=3
            )
            results["retrieve_episodic"] = {
                "status": "✅ PASS" if len(retrieved_episodic) > 0 else "❌ FAIL",
                "count": len(retrieved_episodic)
            }
            
            print("\n📝 Test 3: Testing reinforcement")
            similar = self.memory_orchestrator.memory_store.find_similar_memory(
                collection_name="episodic_memory",
                user_id="test_user",
                content="User: I'm working on SmartForge\nAI: Tell me more!",
                similarity_threshold=0.75
            )
            if similar:
                self.memory_orchestrator.memory_store.reinforce_memory(
                    collection_name="episodic_memory",
                    memory_id=similar["id"],
                    importance_boost=0.1
                )
                results["reinforcement"] = {
                    "status": "✅ PASS",
                    "similarity": round(similar["similarity"], 3)
                }
            else:
                results["reinforcement"] = {"status": "⚠️ SKIP (no similar memory)"}
            
            logger.info("📝 Test 4: Storing semantic memory")
            memory_id2 = self.memory_orchestrator.memory_store.store_semantic_memory(
                user_id="test_user",
                content="User prefers Python for backend development",
                memory_type="preference",
                importance_score=0.9,
                confidence=0.95
            )
            results["insert_semantic"] = {"status": "✅ PASS", "memory_id": memory_id2[:12] if memory_id2 else "None"}
            
            logger.info("📝 Test 5: Retrieving semantic memory")
            retrieved_semantic = self.memory_orchestrator.memory_store.retrieve_semantic_memories(
                user_id="test_user",
                query="What language does the user prefer?",
                limit=3
            )
            results["retrieve_semantic"] = {
                "status": "✅ PASS" if len(retrieved_semantic) > 0 else "❌ FAIL",
                "count": len(retrieved_semantic)
            }
            
            logger.info("📝 Test 6: Checking memory stats")
            stats = self.memory_orchestrator.get_memory_stats("test_user")
            results["memory_stats"] = {
                "status": "✅ PASS",
                "episodic": stats["episodic_count"],
                "semantic": stats["semantic_count"],
                "total": stats["total"]
            }
            
            logger.info("✅ All tests completed!")
            results["overall"] = "✅ MEMORY SYSTEM FUNCTIONAL"
            
        except Exception as e:
            logger.error(f"❌ Test failed with error: {e}", exc_info=True)
            results["overall"] = f"❌ FAILED: {str(e)}"
        
        return results
    
    @modal.asgi_app()
    def fastapi_app(self):
        """Create FastAPI app with WebSocket endpoint"""
        web_app = FastAPI(title="Conversational AI API")
        
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # === RATE LIMITING STATE ===
        last_request_time = {}
        rate_limit_window = 2.0
        
        def check_rate_limit(session_id: str) -> bool:
            """Check if request is within rate limit. Returns True if allowed."""
            current_time = datetime.now().timestamp()
            
            if session_id not in last_request_time:
                last_request_time[session_id] = current_time
                return True
            
            time_since_last = current_time - last_request_time[session_id]
            if time_since_last >= rate_limit_window:
                last_request_time[session_id] = current_time
                return True
            
            return False
        
        @web_app.get("/")
        async def root():
            """Serve frontend interface from file"""
            with open("/root/frontend/index.html", "r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        
        # =====================================================
        # SESSION MANAGEMENT API ENDPOINTS
        # =====================================================
        
        @web_app.get("/api/sessions")
        async def get_sessions():
            """Get all sessions (for history sidebar)"""
            sessions = await list_sessions()
            return {"sessions": sessions}
        
        @web_app.post("/api/sessions/new")
        async def create_new_session():
            """Create a new session and return its ID"""
            import uuid
            from conversation_state import ConversationState
            
            new_session_id = f"session_{uuid.uuid4().hex[:12]}"
            
            update_session_metadata(
                new_session_id,
                ConversationState(session_id=new_session_id),
                transcript=""
            )
            
            return {"session_id": new_session_id}
        
        @web_app.delete("/api/sessions/{session_id}")
        async def delete_session_endpoint(session_id: str):
            """Delete a session"""
            success = delete_session(session_id)
            if success:
                return {"success": True, "message": f"Session {session_id} deleted"}
            else:
                return {"success": False, "message": "Failed to delete session"}
        
        # === HEALTH CHECK ENDPOINTS ===
        @web_app.get("/health")
        async def health_check():
            """Basic health check - Modal's health prober"""
            logger.debug("📊 Health check")
            return {"status": "healthy"}

        @web_app.get("/readiness")
        async def readiness_check():
            """Readiness check - Modal restarts if fails"""
            try:
                memory_available = self.memory_store.is_available()
                
                if memory_available:
                    logger.info("✅ Readiness: Fully ready (with long-term memory)")
                    return {"status": "ready", "memory": "available"}
                else:
                    logger.warning("⚠️ Readiness: Degraded (no long-term memory)")
                    return {"status": "ready", "memory": "unavailable"}
                    
            except Exception as e:
                logger.error(f"❌ Readiness check failed: {e}")
                return {"status": "not_ready", "error": str(e)}
        
        @web_app.get("/liveness")
        async def liveness_check():
            """Liveness check - Modal restarts if fails"""
            try:
                logger.debug("💓 Liveness check")
                return {"status": "alive"}
            except Exception as e:
                logger.error(f"❌ Liveness check failed: {e}")
                raise HTTPException(status_code=500, detail="Not alive")
        
        @web_app.websocket("/ws/conversation")
        async def conversation_endpoint(websocket: WebSocket):
            """WebSocket endpoint for continuous conversation"""
            client_host = websocket.client.host if websocket.client else "unknown"
            logger.info(f"🔗 WebSocket connection from {client_host}")
            
            await websocket.accept()
            session_id = None
            
            try:
                message = await websocket.receive_text()
                
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Invalid JSON: {e}")
                    await websocket.send_json({"error": "Invalid JSON format"})
                    await websocket.close(code=1008)
                    return
                
                if 'audio' not in data or 'session_id' not in data:
                    logger.warning("⚠️ Missing required fields (audio or session_id)")
                    await websocket.send_json({
                        "error": "Missing required fields",
                        "required": ["audio", "session_id"],
                        "optional": ["voice_name"]
                    })
                    await websocket.close(code=1008)
                    return
                
                audio_base64 = data['audio']
                session_id = data['session_id']
                voice_name = data.get('voice_name', 'en-US-DragonV2.1Neural')
                
                if not check_rate_limit(session_id):
                    logger.warning(f"⚠️ Rate limit exceeded for session {session_id}")
                    await websocket.send_json({
                        "error": "Too Many Requests",
                        "message": f"Please wait {rate_limit_window} seconds between requests",
                        "retry_after": rate_limit_window
                    })
                    await websocket.close(code=1008, reason="Rate limit exceeded")
                    return
                
                if not audio_base64 or len(audio_base64) < 100:
                    logger.warning(f"⚠️ Audio too small: {len(audio_base64) if audio_base64 else 0} bytes")
                    await websocket.send_json({"error": "Audio data too small"})
                    await websocket.close(code=1008)
                    return
                
                logger.info(f"📥 Processing - Session: {session_id}, Voice: {voice_name}, Audio: {len(audio_base64)} bytes")
                
                try:
                    result = await self.process_conversation_turn.remote.aio(
                        audio_base64,
                        session_id,
                        voice_name
                    )
                    
                    logger.debug(f"✅ Conversation turn completed - Turn: {result.get('turn_count', '?')}")
                    await websocket.send_json(result)
                    logger.info(f"✅ Response sent to {client_host}")
                    
                except asyncio.TimeoutError:
                    logger.error("❌ Conversation processing timeout")
                    error_response = {
                        "error": "Processing timeout",
                        "transcript": "[Processing took too long]",
                        "llm_reply": "Sorry, I'm taking longer than expected. Could you try again?"
                    }
                    try:
                        await websocket.send_json(error_response)
                    except:
                        pass
                    await websocket.close(code=1011, reason="Processing timeout")
                    
                except Exception as e:
                    logger.error(f"❌ Conversation processing failed: {e}", exc_info=True)
                    error_response = {
                        "error": "Processing failed",
                        "transcript": "[Processing failed]",
                        "llm_reply": "I encountered an error. Please try again."
                    }
                    try:
                        await websocket.send_json(error_response)
                    except:
                        pass
                    await websocket.close(code=1011, reason="Processing error")
                
            except WebSocketDisconnect:
                logger.info(f"ℹ️ WebSocket disconnected from {client_host}")
                
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}", exc_info=True)
                try:
                    await websocket.send_json({
                        "error": "Server error",
                        "details": str(e)[:100]
                    })
                except:
                    pass
                    
            finally:
                try:
                    logger.debug(f"🔌 Closing WebSocket for {client_host}")
                    await websocket.close()
                except Exception as e:
                    logger.debug(f"⚠️ Error closing WebSocket: {e}")
        
        return web_app
