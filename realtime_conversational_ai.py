# realtime_conversational_ai.py (FIXED)

import asyncio
import tempfile
import os
import json
import base64
from typing import Dict
import modal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# =====================================================
# MODAL APP & IMAGE
# =====================================================
app = modal.App("conversational-ai-realtime")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "sox")
    .pip_install(
        "fastapi[standard]",
        "websockets",
        "numpy",
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "transformers",
        "accelerate",
        "openai",
        "azure-cognitiveservices-speech",
        "librosa",
        "soundfile",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_file("fusion.py", remote_path="/root/fusion.py")
    .add_local_file("emotional_trends.py", remote_path="/root/emotional_trends.py")
    .add_local_file("conversation_state.py", remote_path="/root/conversation_state.py")
    .add_local_file("memory_manager.py", remote_path="/root/memory_manager.py")
    .add_local_file("tts_azure.py", remote_path="/root/tts_azure.py")
    .add_local_file("llm_client.py", remote_path="/root/llm_client.py")
    .add_local_file("services.py", remote_path="/root/services.py")
)

# =====================================================
# PERSISTENT SESSION STORAGE (Modal Dict)
# =====================================================
persistent_sessions = modal.Dict.from_name("conversation-sessions", create_if_missing=True)

def get_session(session_id: str) -> Dict:
    """Get or create session state with persistence"""
    try:
        # Try to load existing session
        session_data = persistent_sessions[session_id]
        
        # Deserialize state objects
        from conversation_state import ConversationState
        from emotional_trends import EmotionalStateTracker
        from tts_azure import TTSController
        
        return {
            "conversation": ConversationState.from_dict(session_data["conversation"]),
            "emotional_tracker": EmotionalStateTracker.from_dict(session_data["emotional_tracker"]),
            "tts_controller": TTSController(),  # Fresh controller (stateless enough)
        }
    except KeyError:
        # Create new session
        from conversation_state import ConversationState
        from emotional_trends import EmotionalStateTracker
        from tts_azure import TTSController
        
        return {
            "conversation": ConversationState(session_id=session_id),
            "emotional_tracker": EmotionalStateTracker(),
            "tts_controller": TTSController(),
        }

def save_session(session_id: str, session: Dict):
    """Save session state to persistent storage"""
    persistent_sessions[session_id] = {
        "conversation": session["conversation"].to_dict(),
        "emotional_tracker": session["emotional_tracker"].to_dict(),
    }

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
# MODAL CLASS - GPU BACKEND
# =====================================================

@app.cls(
    image=image,
    gpu="A10G",
    timeout=3600,
    secrets=[modal.Secret.from_name("emotion-env")],
    scaledown_window=600,
)
class ConversationalAI:
    
    @modal.enter()
    def load_models(self):
        """Load all models on container startup"""
        print("🚀 Loading models...")
        
        from services import ASRTextService, AudioEmotionService
        from fusion import PsychologicalFusion
        from memory_manager import MemoryManager
        
        self.asr_service = ASRTextService()
        self.audio_service = AudioEmotionService()
        self.fusion = PsychologicalFusion()
        self.memory_mgr = MemoryManager()  # ✅ Single instance per container
        
        print("✅ All models loaded successfully")
    
    @modal.method()
    async def process_conversation_turn(
        self, 
        audio_data: str,
        session_id: str,
        voice_name: str = "en-US-DragonV2.1Neural"
    ) -> dict:
        """Process a conversation turn with full state management"""
        import asyncio
        from llm_client import psychological_llm_response
        from tts_azure import synthesize_azure_tts
        from services import SER_LABELS
        
        # ✅ Load session state (from persistent storage)
        session = get_session(session_id)
        conv_state = session["conversation"]
        emotional_tracker = session["emotional_tracker"]
        tts_controller = session["tts_controller"]
        
        # Decode audio
        audio_bytes = base64.b64decode(audio_data)
        print(f"🔥 Turn {conv_state.dialogue_state.turn_count + 1} - {len(audio_bytes)} bytes")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wav_path = tmp_file.name
        
        try:
            # Convert audio
            webm_to_wav(audio_bytes, wav_path)
            
            # === PARALLEL ANALYSIS ===
            asr_task = asyncio.to_thread(self.asr_service.run, wav_path)
            audio_task = asyncio.to_thread(self.audio_service.run, wav_path)
            
            asr_result, audio_result = await asyncio.gather(asr_task, audio_task)
            
            transcript = asr_result['transcript']
            print(f"📝 Transcript: {transcript}")
            
            # === FUSION ===
            ser_dict = dict(zip(SER_LABELS, audio_result["ser"]))
            
            instant_psychological_state = self.fusion.fuse(
                text=asr_result["text_emotion"],
                ser=ser_dict,
                ast=audio_result["ast"],
                features={
                    "speech_rate": 3.6,
                    "pause_duration": 0.5,
                    "jitter": 0.1,
                },
            )
            
            print(f"🎭 Instant State: valence={instant_psychological_state['valence']:.2f}, "
                  f"stress={instant_psychological_state['stress']:.2f}")
            
            # === UPDATE EMOTIONAL TRENDS ===
            emotional_tracker.update(instant_psychological_state)
            adaptive_state = emotional_tracker.get_adaptive_state()
            
            print(f"🧠 Mode: {adaptive_state['mode']}, Confidence: {adaptive_state.get('confidence', 0):.2f}")
            print(f"📊 Adaptive State: valence={adaptive_state['valence']:.2f}, "
                  f"stress={adaptive_state['stress']:.2f}")
            
            # === MEMORY & TOPIC DETECTION ===
            topic_shifted, new_topic, topic_confidence = self.memory_mgr.update_topic(transcript)
            
            if topic_shifted:
                conv_state.dialogue_state.update_topic(new_topic, topic_confidence)
                print(f"📌 Topic shift: {new_topic} (confidence: {topic_confidence:.2f})")
            
            # Detect emotional shifts
            emotional_shift_detected, shift_dimension = emotional_tracker.detect_major_shift()
            emotional_shift_magnitude = (
                abs(emotional_tracker.dimensions[shift_dimension].get_normalized_trend()) 
                if shift_dimension else 0.0
            )
            
            if emotional_shift_detected:
                print(f"⚡ Emotional shift detected in {shift_dimension}: {emotional_shift_magnitude:.2f}")
            
            # === EVENT-DRIVEN MEMORY RETRIEVAL ===
            should_retrieve, retrieval_reason = self.memory_mgr.should_retrieve_memory(
                user_query=transcript,
                topic_changed=topic_shifted,
                topic_confidence=topic_confidence,
                emotional_shift_magnitude=emotional_shift_magnitude
            )
            
            if should_retrieve:
                print(f"🗄️ Memory retrieval triggered: {retrieval_reason}")
                # TODO: Implement vector DB retrieval
            
            # === GET CONTEXT FOR LLM ===
            llm_context = conv_state.get_context_for_llm()
            
            # === LLM RESPONSE ===
            print("🤖 Generating response...")
            llm_reply = await asyncio.to_thread(
                psychological_llm_response,
                transcript,
                adaptive_state,
                llm_context
            )
            
            print(f"💬 Response: {llm_reply}")
            
            # === COMPUTE TTS PARAMS (Using session's controller) ===
            tts_params = tts_controller.compute(adaptive_state)
            print(f"🎵 TTS: style={tts_params['style']}, degree={tts_params['styledegree']}, "
                  f"rate={tts_params['rate']}, pitch={tts_params['pitch']}")
            
            # === GENERATE TTS ===
            tts_audio_path = await asyncio.to_thread(
                synthesize_azure_tts,
                llm_reply,
                tts_params,
                "/tmp",
                voice_name
            )
            
            with open(tts_audio_path, "rb") as f:
                tts_audio_bytes = f.read()
            
            os.unlink(tts_audio_path)
            
            # === UPDATE CONVERSATION STATE ===
            conv_state.add_turn(
                transcript, 
                llm_reply, 
                instant_psychological_state,
                user_intent=None,  # Auto-detect from transcript
                topic=conv_state.dialogue_state.primary_topic
            )
            
            # === SAVE SESSION STATE ✅ ===
            save_session(session_id, session)
            
            # === RETURN RESPONSE ===
            return {
                "transcript": transcript,
                "llm_reply": llm_reply,
                "tts_audio": base64.b64encode(tts_audio_bytes).decode('utf-8'),
                "turn_count": conv_state.dialogue_state.turn_count,
                "emotional_mode": adaptive_state['mode'],
                "instant_state": instant_psychological_state,
                "adaptive_state": {
                    k: v for k, v in adaptive_state.items() 
                    if k not in ['trends', 'stability']
                },
                "tts_params": tts_params,
                # Model outputs for debugging/monitoring
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
                        )[:5]),  # Top 5 emotions
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
                        )[:5]) if audio_result["ast"] else {},  # Top 5 events
                        "latency_ms": audio_result['latency'] * 1000,
                    },
                    "fusion": {
                        "instant_state": instant_psychological_state,
                        "features_used": {
                            "speech_rate": 3.6,
                            "pause_duration": 0.5,
                            "jitter": 0.1,
                        }
                    },
                },
                # Memory view data
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
                    }
                }
            }
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)
    
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
        
        @web_app.get("/")
        async def root():
            """Serve enhanced interface with voice selector and memory view"""
            html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Voice Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
            padding: 20px;
            overflow-x: hidden;
        }
        
        .main-container {
            text-align: center;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .debug-panels {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 30px;
            max-width: 1400px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .debug-panel {
            background: rgba(0, 0, 0, 0.3);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .debug-panel::-webkit-scrollbar {
            width: 8px;
        }
        
        .debug-panel::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
        }
        
        .debug-panel::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.3);
            border-radius: 10px;
        }
        
        .debug-panel::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.5);
        }
        
        h1 {
            font-size: 48px;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .subtitle {
            font-size: 18px;
            opacity: 0.9;
            margin-bottom: 30px;
        }
        
        .controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 30px 0;
        }
        
        .btn {
            padding: 15px 40px;
            font-size: 18px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.3);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-start {
            background: #4ade80;
            color: #1e293b;
        }
        
        .btn-stop {
            background: #ef4444;
            color: white;
        }
        
        .assistant-ring {
            width: 200px;
            height: 200px;
            margin: 40px auto;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            border: 4px solid rgba(255, 255, 255, 0.3);
            display: flex;
            justify-content: center;
            align-items: center;
            transition: all 0.3s ease;
        }
        
        .assistant-ring.listening {
            animation: pulse 1.5s infinite;
            background: rgba(74, 222, 128, 0.2);
            border-color: #4ade80;
            box-shadow: 0 0 30px rgba(74, 222, 128, 0.6);
        }
        
        .assistant-ring.processing {
            border-color: #fbbf24;
            box-shadow: 0 0 30px rgba(251, 191, 36, 0.6);
            animation: spin 2s linear infinite;
        }
        
        .assistant-ring.speaking {
            border-color: #60a5fa;
            animation: speak 0.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        
        @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        @keyframes speak {
            0%, 100% { box-shadow: 0 0 20px rgba(96, 165, 250, 0.4); }
            50% { box-shadow: 0 0 40px rgba(96, 165, 250, 0.8); }
        }
        
        .mic-icon {
            font-size: 80px;
        }
        
        .status {
            font-size: 24px;
            margin: 20px 0;
            min-height: 30px;
            font-weight: 500;
        }
        
        .turn-info {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px 20px;
            border-radius: 20px;
            display: inline-block;
            margin: 10px 0;
            font-size: 14px;
        }
        
        .transcript-box, .response-box {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            min-height: 80px;
            backdrop-filter: blur(10px);
            text-align: left;
        }
        
        .response-box {
            background: rgba(255, 255, 255, 0.15);
        }
        
        .state-info {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 20px;
        }
        
        .state-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 8px;
        }
        
        .state-label {
            font-weight: bold;
            display: block;
            margin-bottom: 8px;
            font-size: 12px;
            opacity: 0.8;
        }
        
        .state-value {
            font-size: 18px;
        }
        
        .mode-badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 10px;
        }
        
        .mode-instant {
            background: #fbbf24;
            color: #1e293b;
        }
        
        .mode-trend {
            background: #60a5fa;
            color: #1e293b;
        }
        
        .error {
            background: rgba(239, 68, 68, 0.2);
            color: #fecaca;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            border: 1px solid #ef4444;
        }
        
        .hidden {
            display: none;
        }
        
        /* Voice Selector */
        .voice-selector {
            margin: 20px 0;
        }
        
        .voice-selector label {
            display: block;
            margin-bottom: 8px;
            font-size: 14px;
            opacity: 0.9;
        }
        
        .voice-selector select {
            padding: 10px 15px;
            font-size: 14px;
            border-radius: 10px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            background: rgba(255, 255, 255, 0.1);
            color: white;
            cursor: pointer;
            width: 100%;
            max-width: 350px;
        }
        
        .voice-selector select option {
            background: #1e293b;
            color: white;
        }
        
        /* Debug Panel Headers */
        .panel-title {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 15px;
            text-align: center;
            border-bottom: 2px solid rgba(255, 255, 255, 0.3);
            padding-bottom: 10px;
            color: #fbbf24;
        }
        
        .section-header {
            font-size: 14px;
            font-weight: 600;
            margin: 15px 0 10px 0;
            color: #60a5fa;
            border-left: 3px solid #60a5fa;
            padding-left: 10px;
        }
        
        .data-row {
            font-size: 12px;
            line-height: 1.8;
            padding: 6px 10px;
            background: rgba(255, 255, 255, 0.05);
            margin-bottom: 4px;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
        }
        
        .data-row:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        
        .data-label {
            color: rgba(255, 255, 255, 0.7);
            font-weight: 500;
        }
        
        .data-value {
            color: #fff;
            font-weight: 600;
            font-family: 'Courier New', monospace;
        }
        
        .emotion-bar {
            height: 8px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
            overflow: hidden;
            margin-top: 4px;
        }
        
        .emotion-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #4ade80, #fbbf24, #ef4444);
            transition: width 0.3s ease;
        }
        
        .trend-indicator {
            display: inline-block;
            margin-left: 8px;
            font-weight: bold;
        }
        
        .trend-up {
            color: #ef4444;
        }
        
        .trend-down {
            color: #4ade80;
        }
        
        .trend-stable {
            color: #fbbf24;
        }
        
        @media (max-width: 1200px) {
            .debug-panels {
                grid-template-columns: 1fr;
            }
            
            .debug-panel {
                max-height: 400px;
            }
        }
    </style>
</head>
<body>
    <div class="main-container">
        <h1>🎙️ AI Voice Assistant</h1>
        <p class="subtitle">Psychologically Adaptive Conversation</p>
        
        <!-- Voice Selector -->
        <div class="voice-selector">
            <label for="voiceSelect">🎵 Select Voice:</label>
            <select id="voiceSelect">
                <option value="en-US-DragonV2.1Neural">Dragon V2.1 (Default)</option>
                <option value="en-US-AvaMultilingualNeural">Ava Multilingual</option>
                <option value="en-US-AndrewMultilingualNeural">Andrew Multilingual</option>
                <option value="en-US-EmmaMultilingualNeural">Emma Multilingual</option>
                <option value="en-US-BrianMultilingualNeural">Brian Multilingual</option>
                <option value="en-IN-KavyaNeural">Kavya (Indian)</option>
                <option value="en-IN-AnanyaNeural">Ananya (Indian)</option>
                <option value="en-IN-AashiNeural">Aashi (Indian)</option>
            </select>
        </div>
        
        <div class="turn-info">
            Turn <span id="turnCount">0</span> 
            <span id="modeBadge" class="mode-badge mode-instant">Instant Mode</span>
        </div>
        
        <div id="assistantRing" class="assistant-ring">
            <div class="mic-icon">🎤</div>
        </div>
        
        <div class="status" id="status">Ready to listen</div>
        
        <div class="controls">
            <button id="startBtn" class="btn btn-start">Start Speaking</button>
            <button id="stopBtn" class="btn btn-stop" disabled>Stop Speaking</button>
        </div>
        
        <div id="errorBox" class="error hidden"></div>
        
        <div class="transcript-box hidden" id="transcriptBox">
            <strong>You said:</strong>
            <p id="transcript" style="margin-top: 10px; font-size: 16px;">...</p>
        </div>
        
        <div class="response-box hidden" id="responseBox">
            <strong>AI responds:</strong>
            <p id="response" style="margin-top: 10px; font-size: 16px;">...</p>
        </div>
        
        <div class="state-info hidden" id="stateInfo">
            <div class="state-item">
                <span class="state-label">EMOTION</span>
                <span class="state-value" id="valence">-</span>
            </div>
            <div class="state-item">
                <span class="state-label">ENERGY</span>
                <span class="state-value" id="arousal">-</span>
            </div>
            <div class="state-item">
                <span class="state-label">CLARITY</span>
                <span class="state-value" id="clarity">-</span>
            </div>
            <div class="state-item">
                <span class="state-label">STRESS</span>
                <span class="state-value" id="stress">-</span>
            </div>
        </div>
    </div>
    
    <!-- Debug & Monitoring Panels -->
    <div class="debug-panels">
        <!-- Model Outputs Panel -->
        <div class="debug-panel">
            <div class="panel-title">🤖 Model Outputs</div>
            
            <div class="section-header">🎤 ASR (Whisper)</div>
            <div id="modelASR">
                <div class="data-row">
                    <span class="data-label">Transcript:</span>
                    <span class="data-value">-</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Latency:</span>
                    <span class="data-value">-</span>
                </div>
            </div>
            
            <div class="section-header">💭 Text Emotion (RoBERTa)</div>
            <div id="modelTextEmotion">
                <div class="data-row">
                    <span class="data-label">No data yet</span>
                </div>
            </div>
            
            <div class="section-header">🎵 SER (Wav2Vec2)</div>
            <div id="modelSER">
                <div class="data-row">
                    <span class="data-label">Angry:</span>
                    <span class="data-value">0.00</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Happy:</span>
                    <span class="data-value">0.00</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Neutral:</span>
                    <span class="data-value">0.00</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Sad:</span>
                    <span class="data-value">0.00</span>
                </div>
            </div>
            
            <div class="section-header">🔊 AST (Audio Events)</div>
            <div id="modelAST">
                <div class="data-row">
                    <span class="data-label">No events detected</span>
                </div>
            </div>
            
            <div class="section-header">🎯 Fusion Output</div>
            <div id="modelFusion">
                <div class="data-row">
                    <span class="data-label">Valence:</span>
                    <span class="data-value">0.00</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Arousal:</span>
                    <span class="data-value">0.00</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Stress:</span>
                    <span class="data-value">0.00</span>
                </div>
                <div class="data-row">
                    <span class="data-label">Clarity:</span>
                    <span class="data-value">0.00</span>
                </div>
            </div>
        </div>
        
        <!-- Memory & State Panel -->
        <div class="debug-panel">
            <div class="panel-title">🧠 Memory & State</div>
            
            <div class="section-header">📋 Session Info</div>
            <div class="data-row">
                <span class="data-label">Session ID:</span>
                <span class="data-value" id="memSessionId">-</span>
            </div>
            <div class="data-row">
                <span class="data-label">Turn Count:</span>
                <span class="data-value" id="memTurnCount">0</span>
            </div>
            
            <div class="section-header">🎯 Topic Tracking</div>
            <div class="data-row">
                <span class="data-label">Current Topic:</span>
                <span class="data-value" id="memTopic">general</span>
            </div>
            <div class="data-row">
                <span class="data-label">Confidence:</span>
                <span class="data-value" id="memTopicConf">0%</span>
            </div>
            
            <div class="section-header">📈 Emotional Trends</div>
            <div class="data-row">
                <span class="data-label">Mode:</span>
                <span class="data-value" id="memMode">instant</span>
            </div>
            <div class="data-row">
                <span class="data-label">Confidence:</span>
                <span class="data-value" id="memModeConf">0%</span>
            </div>
            <div class="data-row">
                <span class="data-label">Valence:</span>
                <span>
                    <span class="data-value" id="memValence">0.00</span>
                    <span id="memValenceTrend" class="trend-indicator">→</span>
                </span>
            </div>
            <div class="data-row">
                <span class="data-label">Arousal:</span>
                <span>
                    <span class="data-value" id="memArousal">0.00</span>
                    <span id="memArousalTrend" class="trend-indicator">→</span>
                </span>
            </div>
            <div class="data-row">
                <span class="data-label">Stress:</span>
                <span>
                    <span class="data-value" id="memStress">0.00</span>
                    <span id="memStressTrend" class="trend-indicator">→</span>
                </span>
            </div>
            <div class="data-row">
                <span class="data-label">Clarity:</span>
                <span>
                    <span class="data-value" id="memClarity">0.00</span>
                    <span id="memClarityTrend" class="trend-indicator">→</span>
                </span>
            </div>
            
            <div class="section-header">💬 Recent Turns (Buffer: 5)</div>
            <div id="memRecentTurns">
                <div class="data-row">
                    <span class="data-label">No turns yet</span>
                </div>
            </div>
            
            <div class="section-header">🎭 Dialogue State</div>
            <div class="data-row">
                <span class="data-label">Recent Intents:</span>
                <span class="data-value" id="memIntents">-</span>
            </div>
            <div class="data-row">
                <span class="data-label">Coherence:</span>
                <span class="data-value" id="memCoherence">1.00</span>
            </div>
        </div>
    </div>
    
    <script>
        let mediaRecorder;
        let audioChunks = [];
        let ws;
        let isRecording = false;
        let stream;
        let sessionId = generateSessionId();
        
        const ring = document.getElementById('assistantRing');
        const status = document.getElementById('status');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const transcript = document.getElementById('transcript');
        const response = document.getElementById('response');
        const errorBox = document.getElementById('errorBox');
        const transcriptBox = document.getElementById('transcriptBox');
        const responseBox = document.getElementById('responseBox');
        const stateInfo = document.getElementById('stateInfo');
        const turnCount = document.getElementById('turnCount');
        const modeBadge = document.getElementById('modeBadge');
        
        function generateSessionId() {
            return 'session_' + Math.random().toString(36).substring(2, 15);
        }
        
        function showError(message) {
            errorBox.textContent = '❌ ' + message;
            errorBox.classList.remove('hidden');
            setTimeout(() => {
                errorBox.classList.add('hidden');
            }, 5000);
        }
        
        function updateState(data) {
            stateInfo.classList.remove('hidden');
            
            const state = data.adaptive_state;
            
            document.getElementById('valence').textContent = 
                state.valence > 0 ? `😊 +${state.valence.toFixed(2)}` : `😔 ${state.valence.toFixed(2)}`;
            document.getElementById('arousal').textContent = 
                `⚡ ${(state.arousal * 100).toFixed(0)}%`;
            document.getElementById('clarity').textContent = 
                `💡 ${(state.clarity * 100).toFixed(0)}%`;
            document.getElementById('stress').textContent = 
                state.stress > 0.5 ? `😰 ${(state.stress * 100).toFixed(0)}%` : `😌 ${(state.stress * 100).toFixed(0)}%`;
            
            turnCount.textContent = data.turn_count;
            
            if (data.emotional_mode === 'trend') {
                modeBadge.textContent = 'Trend Mode';
                modeBadge.className = 'mode-badge mode-trend';
            } else {
                modeBadge.textContent = 'Instant Mode';
                modeBadge.className = 'mode-badge mode-instant';
            }
        }
        
        function updateMemoryView(data) {
            if (!data.memory_view) return;
            
            const mem = data.memory_view;
            
            // Session Info
            document.getElementById('memSessionId').textContent = mem.session_id.substring(0, 15) + '...';
            document.getElementById('memTurnCount').textContent = mem.dialogue_state.turn_count;
            
            // Topic Tracking
            document.getElementById('memTopic').textContent = mem.topic_info.current_topic;
            document.getElementById('memTopicConf').textContent = (mem.topic_info.confidence * 100).toFixed(0) + '%';
            
            // Emotional Trends
            document.getElementById('memMode').textContent = mem.emotional_trends.mode;
            document.getElementById('memModeConf').textContent = (mem.emotional_trends.confidence * 100).toFixed(0) + '%';
            
            // Valence
            const valence = mem.emotional_trends.valence;
            document.getElementById('memValence').textContent = valence.current.toFixed(2);
            updateTrendIndicator('memValenceTrend', valence.trend);
            
            // Arousal
            const arousal = mem.emotional_trends.arousal;
            document.getElementById('memArousal').textContent = arousal.current.toFixed(2);
            updateTrendIndicator('memArousalTrend', arousal.trend);
            
            // Stress
            const stress = mem.emotional_trends.stress;
            document.getElementById('memStress').textContent = stress.current.toFixed(2);
            updateTrendIndicator('memStressTrend', stress.trend);
            
            // Clarity
            const clarity = mem.emotional_trends.clarity;
            document.getElementById('memClarity').textContent = clarity.current.toFixed(2);
            updateTrendIndicator('memClarityTrend', clarity.trend);
            
            // Recent Turns
            const turnsContainer = document.getElementById('memRecentTurns');
            if (mem.recent_turns.length > 0) {
                turnsContainer.innerHTML = mem.recent_turns.map((turn, idx) => `
                    <div style="margin-bottom: 12px; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 5px; border-left: 3px solid #60a5fa;">
                        <div style="font-size: 11px; opacity: 0.7; margin-bottom: 4px;">
                            Turn ${data.turn_count - mem.recent_turns.length + idx + 1} | ${turn.topic}
                        </div>
                        <div style="font-size: 12px; margin-bottom: 2px;">
                            👤 ${turn.user}
                        </div>
                        <div style="font-size: 12px; color: #4ade80;">
                            🤖 ${turn.ai}
                        </div>
                    </div>
                `).join('');
            }
            
            // Dialogue State
            if (mem.dialogue_state.recent_intents.length > 0) {
                document.getElementById('memIntents').textContent = mem.dialogue_state.recent_intents.slice(-3).join(', ');
            }
            document.getElementById('memCoherence').textContent = mem.dialogue_state.coherence_score.toFixed(2);
        }
        
        function updateTrendIndicator(elementId, trend) {
            const element = document.getElementById(elementId);
            if (trend > 0.05) {
                element.textContent = '↗️';
                element.className = 'trend-indicator trend-up';
            } else if (trend < -0.05) {
                element.textContent = '↘️';
                element.className = 'trend-indicator trend-down';
            } else {
                element.textContent = '→';
                element.className = 'trend-indicator trend-stable';
            }
        }
        
        function updateModelOutputs(data) {
            if (!data.model_outputs) return;
            
            const models = data.model_outputs;
            
            // ASR Output
            if (models.asr) {
                document.getElementById('modelASR').innerHTML = `
                    <div class="data-row">
                        <span class="data-label">Transcript:</span>
                        <span class="data-value">${models.asr.transcript.substring(0, 50)}...</span>
                    </div>
                    <div class="data-row">
                        <span class="data-label">Latency:</span>
                        <span class="data-value">${models.asr.latency_ms.toFixed(0)}ms</span>
                    </div>
                `;
            }
            
            // Text Emotion Output
            if (models.text_emotion && models.text_emotion.top_emotions) {
                const emotions = models.text_emotion.top_emotions;
                let html = '';
                for (const [emotion, score] of Object.entries(emotions)) {
                    html += `
                        <div class="data-row">
                            <span class="data-label">${emotion}:</span>
                            <span class="data-value">${(score * 100).toFixed(1)}%</span>
                        </div>
                        <div class="emotion-bar">
                            <div class="emotion-bar-fill" style="width: ${score * 100}%"></div>
                        </div>
                    `;
                }
                document.getElementById('modelTextEmotion').innerHTML = html;
            }
            
            // SER Output
            if (models.audio_analysis && models.audio_analysis.ser) {
                const ser = models.audio_analysis.ser;
                document.getElementById('modelSER').innerHTML = `
                    <div class="data-row">
                        <span class="data-label">Angry:</span>
                        <span class="data-value">${(ser.angry * 100).toFixed(1)}%</span>
                    </div>
                    <div class="emotion-bar">
                        <div class="emotion-bar-fill" style="width: ${ser.angry * 100}%"></div>
                    </div>
                    <div class="data-row">
                        <span class="data-label">Happy:</span>
                        <span class="data-value">${(ser.happy * 100).toFixed(1)}%</span>
                    </div>
                    <div class="emotion-bar">
                        <div class="emotion-bar-fill" style="width: ${ser.happy * 100}%"></div>
                    </div>
                    <div class="data-row">
                        <span class="data-label">Neutral:</span>
                        <span class="data-value">${(ser.neutral * 100).toFixed(1)}%</span>
                    </div>
                    <div class="emotion-bar">
                        <div class="emotion-bar-fill" style="width: ${ser.neutral * 100}%"></div>
                    </div>
                    <div class="data-row">
                        <span class="data-label">Sad:</span>
                        <span class="data-value">${(ser.sad * 100).toFixed(1)}%</span>
                    </div>
                    <div class="emotion-bar">
                        <div class="emotion-bar-fill" style="width: ${ser.sad * 100}%"></div>
                    </div>
                    <div class="data-row">
                        <span class="data-label">Latency:</span>
                        <span class="data-value">${models.audio_analysis.latency_ms.toFixed(0)}ms</span>
                    </div>
                `;
            }
            
            // AST Output
            if (models.audio_analysis && models.audio_analysis.ast) {
                const ast = models.audio_analysis.ast;
                if (Object.keys(ast).length > 0) {
                    let html = '';
                    for (const [event, score] of Object.entries(ast)) {
                        html += `
                            <div class="data-row">
                                <span class="data-label">${event}:</span>
                                <span class="data-value">${(score * 100).toFixed(1)}%</span>
                            </div>
                            <div class="emotion-bar">
                                <div class="emotion-bar-fill" style="width: ${score * 100}%"></div>
                            </div>
                        `;
                    }
                    document.getElementById('modelAST').innerHTML = html;
                } else {
                    document.getElementById('modelAST').innerHTML = `
                        <div class="data-row">
                            <span class="data-label">No events detected</span>
                        </div>
                    `;
                }
            }
            
            // Fusion Output
            if (models.fusion && models.fusion.instant_state) {
                const state = models.fusion.instant_state;
                document.getElementById('modelFusion').innerHTML = `
                    <div class="data-row">
                        <span class="data-label">Valence:</span>
                        <span class="data-value">${state.valence.toFixed(3)}</span>
                    </div>
                    <div class="data-row">
                        <span class="data-label">Arousal:</span>
                        <span class="data-value">${state.arousal.toFixed(3)}</span>
                    </div>
                    <div class="data-row">
                        <span class="data-label">Stress:</span>
                        <span class="data-value">${state.stress.toFixed(3)}</span>
                    </div>
                    <div class="data-row">
                        <span class="data-label">Clarity:</span>
                        <span class="data-value">${state.clarity.toFixed(3)}</span>
                    </div>
                `;
            }
        }
        
        async function startRecording() {
            if (isRecording) return;
            
            try {
                transcriptBox.classList.add('hidden');
                responseBox.classList.add('hidden');
                
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws/conversation`);
                
                ws.onopen = () => {
                    console.log('✅ WebSocket connected');
                };
                
                ws.onmessage = async (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        
                        if (data.error) {
                            showError(data.error);
                            ring.className = 'assistant-ring';
                            status.textContent = 'Ready to listen';
                            startBtn.disabled = false;
                            stopBtn.disabled = true;
                            return;
                        }
                        
                        console.log('📨 Received:', data);
                        
                        transcriptBox.classList.remove('hidden');
                        transcript.textContent = data.transcript;
                        
                        responseBox.classList.remove('hidden');
                        response.textContent = data.llm_reply;
                        
                        updateState(data);
                        updateMemoryView(data);
                        updateModelOutputs(data);
                        
                        if (data.tts_audio) {
                            const audioData = atob(data.tts_audio);
                            const audioArray = new Uint8Array(audioData.length);
                            for (let i = 0; i < audioData.length; i++) {
                                audioArray[i] = audioData.charCodeAt(i);
                            }
                            
                            const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
                            const audioUrl = URL.createObjectURL(audioBlob);
                            const audio = new Audio(audioUrl);
                            
                            ring.className = 'assistant-ring speaking';
                            status.textContent = '🔊 Speaking...';
                            
                            audio.play();
                            audio.onended = () => {
                                ring.className = 'assistant-ring';
                                status.textContent = 'Ready to listen';
                                startBtn.disabled = false;
                                stopBtn.disabled = true;
                                ws.close();
                            };
                        } else {
                            ring.className = 'assistant-ring';
                            status.textContent = 'Ready to listen';
                            startBtn.disabled = false;
                            stopBtn.disabled = true;
                            ws.close();
                        }
                    } catch (e) {
                        console.error('Error:', e);
                        showError('Error processing response');
                        ring.className = 'assistant-ring';
                        status.textContent = 'Ready to listen';
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    showError('Connection error');
                    ring.className = 'assistant-ring';
                    status.textContent = 'Ready to listen';
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                };
                
                stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        channelCount: 1,
                        sampleRate: 16000,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true
                    } 
                });
                
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus'
                });
                
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    
                    console.log(`📤 Sending ${arrayBuffer.byteLength} bytes`);
                    
                    ring.className = 'assistant-ring processing';
                    status.textContent = '🤔 Processing...';
                    
                    const base64Audio = btoa(
                        new Uint8Array(arrayBuffer).reduce(
                            (data, byte) => data + String.fromCharCode(byte),
                            ''
                        )
                    );
                    
                    const selectedVoice = document.getElementById('voiceSelect').value;
                    
                    ws.send(JSON.stringify({ 
                        audio: base64Audio,
                        session_id: sessionId,
                        voice_name: selectedVoice
                    }));
                    
                    stream.getTracks().forEach(track => track.stop());
                };
                
                mediaRecorder.start();
                isRecording = true;
                
                ring.className = 'assistant-ring listening';
                status.textContent = '🎙️ Listening... (click Stop when done)';
                startBtn.disabled = true;
                stopBtn.disabled = false;
                
            } catch (error) {
                console.error('Error:', error);
                showError('Microphone access denied');
                ring.className = 'assistant-ring';
                status.textContent = 'Ready to listen';
                startBtn.disabled = false;
                stopBtn.disabled = true;
            }
        }
        
        function stopRecording() {
            if (!isRecording || !mediaRecorder) return;
            
            mediaRecorder.stop();
            isRecording = false;
            stopBtn.disabled = true;
        }
        
        startBtn.addEventListener('click', startRecording);
        stopBtn.addEventListener('click', stopRecording);
    </script>
</body>
</html>
            """
            return HTMLResponse(content=html_content)
        
        @web_app.websocket("/ws/conversation")
        async def conversation_endpoint(websocket: WebSocket):
            """WebSocket endpoint for continuous conversation"""
            await websocket.accept()
            print("✅ WebSocket connected")
            
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if 'audio' not in data or 'session_id' not in data:
                    await websocket.send_json({"error": "Missing audio or session_id"})
                    await websocket.close()
                    return
                
                audio_base64 = data['audio']
                session_id = data['session_id']
                voice_name = data.get('voice_name', 'en-US-DragonV2.1Neural')
                
                print(f"📥 Session: {session_id}, Voice: {voice_name}")
                
                # Process conversation turn
                result = await self.process_conversation_turn.remote.aio(
                    audio_base64,
                    session_id,
                    voice_name
                )
                
                await websocket.send_json(result)
                print("✅ Response sent")
                
            except WebSocketDisconnect:
                print("❌ WebSocket disconnected")
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                try:
                    await websocket.send_json({"error": str(e)})
                except:
                    pass
            finally:
                try:
                    await websocket.close()
                except:
                    pass
        
        return web_app