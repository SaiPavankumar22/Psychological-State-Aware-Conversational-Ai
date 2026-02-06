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
        session_id: str
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
                "/tmp"
            )
            
            with open(tts_audio_path, "rb") as f:
                tts_audio_bytes = f.read()
            
            os.unlink(tts_audio_path)
            
            # === UPDATE CONVERSATION STATE ===
            conv_state.add_turn(
                transcript, 
                llm_reply, 
                instant_psychological_state,
                user_intent="unknown",
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
            """Serve enhanced interface with start/stop controls"""
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
            display: flex;
            justify-content: center;
            align-items: center;
            color: white;
            padding: 20px;
        }
        
        .container {
            text-align: center;
            max-width: 800px;
            width: 100%;
            padding: 40px;
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
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ AI Voice Assistant</h1>
        <p class="subtitle">Psychologically Adaptive Conversation</p>
        
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
                    
                    ws.send(JSON.stringify({ 
                        audio: base64Audio,
                        session_id: sessionId
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
                
                print(f"📥 Session: {session_id}")
                
                # Process conversation turn
                result = await self.process_conversation_turn.remote.aio(
                    audio_base64,
                    session_id
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