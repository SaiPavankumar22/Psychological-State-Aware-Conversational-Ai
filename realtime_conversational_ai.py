# realtime_conversational_ai.py

import asyncio
import tempfile
import wave
import os
import json
import base64
from pathlib import Path
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
    .add_local_dir(".", remote_path="/root")
)

# =====================================================
# AUDIO UTILITIES
# =====================================================
TARGET_SAMPLE_RATE = 16000

def webm_to_wav(webm_bytes, output_path: str) -> str:
    """Convert WebM audio to WAV using ffmpeg"""
    import subprocess
    
    webm_temp = output_path.replace('.wav', '.webm')
    
    # Write WebM bytes to temp file
    with open(webm_temp, 'wb') as f:
        f.write(webm_bytes)
    
    # Convert using ffmpeg
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
    scaledown_window=300,  # Keep warm for 5 minutes (FIXED)
)
class ConversationalAI:
    
    @modal.enter()
    def load_models(self):
        """Load all models on container startup"""
        print("🚀 Loading models...")
        
        from services import ASRTextService, AudioEmotionService
        from fusion import PsychologicalFusion
        
        self.asr_service = ASRTextService()
        self.audio_service = AudioEmotionService()
        self.fusion = PsychologicalFusion()
        
        print("✅ All models loaded successfully")
    
    @modal.method()
    async def process_audio_chunk(self, audio_data: str) -> dict:  # FIXED: string instead of bytes
        """
        Process a single audio chunk through the entire pipeline
        audio_data: base64 encoded audio bytes
        Returns: dict with state, transcript, reply, and TTS audio
        """
        import asyncio
        from llm_client import psychological_llm_response
        from tts_azure import synthesize_azure_tts
        from fusion import azure_tts_input
        from services import SER_LABELS
        
        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data)
        print(f"📥 Received {len(audio_bytes)} bytes of audio")
        
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            wav_path = tmp_file.name
        
        try:
            # Convert WebM to WAV
            webm_to_wav(audio_bytes, wav_path)
            print(f"✅ Audio converted to WAV: {wav_path}")
            
            # Run ASR + Text Emotion and Audio Analysis in parallel
            print("🔄 Running parallel analysis...")
            asr_task = asyncio.to_thread(self.asr_service.run, wav_path)
            audio_task = asyncio.to_thread(self.audio_service.run, wav_path)
            
            asr_result, audio_result = await asyncio.gather(asr_task, audio_task)
            
            print(f"📝 Transcript: {asr_result['transcript']}")
            
            # Fusion
            ser_dict = dict(zip(SER_LABELS, audio_result["ser"]))
            
            psychological_state = self.fusion.fuse(
                text=asr_result["text_emotion"],
                ser=ser_dict,
                ast=audio_result["ast"],
                features={
                    "speech_rate": 3.6,
                    "pause_duration": 0.5,
                    "jitter": 0.1,
                },
            )
            
            print(f"🧠 Psychological State: {psychological_state}")
            
            # Generate LLM response
            print("🤖 Generating LLM response...")
            llm_reply = await asyncio.to_thread(
                psychological_llm_response,
                asr_result["transcript"],
                psychological_state
            )
            
            print(f"💬 LLM Reply: {llm_reply}")
            
            # Generate TTS with emotional parameters
            print("🔊 Generating TTS...")
            tts_params = azure_tts_input(psychological_state)
            tts_audio_path = await asyncio.to_thread(
                synthesize_azure_tts,
                llm_reply,
                tts_params,
                "/tmp"
            )
            
            # Read TTS audio
            with open(tts_audio_path, "rb") as f:
                tts_audio_bytes = f.read()
            
            print(f"✅ TTS generated: {len(tts_audio_bytes)} bytes")
            
            # Cleanup
            os.unlink(tts_audio_path)
            
            return {
                "transcript": asr_result["transcript"],
                "psychological_state": psychological_state,
                "llm_reply": llm_reply,
                "tts_audio": base64.b64encode(tts_audio_bytes).decode('utf-8'),
                "latencies": {
                    "asr": asr_result["latency"],
                    "audio_analysis": audio_result["latency"],
                }
            }
            
        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        finally:
            # Cleanup temp file
            if os.path.exists(wav_path):
                os.unlink(wav_path)
    
    @modal.asgi_app()
    def fastapi_app(self):
        """Create FastAPI app with WebSocket endpoint"""
        web_app = FastAPI(title="Conversational AI API")
        
        # Add CORS middleware
        web_app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @web_app.get("/")
        async def root():
            """Serve Alexa-like interface"""
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
            max-width: 700px;
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
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
        }
        
        .assistant-ring:hover {
            transform: scale(1.05);
            border-color: rgba(255, 255, 255, 0.6);
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
        
        .instructions {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            backdrop-filter: blur(10px);
        }
        
        .instructions h3 {
            margin-bottom: 15px;
            font-size: 20px;
        }
        
        .instruction-steps {
            text-align: left;
            display: inline-block;
        }
        
        .instruction-steps li {
            margin: 10px 0;
            padding-left: 10px;
        }
        
        .transcript-box {
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
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            min-height: 80px;
            backdrop-filter: blur(10px);
            text-align: left;
        }
        
        .state-info {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 20px;
        }
        
        .state-item {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 8px;
            font-size: 14px;
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
        
        @media (max-width: 768px) {
            h1 { font-size: 36px; }
            .assistant-ring { width: 150px; height: 150px; }
            .mic-icon { font-size: 60px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎙️ AI Voice Assistant</h1>
        <p class="subtitle">Psychologically Adaptive AI Companion</p>
        
        <div id="assistantRing" class="assistant-ring">
            <div class="mic-icon">🎤</div>
        </div>
        
        <div class="status" id="status">Ready to listen</div>
        
        <div id="errorBox" class="error hidden"></div>
        
        <div class="instructions">
            <h3>📋 How to Use</h3>
            <ol class="instruction-steps">
                <li><strong>Click</strong> the microphone to start recording</li>
                <li><strong>Speak</strong> naturally (it will auto-stop after 5 seconds of silence)</li>
                <li><strong>Wait</strong> for the AI to process and respond</li>
                <li><strong>Listen</strong> to the emotionally-adaptive response</li>
            </ol>
        </div>
        
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
                <span class="state-label">EMOTION (Valence)</span>
                <span class="state-value" id="valence">-</span>
            </div>
            <div class="state-item">
                <span class="state-label">ENERGY (Arousal)</span>
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
        let silenceTimeout;
        let stream;
        
        const ring = document.getElementById('assistantRing');
        const status = document.getElementById('status');
        const transcript = document.getElementById('transcript');
        const response = document.getElementById('response');
        const errorBox = document.getElementById('errorBox');
        const transcriptBox = document.getElementById('transcriptBox');
        const responseBox = document.getElementById('responseBox');
        const stateInfo = document.getElementById('stateInfo');
        
        function showError(message) {
            errorBox.textContent = '❌ ' + message;
            errorBox.classList.remove('hidden');
            setTimeout(() => {
                errorBox.classList.add('hidden');
            }, 5000);
        }
        
        function updateState(state) {
            stateInfo.classList.remove('hidden');
            
            document.getElementById('valence').textContent = 
                state.valence > 0 ? `😊 +${state.valence.toFixed(2)}` : `😔 ${state.valence.toFixed(2)}`;
            document.getElementById('arousal').textContent = 
                `⚡ ${(state.arousal * 100).toFixed(0)}%`;
            document.getElementById('clarity').textContent = 
                `💡 ${(state.clarity * 100).toFixed(0)}%`;
            document.getElementById('stress').textContent = 
                state.stress > 0.5 ? `😰 ${(state.stress * 100).toFixed(0)}%` : `😌 ${(state.stress * 100).toFixed(0)}%`;
        }
        
        async function startRecording() {
            if (isRecording) {
                stopRecording();
                return;
            }
            
            try {
                // Hide previous results
                transcriptBox.classList.add('hidden');
                responseBox.classList.add('hidden');
                stateInfo.classList.add('hidden');
                
                // Connect WebSocket
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
                            return;
                        }
                        
                        console.log('📨 Received response:', data);
                        
                        // Show transcript
                        transcriptBox.classList.remove('hidden');
                        transcript.textContent = data.transcript || 'No speech detected';
                        
                        // Show response
                        responseBox.classList.remove('hidden');
                        response.textContent = data.llm_reply || 'No response generated';
                        
                        // Update psychological state
                        if (data.psychological_state) {
                            updateState(data.psychological_state);
                        }
                        
                        // Play TTS audio
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
                                ws.close();
                            };
                        } else {
                            ring.className = 'assistant-ring';
                            status.textContent = 'Ready to listen';
                            ws.close();
                        }
                    } catch (e) {
                        console.error('Error parsing message:', e);
                        showError('Error processing response: ' + e.message);
                        ring.className = 'assistant-ring';
                        status.textContent = 'Ready to listen';
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('❌ WebSocket error:', error);
                    showError('Connection error. Please try again.');
                    ring.className = 'assistant-ring';
                    status.textContent = 'Ready to listen';
                };
                
                ws.onclose = () => {
                    console.log('WebSocket closed');
                };
                
                // Start recording
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
                    status.textContent = '🤔 Processing your speech...';
                    
                    // Convert to base64 and send as JSON
                    const base64Audio = btoa(
                        new Uint8Array(arrayBuffer).reduce(
                            (data, byte) => data + String.fromCharCode(byte),
                            ''
                        )
                    );
                    
                    ws.send(JSON.stringify({ audio: base64Audio }));
                    
                    // Stop all tracks
                    stream.getTracks().forEach(track => track.stop());
                };
                
                mediaRecorder.start();
                isRecording = true;
                
                ring.className = 'assistant-ring listening';
                status.textContent = '🎙️ Listening... (will auto-stop)';
                
                // Auto-stop after 10 seconds (max recording time)
                setTimeout(() => {
                    if (isRecording) {
                        stopRecording();
                    }
                }, 10000);
                
            } catch (error) {
                console.error('Error starting recording:', error);
                showError('Microphone access denied. Please allow microphone access.');
                ring.className = 'assistant-ring';
                status.textContent = 'Ready to listen';
            }
        }
        
        function stopRecording() {
            if (!isRecording || !mediaRecorder) return;
            
            mediaRecorder.stop();
            isRecording = false;
            
            clearTimeout(silenceTimeout);
        }
        
        ring.addEventListener('click', () => {
            startRecording();
        });
        
        // Keyboard shortcut: Space bar to toggle recording
        document.addEventListener('keydown', (e) => {
            if (e.code === 'Space' && !e.repeat) {
                e.preventDefault();
                startRecording();
            }
        });
    </script>
</body>
</html>
            """
            return HTMLResponse(content=html_content)
        
        @web_app.websocket("/ws/conversation")
        async def conversation_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time conversation"""
            await websocket.accept()
            print("✅ WebSocket connection established")
            
            try:
                # Receive audio data as JSON
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if 'audio' not in data:
                    await websocket.send_json({"error": "No audio data in message"})
                    await websocket.close()
                    return
                
                audio_base64 = data['audio']
                print(f"📥 Received audio data (base64 length: {len(audio_base64)})")
                
                # Process the audio
                result = await self.process_audio_chunk.remote.aio(audio_base64)
                
                # Send result back
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

"""
```

## 🎯 How to Use the Interface (Step-by-Step)

### **Simple 1-Click Interaction (Like Alexa)**

1. **Click the Microphone** 🎤
   - The ring turns **GREEN** and starts pulsing
   - Status shows: "🎙️ Listening... (will auto-stop)"

2. **Speak Naturally** 🗣️
   - Just talk like you would to Alexa
   - Example: "Hello, how are you today?"
   - Example: "I'm feeling stressed about work"
   - Example: "Tell me a joke"

3. **It Auto-Stops** ⏸️
   - After you finish speaking, wait a moment
   - OR it will auto-stop after 10 seconds maximum
   - NO need to click stop button!

4. **Processing Happens** 🤔
   - Ring turns **YELLOW** and spins
   - Status shows: "🤔 Processing your speech..."
   - This takes 5-15 seconds

5. **AI Responds** 🔊
   - Ring turns **BLUE** and pulses
   - Status shows: "🔊 Speaking..."
   - You'll see:
     - What you said (transcript)
     - AI's reply (text)
     - Your emotional state (valence, arousal, clarity, stress)
   - Audio plays automatically

6. **Ready for Next Turn** 🔄
   - After audio finishes, status shows: "Ready to listen"
   - Click microphone again to continue conversation

### **Alternative: Keyboard Shortcut**

- Press **SPACEBAR** to start/stop recording
- Same behavior as clicking the microphone

## 📋 Example Conversations

### **Example 1: Happy Greeting**
```
You: "Hey! I'm so excited about my new project!"

AI detects:
- Valence: +0.7 (positive)
- Arousal: 0.6 (high energy)
- Clarity: 0.8 (clear)
- Stress: 0.2 (low)

AI responds (cheerful style):
"That's wonderful to hear! I can tell you're really enthusiastic 
about it. What's your new project about?"
```

### **Example 2: Stressed User**
```
You: "I'm so overwhelmed... I don't know what to do..."

AI detects:
- Valence: -0.6 (negative)
- Arousal: 0.7 (high energy)
- Clarity: 0.3 (confused)
- Stress: 0.8 (high)

AI responds (calm, soothing style):
"I can hear that you're feeling stressed. Take a deep breath. 
Would you like to talk through what's overwhelming you?"
```

### **Example 3: Neutral Question**
```
You: "What's the weather like today?"

AI detects:
- Valence: 0.0 (neutral)
- Arousal: 0.3 (low energy)
- Clarity: 0.9 (very clear)
- Stress: 0.1 (calm)

AI responds (professional style):
"I don't have access to current weather data, but I'd be 
happy to help you with something else!"""