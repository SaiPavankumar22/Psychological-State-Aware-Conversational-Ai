import time
import asyncio
import tempfile
import wave
import numpy as np
import modal

from fastapi import FastAPI, WebSocket
from services import ASRTextService, AudioEmotionService, SER_LABELS
from fusion import PsychologicalFusion, azure_tts_input
from llm_client import psychological_llm_response
from tts_azure import synthesize_azure_tts

# =====================================================
# MODAL APP
# =====================================================
app = modal.App("emotion-system-realtime")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "fastapi",
        "uvicorn",
        "numpy",
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "transformers",
        "accelerate",
        "openai",
        "azure-cognitiveservices-speech",
        "librosa",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_dir(".", remote_path="/root", ignore=["_pycache_"],)
)

# =====================================================
# AUDIO UTILS
# =====================================================
TARGET_SAMPLE_RATE = 16000

def waveform_from_pcm_bytes(raw: bytes):
    audio = np.frombuffer(raw, dtype=np.float32)
    return audio, TARGET_SAMPLE_RATE

def write_wav_int16(path, audio, sr):
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())

# =====================================================
# MODAL CLASS (GPU CONTAINER)
# =====================================================
@app.cls(
    image=image,
    gpu="A10G",  # still needed for SER + AST
    timeout=600,
    startup_timeout=300,
    secrets=[modal.Secret.from_name("emotion-env")],
)
class RealtimePsychologist:

    @modal.enter()
    def load_models(self):
        print("✅ Models loaded, container ready")

        self.asr = ASRTextService()          # Whisper API
        self.audio = AudioEmotionService()  # GPU SER + AST
        self.fusion = PsychologicalFusion()
        print("✅ Models loaded, ready to process requests")


    #@modal.concurrent(max_inputs=10)
    @modal.asgi_app()
    def fastapi_app(self):
        api = FastAPI() 

        @api.websocket("/ws/audio")
        async def audio_ws(ws: WebSocket):
            await ws.accept()
            buffer = bytearray()

            while True:
                msg = await ws.receive()
                if msg.get("bytes"):
                    buffer.extend(msg["bytes"])
                elif msg.get("text") == "END":
                    break

            audio, sr = waveform_from_pcm_bytes(bytes(buffer))

            def pipeline():
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    write_wav_int16(f.name, audio, sr)
                    wav_path = f.name

                # --- ASR (Whisper API) ---
                asr_res = self.asr.run(wav_path)

                # --- SER + AST (GPU) ---
                audio_res = self.audio.run(wav_path)
                ser_dict = dict(zip(SER_LABELS, audio_res["ser"]))

                # --- Psychological Fusion ---
                state = self.fusion.fuse(
                    text=asr_res["text_emotion"],
                    ser=ser_dict,
                    ast=audio_res["ast"],
                    features={
                        "speech_rate": 3.6,
                        "pause_duration": 0.5,
                        "jitter": 0.1,
                    },
                )

                # --- LLM + TTS ---
                reply = psychological_llm_response(
                    asr_res["transcript"], state
                )
                tts_params = azure_tts_input(state)
                tts_path = synthesize_azure_tts(
                    reply, tts_params, "/tmp"
                )

                return state, reply, tts_path

            state, reply, tts_path = await asyncio.to_thread(pipeline)

            await ws.send_json({
                "psychological_state": state,
                "llm_reply": reply,
            })

            with open(tts_path, "rb") as f:
                await ws.send_bytes(f.read())

            await ws.close()

        return api
