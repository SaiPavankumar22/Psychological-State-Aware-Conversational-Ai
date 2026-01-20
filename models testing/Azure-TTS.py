import azure.cognitiveservices.speech as speechsdk
import uuid

# ----------------------------
# CONFIG
# ----------------------------
AZURE_TTS_KEY = "your_azure_tts_key"
AZURE_REGION = "centralindia"  # e.g. "eastus"

VOICE_NAME = "en-US-JennyNeural"  # Change if needed
OUTPUT_FILE = f"tts_output_{uuid.uuid4().hex}.wav"


def synthesize_speech(
    text: str,
    style: str,
    styledegree: float,
    rate: float,
    pitch: float,
    volume: str
):
    """
    style        : e.g. 'cheerful', 'sad', 'angry', 'excited'
    styledegree  : float (0.1 - 2.0)
    rate         : float (1.0 = normal)
    pitch        : int (percentage, e.g. 10 or -5)
    volume       : 'silent' | 'x-soft' | 'soft' | 'medium' | 'loud' | 'x-loud'
    """

    # Azure speech config
    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_TTS_KEY,
        region=AZURE_REGION
    )

    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm
    )

    audio_config = speechsdk.audio.AudioOutputConfig(filename=OUTPUT_FILE)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    # ----------------------------
    # SSML
    # ----------------------------
    ssml = f"""
    <speak version="1.0"
        xmlns="http://www.w3.org/2001/10/synthesis"
        xmlns:mstts="http://www.w3.org/2001/mstts"
        xml:lang="en-US">
        
        <voice name="{VOICE_NAME}">
            <mstts:express-as
                style="{style}"
                styledegree="{round(styledegree, 2)}">
                
                <prosody
                    rate="{int(rate * 100)}%"
                    pitch="{int(pitch)}%"
                    volume="{volume}">
                    
                    {text}
                
                </prosody>
            </mstts:express-as>
        </voice>
    </speak>
    """

    # Synthesize
    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"✅ Audio generated: {OUTPUT_FILE}")
        return OUTPUT_FILE

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        print("❌ Speech synthesis canceled")
        print("Reason:", cancellation.reason)
        if cancellation.error_details:
            print("Error details:", cancellation.error_details)
        return None


# ----------------------------
# EXAMPLE USAGE
# ----------------------------
if __name__ == "__main__":
    synthesize_speech(
        text="Hello! This is Azure neural text to speech with emotional style control.",
        style="calm",
        styledegree=1.5,
        rate=1.1,
        pitch=5,
        volume="medium"
    )
