import os
import uuid

import azure.cognitiveservices.speech as speechsdk


# =====================================================
# AZURE CONFIG
# =====================================================
AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY")
AZURE_REGION = os.getenv("AZURE_TTS_REGION", "centralindia")  # e.g. "eastus"
VOICE_NAME = os.getenv("AZURE_TTS_VOICE", "en-US-JennyNeural")

# Audio format
AUDIO_FORMAT = speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm


def _get_speech_config() -> speechsdk.SpeechConfig:
    """
    Build a SpeechConfig from environment variables, with basic validation.
    """
    if not AZURE_TTS_KEY:
        raise RuntimeError(
            "AZURE_TTS_KEY environment variable is not set. "
            "Please export it before running the application."
        )

    speech_config = speechsdk.SpeechConfig(
        subscription=AZURE_TTS_KEY,
        region=AZURE_REGION,
    )
    speech_config.set_speech_synthesis_output_format(AUDIO_FORMAT)
    return speech_config


# =====================================================
# AZURE TTS SYNTHESIS
# =====================================================
def synthesize_azure_tts(
    text: str,
    tts_params: dict,
    output_dir: str
) -> str:
    """
    Converts text to speech using Azure Neural TTS with emotional control.

    Args:
        text (str): Text to synthesize
        tts_params (dict): Output from azure_tts_input(state)
        output_dir (str): Local folder to save WAV file

    Returns:
        str: Path to generated WAV file
    """

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"tts_{uuid.uuid4().hex}.wav"
    )

    # ----------------------------
    # Azure Speech Config
    # ----------------------------
    speech_config = _get_speech_config()

    audio_config = speechsdk.audio.AudioOutputConfig(
        filename=output_file
    )

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config
    )

    # ----------------------------
    # SSML (Emotion-Aware)
    # ----------------------------
    ssml = f"""
<speak version="1.0"
    xmlns="http://www.w3.org/2001/10/synthesis"
    xmlns:mstts="http://www.w3.org/2001/mstts"
    xml:lang="en-US">

    <voice name="{VOICE_NAME}">
        <mstts:express-as
            style="{tts_params['style']}"
            styledegree="{tts_params['styledegree']}">

            <prosody
                rate="{tts_params['rate']}"
                pitch="{tts_params['pitch']}"
                volume="{tts_params['volume']}">

                {text}

            </prosody>
        </mstts:express-as>
    </voice>
</speak>
"""

    # ----------------------------
    # Synthesize
    # ----------------------------
    result = synthesizer.speak_ssml_async(ssml).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print(f"🔊 Azure TTS audio saved: {output_file}")
        return output_file

    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        print("❌ Azure TTS synthesis canceled")
        print("Reason:", cancellation.reason)
        if cancellation.error_details:
            print("Error details:", cancellation.error_details)
        return None

    else:
        print("❌ Unknown Azure TTS failure")
        return None
