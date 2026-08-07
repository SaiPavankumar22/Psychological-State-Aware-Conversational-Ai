import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "soundfile",
        "librosa",
        "transformers",
        "accelerate",
        "numpy",
        "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # 🔴 Mount ONLY audio files
    .add_local_dir(
        "D:/All/Emotion psychologist/audio-files",
        remote_path="/audio",
    )
)
