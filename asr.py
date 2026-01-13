import modal

app = modal.App("canary-asr")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch==2.6.0",
        "torchaudio==2.6.0",
        "soundfile",
        "librosa",
        "sacrebleu",
        "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .add_local_dir(
        "D:/All/Emotion psychologist",
        remote_path="/audio",
    )
)

@app.function(gpu="a100", image=image, timeout=900)
def transcribe(audio_path: str):
    import torch
    import librosa
    from nemo.collections.speechlm2.models import SALM

    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to("cuda")
    audio_lens = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to("cuda")

    model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
    model = model.to("cuda").eval()

    output_ids = model.generate(
        prompts=[[
            {
                "role": "user",
                "content": f"Transcribe the following: {model.audio_locator_tag}",
            }
        ]],
        audios=audio_tensor,
        audio_lens=audio_lens,
        max_new_tokens=256,
    )

    return model.tokenizer.ids_to_text(output_ids[0].cpu())



@app.local_entrypoint()
def main(audio_path: str):
    result = transcribe.remote(audio_path)
    print("\n========== TRANSCRIPT ==========\n")
    print(result)
    print("\n================================\n")
