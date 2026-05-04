"""Optional: feed each (voice_audio, query) into a Large Audio-Language Model
and save the model's response next to the gold preferred_response.

This is the actual research probe — does the model produce a response that
matches the speaker's preferred style WITHOUT being told who the speaker is?

Requires GPU (or patient CPU) and:
    pip install transformers torch librosa soundfile

Skips gracefully if dependencies are missing.
"""

import json
from pathlib import Path


def evaluate(
    triplets_path: Path,
    model_id: str = "Qwen/Qwen2-Audio-7B-Instruct",
    limit: int | None = None,
) -> list[dict] | None:
    try:
        import torch
        import librosa
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    except ImportError as e:
        print(f"[skip eval] missing deps: {e}")
        print("  install with: pip install transformers torch librosa soundfile")
        return None

    triplets = json.loads(Path(triplets_path).read_text())
    if limit:
        triplets = triplets[:limit]

    print(f"loading {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, device_map="auto", torch_dtype=dtype
    )

    sr = processor.feature_extractor.sampling_rate
    results: list[dict] = []
    for t in triplets:
        audio, _ = librosa.load(t["voice_audio_path"], sr=sr)
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": t["voice_audio_path"]},
                {"type": "text", "text": "Listen to the speaker and answer their question."},
            ],
        }]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(
            text=text, audios=[audio], return_tensors="pt", padding=True
        ).to(model.device)

        out_ids = model.generate(**inputs, max_new_tokens=256)
        out_ids = out_ids[:, inputs.input_ids.size(1):]
        model_response = processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

        results.append({**t, "model_response": model_response})
        print(f"\n[{t['character_id']} / {t['query_id']}]")
        print(f"  gold:  {t['preferred_response'][:120]}")
        print(f"  model: {model_response[:120]}")

    out_path = Path(triplets_path).with_name("eval_results.json")
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {out_path}")
    return results
