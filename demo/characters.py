"""Synthetic character profile generator.

Uses a local LLM (Ollama) to invent diverse personas and assigns each one an
edge-tts voice that loosely matches their profile.
"""

import json
from pathlib import Path

from demo.llm import generate

VOICES = [
    {"id": "en-US-AriaNeural", "lang": "en", "gender": "female", "style": "cheerful young"},
    {"id": "en-US-GuyNeural", "lang": "en", "gender": "male", "style": "mature confident"},
    {"id": "en-US-JennyNeural", "lang": "en", "gender": "female", "style": "friendly warm"},
    {"id": "en-US-DavisNeural", "lang": "en", "gender": "male", "style": "young enthusiastic"},
    {"id": "en-GB-SoniaNeural", "lang": "en", "gender": "female", "style": "calm British"},
    {"id": "en-GB-RyanNeural", "lang": "en", "gender": "male", "style": "thoughtful British"},
    {"id": "en-AU-NatashaNeural", "lang": "en", "gender": "female", "style": "Australian relaxed"},
    {"id": "en-IN-NeerjaNeural", "lang": "en", "gender": "female", "style": "Indian-English clear"},
]

PROMPT = """Generate {n} diverse character profiles for a personalized audio-language model dataset.

Each character must be DISTINCT in communication style so that the SAME question would receive
genuinely DIFFERENT preferred responses across characters.

Return ONLY a valid JSON array. No prose before or after. Each object has exactly these keys:
- "name": short first name
- "age": int (range 18-70)
- "gender": "male" | "female" | "non-binary"
- "occupation": string
- "background": one-sentence personal context (interests, life situation)
- "response_style": object with:
    - "length": "short" | "medium" | "long"
    - "tone": "formal" | "casual" | "warm" | "witty" | "analytical" | "blunt"
    - "uses_examples": boolean
    - "expertise_level": "layperson" | "intermediate" | "expert"
- "voice_id": one of {voices}

Diversify across age, gender, occupation, and especially response_style.
Output the JSON array now:
"""


def generate_characters(n: int, output_path: Path) -> list[dict]:
    voice_ids = [v["id"] for v in VOICES]
    text = generate(PROMPT.format(n=n, voices=voice_ids), max_tokens=4096, temperature=0.9)

    start, end = text.find("["), text.rfind("]") + 1
    if start == -1 or end <= start:
        raise ValueError(f"no JSON array in model output:\n{text}")
    chars = json.loads(text[start:end])

    valid_voices = {v["id"] for v in VOICES}
    for i, c in enumerate(chars):
        c["id"] = f"char_{i+1:03d}"
        if c.get("voice_id") not in valid_voices:
            c["voice_id"] = VOICES[i % len(VOICES)]["id"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(chars, indent=2, ensure_ascii=False))
    return chars
