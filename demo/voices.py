"""Voice synthesis with edge-tts.

Each (character, query) pair produces one audio file: the character's voice
reading that query out loud. Free, no API key, but needs network access.
"""

import asyncio
from pathlib import Path

import edge_tts


async def _synth(text: str, voice: str, out_path: Path) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(out_path))


def synthesize(text: str, voice: str, out_path: Path) -> None:
    asyncio.run(_synth(text, voice, out_path))


def synthesize_dataset(triplets: list[dict], audio_dir: Path) -> list[dict]:
    audio_dir.mkdir(parents=True, exist_ok=True)
    for t in triplets:
        out_path = audio_dir / f"{t['character_id']}_{t['query_id']}.mp3"
        if not out_path.exists():
            synthesize(t["query_text"], t["voice_id"], out_path)
            print(f"  synthesized {out_path.name}")
        t["voice_audio_path"] = str(out_path)
    return triplets
