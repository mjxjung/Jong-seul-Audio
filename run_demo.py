"""End-to-end demo for the personalized Audio-LM dataset pipeline.

Pipeline:
  1. Generate synthetic character profiles  (Claude)
  2. Generate per-character preferred responses for each query  (Claude, roleplay)
  3. Synthesize each query in each character's voice  (edge-tts)
  4. (optional) Run the audio through a Large Audio-LM and compare  (Qwen2-Audio)

Each stage is cached on disk; rerunning skips work that already finished.
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from demo.characters import generate_characters
from demo.llm import health_check
from demo.queries import QUERIES
from demo.responses import generate_responses
from demo.voices import synthesize_dataset


def main() -> None:
    load_dotenv()
    health_check()

    p = argparse.ArgumentParser()
    p.add_argument("--characters", type=int, default=4, help="number of synthetic characters")
    p.add_argument("--queries", type=int, default=5, help=f"number of queries (max {len(QUERIES)})")
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--evaluate", action="store_true", help="also run Qwen2-Audio inference")
    p.add_argument("--eval-limit", type=int, default=4, help="cap eval to first N triplets")
    args = p.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    chars_path = args.data_dir / "characters.json"
    triplets_path = args.data_dir / "triplets.json"
    audio_dir = args.data_dir / "audio"

    print(f"\n[1/3] characters -> {chars_path}")
    if chars_path.exists():
        characters = json.loads(chars_path.read_text())
        print(f"  cached ({len(characters)} characters)")
    else:
        characters = generate_characters(args.characters, chars_path)
        print(f"  generated {len(characters)} characters")

    queries = QUERIES[:args.queries]
    print(f"\n[2/3] preferred responses ({len(characters)} x {len(queries)}) -> {triplets_path}")
    triplets = generate_responses(characters, queries, triplets_path)

    print(f"\n[3/3] voice synthesis -> {audio_dir}")
    triplets = synthesize_dataset(triplets, audio_dir)
    triplets_path.write_text(json.dumps(triplets, indent=2, ensure_ascii=False))

    print("\n--- preview: same query, different preferred responses ---")
    if triplets:
        first_q = triplets[0]["query_id"]
        char_lookup = {c["id"]: c for c in characters}
        for t in triplets:
            if t["query_id"] != first_q:
                continue
            c = char_lookup.get(t["character_id"], {})
            print(f"\n[{t['character_id']}] {c.get('name', '?')}, "
                  f"{c.get('age', '?')} {c.get('gender', '?')}, {c.get('occupation', '?')}")
            print(f"  voice: {t['voice_id']}  audio: {t['voice_audio_path']}")
            print(f"  Q: {t['query_text']}")
            print(f"  A: {t['preferred_response']}")

    if args.evaluate:
        print("\n[4/4] audio-LM evaluation")
        from demo.evaluate import evaluate
        evaluate(triplets_path, limit=args.eval_limit)


if __name__ == "__main__":
    main()
