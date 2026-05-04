"""Generate the preferred response for every (character, query) pair.

The local LLM is asked to roleplay as the character and produce the response
THAT character would prefer to receive. Cached incrementally.
"""

import json
from pathlib import Path

from demo.llm import generate

PROMPT = """You are roleplaying as the following character. Respond to the user's question
the way THIS character would PREFER to receive an answer — match their preferred
length, tone, and level of detail. Stay in character.

CHARACTER:
{character}

USER QUESTION:
{query}

Respond directly in 1-4 sentences. No preamble, no meta-commentary, no quoting the question.
Answer:
"""


def _key(char_id: str, query_id: str) -> tuple[str, str]:
    return (char_id, query_id)


def generate_responses(
    characters: list[dict],
    queries: list[dict],
    output_path: Path,
) -> list[dict]:
    existing: dict[tuple[str, str], dict] = {}
    if output_path.exists():
        for t in json.loads(output_path.read_text()):
            existing[_key(t["character_id"], t["query_id"])] = t

    triplets: list[dict] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for char in characters:
        char_view = {k: v for k, v in char.items() if k != "voice_id"}
        char_str = json.dumps(char_view, ensure_ascii=False, indent=2)
        for q in queries:
            k = _key(char["id"], q["id"])
            if k in existing:
                triplets.append(existing[k])
                continue

            response = generate(
                PROMPT.format(character=char_str, query=q["text"]),
                max_tokens=512,
                temperature=0.7,
            )
            triplet = {
                "character_id": char["id"],
                "query_id": q["id"],
                "query_text": q["text"],
                "preferred_response": response,
                "voice_id": char["voice_id"],
            }
            triplets.append(triplet)
            output_path.write_text(json.dumps(triplets, indent=2, ensure_ascii=False))
            print(f"  generated {char['id']} x {q['id']}")

    output_path.write_text(json.dumps(triplets, indent=2, ensure_ascii=False))
    return triplets
