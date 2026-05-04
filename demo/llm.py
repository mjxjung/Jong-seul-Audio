"""Local-LLM client (Ollama).

One-stop wrapper around Ollama's /api/generate endpoint so the rest of the
pipeline doesn't need to know which backend is being used.

Run a model first:
    ollama pull llama3.1:8b
    ollama serve     # usually auto-started by the desktop app
"""

import os

import requests


def _host() -> str:
    return os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")


def _model() -> str:
    return os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def generate(prompt: str, *, max_tokens: int = 1024, temperature: float = 0.8) -> str:
    """Single-shot completion. Returns the model's text."""
    r = requests.post(
        f"{_host()}/api/generate",
        json={
            "model": _model(),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["response"].strip()


def health_check() -> None:
    """Raise SystemExit with a clear message if Ollama isn't reachable."""
    try:
        r = requests.get(f"{_host()}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise SystemExit(
            f"cannot reach Ollama at {_host()} ({e}).\n"
            "  install: https://ollama.com  |  start: `ollama serve`\n"
            f"  pull model: `ollama pull {_model()}`"
        )
    tags = {m["name"] for m in r.json().get("models", [])}
    if _model() not in tags and not any(t.startswith(_model().split(":")[0]) for t in tags):
        raise SystemExit(
            f"model '{_model()}' not pulled. run: `ollama pull {_model()}`\n"
            f"  available: {sorted(tags) or '(none)'}"
        )
