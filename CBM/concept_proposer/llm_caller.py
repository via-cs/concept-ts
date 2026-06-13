"""
Thin Anthropic-API wrapper used by ``propose_concepts.py``.

Mirrors ``prototype_proposing_loop/llm_proposer.py`` but only does the
one-shot call needed for CBM atomic-concept proposal — no revision loop.

Reads the API key from ``ANTHROPIC_API_KEY`` (or the optional hard-coded
fallback below, convenient inside Colab).
"""

from __future__ import annotations

import json
import os
import re

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# Optional fallback so you don't have to export ANTHROPIC_API_KEY every time
# in Colab. Leave as None for env-var-only behavior.
_HARDCODED_API_KEY: str | None = None


def _get_client() -> "anthropic.Anthropic":
    if not HAS_ANTHROPIC:
        raise RuntimeError(
            "anthropic package not installed. Run: pip install anthropic"
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY") or _HARDCODED_API_KEY
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Either export it, or assign "
            "llm_caller._HARDCODED_API_KEY in this file."
        )
    return anthropic.Anthropic(api_key=api_key)


def call_claude(prompt: str,
                system_prompt: str = "",
                model: str = "claude-sonnet-4-20250514",
                max_tokens: int = 16384) -> str:
    """One-shot call. Returns the assistant message text."""
    client = _get_client()
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    resp = client.messages.create(**kwargs)
    return resp.content[0].text


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def extract_json_object(response_text: str) -> dict:
    """Pull the first JSON object out of a Claude response. Tolerates markdown
    fencing and a small amount of surrounding text. Returns a dict — for the
    two-set CBM prompt we expect keys 'lookback_concepts' and 'forecast_concepts'.
    """
    # Fenced block: ```json { ... } ```
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", response_text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Bare object — greedy match for the largest {...} in the response
    bare = re.search(r"(\{[\s\S]*\})", response_text)
    if bare:
        try:
            return json.loads(bare.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not extract a JSON object from the LLM response. "
        f"First 500 chars: {response_text[:500]!r}"
    )


def validate_concept_schema(c: dict) -> None:
    """Light schema check — fail loud on missing required fields."""
    for key in ("id", "name", "function"):
        if key not in c:
            raise ValueError(f"concept missing required field {key!r}: {c}")
    if not isinstance(c["function"], str) or "matches" not in c["function"]:
        raise ValueError(
            f"concept {c.get('name')!r} has invalid function field "
            "(must be a string defining `def matches(seg)` or `def matches(sub_seg)`)"
        )
