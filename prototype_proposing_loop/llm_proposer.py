"""
LLM Proposer: interface with Claude for the function-based pipeline.

Given dataset context (global stats + cluster summaries), Claude returns a
JSON array of prototype objects, each containing a Python `select_segments`
function as a string. We parse the JSON and pass it back to the loop.

API key: read from ANTHROPIC_API_KEY (env var). Fall back to a hard-coded
constant if the env var isn't set, for convenience inside Colab.
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


# Optional fallback (set this if you don't want to export ANTHROPIC_API_KEY)
_HARDCODED_API_KEY: str | None = None


def _get_client() -> "anthropic.Anthropic":
    if not HAS_ANTHROPIC:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
    api_key = os.environ.get("ANTHROPIC_API_KEY") or _HARDCODED_API_KEY
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Either export it, or assign "
            "llm_proposer._HARDCODED_API_KEY in this file."
        )
    return anthropic.Anthropic(api_key=api_key)


def call_claude(prompt: str, system_prompt: str = "",
                model: str = "claude-sonnet-4-20250514",
                max_tokens: int = 8192) -> str:
    client = _get_client()
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    response = client.messages.create(**kwargs)
    return response.content[0].text


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def extract_json_array(response_text: str) -> list:
    """Pull the JSON array out of a Claude response. Handles markdown fencing."""
    fenced = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", response_text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    bare = re.search(r"(\[[\s\S]*\])", response_text)
    if bare:
        try:
            return json.loads(bare.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError(
        f"Could not extract JSON array from LLM response. "
        f"Preview: {response_text[:500]}..."
    )


def _validate_prototype(p: dict) -> None:
    """Light schema check — fail loud on missing essentials."""
    for key in ("prototype_id", "name", "function"):
        if key not in p:
            raise ValueError(f"prototype missing required field '{key}': {p}")
    if not isinstance(p["function"], str) or "select_segments" not in p["function"]:
        raise ValueError(
            f"prototype '{p.get('name')}' has invalid function field "
            f"(must be a string defining select_segments)"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def propose_initial_prototypes(n_prototypes: int, global_statistics: dict,
                                cluster_summaries: list, metadata: dict,
                                initial_prompt_template: str) -> list:
    """First-round prototype generation."""
    prompt = initial_prompt_template.format(
        n_prototypes=n_prototypes,
        n_clusters=len(cluster_summaries),
        global_statistics=json.dumps(global_statistics, indent=2),
        cluster_summaries=json.dumps(cluster_summaries, indent=2),
        channel_descriptions=json.dumps(metadata.get("channels", {}), indent=2),
        dataset_name=metadata.get("dataset_name", "unknown"),
        target_variable=metadata.get("target_variable", "load"),
        target_description=metadata.get("target_description", ""),
        time_resolution=metadata.get("time_resolution", "15 minutes"),
        input_length=metadata.get("input_length", 384),
        output_length=metadata.get("output_length", 96),
        total_segments=metadata.get("total_segments", 0),
        training_rows=metadata.get("training_rows", 0),
    )
    system_prompt = (
        "You are a domain expert in time series forecasting. You produce "
        "Python filter functions and structured JSON. Always return valid JSON."
    )
    print(f"Calling Claude to propose {n_prototypes} prototypes...")
    response = call_claude(prompt, system_prompt)
    prototypes = extract_json_array(response)
    for p in prototypes:
        _validate_prototype(p)
    print(f"  Received {len(prototypes)} prototypes.")
    return prototypes


def propose_revisions(current_prototypes: list, feedback_text: str,
                       revision_prompt_template: str) -> list:
    """Revision round: ask Claude to fix flagged prototypes."""
    prompt = revision_prompt_template.format(
        current_prototypes=json.dumps(current_prototypes, indent=2),
        feedback=feedback_text,
    )
    system_prompt = (
        "You are revising prototype filter functions based on validation "
        "feedback. Return the COMPLETE list of all prototypes (revised + "
        "unchanged) as a JSON array."
    )
    print("Calling Claude for revision...")
    response = call_claude(prompt, system_prompt)
    prototypes = extract_json_array(response)
    for p in prototypes:
        _validate_prototype(p)
    print(f"  Received {len(prototypes)} prototypes.")
    return prototypes
