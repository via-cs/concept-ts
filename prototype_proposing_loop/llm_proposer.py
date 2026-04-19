"""
LLM Proposer: Interfaces with Claude API to generate and revise prototype specs.

Two modes:
1. Initial proposal: Given dataset context + global stats, propose N prototypes
2. Revision: Given previous specs + feedback report, revise low-quality prototypes

API key is loaded from environment variable ANTHROPIC_API_KEY.
"""

import os
import json
import re

# ============================================================
# TODO: pip install anthropic
# ============================================================

Claude_API_KEY = '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'

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("WARNING: anthropic package not installed. Run: pip install anthropic")


def get_client():
    """Create an Anthropic client. Reads API key from environment."""
    if not HAS_ANTHROPIC:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = Claude_API_KEY
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. "
            "Export it: export ANTHROPIC_API_KEY='sk-ant-...'"
        )

    return anthropic.Anthropic(api_key=api_key)


def call_claude(prompt: str, system_prompt: str = "", model: str = "claude-sonnet-4-20250514") -> str:
    """Send a prompt to Claude and return the response text.

    Args:
        prompt: The user message
        system_prompt: System-level instructions
        model: Model identifier (default: claude-sonnet-4-20250514)

    Returns:
        The assistant's response text
    """
    client = get_client()

    messages = [{"role": "user", "content": prompt}]

    kwargs = {
        "model": model,
        "max_tokens": 8192,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)

    return response.content[0].text


def extract_json_from_response(response_text: str) -> list:
    """Extract JSON array from Claude's response text.

    Claude often wraps JSON in markdown code blocks. This handles that.
    """
    # Try to find JSON array in code blocks first
    code_block_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response_text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find a bare JSON array
    array_match = re.search(r'(\[[\s\S]*\])', response_text)
    if array_match:
        try:
            return json.loads(array_match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not extract JSON array from LLM response. "
        f"Response preview: {response_text[:500]}..."
    )


def propose_initial_prototypes(
    n_prototypes: int,
    global_statistics: dict,
    metadata: dict,
    initial_prompt_template: str,
) -> list:
    """Ask Claude to propose N initial prototype specs.

    Args:
        n_prototypes: Number of prototypes to generate
        global_statistics: Global stats dict from Part A preprocessing
        metadata: Dataset metadata dict from Part A
        initial_prompt_template: The prompt template string

    Returns:
        List of prototype spec dicts
    """
    # Format the global stats into a compact string for the prompt
    stats_str = json.dumps(global_statistics, indent=2)
    channels_str = json.dumps(metadata.get("channels", {}), indent=2)

    prompt = initial_prompt_template.format(
        n_prototypes=n_prototypes,
        global_statistics=stats_str,
        channel_descriptions=channels_str,
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
        "You are a domain expert in time series forecasting. "
        "You generate structured JSON prototype specifications based on domain knowledge "
        "and dataset statistics. Always return valid JSON."
    )

    print(f"Calling Claude to propose {n_prototypes} prototypes...")
    response_text = call_claude(prompt, system_prompt)

    prototypes = extract_json_from_response(response_text)
    print(f"  Received {len(prototypes)} prototypes from Claude")

    return prototypes


def propose_revisions(
    current_prototypes: list,
    feedback_text: str,
    revision_prompt_template: str,
) -> list:
    """Ask Claude to revise low-quality prototypes based on feedback.

    Args:
        current_prototypes: The full list of current prototype specs
        feedback_text: Feedback report from feedback_generator
        revision_prompt_template: The revision prompt template

    Returns:
        Updated list of all prototype specs (revised + unchanged)
    """
    current_specs_str = json.dumps(current_prototypes, indent=2)

    prompt = revision_prompt_template.format(
        current_prototypes=current_specs_str,
        feedback=feedback_text,
    )

    system_prompt = (
        "You are a domain expert in time series forecasting. "
        "You are revising prototype specifications based on validation feedback. "
        "Return the COMPLETE list of all prototypes (both revised and unchanged) as a JSON array. "
        "For revised prototypes, adjust the requirements to be less restrictive "
        "using the percentile data provided, while preserving the prototype's conceptual identity."
    )

    print("Calling Claude to revise prototypes...")
    response_text = call_claude(prompt, system_prompt)

    prototypes = extract_json_from_response(response_text)
    print(f"  Received {len(prototypes)} prototypes from Claude")

    return prototypes
