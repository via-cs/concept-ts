"""
Prompt templates for the prototype proposing loop.

Two templates:
1. INITIAL_PROMPT: First-round prototype generation
2. REVISION_PROMPT: Revision based on validation feedback
"""

INITIAL_PROMPT = """You are a domain expert in electricity load forecasting for a province in northern China. You are helping design a prototype-based forecasting model.

## Context

We have a dataset of electricity load at {time_resolution} resolution. The model uses a {input_length}-step lookback window to forecast the next {output_length} steps. The training set has {training_rows} rows, segmented into {total_segments} non-overlapping windows.

Dataset: {dataset_name}
Target: {target_variable} — {target_description}

## Available Feature Channels

{channel_descriptions}

## Global Statistics Across Training Set

{global_statistics}

## Your Task

Design exactly {n_prototypes} prototypes that together cover the major electricity load regimes in this dataset. Each prototype represents a distinct, recurring pattern in the daily load curve (the {output_length}-step forecast).

For each prototype, provide:

1. **name**: A short snake_case identifier (e.g., "cold_winter_workday")
2. **description**: 2-3 sentences explaining what this prototype captures and why it matters.
3. **channel_requirements**: For each relevant feature channel, specify a filtering condition:

For channels that matter:
```json
"channel_name": {{
  "operator": "<" | ">" | "==" | "in" | "between",
  "threshold": <number>,
  "values": [<list>],
  "low": <number>, "high": <number>,
  "stat": "mean" | "mode",
  "reasoning": "Why this condition matters"
}}
```

For channels that don't matter:
```json
"channel_name": {{
  "operator": null,
  "reasoning": "Why irrelevant"
}}
```

4. **expected_load_characteristics**: What the daily load curve should roughly look like.

## Important Guidelines

- The {n_prototypes} prototypes should be MUTUALLY DISTINCT and COLLECTIVELY EXHAUSTIVE.
- Focus on REGIME-BASED conditions (sustained states over 4 days) rather than momentary transitions.
- Use the global statistics above to set REALISTIC thresholds. For example, if temperature p25=-0.48, setting a threshold of < -20 would match almost nothing.
- Not every channel needs a requirement. Only include channels that genuinely distinguish this prototype.
- For discrete channels (is_workday, solar_month, etc.), use "stat": "mode".
- For continuous channels (Skin_temperature, etc.), use "stat": "mean".
- Each prototype should match roughly 3-15% of the total segments. Avoid overly restrictive requirements.
- You do NOT need to list irrelevant channels. Only include channels with actual requirements.

## Output Format

Return a JSON array of {n_prototypes} prototype objects. Example structure:

```json
[
  {{
    "prototype_id": 1,
    "name": "cold_winter_workday",
    "description": "...",
    "channel_requirements": {{
      "Skin_temperature": {{"operator": "<", "threshold": 0, "stat": "mean", "reasoning": "..."}},
      "solar_month": {{"operator": "in", "values": [1, 2, 11, 12], "stat": "mode", "reasoning": "..."}},
      "is_workday": {{"operator": "==", "threshold": 1, "stat": "mode", "reasoning": "..."}}
    }},
    "expected_load_characteristics": {{
      "mean_range": [3500, 5600],
      "description": "High overnight heating load, dual peaks at morning and evening"
    }}
  }}
]
```

Return ONLY the JSON array, no other text.
"""


REVISION_PROMPT = """You previously proposed the following prototype specifications for an electricity load forecasting model:

{current_prototypes}

## Validation Results

We ran these prototypes against the actual dataset segment bank. Here is the evaluation report:

{feedback}

## Your Task

Revise the prototypes marked as "REVISION NEEDED" so they match enough segments. Use the percentile data in the report to set more realistic thresholds.

Rules:
1. Keep each prototype's conceptual identity (don't change what it represents, just relax overly strict thresholds).
2. Use the percentile data provided to choose thresholds that are realistic for this dataset.
3. Return the COMPLETE list of ALL prototypes (both revised and unchanged) as a JSON array.
4. Use the exact same JSON format as before.
5. For revised prototypes, update the "reasoning" field to explain your changes.

Return ONLY the JSON array, no other text.
"""
