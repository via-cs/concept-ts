"""
Shared helpers for ProtoLLM interpretability experiments.

This module is imported by ``aggregate_alignment.py`` and ``case_studies.py``.
It provides:

  * ``load_encoder_and_prototype``  – build encoder + Prototype with the
                                      trained checkpoints loaded.
  * ``load_llm_prototypes``         – read the ``final_prototypes_*.json`` the
                                      LLM proposed and return only the
                                      anchored (non-blank) layer-0 prototypes.
  * ``build_test_segments``         – run the Part-A preprocessing pipeline on
                                      the test slice of the CSV, returning a
                                      list[dict] with the same schema as the
                                      training-segments JSON the LLM saw.
  * ``find_matching_test_segments`` – run an LLM-supplied filter function
                                      against the test segments.
  * ``forward_capture``             – batched forward pass on a chosen list of
                                      dataset window indices, returning
                                      ``{layer0_score, prediction, target,
                                      input_data}`` tensors.
  * ``metadata_for_segment``        – pull human-readable metadata (date,
                                      day-type, mean temp, mean load, season)
                                      from a test segment dict.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make the project root importable when this file is executed via
# ``PYTHONPATH=. python experiments/...``.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataloader.dataloader import LOF_Dataset            # noqa: E402
from model.embedding import Embedding_Learning           # noqa: E402
from model.prototype import Prototype                    # noqa: E402
from preprocessing.data_preprocessing import (            # noqa: E402
    CHANNEL_META,
    segment_time_series,
)
from prototype_proposing_loop.function_executor import (  # noqa: E402
    execute_select_function,
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_encoder_and_prototype(
    *,
    pretrain_checkpoint: str,
    prototype_checkpoint: str,
    prototype_num: int,
    input_dim: int,
    output_dim: int,
    embed_dim: int,
    input_length: int,
    output_length: int,
    num_layer: int,
    dropout: float,
    bottle_dim_p: float,
    prototype_split_num: int,
    top_k_activate: int,
    split_ratio: float,
    device: str,
):
    """Instantiate encoder + Prototype and load the trained checkpoints.

    The Prototype is instantiated with ``layer_num=1`` (the default). If the
    saved checkpoint has been hierarchically split (layer_num > 1) the extra
    layer-N tensors are loaded non-strictly. This is intentional — every
    interpretability experiment in this folder only inspects the LLM-named
    LAYER-0 prototypes, so the deeper layers are not needed and their absence
    is harmless.
    """
    os.environ.setdefault("device", device)

    encoder = Embedding_Learning(
        input_dim=input_dim,
        mid_dim=embed_dim,
        ext_dim=input_dim - output_dim,
        output_dim=output_dim,
        bottle_dim_p=bottle_dim_p,
        input_length=input_length,
        output_length=output_length,
        num_blocks=num_layer,
        dropout=dropout,
        norm_type="B",
        use_rev_in=1,
    ).to(device)

    prototype_model = Prototype(
        prototype_num=prototype_num,
        prototype_dim=embed_dim,
        output_dim=output_dim,
        input_length=output_length,
        prototype_split_num=prototype_split_num,
        top_k_activate=top_k_activate,
        split_ratio=split_ratio,
    ).to(device)

    enc_state = torch.load(pretrain_checkpoint, map_location=device)
    enc_result = encoder.load_state_dict(enc_state, strict=False)
    _warn_about_state_diff("encoder", enc_result)

    proto_state = torch.load(prototype_checkpoint, map_location=device)
    # The Prototype was instantiated with layer_num=1, so only layer-0
    # entries in the ParameterLists / ModuleList exist on our model object.
    # The trained checkpoint may have additional layers (after hierarchical
    # splitting) — keep only the layer-0 keys to avoid silently dropping
    # tensors via strict=False (which would mask other bugs too).
    keep = {k: v for k, v in proto_state.items() if _is_layer0_param(k)}
    proto_result = prototype_model.load_state_dict(keep, strict=False)
    _warn_about_state_diff("prototype", proto_result)

    encoder.eval()
    prototype_model.eval()
    return encoder, prototype_model


_LAYERED_TAGS = ("prototype_embedding", "prototype_value", "layer_bias")


def _is_layer0_param(key: str) -> bool:
    """True iff this state_dict key belongs to layer 0 of a layered
    container (prototype_embedding / prototype_value / layer_bias). Keys
    not inside any of those containers are kept unconditionally (e.g.
    ``alpha`` or ``matching_head.weight``).

    State-dict keys look like:
        prototype.prototype_embedding.0           → keep
        prototype.prototype_embedding.1           → drop
        prototype.layer_bias.0.0.weight           → keep
        prototype.layer_bias.2.2.weight           → drop
        alpha                                     → keep (not layered)
    """
    for tag in _LAYERED_TAGS:
        marker = f"{tag}."
        i = key.find(marker)
        if i < 0:
            continue
        # The character right after the marker is the layer index. Take the
        # full integer token up to the next ``.`` or end-of-string.
        rest = key[i + len(marker):]
        idx_token = rest.split(".", 1)[0]
        return idx_token == "0"
    return True   # not inside a layered container → keep


def _warn_about_state_diff(label: str, result) -> None:
    missing = list(result.missing_keys) if hasattr(result, "missing_keys") else []
    unexpected = list(result.unexpected_keys) if hasattr(result, "unexpected_keys") else []
    if missing:
        print(f"  [{label}] missing keys ({len(missing)}): "
              + ", ".join(missing[:6]) + (" …" if len(missing) > 6 else ""))
    if unexpected:
        print(f"  [{label}] ignored unexpected keys ({len(unexpected)}): "
              + ", ".join(unexpected[:6]) + (" …" if len(unexpected) > 6 else ""))


# ---------------------------------------------------------------------------
# LLM prototype metadata
# ---------------------------------------------------------------------------

def load_llm_prototypes(final_prototypes_path: str) -> list[dict]:
    """Read final_prototypes_<timestamp>.json and return the anchored
    layer-0 prototypes in the same order they sit inside ``init_embedding.pt``
    / ``prototype_embedding[0]``.

    Skips blank prototypes (those added by --num_blank when building init).
    """
    with open(final_prototypes_path) as f:
        data = json.load(f)

    # Common JSON shapes seen across runs:
    #   {"prototypes": [...]}, {"final_prototypes": [...]}, or a plain list.
    if isinstance(data, list):
        protos = data
    elif "prototypes" in data:
        protos = data["prototypes"]
    elif "final_prototypes" in data:
        protos = data["final_prototypes"]
    else:
        raise ValueError(
            f"Could not find prototype list inside {final_prototypes_path}. "
            f"Top-level keys: {list(data.keys())}"
        )

    out = []
    for i, p in enumerate(protos):
        name = p.get("name") or p.get("prototype_id") or f"proto_{i}"
        out.append({
            "anchor_idx": i,                       # row in init_embedding.pt
            "name": str(name),
            "function": p.get("function", ""),
            "rubric": p.get("rubric") or p.get("description") or "",
        })
    return out


# ---------------------------------------------------------------------------
# Test-set segments (same schema as the train segments the LLM saw)
# ---------------------------------------------------------------------------

def build_test_segments(
    *,
    csv_path: str,
    input_length: int,
    output_length: int,
    stride: int = 96,
    split: str = "test",
) -> list[dict]:
    """Build a list of segment dicts over the requested split of the CSV.

    Uses the SAME pipeline ``preprocessing/data_preprocessing.py`` runs on the
    training portion (and the same 70/10/20 row partition), so the resulting
    schema is identical to the JSON the LLM saw at proposal time. The
    ``start_idx`` of each returned segment is relative to the split's own
    dataframe — i.e. you can use it directly as a ``LOF_Dataset(split=split)``
    window index.

    ``split``:
      * "test"   – rows [int(0.8*N) : N]
      * "vali"   – rows [int(0.7*N) : int(0.8*N)]
      * "train"  – rows [0 : int(0.7*N)]
    """
    df = pd.read_csv(csv_path, encoding="utf-8")
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.8 * n)

    if split == "train":
        sub = df.iloc[:train_end].reset_index(drop=True)
    elif split == "vali":
        sub = df.iloc[train_end:val_end].reset_index(drop=True)
    elif split == "test":
        sub = df.iloc[val_end:n].reset_index(drop=True)
    else:
        raise ValueError(f"unknown split: {split!r}")

    return segment_time_series(
        sub,
        input_length=input_length,
        output_length=output_length,
        stride=stride,
    )


def find_matching_test_segments(
    prototype_spec: dict, test_segments: list[dict]
) -> tuple[list[int], str | None]:
    """Run ``prototype_spec['function']`` against the test segments.

    Returns (matching_indices, error). ``error`` is ``None`` on success.
    """
    code = prototype_spec.get("function", "")
    if not code.strip():
        return [], "empty function"
    result = execute_select_function(code, test_segments)
    return (result["matched_indices"] or []), result["error"]


# ---------------------------------------------------------------------------
# Forward pass on chosen window indices
# ---------------------------------------------------------------------------

@torch.no_grad()
def forward_capture(
    encoder: nn.Module,
    prototype_model: nn.Module,
    dataset: LOF_Dataset,
    window_indices: list[int],
    device: str,
    batch_size: int = 32,
) -> dict:
    """Run forward on the given dataset window indices, capture per-sample
    layer-0 scores + predictions.

    Returns
    -------
    dict with tensors:
        layer0_score : (K, P)    softmax score over layer-0 prototypes
        prediction   : (K, 96, 1) forecast in normalized space
        target       : (K, 96, 1) ground-truth in normalized space
        input_data   : (K, 480, in_dim) raw model input (for replotting)
    """
    encoder.eval()
    prototype_model.eval()

    all_layer0, all_pred, all_target, all_input = [], [], [], []
    for chunk_start in range(0, len(window_indices), batch_size):
        chunk = window_indices[chunk_start:chunk_start + batch_size]
        items = [dataset[i] for i in chunk]
        x = torch.stack([it[0] for it in items], dim=0).to(device)
        y = torch.stack([it[1] for it in items], dim=0).to(device)

        # Encoder forward
        _, encoder_out = encoder(x)

        # Prototype layer forward — we want the per-layer scores out, so call
        # the inner Prototype_Layer directly after applying the align step
        # that Prototype.forward would have applied.
        use_align = (prototype_model.adapter_mode != "none")
        embd = encoder_out
        if use_align:
            embd = prototype_model.align_layer(x, embd, prototype_model.input_length)
        all_ans, all_score = prototype_model.prototype(embd)
        if use_align:
            all_ans = prototype_model.align_layer.backward(
                x, all_ans, prototype_model.input_length
            )

        # Layer-0 score: shape (B, 1, P0). Squeeze the singleton "1" dim.
        layer0_score = all_score[0].squeeze(1).cpu()    # (B, P0)
        # Final prediction = top layer's ans.
        prediction = all_ans[-1].cpu()                  # (B, 96, O)

        all_layer0.append(layer0_score)
        all_pred.append(prediction)
        all_target.append(y.cpu())
        all_input.append(x.cpu())

    return {
        "layer0_score": torch.cat(all_layer0, dim=0),
        "prediction":   torch.cat(all_pred,   dim=0),
        "target":       torch.cat(all_target, dim=0),
        "input_data":   torch.cat(all_input,  dim=0),
    }


# ---------------------------------------------------------------------------
# Metadata extraction from a segment dict
# ---------------------------------------------------------------------------

def metadata_for_segment(seg: dict) -> dict:
    """Pull a small, human-readable summary out of a test segment dict."""
    tc = seg.get("time_context", {})
    fcs = tc.get("forecast_start", {})
    lb = seg.get("lookback_stats", {})
    cs = seg.get("cross_channel_stats", {})

    def _get(d, *path, default=None):
        for p in path:
            d = d.get(p) if isinstance(d, dict) else None
            if d is None:
                return default
        return d

    return {
        "segment_id": seg.get("segment_id"),
        "start_idx":  seg.get("start_idx"),
        "season":     tc.get("season"),
        "solar_month": fcs.get("solar_month"),
        "solar_day":  fcs.get("solar_day"),
        "forecast_hour": fcs.get("hour"),
        "day_type":   cs.get("uniform_day_type") or
                      ("workday" if not cs.get("contains_holiday") else "holiday"),
        "is_holiday": bool(cs.get("contains_holiday", False)),
        "mean_skin_temp_C": _get(lb, "Skin_temperature", "mean"),
        "mean_load":        _get(lb, "load", "mean"),
        "mean_humidity_pct": _get(lb, "Relative_humidity", "mean"),
    }
