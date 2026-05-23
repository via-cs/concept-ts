"""
diagnose_geometry.py — measure how far the trained layer-0 prototypes have
drifted from the LLM-init anchors.

Why this matters
----------------
``case_studies.py`` and ``aggregate_alignment.py`` both assume the
LLM-supplied names (``cold_winter_workday`` etc.) still describe the
geometry of the layer-0 prototypes after training. If the anchor weight
(``--anchor_lambda_emb``, ``--anchor_lambda_val`` in prototype_learning.py)
was small, the names may be vestigial labels on parameters that have moved
far from where the LLM put them — in which case the interpretability
claims based on those names lose their grounding.

This script loads:
  * the trained prototype state dict          (--prototype_checkpoint)
  * the LLM-init anchors  μ₀, p₀, anchor_mask (--init_embedding,
                                              --init_value, --anchor_mask)

and reports, per prototype:
  * For ANCHORED prototypes (anchor_mask == 1):
      - cos(μ, μ₀)                — 1.0 = no drift, 0.0 = orthogonal
      - rel_drift_μ = ‖μ−μ₀‖/‖μ₀‖
      - rel_drift_p = ‖p−p₀‖/‖p₀‖
  * For BLANK prototypes (anchor_mask == 0):
      - cos to nearest anchored μ₀ and its name — answers "has this
        blank specialised into anyone's neighborhood?"

Outputs
-------
  * ``<output_dir>/geometry_diagnosis.csv`` — one row per layer-0 prototype.
  * ``<output_dir>/geometry_diagnosis.png`` — two-panel bar chart.
  * Console summary of aggregate stats.

No forward pass and no encoder needed; this is a pure tensor comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Optional prototype-name loading (supports the new function-based pipeline
# and the legacy prototype_specs/ directory layout).
# ---------------------------------------------------------------------------

def _load_names_from_final_prototypes(path: str) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        protos = data
    elif "prototypes" in data:
        protos = data["prototypes"]
    elif "final_prototypes" in data:
        protos = data["final_prototypes"]
    else:
        raise ValueError(
            f"Cannot find prototype list in {path}; "
            f"top-level keys: {list(data.keys())}"
        )
    return [str(p.get("name") or p.get("prototype_id") or f"proto_{i}")
            for i, p in enumerate(protos)]


def _load_names_from_specs_dir(dir_path: str) -> list[str]:
    """Legacy ``LLM_prototype/prototype_specs/<dataset>_prototype_NNN.json``
    layout — one spec file per prototype, sorted by filename so anchor_idx
    matches insertion order into ``init_embedding.pt``."""
    files = sorted(p for p in Path(dir_path).iterdir()
                   if p.is_file() and p.suffix == ".json")
    names = []
    for i, f in enumerate(files):
        with open(f) as fh:
            spec = json.load(fh)
        names.append(str(spec.get("name") or spec.get("prototype_id") or f"proto_{i}"))
    return names


def load_prototype_names(args) -> list[str] | None:
    if args.final_prototypes:
        return _load_names_from_final_prototypes(args.final_prototypes)
    if args.prototype_specs_dir:
        return _load_names_from_specs_dir(args.prototype_specs_dir)
    return None


# ---------------------------------------------------------------------------
# Geometry math
# ---------------------------------------------------------------------------

def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Cosine similarity of two 1-D tensors (handles zero-norm safely)."""
    na = a.norm().clamp(min=eps)
    nb = b.norm().clamp(min=eps)
    return float((a @ b) / (na * nb))


def _rel_drift(curr: torch.Tensor, anchor: torch.Tensor, eps: float = 1e-12) -> float:
    """Relative drift = ‖curr − anchor‖ / ‖anchor‖."""
    return float((curr - anchor).norm() / anchor.norm().clamp(min=eps))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prototype_checkpoint", type=str, required=True,
                   help="Trained prototype state dict (.pth saved by "
                        "exp/prototype_learning.py).")
    p.add_argument("--init_embedding",  type=str, required=True,
                   help="LLM-init μ₀ tensor (init_embedding.pt).")
    p.add_argument("--init_value",      type=str, required=True,
                   help="LLM-init p₀ tensor (init_value.pt).")
    p.add_argument("--anchor_mask",     type=str, required=True,
                   help="LLM-init anchor mask (anchor_mask.pt).")
    p.add_argument("--final_prototypes", type=str, default=None,
                   help="Optional: final_prototypes_<timestamp>.json for "
                        "per-prototype names.")
    p.add_argument("--prototype_specs_dir", type=str, default=None,
                   help="Optional: legacy LLM_prototype/prototype_specs/ "
                        "directory for per-prototype names (one spec JSON "
                        "per prototype, sorted by filename).")
    p.add_argument("--output_dir", type=str, default="./experiments/output")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- Load tensors (CPU only — pure geometry comparison) -------------
    state = torch.load(args.prototype_checkpoint, map_location="cpu")

    KEY_MU = "prototype.prototype_embedding.0"
    KEY_P  = "prototype.prototype_value.0"
    if KEY_MU not in state or KEY_P not in state:
        raise KeyError(
            f"Expected keys '{KEY_MU}' and '{KEY_P}' in the prototype "
            f"checkpoint. Found: {list(state.keys())[:8]} ..."
        )
    mu_curr = state[KEY_MU].cpu().float()      # (P, D)
    p_curr  = state[KEY_P].cpu().float()       # (P, L, O)

    mu_anchor = torch.load(args.init_embedding, map_location="cpu").float()    # (P, D)
    p_anchor  = torch.load(args.init_value,     map_location="cpu").float()    # (P, L, O)
    mask      = torch.load(args.anchor_mask,    map_location="cpu").float()    # (P,)

    if not (mu_curr.shape == mu_anchor.shape):
        raise ValueError(f"μ shape mismatch: trained {tuple(mu_curr.shape)} "
                         f"vs anchor {tuple(mu_anchor.shape)}")
    if not (p_curr.shape == p_anchor.shape):
        raise ValueError(f"p shape mismatch: trained {tuple(p_curr.shape)} "
                         f"vs anchor {tuple(p_anchor.shape)}")

    P = mu_curr.shape[0]
    print(f"loaded {P} layer-0 prototypes  "
          f"(μ dim={mu_curr.shape[1]}, p shape={tuple(p_curr.shape[1:])})")

    names = load_prototype_names(args)
    if names is None:
        names = [f"proto_{i}" for i in range(P)]
    elif len(names) < P:
        # Likely num_blank > 0 — anchored protos come first, blanks unnamed.
        names = names + [f"blank_{i}" for i in range(P - len(names))]
    elif len(names) > P:
        names = names[:P]
    n_named = sum(1 for n in names if not n.startswith("blank_"))
    print(f"loaded {n_named} prototype names  "
          f"(remaining {P - n_named} treated as blank)")

    # ----- Per-prototype geometry -----------------------------------------
    anchored_indices = [i for i in range(P) if mask[i] > 0.5]
    blank_indices    = [i for i in range(P) if mask[i] <= 0.5]

    # Flatten p for cosine / drift on a single vector
    p_curr_flat   = p_curr.reshape(P, -1)
    p_anchor_flat = p_anchor.reshape(P, -1)

    rows = []
    for i in range(P):
        is_anchored = bool(mask[i] > 0.5)
        row = {
            "anchor_idx": i,
            "name":       names[i],
            "anchored":   is_anchored,
        }
        if is_anchored:
            row["cos_mu"]       = round(_cosine(mu_curr[i], mu_anchor[i]), 4)
            row["rel_drift_mu"] = round(_rel_drift(mu_curr[i], mu_anchor[i]), 4)
            row["cos_p"]        = round(_cosine(p_curr_flat[i], p_anchor_flat[i]), 4)
            row["rel_drift_p"]  = round(_rel_drift(p_curr_flat[i], p_anchor_flat[i]), 4)
            row["nearest_anchored"]      = None
            row["nearest_anchored_cos"]  = None
        else:
            # Blank — measure cos to the nearest anchored μ₀ (semantic
            # "has this blank specialised into someone's neighborhood?")
            if anchored_indices:
                cos_vals = [(_cosine(mu_curr[i], mu_anchor[j]), j)
                            for j in anchored_indices]
                cos_vals.sort(reverse=True)   # highest cosine first
                best_cos, best_j = cos_vals[0]
                row["nearest_anchored"]     = names[best_j]
                row["nearest_anchored_cos"] = round(best_cos, 4)
            else:
                row["nearest_anchored"]     = None
                row["nearest_anchored_cos"] = None
            row["cos_mu"]       = None
            row["rel_drift_mu"] = None
            row["cos_p"]        = None
            row["rel_drift_p"]  = None
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "geometry_diagnosis.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")

    # ----- Console summary -------------------------------------------------
    def _stats(vals, label):
        finite = [v for v in vals if v is not None and v == v]
        if not finite:
            return f"  {label:24s}  (no data)"
        mn = min(finite); mx = max(finite)
        mean = sum(finite) / len(finite)
        med = sorted(finite)[len(finite) // 2]
        return (f"  {label:24s}  mean={mean:+.4f}  "
                f"median={med:+.4f}  min={mn:+.4f}  max={mx:+.4f}  "
                f"n={len(finite)}")

    print("\n── Anchored prototypes ─────────────────────────────────────────")
    print(_stats([r["cos_mu"]       for r in rows if r["anchored"]],
                 "cos(μ, μ₀)"))
    print(_stats([r["rel_drift_mu"] for r in rows if r["anchored"]],
                 "rel_drift_μ"))
    print(_stats([r["cos_p"]        for r in rows if r["anchored"]],
                 "cos(p, p₀)"))
    print(_stats([r["rel_drift_p"]  for r in rows if r["anchored"]],
                 "rel_drift_p"))

    if blank_indices:
        print("\n── Blank prototypes (cosine to nearest anchored μ₀) ────────────")
        for r in rows:
            if not r["anchored"]:
                tag = (f"{r['nearest_anchored_cos']:+.4f} → "
                       f"{r['nearest_anchored']}") if r["nearest_anchored"] else "(no anchored prototypes)"
                print(f"  {r['name']:24s} {tag}")

    print("\nInterpretation: cos(μ, μ₀) close to 1.0 means the prototype "
          "still lives where the LLM put it — the name remains "
          "geometrically grounded. Values near 0 mean the prototype has "
          "drifted to an unrelated direction; the name is then vestigial.")

    # ----- Plot ------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(max(7, P * 0.6), 6.2), sharex=True)
    xs = list(range(P))
    colors = ["tab:blue" if r["anchored"] else "0.7" for r in rows]

    cos_vals = []
    for r in rows:
        if r["anchored"]:
            cos_vals.append(r["cos_mu"])
        else:
            cos_vals.append(r["nearest_anchored_cos"]
                            if r["nearest_anchored_cos"] is not None else 0.0)
    axes[0].bar(xs, cos_vals, color=colors)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_ylim(-1.0, 1.05)
    axes[0].set_ylabel("cos(μ, μ₀)\n[for blanks: cos to nearest anchor]")
    axes[0].set_title("Layer-0 prototype geometry vs. LLM anchors")

    drift_p_vals = [r["rel_drift_p"] if r["rel_drift_p"] is not None else 0.0
                    for r in rows]
    axes[1].bar(xs, drift_p_vals, color=colors)
    axes[1].set_ylabel("‖p − p₀‖ / ‖p₀‖")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([n if len(n) <= 18 else n[:15] + "…" for n in names],
                            rotation=45, ha="right", fontsize=8)
    axes[1].set_xlabel("prototype")

    fig.tight_layout()
    png_path = out_dir / "geometry_diagnosis.png"
    fig.savefig(png_path, dpi=130)
    plt.close(fig)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
