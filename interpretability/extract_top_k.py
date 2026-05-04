"""
Interpretability tool: for each learned prototype, find the top-K training
segments whose encoder match_key is most similar to it, then plot every
feature channel of those K segments in a per-prototype subfolder.

Pipeline
--------
1. Load the pretrained encoder (from embedding_learning.py).
2. Parse the trained prototype checkpoint (from prototype_learning.py)
   and extract prototype_embedding / prototype_value at the chosen layer
   (default = deepest layer in the checkpoint).
3. Run the encoder over every training segment, take match_key as
   encoder.mean(dim=1)  -- this matches the C1 architecture currently
   active in layer/prototype.py.
4. For each prototype p_i, rank all segments by similarity(match_key, p_i)
   and keep the top-K.
5. Save a lightweight JSON of (segment indices, similarity scores) and
   emit one PNG per (prototype, feature) showing the K curves overlaid
   on top of their average. For the target feature ("load"), also overlay
   the prototype's learned forecast curve (prototype_value).

Output layout
-------------
output_dir/
    topk_results.json
    plots/
        prototype_00/
            feature_00_<name>.png
            feature_01_<name>.png
            ...
            feature_22_load.png
        prototype_01/
            ...
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make `python interpretability/extract_top_k.py` work from the repo root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.embedding import Embedding_Learning
from dataloader.dataloader import LOF_Dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Find top-K training segments most similar to each prototype."
    )

    # Required paths
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset CSV (e.g. ./dataset/LOF/RE.csv)")
    parser.add_argument("--encoder_checkpoint", type=str, required=True,
                        help="Pretrained encoder .pth from embedding_learning.py")
    parser.add_argument("--prototype_checkpoint", type=str, required=True,
                        help="Trained prototype model .pth from prototype_learning.py")

    # Interpretation knobs
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of most-similar segments to retrieve per prototype")
    parser.add_argument("--layer", type=int, default=-1,
                        help="Which prototype layer to interpret. -1 = deepest layer in the checkpoint.")
    parser.add_argument("--similarity", type=str, default="dot",
                        choices=["dot", "cosine", "l2"],
                        help="Similarity metric. 'dot' matches what the model uses internally.")
    parser.add_argument("--inverse_scale_target", action="store_true",
                        help="Plot the target ('load') in original units. "
                             "Exogenous features are always plotted in raw units (never scaled by the dataloader).")

    # Output
    parser.add_argument("--output_dir", type=str, default="./interpretability/output")
    parser.add_argument("--no_plots", action="store_true", help="Skip plot generation")

    # Model architecture (defaults match scripts/train.sh)
    parser.add_argument("--input_length", type=int, default=96 * 4)
    parser.add_argument("--output_length", type=int, default=96)
    parser.add_argument("--input_dim", type=int, default=23)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bottle_dim_p", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_feature_names(csv_path, input_dim):
    """Best-effort recovery of column names from the dataset CSV.

    LOF_Dataset uses "load" as the target and concatenates input_data columns
    as [<exogenous in CSV order>, load]. We mirror that order here so the
    feature names match input_data's columns exactly.
    """
    try:
        df_head = pd.read_csv(csv_path, nrows=0)
        cols = list(df_head.columns)
        if "load" in cols and len(cols) == input_dim:
            exog = [c for c in cols if c != "load"]
            return exog + ["load"], (input_dim - 1)
    except Exception as e:
        print(f"  [warn] couldn't parse CSV column names ({e}); using generic names")
    # Fallback
    names = [f"channel_{i}" for i in range(input_dim - 1)] + ["load"]
    return names, (input_dim - 1)


@torch.no_grad()
def compute_match_keys(model, dataset, batch_size, device):
    """Run the encoder over every segment and return per-segment match_key."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    keys = []
    for data, _ in tqdm(loader, desc="Encoding training segments"):
        data = data.to(device)
        _, encoder = model(data)            # (B, output_length, mid_dim)
        match_key = encoder.mean(dim=1)     # (B, mid_dim)  -- C1 matching key
        keys.append(match_key.cpu())
    return torch.cat(keys, dim=0)


def compute_similarity(match_keys, prototype_embedding, mode="dot"):
    """Return (N, P) similarity matrix. Higher = more similar."""
    if mode == "dot":
        return match_keys @ prototype_embedding.T
    if mode == "cosine":
        mk = torch.nn.functional.normalize(match_keys, dim=-1)
        pe = torch.nn.functional.normalize(prototype_embedding, dim=-1)
        return mk @ pe.T
    if mode == "l2":
        return -torch.cdist(match_keys, prototype_embedding)
    raise ValueError(f"Unknown similarity mode: {mode}")


def list_prototype_layers(state_dict):
    """Return sorted list of prototype-layer indices in a state dict."""
    layers = []
    i = 0
    while f"prototype.prototype_embedding.{i}" in state_dict:
        layers.append(i)
        i += 1
    return layers


def safe_filename(s: str) -> str:
    """Sanitise a feature name for use in a filename."""
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_per_feature(p_idx, topk_segments, prototype_value_target,
                      feature_names, target_idx, input_length, output_length,
                      save_dir, inverse_scale_target=False, inver_data_fn=None):
    """For each feature channel, save one PNG in `save_dir`.

    Each plot contains:
      - K faint curves (one per top-K segment), spanning lookback + forecast
      - bold mean lookback + mean forecast curves
      - vertical line at the lookback/forecast boundary
      - for the target ("load") channel only: a dashed green line showing
        prototype_value (the model's learned forecast for this prototype)
    """
    import matplotlib.pyplot as plt

    K = len(topk_segments)
    t_lb = np.arange(input_length)
    t_fc = np.arange(input_length, input_length + output_length)

    for c, fname in enumerate(feature_names):
        fig, ax = plt.subplots(figsize=(12, 4))

        is_target = (c == target_idx)

        if is_target:
            # Last column of input_data is target * mask → first input_length steps
            # are the unmasked lookback target; last output_length are zeros.
            # The actual unmasked forecast lives in seg["forecast_target"].
            curves_lb, curves_fc = [], []
            for seg in topk_segments:
                lb = seg["input_data"][:input_length, c].astype(np.float64)
                fc = seg["forecast_target"].astype(np.float64)
                if inverse_scale_target and inver_data_fn is not None:
                    lb = inver_data_fn(lb)
                    fc = inver_data_fn(fc)
                curves_lb.append(lb)
                curves_fc.append(fc)
                ax.plot(t_lb, lb, color="steelblue", alpha=0.25, lw=0.8)
                ax.plot(t_fc, fc, color="crimson",   alpha=0.25, lw=0.8)

            avg_lb = np.mean(curves_lb, axis=0)
            avg_fc = np.mean(curves_fc, axis=0)
            ax.plot(t_lb, avg_lb, color="steelblue", lw=2.2,
                    label=f"mean lookback (K={K})")
            ax.plot(t_fc, avg_fc, color="crimson", lw=2.2,
                    label=f"mean forecast (K={K})")

            if prototype_value_target is not None:
                pv = prototype_value_target.astype(np.float64)
                if inverse_scale_target and inver_data_fn is not None:
                    pv = inver_data_fn(pv)
                ax.plot(t_fc, pv, color="green", lw=2.2, linestyle="--",
                        label="prototype_value (model output)")

            ylabel = f"{fname} ({'original' if inverse_scale_target else 'z-scored'})"

        else:
            # Exogenous channels are NOT scaled by LOF_Dataset, and have actual
            # values for both lookback and forecast portions of the window.
            curves_full = []
            for seg in topk_segments:
                full = seg["input_data"][:, c].astype(np.float64)
                curves_full.append(full)
                ax.plot(t_lb, full[:input_length], color="steelblue", alpha=0.25, lw=0.8)
                ax.plot(t_fc, full[input_length:], color="crimson",   alpha=0.25, lw=0.8)

            avg_full = np.mean(curves_full, axis=0)
            ax.plot(t_lb, avg_full[:input_length], color="steelblue", lw=2.2,
                    label=f"mean lookback (K={K})")
            ax.plot(t_fc, avg_full[input_length:], color="crimson", lw=2.2,
                    label=f"mean forecast (K={K})")

            ylabel = f"{fname} (raw)"

        ax.axvline(x=input_length, color="gray", linestyle=":", alpha=0.6,
                   label="lookback / forecast boundary")
        ax.set_title(f"Prototype {p_idx}  |  feature: {fname}")
        ax.set_xlabel("Time step")
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        out_name = f"feature_{c:02d}_{safe_filename(fname)}.png"
        fig.savefig(os.path.join(save_dir, out_name), dpi=80)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["device"] = device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(f"Using device: {device}")

    # 1. Encoder ------------------------------------------------------------
    encoder_model = Embedding_Learning(
        input_dim=args.input_dim,
        mid_dim=args.embed_dim,
        ext_dim=args.input_dim - args.output_dim,
        output_dim=args.output_dim,
        bottle_dim_p=args.bottle_dim_p,
        input_length=args.input_length,
        output_length=args.output_length,
        num_blocks=args.num_layer,
        dropout=args.dropout,
        norm_type="B",
        use_rev_in=1,
    ).to(device)
    encoder_model.load_state_dict(torch.load(args.encoder_checkpoint, map_location=device))
    print(f"Loaded encoder      <- {args.encoder_checkpoint}")

    # 2. Prototype tensors --------------------------------------------------
    state_dict = torch.load(args.prototype_checkpoint, map_location=device)
    available_layers = list_prototype_layers(state_dict)
    if not available_layers:
        raise RuntimeError("No prototype layers found in checkpoint state_dict.")

    layer_idx = available_layers[-1] if args.layer < 0 else args.layer
    if layer_idx not in available_layers:
        raise ValueError(f"Layer {layer_idx} not in checkpoint. Available: {available_layers}")

    prototype_embedding = state_dict[f"prototype.prototype_embedding.{layer_idx}"].cpu()
    prototype_value     = state_dict[f"prototype.prototype_value.{layer_idx}"].cpu()
    P, D = prototype_embedding.shape
    print(f"Loaded prototypes   <- {args.prototype_checkpoint}")
    print(f"  using layer {layer_idx} of {available_layers}")
    print(f"  prototype_embedding: {tuple(prototype_embedding.shape)}")
    print(f"  prototype_value:     {tuple(prototype_value.shape)}")

    # 3. Training data + feature names -------------------------------------
    train_data = LOF_Dataset(args.input_length, args.output_length, "train",
                             args.input_dim, args.dataset)
    print(f"Loaded {len(train_data)} training segments from {args.dataset}")

    feature_names, target_idx = read_feature_names(args.dataset, args.input_dim)
    print(f"Feature names ({len(feature_names)}): "
          f"{feature_names[:3]} ... {feature_names[-3:]}  (target idx = {target_idx})")

    # 4. Per-segment match keys --------------------------------------------
    match_keys = compute_match_keys(encoder_model, train_data, args.batch_size, device)
    print(f"match_keys: {tuple(match_keys.shape)}")

    # 5. Similarity ---------------------------------------------------------
    sim = compute_similarity(match_keys, prototype_embedding, args.similarity)
    print(f"similarity matrix ({args.similarity}): {tuple(sim.shape)}")

    # 6. Top-K per prototype + plots ---------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    plot_root = os.path.join(args.output_dir, "plots")
    if not args.no_plots:
        os.makedirs(plot_root, exist_ok=True)

    results = {
        "metadata": {
            "dataset": args.dataset,
            "encoder_checkpoint": args.encoder_checkpoint,
            "prototype_checkpoint": args.prototype_checkpoint,
            "layer": int(layer_idx),
            "available_layers": available_layers,
            "top_k": int(args.top_k),
            "similarity_mode": args.similarity,
            "n_prototypes": int(P),
            "embed_dim": int(D),
            "n_training_segments": int(match_keys.shape[0]),
            "feature_names": feature_names,
            "target_idx": int(target_idx),
            "inverse_scale_target": bool(args.inverse_scale_target),
        },
        "segments_per_prototype": {},
    }

    print(f"\nFinding top-{args.top_k} segments per prototype + plotting per-feature curves:")
    for p_idx in range(P):
        scores = sim[:, p_idx]
        topk_vals, topk_idx = torch.topk(scores, args.top_k)
        topk_idx  = topk_idx.tolist()
        topk_vals = topk_vals.tolist()

        # Fetch the K segments' raw data
        topk_segments = []
        for rank, (idx, score) in enumerate(zip(topk_idx, topk_vals)):
            input_data, target = train_data[idx]
            topk_segments.append({
                "rank": rank + 1,
                "segment_index": int(idx),
                "similarity_score": float(score),
                "input_data": input_data.numpy(),                  # (T, in_dim)
                "forecast_target": target.squeeze(-1).numpy(),     # (output_length,)
            })

        # Save lightweight metadata only (no curves, to keep JSON small)
        results["segments_per_prototype"][p_idx] = [
            {"rank": s["rank"], "segment_index": s["segment_index"],
             "similarity_score": s["similarity_score"]}
            for s in topk_segments
        ]

        # Plots
        if not args.no_plots:
            proto_dir = os.path.join(plot_root, f"prototype_{p_idx:02d}")
            os.makedirs(proto_dir, exist_ok=True)

            pv_target = prototype_value[p_idx].squeeze(-1).numpy().astype(np.float64)
            plot_per_feature(
                p_idx=p_idx,
                topk_segments=topk_segments,
                prototype_value_target=pv_target,
                feature_names=feature_names,
                target_idx=target_idx,
                input_length=args.input_length,
                output_length=args.output_length,
                save_dir=proto_dir,
                inverse_scale_target=args.inverse_scale_target,
                inver_data_fn=train_data.inver_data,
            )

        head = ", ".join(f"{i}({s:.3f})" for i, s in zip(topk_idx[:3], topk_vals[:3]))
        print(f"  prototype {p_idx:2d}: top3 = [{head}, ...]"
              + ("" if args.no_plots else f"  -> {os.path.join(plot_root, f'prototype_{p_idx:02d}')}/"))

    # 7. Save JSON ----------------------------------------------------------
    out_json = os.path.join(args.output_dir, "topk_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON  -> {out_json}")
    if not args.no_plots:
        print(f"Saved plots -> {plot_root}/  (one subfolder per prototype, "
              f"{args.input_dim} feature plots each)")


if __name__ == "__main__":
    main()
