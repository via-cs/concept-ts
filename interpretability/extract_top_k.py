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
    parser.add_argument("--min_gap", type=int, default=-1,
                        help="Minimum index distance between any two selected top-K segments. "
                             "Two segments at indices s_a, s_b overlap iff |s_a - s_b| < length. "
                             "-1 (default) sets min_gap = input_length + output_length, "
                             "which guarantees zero time-step overlap. "
                             "Pass 0 to disable the constraint and recover the old behaviour.")
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
    """Run the encoder over every segment and return per-segment match_key.

    The encoder now returns a ``(B, mid_dim)`` time-averaged feature vector
    directly, so the match key is the encoder output without further reduction.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    keys = []
    for data, _ in tqdm(loader, desc="Encoding training segments"):
        data = data.to(device)
        _, encoder = model(data)            # (B, mid_dim) for ConvEmbedding,
                                             # (B, output_length, mid_dim) for MixerEmbedding
        if encoder.dim() == 3:
            encoder = encoder.mean(dim=1)   # average over time for Mixer encoder
        keys.append(encoder.cpu())
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


def compute_curve_stats(seg_curves, eps=1e-9):
    """Per-feature coherence over K segment curves of ONE feature.

    Parameters
    ----------
    seg_curves : np.ndarray, shape (K, T)
        K segments' curves for a single feature, all aligned to relative
        time-step (column 0 = first step of segment, etc.).

    Returns
    -------
    mean_std : float
        Mean over time of per-timestep std across the K curves. Same units
        as the input. Lower = K curves more aligned at every step.
    mean_cv : float
        Mean over time of per-timestep CV = std / (|mean| + eps).
        Dimensionless. Useful for cross-feature comparison.

    Returns NaN for both metrics if K < 2 (std undefined for a single sample).
    """
    if seg_curves.shape[0] < 2:
        return float("nan"), float("nan")
    mean_t = seg_curves.mean(axis=0)                       # (T,)
    std_t  = seg_curves.std(axis=0, ddof=0)                # (T,)
    cv_t   = std_t / (np.abs(mean_t) + eps)                # (T,)
    return float(std_t.mean()), float(cv_t.mean())


@torch.no_grad()
def compute_baseline_curve_stats(dataset, input_length, output_length,
                                  target_idx, n_features, eps=1e-9):
    """Stream all training segments, accumulate per-timestep mean/std per
    feature, and reduce to per-feature scalar coherence metrics.

    Returns
    -------
    mean_std : np.ndarray, shape (n_features,)
        Per-feature std averaged over time, computed as if all training
        segments were the cohort (the natural N-baseline for the K-coherence).
    mean_cv : np.ndarray, shape (n_features,)
        Per-feature CV averaged over time.
    n_segments : int
        Number of training segments used (= len(dataset)).
    """
    T = input_length + output_length
    sum_t    = np.zeros((T, n_features), dtype=np.float64)
    sum_sq_t = np.zeros((T, n_features), dtype=np.float64)
    n = 0

    for idx in tqdm(range(len(dataset)), desc="Baseline coherence (full train set)"):
        input_data, target = dataset[idx]
        full = input_data.numpy().astype(np.float64)
        # Replace the dataloader's zero-mask in the target's forecast portion
        # with the actual forecast values so the curve is complete.
        full[input_length:, target_idx] = target.squeeze(-1).numpy().astype(np.float64)
        sum_t    += full
        sum_sq_t += full * full
        n += 1

    mean_t = sum_t / n
    var_t  = np.maximum(sum_sq_t / n - mean_t * mean_t, 0.0)
    std_t  = np.sqrt(var_t)
    cv_t   = std_t / (np.abs(mean_t) + eps)
    return std_t.mean(axis=0), cv_t.mean(axis=0), n


def topk_non_overlapping(scores, k, min_gap):
    """Greedy non-overlap selection: return up to k indices whose pairwise
    distance is at least `min_gap`, ranked by descending score.

    Implements standard non-maximum suppression in 1-D index space. Two
    segments at indices s_a and s_b in a sliding-window dataset overlap iff
    |s_a - s_b| < length, so passing min_gap = input_length + output_length
    guarantees the selected segments share no time steps.

    Returns
    -------
    selected_idx : list[int]
        Up to k segment indices (may be fewer if non-overlap can't be met).
    selected_vals : list[float]
        Their corresponding similarity scores.
    """
    if min_gap <= 0:
        topk_vals, topk_idx = torch.topk(scores, k)
        return topk_idx.tolist(), topk_vals.tolist()

    sorted_vals, sorted_idx = torch.sort(scores, descending=True)
    sorted_idx_list = sorted_idx.tolist()
    sorted_vals_list = sorted_vals.tolist()

    selected_idx, selected_vals = [], []
    for idx, val in zip(sorted_idx_list, sorted_vals_list):
        if all(abs(idx - s) >= min_gap for s in selected_idx):
            selected_idx.append(idx)
            selected_vals.append(val)
            if len(selected_idx) == k:
                break
    return selected_idx, selected_vals


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

    # 4b. Baseline coherence over the entire training set ------------------
    baseline_std, baseline_cv, n_baseline = compute_baseline_curve_stats(
        train_data, args.input_length, args.output_length,
        target_idx=target_idx, n_features=args.input_dim,
    )
    print(f"Baseline computed over {n_baseline} training segments.")

    # 5. Similarity ---------------------------------------------------------
    sim = compute_similarity(match_keys, prototype_embedding, args.similarity)
    print(f"similarity matrix ({args.similarity}): {tuple(sim.shape)}")

    # 6. Top-K per prototype + plots ---------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    plot_root = os.path.join(args.output_dir, "plots")
    if not args.no_plots:
        os.makedirs(plot_root, exist_ok=True)

    # Resolve min_gap. -1 sentinel ⇒ enforce full non-overlap (length-apart).
    length = args.input_length + args.output_length
    min_gap = length if args.min_gap < 0 else args.min_gap
    if min_gap > 0:
        print(f"Non-overlap constraint: any two selected top-K segments must be "
              f"at least {min_gap} indices apart "
              f"(length = {length}, so this is "
              f"{'zero' if min_gap >= length else 'partial'} time-step overlap).")
    else:
        print("Non-overlap constraint: DISABLED (segments may share time steps).")

    results = {
        "metadata": {
            "dataset": args.dataset,
            "encoder_checkpoint": args.encoder_checkpoint,
            "prototype_checkpoint": args.prototype_checkpoint,
            "layer": int(layer_idx),
            "available_layers": available_layers,
            "top_k": int(args.top_k),
            "min_gap": int(min_gap),
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

    coherence_records = []   # one row per (prototype, feature) for the CSV

    print(f"\nFinding top-{args.top_k} non-overlapping segments per prototype + "
          f"plotting per-feature curves + computing coherence:")
    for p_idx in range(P):
        scores = sim[:, p_idx]
        topk_idx, topk_vals = topk_non_overlapping(scores, args.top_k, min_gap)

        if len(topk_idx) < args.top_k:
            print(f"  [warn] prototype {p_idx}: only {len(topk_idx)} non-overlapping "
                  f"segments available (wanted {args.top_k}). "
                  f"Consider lowering --min_gap or --top_k.")

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

        # ---- Per-feature coherence over the K matched segments -----------
        T = args.input_length + args.output_length
        K = len(topk_segments)
        # Stack the K full curves: (K, T, n_features)
        if K >= 1:
            curves_KTC = np.zeros((K, T, args.input_dim), dtype=np.float64)
            for i, seg in enumerate(topk_segments):
                full = seg["input_data"].astype(np.float64)
                # Replace zero-masked target forecast with the actual forecast
                full[args.input_length:, target_idx] = \
                    seg["forecast_target"].astype(np.float64)
                curves_KTC[i] = full

            for c, fname in enumerate(feature_names):
                mean_std_K, mean_cv_K = compute_curve_stats(curves_KTC[:, :, c])
                std_n = float(baseline_std[c])
                cv_n = float(baseline_cv[c])
                std_ratio = (mean_std_K / (std_n + 1e-9)) if not np.isnan(mean_std_K) else float("nan")
                cv_ratio  = (mean_cv_K  / (cv_n  + 1e-9)) if not np.isnan(mean_cv_K)  else float("nan")
                coherence_records.append({
                    "prototype_id": int(p_idx),
                    "feature_idx": int(c),
                    "feature_name": fname,
                    "is_target": bool(c == target_idx),
                    "n_segments_K": int(K),
                    "n_segments_N": int(n_baseline),
                    "mean_std_K": mean_std_K,
                    "mean_std_N": std_n,
                    "std_ratio_K_over_N": std_ratio,
                    "mean_cv_K": mean_cv_K,
                    "mean_cv_N": cv_n,
                    "cv_ratio_K_over_N": cv_ratio,
                })

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

    # 8. Save coherence CSV (wide format) ----------------------------------
    # Layout:
    #   row index = feature_name  (one row per feature channel)
    #   columns   = "prototype 0", "prototype 1", ..., "prototype N-1", "all_training_sets"
    #   each cell = "A(B)"  where A = mean_std and B = mean_cv
    #     - A (mean_std): per-timestep std across the cohort, averaged over time.
    #       Lower means the cohort's curves are tighter at every step.
    #     - B (mean_cv):  per-timestep CV (std / |mean|), averaged over time.
    #       Dimensionless, useful for cross-feature comparison; unstable when
    #       the per-timestep mean is near zero (e.g. for the z-scored target).
    # The "all_training_sets" column is the baseline cohort (every training
    # segment), so each row's prototype columns become directly comparable
    # to the right-most column without any mental arithmetic.
    def _fmt(std_v, cv_v):
        if std_v is None or (isinstance(std_v, float) and np.isnan(std_v)):
            return "n/a"
        return f"{std_v:.3g}({cv_v:.3g})"

    rec_lookup = {(r["prototype_id"], r["feature_idx"]): r for r in coherence_records}
    wide_rows = {}
    proto_cols = [f"prototype {p_idx}" for p_idx in range(P)]
    for c, fname in enumerate(feature_names):
        row = {}
        for p_idx in range(P):
            r = rec_lookup.get((p_idx, c))
            row[f"prototype {p_idx}"] = (
                _fmt(r["mean_std_K"], r["mean_cv_K"]) if r is not None else "n/a"
            )
        row["all_training_sets"] = _fmt(float(baseline_std[c]), float(baseline_cv[c]))
        wide_rows[fname] = row

    wide_df = pd.DataFrame.from_dict(wide_rows, orient="index",
                                      columns=proto_cols + ["all_training_sets"])
    wide_df.index.name = "feature_name"

    out_csv = os.path.join(args.output_dir, "per_feature_coherence.csv")
    wide_df.to_csv(out_csv)
    print(f"Saved coherence -> {out_csv}  "
          f"(rows = {len(wide_df)} features × cols = {len(wide_df.columns)} "
          f"= {P} prototypes + 1 baseline)")
    print("Each cell is formatted as  std(cv)  : "
          "lower std = curves more aligned at every timestep; "
          "lower cv = tighter relative spread.")

    # Brief summary: which features each prototype concentrates most
    long_df = pd.DataFrame(coherence_records)
    print("\nPer-prototype top-3 most concentrated features (smallest std vs baseline):")
    for p_idx, group in long_df.groupby("prototype_id"):
        ranked = group.sort_values("std_ratio_K_over_N").head(3)
        head = ", ".join(
            f"{r.feature_name}({r.std_ratio_K_over_N:.2f}x baseline)"
            for r in ranked.itertuples()
        )
        print(f"  prototype {p_idx:2d}: {head}")


if __name__ == "__main__":
    main()
