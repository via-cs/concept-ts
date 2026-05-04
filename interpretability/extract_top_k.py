"""
Interpretability tool: for each learned prototype, find the top-K training
segments whose encoder match_key is most similar to it.

Pipeline:
    1. Load the pretrained encoder (from embedding_learning.py).
    2. Parse the trained prototype checkpoint (from prototype_learning.py)
       and pull out prototype_embedding / prototype_value tensors at the
       chosen layer (default = deepest layer in the checkpoint).
    3. Run the encoder over every training segment, take match_key as
       encoder.mean(dim=1)  -- this matches the C1 architecture currently
       active in layer/prototype.py.
    4. For each prototype p_i, rank all segments by similarity(match_key, p_i)
       and keep the top-K.
    5. Save segment indices, similarity scores, and target curves to JSON,
       and emit a per-prototype plot overlaying the K curves alongside the
       prototype's learned forecast (prototype_value).

Usage from repo root:
    python -u ./interpretability/extract_top_k.py \
        --dataset ./dataset/LOF/RE.csv \
        --encoder_checkpoint ./checkpoint/embedding_learning/RE_OL96_IL384_L4_D0.5_EMB128_BDP0.1.pth \
        --prototype_checkpoint ./checkpoint/prototype_learning/RE_OL96_IL384_PN6_SR0.4.pth \
        --top_k 10 \
        --output_dir ./interpretability/output/RE
"""

import argparse
import json
import os
import sys

import numpy as np
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
    parser.add_argument("--inverse_scale", action="store_true",
                        help="Plot target curves in original units (un-scale via StandardScaler).")

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
# Core
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_match_keys(model, dataset, batch_size, device):
    """Run the encoder over every segment and return per-segment match_key.

    Returns
    -------
    Tensor of shape (N_segments, mid_dim).
    """
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
    """Return sorted list of prototype-layer indices found in a state dict."""
    layers = []
    i = 0
    while f"prototype.prototype_embedding.{i}" in state_dict:
        layers.append(i)
        i += 1
    return layers


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_prototype_topk(p_idx, topk_segments, prototype_value_curve,
                         input_length, output_length, save_path):
    """Overlay the K segment curves, their average, and the prototype's
    learned forecast curve, all on one axis."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))

    t_lb = np.arange(input_length)
    t_fc = np.arange(input_length, input_length + output_length)

    # Faint individual curves
    for seg in topk_segments:
        ax.plot(t_lb, seg["lookback_target"], color="steelblue", alpha=0.25, lw=0.8)
        ax.plot(t_fc, seg["forecast_target"], color="crimson",   alpha=0.25, lw=0.8)

    # Bold averages across the K
    avg_lb = np.mean([s["lookback_target"] for s in topk_segments], axis=0)
    avg_fc = np.mean([s["forecast_target"] for s in topk_segments], axis=0)
    ax.plot(t_lb, avg_lb, color="steelblue", lw=2.2,
            label=f"mean lookback (K={len(topk_segments)})")
    ax.plot(t_fc, avg_fc, color="crimson",   lw=2.2,
            label=f"mean forecast (K={len(topk_segments)})")

    # Prototype's learned forecast curve (prototype_value)
    if prototype_value_curve is not None:
        ax.plot(t_fc, prototype_value_curve, color="green", lw=2.2,
                linestyle="--", label="prototype_value (model output)")

    ax.axvline(x=input_length, color="gray", linestyle=":", alpha=0.6,
               label="lookback / forecast boundary")
    ax.set_title(f"Prototype {p_idx} — top-{len(topk_segments)} most-similar segments")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Target value")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=80)
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

    # 3. Training data ------------------------------------------------------
    train_data = LOF_Dataset(args.input_length, args.output_length, "train",
                             args.input_dim, args.dataset)
    print(f"Loaded {len(train_data)} training segments from {args.dataset}")

    # 4. Per-segment match keys --------------------------------------------
    match_keys = compute_match_keys(encoder_model, train_data, args.batch_size, device)
    print(f"match_keys: {tuple(match_keys.shape)}")

    # 5. Similarity ---------------------------------------------------------
    sim = compute_similarity(match_keys, prototype_embedding, args.similarity)
    print(f"similarity matrix ({args.similarity}): {tuple(sim.shape)}")

    # 6. Top-K per prototype ------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, "plots")
    if not args.no_plots:
        os.makedirs(plot_dir, exist_ok=True)

    results = {
        "metadata": {
            "dataset": args.dataset,
            "encoder_checkpoint": args.encoder_checkpoint,
            "prototype_checkpoint": args.prototype_checkpoint,
            "layer": layer_idx,
            "available_layers": available_layers,
            "top_k": args.top_k,
            "similarity_mode": args.similarity,
            "n_prototypes": int(P),
            "embed_dim": int(D),
            "n_training_segments": int(match_keys.shape[0]),
            "inverse_scale": bool(args.inverse_scale),
        },
        "segments_per_prototype": {},
    }

    print(f"\nFinding top-{args.top_k} per prototype:")
    for p_idx in range(P):
        scores = sim[:, p_idx]
        topk_vals, topk_idx = torch.topk(scores, args.top_k)
        topk_idx  = topk_idx.tolist()
        topk_vals = topk_vals.tolist()

        topk_segments = []
        for rank, (idx, score) in enumerate(zip(topk_idx, topk_vals)):
            input_data, target = train_data[idx]
            # Last column of input_data is target * mask: first input_length steps
            # are the unmasked lookback target, last output_length are zeroed.
            lb_tgt = input_data[: args.input_length, -1].numpy().astype(np.float64)
            fc_tgt = target.squeeze(-1).numpy().astype(np.float64)

            if args.inverse_scale:
                lb_tgt = train_data.inver_data(lb_tgt)
                fc_tgt = train_data.inver_data(fc_tgt)

            topk_segments.append({
                "rank": rank + 1,
                "segment_index": int(idx),
                "similarity_score": float(score),
                "lookback_target": lb_tgt.tolist(),
                "forecast_target": fc_tgt.tolist(),
            })

        results["segments_per_prototype"][p_idx] = topk_segments

        if not args.no_plots:
            pv = prototype_value[p_idx].squeeze(-1).numpy().astype(np.float64)
            if args.inverse_scale:
                pv = train_data.inver_data(pv)
            plot_prototype_topk(
                p_idx=p_idx,
                topk_segments=topk_segments,
                prototype_value_curve=pv,
                input_length=args.input_length,
                output_length=args.output_length,
                save_path=os.path.join(plot_dir, f"prototype_{p_idx:02d}.png"),
            )

        head = ", ".join(f"{i}({s:.3f})" for i, s in zip(topk_idx[:3], topk_vals[:3]))
        print(f"  prototype {p_idx:2d}: top3 = [{head}, ...]")

    # 7. Save JSON ----------------------------------------------------------
    out_json = os.path.join(args.output_dir, "topk_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON  -> {out_json}")
    if not args.no_plots:
        print(f"Saved plots -> {plot_dir}/")


if __name__ == "__main__":
    main()
