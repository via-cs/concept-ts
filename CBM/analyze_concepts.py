"""
Post-hoc concept analysis for a TRAINED Concept Bottleneck Model checkpoint.
Does NOT touch training; load a saved best checkpoint and run three
interpretability experiments on the learned STATIC concept embeddings:

  (2) CAUSAL EFFECT CURVES
      For every concept, force its activation to 1 vs 0 (all others at their
      predicted values) and average the resulting forecast change over a test
      subsample -> a per-concept H-step "effect curve" in µg/m³. Saved as line
      plots (top-K by |effect|) and a concept x forecast-hour effect heatmap.

  (3) CAUSAL-EFFECT SIMILARITY HEATMAP
      Correlate the (2) effect curves across concepts (Pearson) and cluster.
      Concepts that move the forecast the same way sit together; anti-similar
      pairs are antagonists. Axes are colored by the LLM concept category.

  (5) COUNTERFACTUAL INTERVENTION (contrasting pair + activation sweep)
      Pick a clean (low-PM2.5) and a polluted (high-PM2.5) test window, choose
      one target concept, and sweep its forced activation p in {0,.25,.5,.75,1}
      (whole-horizon: forced at every sub-window). Overlay the resulting
      forecasts + ground truth -> a "dial the concept up" editability figure.

RECOMMENDED SETTING: run on a model trained with --static_concept_emb and
--residual_dim 0 (pure static CBM). If the checkpoint IS a hybrid (residual
on), pass --disable_residual to zero the residual so the effects reflect the
concepts alone (the residual does not respond to concept interventions and
would otherwise dampen every effect).

The architecture args below MUST match the training run (same concepts file,
segments, pos-rate band, caps, and model widths) so the checkpoint loads into
an identically-shaped model.

Run
---
    PYTHONPATH=. python -u CBM/analyze_concepts.py \
        --dataset       ./dataset/PRSA_data/Aotizhongxin.csv \
        --concepts_path ./CBM/output/concepts_validated_<TS>.json \
        --segments_path ./preprocessing/output/Aotizhongxin/Aotizhongxin_full_segments.json \
        --checkpoint_dir ./checkpoint/cbm \
        --input_length 96 --output_length 96 --input_dim 15 --output_dim 1 \
        --num_layer 4 --dropout 0.5 --embed_dim 128 \
        --n_subwindows 8 --concept_emb_dim 16 --static_concept_emb \
        --residual_dim 0 \
        --min_pos_rate 0.10 --max_pos_rate 0.65 \
        --out_dir ./checkpoint/cbm/concept_analysis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from CBM.concept_bottleneck import ConceptBottleneckModel
from CBM.concept_labels import build_window_label_tables
from dataloader.dataloader import LOF_Dataset


# ---------------------------------------------------------------------------
# CLI (architecture args mirror train_cbm.py; keep them equal to your run)
# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset",        type=str, required=True)
    p.add_argument("--concepts_path",  type=str, required=True)
    p.add_argument("--segments_path",  type=str, required=True,
                   help="TRAIN segments JSON (used only to recover K / concept meta).")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="Directory containing cbm.pth (train_cbm.py's best checkpoint).")
    p.add_argument("--checkpoint_name", type=str, default="cbm.pth")
    p.add_argument("--out_dir",        type=str, default=None,
                   help="Where to save figures (default <checkpoint_dir>/concept_analysis).")
    # --- architecture (MUST match training) ---
    p.add_argument("--input_length",   type=int, default=96)
    p.add_argument("--output_length",  type=int, default=96)
    p.add_argument("--input_dim",      type=int, default=15)
    p.add_argument("--output_dim",     type=int, default=1)
    p.add_argument("--num_layer",      type=int, default=4)
    p.add_argument("--dropout",        type=float, default=0.5)
    p.add_argument("--embed_dim",      type=int, default=128)
    p.add_argument("--bottle_dim_p",   type=float, default=0.1)
    p.add_argument("--n_subwindows",   type=int, default=8)
    p.add_argument("--concept_emb_dim", type=int, default=16)
    p.add_argument("--static_concept_emb", action="store_true")
    p.add_argument("--residual_dim",   type=int, default=0)
    p.add_argument("--decoder_hidden", type=int, default=512)
    p.add_argument("--decoder_depth",  type=int, default=3)
    p.add_argument("--decoder_n_blocks", type=int, default=3)
    p.add_argument("--decoder_feedback", type=int, default=1)
    p.add_argument("--sw_id_dim",      type=int, default=16)
    p.add_argument("--sw_gru_hidden",  type=int, default=32)
    p.add_argument("--sw_summary_dim", type=int, default=64)
    p.add_argument("--attn_pool_hidden", type=int, default=32)
    p.add_argument("--randint_p",      type=float, default=0.25)
    p.add_argument("--experiment_mode", type=str, default="future_aware",
                   choices=["future_aware", "lookback_only"])
    p.add_argument("--single_forecast_bottleneck", action="store_true")
    p.add_argument("--min_pos_rate",   type=float, default=0.10)
    p.add_argument("--max_pos_rate",   type=float, default=0.65)
    p.add_argument("--cap_lb_concepts",   type=int, default=0)
    p.add_argument("--cap_fc_concepts",   type=int, default=0)
    p.add_argument("--cap_gfc_concepts",  type=int, default=0)
    p.add_argument("--cap_lb_sw_concepts", type=int, default=0)
    # --- analysis knobs ---
    p.add_argument("--disable_residual", action="store_true",
                   help="Zero the residual channel during analysis (recommended "
                        "if the checkpoint is a hybrid model, to isolate concepts).")
    p.add_argument("--n_eval_windows", type=int, default=256,
                   help="Test windows sampled to average the causal effects over.")
    p.add_argument("--eval_chunk",     type=int, default=64)
    p.add_argument("--top_k_curves",   type=int, default=10,
                   help="How many concepts (by |mean effect|) to draw as line curves.")
    p.add_argument("--sweep", type=str, default="0,0.25,0.5,0.75,1.0",
                   help="Forced-activation values for the (5) counterfactual sweep.")
    p.add_argument("--target_concept", type=str, default="",
                   help="Concept NAME for (5). Default: the most influential forecast concept.")
    p.add_argument("--target_layer",   type=str, default="fc",
                   choices=["lb", "fc", "gfc"],
                   help="Which bottleneck --target_concept lives in (default fc).")
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--base_year",      type=int, default=2013)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Build model + data, load checkpoint
# ---------------------------------------------------------------------------
def build(args, device):
    train_data = LOF_Dataset(args.input_length, args.output_length, "train",
                             args.input_dim, args.dataset)
    test_data  = LOF_Dataset(args.input_length, args.output_length, "test",
                             args.input_dim, args.dataset)

    (lb, fc, gfc, lb_sw,
     lb_meta, fc_meta, gfc_meta, lb_sw_meta) = build_window_label_tables(
        concepts_path=args.concepts_path, segments_path=args.segments_path,
        n_windows=len(train_data), only_ok=True,
        single_forecast_bottleneck=args.single_forecast_bottleneck,
        min_pos_rate=args.min_pos_rate, max_pos_rate=args.max_pos_rate,
        cap_lb=args.cap_lb_concepts, cap_fc=args.cap_fc_concepts,
        cap_gfc=args.cap_gfc_concepts, cap_lb_sw=args.cap_lb_sw_concepts,
    )
    K_lb  = lb.shape[1]
    K_gfc = gfc.shape[1]
    K_lb_sw         = lb_sw.shape[2] if lb_sw.ndim == 3 else 0
    K_lb_sw_windows = lb_sw.shape[1] if lb_sw.ndim == 3 else 0
    if args.single_forecast_bottleneck:
        K_fc = fc.shape[1]
    else:
        K_sw = fc.shape[1] if fc.ndim == 3 else 0
        K_fc = fc.shape[2] if fc.ndim == 3 else 0
        if K_sw and K_sw != args.n_subwindows:
            print(f"  note: n_subwindows {args.n_subwindows} -> {K_sw} (from segments).")
            args.n_subwindows = K_sw
    print(f"  K_lookback={K_lb}, K_forecast={K_fc}, K_global={K_gfc}, K_lb_sw={K_lb_sw}")

    model = ConceptBottleneckModel(
        input_dim=args.input_dim, mid_dim=args.embed_dim,
        ext_dim=args.input_dim - args.output_dim, output_dim=args.output_dim,
        bottle_dim_p=args.bottle_dim_p, input_length=args.input_length,
        output_length=args.output_length, num_blocks=args.num_layer,
        dropout=args.dropout,
        n_lookback_concepts=K_lb, n_forecast_concepts=K_fc,
        n_global_forecast_concepts=K_gfc, n_subwindows=args.n_subwindows,
        experiment_mode=args.experiment_mode,
        n_lookback_subwindows=K_lb_sw_windows if K_lb_sw_windows > 0 else args.n_subwindows,
        n_lookback_subwindow_concepts=K_lb_sw,
        concept_emb_dim=args.concept_emb_dim,
        static_concept_emb=args.static_concept_emb, residual_dim=args.residual_dim,
        decoder_hidden=args.decoder_hidden, decoder_depth=args.decoder_depth,
        decoder_n_blocks=args.decoder_n_blocks, decoder_feedback=bool(args.decoder_feedback),
        sub_window_id_dim=args.sw_id_dim, sw_gru_hidden=args.sw_gru_hidden,
        sw_summary_dim=args.sw_summary_dim,
        single_forecast_bottleneck=args.single_forecast_bottleneck,
        use_rev_in=1, norm_type="B",
        attn_pool_hidden=args.attn_pool_hidden, randint_p=args.randint_p,
    ).to(device)

    ckpt = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    if not args.static_concept_emb:
        print("  WARNING: --static_concept_emb not set; effects reflect input-"
              "conditional embeddings and are NOT the clean static-concept effects.")
    return model, test_data, {"lb": lb_meta, "fc": fc_meta, "gfc": gfc_meta}


def _eval_windows(test_data, n, seed):
    N = len(test_data)
    rng = np.random.RandomState(seed)
    idx = rng.choice(N, min(n, N), replace=False)
    xs = torch.stack([test_data[int(i)][0] for i in idx])       # (n, T, C)
    ys = torch.stack([test_data[int(i)][1] for i in idx])       # (n, H, 1)
    return idx, xs, ys


# ---------------------------------------------------------------------------
# (2) causal effect curves
# ---------------------------------------------------------------------------
@torch.no_grad()
def causal_effects(model, xs, device, metas, scale, disable_residual, chunk):
    """Return list of dicts: {name, category, layer, idx, eff (H,)} in µg/m³."""
    n = xs.shape[0]
    out = []
    for layer in ("lb", "gfc", "fc"):
        meta = metas[layer]
        for i in range(len(meta)):
            acc = None
            for s in range(0, n, chunk):
                xb = xs[s:s + chunk].to(device)
                y1 = model(xb, disable_residual=disable_residual,
                           concept_override={layer: {i: 1.0}})["y_hat"][:, :, 0]
                y0 = model(xb, disable_residual=disable_residual,
                           concept_override={layer: {i: 0.0}})["y_hat"][:, :, 0]
                d = (y1 - y0).sum(dim=0).cpu()                    # (H,)
                acc = d if acc is None else acc + d
            eff = (acc / n).numpy() * scale                      # (H,) µg/m³
            c = meta[i]
            out.append({"name": c.get("name", f"{layer}{i}"),
                        "category": c.get("category", ""),
                        "layer": layer, "idx": i, "eff": eff})
    return out


def _cat_colors(cats):
    import matplotlib.pyplot as plt
    uniq = sorted(set(cats))
    palette = plt.get_cmap("tab20", max(1, len(uniq)))
    cmap = {c: palette(i) for i, c in enumerate(uniq)}
    return [cmap[c] for c in cats], cmap


def plot_effects(effects, out_dir, top_k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    E = np.stack([e["eff"] for e in effects])                    # (C, H)
    names = [f"[{e['layer']}] {e['name']}" for e in effects]
    mean_abs = np.abs(E).mean(axis=1)
    order = np.argsort(-mean_abs)

    # --- line curves for the top-K most influential concepts ---
    fig, ax = plt.subplots(figsize=(11, 6))
    H = E.shape[1]
    for j in order[:top_k]:
        ax.plot(np.arange(H), E[j], lw=1.8, label=names[j])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("forecast hour"); ax.set_ylabel("Δ forecast PM2.5 (µg/m³)")
    ax.set_title(f"Causal effect curves: top-{top_k} concepts by |mean effect|\n"
                 "(force concept ON vs OFF, avg over test windows)")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "causal_effect_curves.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- concept x hour signed-effect heatmap (all concepts, sorted by mean) ---
    mean_signed = E.mean(axis=1)
    hz = np.argsort(-mean_signed)
    vmax = float(np.abs(E).max()) or 1.0
    fig, ax = plt.subplots(figsize=(12, max(5, 0.28 * len(effects))))
    im = ax.imshow(E[hz], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(effects))); ax.set_yticklabels([names[j] for j in hz], fontsize=6)
    ax.set_xlabel("forecast hour"); ax.set_title("Per-concept causal effect on the forecast (µg/m³)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Δ PM2.5 (µg/m³)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "causal_effect_heatmap.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)

    # --- importance bar (mean |effect|) ---
    fig, ax = plt.subplots(figsize=(11, max(4, 0.24 * len(effects))))
    cols, _ = _cat_colors([effects[j]["category"] for j in order])
    ax.barh(range(len(effects)), mean_abs[order], color=cols)
    ax.set_yticks(range(len(effects))); ax.set_yticklabels([names[j] for j in order], fontsize=6)
    ax.invert_yaxis(); ax.set_xlabel("mean |Δ forecast| (µg/m³)")
    ax.set_title("Concept importance (mean absolute causal effect)")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "concept_importance.png"),
                                    dpi=130, bbox_inches="tight"); plt.close(fig)
    return E, names


# ---------------------------------------------------------------------------
# (3) causal-effect similarity heatmaps
# ---------------------------------------------------------------------------
def _cluster_order(S, fallback_key):
    """Row/col order from hierarchical clustering on distance 1 - S."""
    import numpy as np
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        D = 1.0 - S
        np.fill_diagonal(D, 0.0)
        D = np.clip((D + D.T) / 2.0, 0.0, 2.0)
        return leaves_list(linkage(squareform(D, checks=False), method="average"))
    except Exception:
        return np.argsort(-fallback_key)


def _draw_similarity(S, names, effects, order, out_path, title, cbar_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    So = S[order][:, order]                                       # reorder rows + cols
    cats = [effects[j]["category"] for j in order]
    cols, _ = _cat_colors(cats)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(So, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels([names[j] for j in order], fontsize=5, rotation=90)
    ax.set_yticklabels([names[j] for j in order], fontsize=5)
    for k, c in enumerate(cols):
        ax.add_patch(plt.Rectangle((k - 0.5, -1.5), 1, 1, color=c, clip_on=False))
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=130, bbox_inches="tight"); plt.close(fig)


def plot_similarity(E, names, effects, out_dir):
    """Two complementary similarity heatmaps from the causal effect curves E (C, H):

    1. SIGNED (primary, "semantic"): cosine of the RAW (un-centered) effect
       curves. Positive => the two concepts push the forecast the SAME
       direction with a similar temporal profile (both raise / both lower);
       negative => antagonists (one raises, one lowers). This is what you want
       for "similar concepts contribute the same way."
    2. SHAPE (supplementary): Pearson corr (mean-centered) -- ignores direction,
       compares only the temporal shape. Useful to see when two concepts share
       a profile but push opposite ways (that's the summer-vs-winter case).
    """
    import numpy as np

    # --- SIGNED: cosine of un-centered curves (keeps sign + shape) ---
    n_raw = np.linalg.norm(E, axis=1, keepdims=True); n_raw[n_raw == 0] = 1.0
    S_signed = (E / n_raw) @ (E / n_raw).T                       # (C, C) in [-1,1]

    # --- SHAPE: Pearson (mean-centered) ---
    Ez = E - E.mean(axis=1, keepdims=True)
    n_z = np.linalg.norm(Ez, axis=1, keepdims=True); n_z[n_z == 0] = 1.0
    S_shape = (Ez / n_z) @ (Ez / n_z).T

    mean_signed = E.mean(axis=1)
    order_signed = _cluster_order(S_signed, mean_signed)
    order_shape  = _cluster_order(S_shape,  mean_signed)

    _draw_similarity(
        S_signed, names, effects, order_signed,
        os.path.join(out_dir, "effect_similarity_signed_heatmap.png"),
        "Causal-effect similarity (SIGNED) between concepts\n"
        "(cosine of raw effect curves: +1 = same direction & shape, -1 = antagonist; "
        "clustered; top strip = category)",
        "signed effect-curve cosine",
    )
    _draw_similarity(
        S_shape, names, effects, order_shape,
        os.path.join(out_dir, "effect_similarity_shape_heatmap.png"),
        "Causal-effect similarity (SHAPE only) between concepts\n"
        "(Pearson corr; direction removed; clustered; top strip = category)",
        "effect-curve correlation (shape)",
    )


# ---------------------------------------------------------------------------
# (5) counterfactual intervention: contrasting pair + activation sweep
# ---------------------------------------------------------------------------
@torch.no_grad()
def counterfactual(model, test_data, device, metas, effects, args,
                   scale, mean, disable_residual):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    H = args.output_length
    # ----- choose target concept -----
    layer = args.target_layer
    if args.target_concept:
        idxs = [i for i, c in enumerate(metas[layer]) if c.get("name") == args.target_concept]
        if not idxs:
            raise ValueError(f"target_concept {args.target_concept!r} not in layer {layer}.")
        tgt_idx = idxs[0]
    else:
        fc_eff = [e for e in effects if e["layer"] == layer]
        if not fc_eff:
            raise ValueError(f"no concepts in layer {layer} to auto-pick.")
        best = max(fc_eff, key=lambda e: np.abs(e["eff"]).mean())
        tgt_idx = best["idx"]
    tgt_name = metas[layer][tgt_idx].get("name", f"{layer}{tgt_idx}")

    # ----- choose contrasting windows (clean vs polluted) from a scan -----
    rng = np.random.RandomState(args.seed)
    cand = rng.choice(len(test_data), min(2000, len(test_data)), replace=False)
    gt_mean = []
    for i in cand:
        _, y = test_data[int(i)]
        gt_mean.append(float(y[:, 0].mean()) * scale + mean)      # raw µg/m³
    gt_mean = np.array(gt_mean)
    clean_w   = int(cand[int(np.argmin(gt_mean))])
    polluted_w = int(cand[int(np.argmax(gt_mean))])

    sweep = [float(s) for s in args.sweep.split(",") if s.strip()]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), squeeze=False)
    axes = axes[0]
    for ax, (w, tag) in zip(axes, [(clean_w, "clean"), (polluted_w, "polluted")]):
        x, y = test_data[w]
        x = x.unsqueeze(0).to(device)
        gt = y[:, 0].numpy() * scale + mean
        ax.plot(np.arange(H), gt, color="k", lw=2.6, label="ground truth", zorder=5)
        cmap = plt.get_cmap("viridis")
        for si, sv in enumerate(sweep):
            yh = model(x, disable_residual=disable_residual,
                       concept_override={layer: {tgt_idx: sv}})["y_hat"][0, :, 0]
            yh = yh.cpu().numpy() * scale + mean
            ax.plot(np.arange(H), yh, color=cmap(si / max(1, len(sweep) - 1)),
                    lw=1.6, label=f"p={sv:g}")
        ax.set_title(f"{tag} window {w}  (GT mean {gt.mean():.0f} µg/m³)\n"
                     f"sweep concept [{layer}] {tgt_name}")
        ax.set_xlabel("forecast hour"); ax.set_ylabel("PM2.5 (µg/m³)")
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "counterfactual_sweep.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5] target concept = [{layer}] {tgt_name}; windows clean={clean_w}, polluted={polluted_w}")


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir = args.out_dir or os.path.join(args.checkpoint_dir, "concept_analysis")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Using device: {device}\nOutput -> {args.out_dir}")
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("\n── Building model + loading checkpoint ──")
    model, test_data, metas = build(args, device)
    scale = float(test_data.ans_type.scale_[0])
    mean  = float(test_data.ans_type.mean_[0])
    dr = bool(args.disable_residual)

    print(f"\n── Sampling {args.n_eval_windows} test windows ──")
    _, xs, _ = _eval_windows(test_data, args.n_eval_windows, args.seed)

    print("── (2) Causal effect curves ──")
    effects = causal_effects(model, xs, device, metas, scale, dr, args.eval_chunk)
    E, names = plot_effects(effects, args.out_dir, args.top_k_curves)

    print("── (3) Causal-effect similarity heatmap ──")
    plot_similarity(E, names, effects, args.out_dir)

    print("── (5) Counterfactual intervention (contrasting pair + sweep) ──")
    counterfactual(model, test_data, device, metas, effects, args, scale, mean, dr)

    print(f"\nDone. Figures in {args.out_dir}:")
    for f in ("causal_effect_curves.png", "causal_effect_heatmap.png",
              "concept_importance.png",
              "effect_similarity_signed_heatmap.png",  # SIGNED (semantic): same-direction concepts cluster
              "effect_similarity_shape_heatmap.png",   # SHAPE only (Pearson)
              "counterfactual_sweep.png"):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
