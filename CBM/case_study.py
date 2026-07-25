"""
Case-study figures + tables for a TRAINED Concept Bottleneck Model checkpoint.
Standalone (does not touch training): train as usual, then point this at the
saved best checkpoint (cbm.pth).

Produces the four paper "Case study" pieces:

  #1 SINGLE-WINDOW WALKTHROUGH  (best / lowest-MAE windows)
     For each chosen window: PM2.5 lookback + forecast (ground truth vs full
     model) plus a few key covariates, and the top-K activated concepts in each
     of the three bottlenecks (lookback / per-sub-window / global-forecast).
     -> case1_best_w<W>.png

  #2 CONCEPT PROBABILITY SWEEP  (counterfactual "dial the concept")
     Pick one concept and one window; sweep its forced activation p in
     {0..1} (whole-horizon) and overlay the resulting forecasts + ground truth.
     Also a small table: flip each top-activated concept -> ΔMAE, Δmean.
     -> case2_sweep.png (+ case2_flip_table.csv)

  #3 CONCEPT FIDELITY  (precision@K: do top-activation windows satisfy the rule?)
     For every concept, take the K windows with the highest activation and
     report the fraction whose GROUND-TRUTH label (the LLM rule) is True.
     Needs --test_segments_path.
     -> case3_fidelity.csv, case3_fidelity_bar.png (+ optional feature panels)

  #4 FAILURE ANALYSIS  (worst / highest-MAE windows)
     Same walkthrough figure as #1 but for the worst forecasts, so you can see
     which concepts did/didn't fire when the model was wrong.
     -> case4_worst_w<W>.png

RECOMMENDED: run on a model trained with --static_concept_emb --residual_dim 0
(pure static CBM) so the concepts carry the forecast and the story is clean.
If the checkpoint is a hybrid, pass --disable_residual to zero the residual.

The architecture args MUST match the training run (same concepts, segments,
pos-rate band, caps, and widths) so the checkpoint loads.

Run
---
    PYTHONPATH=. python -u CBM/case_study.py \
        --dataset ./dataset/PRSA_data/Aotizhongxin.csv \
        --concepts_path ./CBM/output/concepts_validated_<TS>.json \
        --segments_path ./preprocessing/output/Aotizhongxin/Aotizhongxin_full_segments.json \
        --test_segments_path ./preprocessing/output/Aotizhongxin/Aotizhongxin_full_segments_test.json \
        --checkpoint_dir ./checkpoint/cbm \
        --input_length 96 --output_length 96 --input_dim 15 --output_dim 1 \
        --num_layer 4 --dropout 0.5 --embed_dim 128 \
        --n_subwindows 8 --concept_emb_dim 16 --static_concept_emb --residual_dim 0 \
        --min_pos_rate 0.10 --max_pos_rate 0.65 \
        --features TEMP,WSPM,PM10 --out_dir ./checkpoint/cbm/case_study
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from CBM.concept_bottleneck import ConceptBottleneckModel
from CBM.concept_labels import build_window_label_tables
from dataloader.dataloader import LOF_Dataset


# PRSA Aotizhongxin column layout of the (T, C) window tensor.
FEATCOL = {"year": 0, "month": 1, "day": 2, "hour": 3, "PM10": 4, "SO2": 5,
           "NO2": 6, "CO": 7, "O3": 8, "TEMP": 9, "PRES": 10, "DEWP": 11,
           "RAIN": 12, "WSPM": 13, "PM2.5": 14}
PM25_COL = 14

# Colors (colorblind-friendly, print-safe).
C_GT   = "#2ca02c"   # ground truth (green)
C_FULL = "#d62728"   # full model (red)
C_COV  = "#1f77b4"   # covariate (blue)
C_LB   = "#0f766e"   # lookback concept header (teal)
C_FC   = "#b45309"   # forecast-sw concept header (amber)
C_GFC  = "#6d28d9"   # global-forecast concept header (violet)

# Compact axis labels for the thin covariate strips in #1/#4.
FRIENDLY = {"TEMP": "Temp\n(°C)", "PRES": "Press\n(hPa)", "DEWP": "Dew pt\n(°C)",
            "RAIN": "Precip\n(mm)", "WSPM": "Wind\n(m/s)", "PM10": "PM10", "O3": "O3",
            "NO2": "NO2", "SO2": "SO2", "CO": "CO"}
# Keyword groups used to auto-pick "important" concepts to sweep in #2.
SWEEP_GROUPS = {
    "temperature": ["temp", "hot", "cold", "warming", "cooling", "heat", "winter", "summer"],
    "humidity":    ["humid", "dewp", "dew", "rain"],
    "pollutant":   ["pm10", "pm25", "pm2", "_co", "co_", "no2", "so2", "o3", "ozone",
                    "pollut", "carryover"],
    "wind":        ["wind", "calm", "stagn", "windy"],
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _season(m):
    return {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
            5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn",
            10: "Autumn", 11: "Autumn"}.get(int(m), "?")


def _timestamps(x, base_year):
    yr = x[:, 0].round().long().tolist(); mo = x[:, 1].round().long().tolist()
    dy = x[:, 2].round().long().tolist(); hr = x[:, 3].round().long().tolist()
    out = []
    for y, m, d, h in zip(yr, mo, dy, hr):
        try:
            out.append(datetime(base_year + int(y), int(m), int(d), int(h)))
        except Exception:
            out.append((out[-1] + timedelta(hours=1)) if out else datetime(base_year, 1, 1))
    return out


def _fmt(ts):
    s = ts.strftime("%Y-%m-%d %I:%M%p").replace("AM", "am").replace("PM", "pm")
    d, t = s.split(" ", 1); hh, rest = t.split(":", 1)
    return f"{d} {int(hh)}:{rest}"


def _top(acts_1d, meta, k):
    order = np.argsort(-acts_1d)[:k]
    return [(float(acts_1d[j]), meta[j].get("name", f"c{j}")) for j in order]


# ---------------------------------------------------------------------------
# build model + data + labels
# ---------------------------------------------------------------------------
def build(args, device):
    train_data = LOF_Dataset(args.input_length, args.output_length, "train", args.input_dim, args.dataset)
    test_data  = LOF_Dataset(args.input_length, args.output_length, "test",  args.input_dim, args.dataset)

    (lb, fc, gfc, lb_sw, lb_meta, fc_meta, gfc_meta, lb_sw_meta) = build_window_label_tables(
        concepts_path=args.concepts_path, segments_path=args.segments_path,
        n_windows=len(train_data), only_ok=True,
        single_forecast_bottleneck=args.single_forecast_bottleneck,
        min_pos_rate=args.min_pos_rate, max_pos_rate=args.max_pos_rate,
        cap_lb=args.cap_lb_concepts, cap_fc=args.cap_fc_concepts,
        cap_gfc=args.cap_gfc_concepts, cap_lb_sw=args.cap_lb_sw_concepts,
    )
    K_lb, K_gfc = lb.shape[1], gfc.shape[1]
    K_lb_sw = lb_sw.shape[2] if lb_sw.ndim == 3 else 0
    K_lb_sw_windows = lb_sw.shape[1] if lb_sw.ndim == 3 else 0
    if args.single_forecast_bottleneck:
        K_fc = fc.shape[1]
    else:
        K_sw = fc.shape[1] if fc.ndim == 3 else 0
        K_fc = fc.shape[2] if fc.ndim == 3 else 0
        if K_sw and K_sw != args.n_subwindows:
            args.n_subwindows = K_sw
    print(f"  K_lookback={K_lb}, K_forecast={K_fc}, K_global={K_gfc}")

    model = ConceptBottleneckModel(
        input_dim=args.input_dim, mid_dim=args.embed_dim,
        ext_dim=args.input_dim - args.output_dim, output_dim=args.output_dim,
        bottle_dim_p=args.bottle_dim_p, input_length=args.input_length,
        output_length=args.output_length, num_blocks=args.num_layer, dropout=args.dropout,
        n_lookback_concepts=K_lb, n_forecast_concepts=K_fc,
        n_global_forecast_concepts=K_gfc, n_subwindows=args.n_subwindows,
        experiment_mode=args.experiment_mode,
        n_lookback_subwindows=K_lb_sw_windows if K_lb_sw_windows > 0 else args.n_subwindows,
        n_lookback_subwindow_concepts=K_lb_sw,
        concept_emb_dim=args.concept_emb_dim, static_concept_emb=args.static_concept_emb,
        residual_dim=args.residual_dim, decoder_hidden=args.decoder_hidden,
        decoder_depth=args.decoder_depth, decoder_n_blocks=args.decoder_n_blocks,
        decoder_feedback=bool(args.decoder_feedback), sub_window_id_dim=args.sw_id_dim,
        sw_gru_hidden=args.sw_gru_hidden, sw_summary_dim=args.sw_summary_dim,
        single_forecast_bottleneck=args.single_forecast_bottleneck,
        use_rev_in=1, norm_type="B", attn_pool_hidden=args.attn_pool_hidden,
        randint_p=args.randint_p,
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.checkpoint_name),
                                     map_location=device))
    model.eval()

    # Test labels (for #3 fidelity) if test segments provided.
    labels = None
    if args.test_segments_path:
        lb_t, fc_t, gfc_t, lb_sw_t, *_ = build_window_label_tables(
            concepts_path=args.concepts_path, segments_path=args.test_segments_path,
            n_windows=len(test_data), only_ok=True,
            single_forecast_bottleneck=args.single_forecast_bottleneck,
            min_pos_rate=args.min_pos_rate, max_pos_rate=args.max_pos_rate,
            cap_lb=args.cap_lb_concepts, cap_fc=args.cap_fc_concepts,
            cap_gfc=args.cap_gfc_concepts, cap_lb_sw=args.cap_lb_sw_concepts,
        )
        labels = {"lb": lb_t, "fc": fc_t, "gfc": gfc_t}
    metas = {"lb": lb_meta, "fc": fc_meta, "gfc": gfc_meta}
    return model, test_data, metas, labels


@torch.no_grad()
def collect(model, test_data, device, disable_residual, chunk=128):
    """One pass over the test set. Returns z-scored forecasts/GT + activations."""
    N = len(test_data)
    yh, gt, lb, fc, gfc = [], [], [], [], []
    for s in range(0, N, chunk):
        idx = list(range(s, min(s + chunk, N)))
        xb = torch.stack([test_data[i][0] for i in idx]).to(device)
        yb = torch.stack([test_data[i][1] for i in idx])
        out = model(xb, disable_residual=disable_residual)
        yh.append(out["y_hat"][:, :, 0].cpu()); gt.append(yb[:, :, 0])
        lb.append(out["lb_acts"].cpu())
        fc.append(out["fc_acts"].cpu() if out["fc_acts"].numel() > 0
                  else torch.zeros(len(idx), 0, 0))
        gfc.append(out["gfc_acts"].cpu() if out["gfc_acts"].numel() > 0
                   else torch.zeros(len(idx), 0))
    return (torch.cat(yh).numpy(), torch.cat(gt).numpy(),
            torch.cat(lb).numpy(),
            (torch.cat(fc).numpy() if fc[0].ndim == 3 else np.zeros((N, 0, 0))),
            torch.cat(gfc).numpy())


# ---------------------------------------------------------------------------
# #1 / #4  single-window walkthrough figure
# ---------------------------------------------------------------------------
def plot_window(w, test_data, yh, gt, lb_acts, fc_acts_mean, gfc_acts, metas,
                scale, mean, args, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    x, _ = test_data[w]
    x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    L, H = args.input_length, args.output_length
    ts = _timestamps(x, args.base_year)
    season = _season(int(x[L, 1].item()))
    mae = float(np.abs(gt[w] - yh[w]).mean()) * scale

    # ALL requested covariates as THIN strips (readable trend/shape).
    feats = [f for f in args.features.split(",") if f.strip() in FEATCOL]
    nrow = 1 + len(feats)

    fig = plt.figure(figsize=(14, 3.0 + 0.95 * len(feats)))
    gs = fig.add_gridspec(nrow, 2, width_ratios=[2.4, 1.0],
                          height_ratios=[2.4] + [0.5] * len(feats),
                          hspace=0.30, wspace=0.16)

    t_full = np.arange(L + H)
    t_fc = np.arange(L, L + H)

    # ---- PM2.5 main panel (tall) ----
    ax = fig.add_subplot(gs[0, 0])
    pm_lb = x[:L, PM25_COL].numpy() * scale + mean
    gt_r = gt[w] * scale + mean
    yh_r = yh[w] * scale + mean
    ax.plot(np.arange(L), pm_lb, color="#555555", lw=1.7, label="PM2.5 lookback")
    ax.plot(t_fc, gt_r, color=C_GT, lw=2.5, label="ground truth")
    ax.plot(t_fc, yh_r, color=C_FULL, lw=2.0, ls="--", label="model forecast")
    ax.axvline(L, color="k", lw=1.0, ls=":", alpha=0.7)
    ax.set_ylabel("PM2.5 (µg/m³)", fontsize=10)
    ax.set_title("PM2.5 (target)", fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.margins(x=0.01); ax.grid(alpha=0.25); ax.tick_params(labelbottom=False)

    # ---- thin covariate strips (all in one figure) ----
    for i, f in enumerate(feats):
        axc = fig.add_subplot(gs[i + 1, 0], sharex=ax)
        axc.plot(t_full, x[:, FEATCOL[f]].numpy(), color=C_COV, lw=1.15)
        axc.axvline(L, color="k", lw=0.9, ls=":", alpha=0.7)
        axc.set_ylabel(FRIENDLY.get(f, f), fontsize=8, rotation=0, ha="right", va="center", labelpad=26)
        axc.yaxis.set_major_locator(MaxNLocator(3)); axc.tick_params(labelsize=7)
        axc.margins(x=0.01); axc.grid(alpha=0.18)
        if i == len(feats) - 1:
            axc.set_xlabel("hours from window start   (dotted line = forecast start)", fontsize=9)
        else:
            axc.tick_params(labelbottom=False)

    # ---- right column: period box + top-K concepts per bottleneck ----
    axt = fig.add_subplot(gs[:, 1]); axt.axis("off")
    period = (f"Forecast window\n {_fmt(ts[L])}\n   →  {_fmt(ts[-1])}\n"
              f"Lookback start\n {_fmt(ts[0])}\nSeason:  {season}")
    axt.text(0.0, 0.998, period, fontsize=8.5, family="monospace", va="top",
             transform=axt.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f4f4f4", ec="#bbbbbb"))
    k = args.top_concepts
    blocks = [
        ("LOOKBACK concepts (history)", C_LB, _top(lb_acts[w], metas["lb"], k)),
        (f"FORECAST sub-window concepts (mean over {args.n_subwindows})", C_FC,
         _top(fc_acts_mean[w], metas["fc"], k) if fc_acts_mean.shape[1] else []),
        ("GLOBAL-FORECAST concepts (whole horizon)", C_GFC,
         _top(gfc_acts[w], metas["gfc"], k) if gfc_acts.shape[1] else []),
    ]
    y0 = 0.80
    for title, col, rows in blocks:
        axt.text(0.0, y0, title, fontsize=9.5, fontweight="bold", color=col,
                 transform=axt.transAxes, va="top")
        y0 -= 0.038
        for p, name in rows:
            axt.text(0.02, y0, f"{p:4.2f}", fontsize=8.5, family="monospace",
                     color="#111", transform=axt.transAxes, va="top")
            axt.text(0.16, y0, name, fontsize=8, transform=axt.transAxes, va="top")
            y0 -= 0.029
        y0 -= 0.024

    ttl_col = C_FULL if tag == "worst" else "#111"
    fig.suptitle(f"{'FAILURE CASE' if tag == 'worst' else 'CASE STUDY'}"
                 f"   ·   {season}   ·   forecast MAE = {mae:.1f} µg/m³",
                 fontsize=13, fontweight="bold", color=ttl_col, y=0.995)
    fig.savefig(os.path.join(args.out_dir, f"case{'4_worst' if tag == 'worst' else '1_best'}_w{w}.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{tag}] window {w}: MAE {mae:.1f} µg/m³ ({season})")


# ---------------------------------------------------------------------------
# #2  concept probability sweeps (several important concepts) + flip table
# ---------------------------------------------------------------------------
def _select_nonoverlap(order, n, gap):
    """Pick n windows from `order` keeping index gaps >= `gap` (so the chosen
    cases don't overlap in time / aren't near-duplicate slices)."""
    picked = []
    for w in order:
        w = int(w)
        if all(abs(w - p) >= gap for p in picked):
            picked.append(w)
            if len(picked) >= n:
                break
    return picked


def _pick_sweep_targets(metas, act_win, n, override):
    """Return up to n (layer, idx, name) concepts to sweep. Either an explicit
    --sweep_concepts list ('layer:name' or 'name'), or one representative
    (highest mean activation) per SWEEP_GROUPS keyword group (temperature /
    humidity / pollutant / wind)."""
    if override.strip():
        out = []
        for tok in override.split(","):
            tok = tok.strip()
            if not tok:
                continue
            layer, name = (tok.split(":", 1) if ":" in tok else ("fc", tok))
            hits = [i for i, c in enumerate(metas.get(layer, [])) if c.get("name") == name]
            if hits:
                out.append((layer, hits[0], name))
            else:
                print(f"  [2] warn: sweep concept {tok!r} not found; skipping.")
        return out[:n]
    picked, used = [], set()
    for _grp, kws in SWEEP_GROUPS.items():
        best, bestval = None, -1.0
        for layer in ("fc", "gfc", "lb"):
            meta, a = metas[layer], act_win[layer]
            if a.shape[1] == 0:
                continue
            for i, c in enumerate(meta):
                if (layer, i) in used:
                    continue
                nm = c.get("name", "").lower()
                if any(kw in nm for kw in kws):
                    val = float(a[:, i].mean())
                    if val > bestval:
                        bestval, best = val, (layer, i, c.get("name", f"{layer}{i}"))
        if best:
            picked.append(best); used.add((best[0], best[1]))
        if len(picked) >= n:
            break
    return picked[:n]


@torch.no_grad()
def concept_sweep(model, test_data, device, metas, lb_acts, fc_mean, gfc_acts, gt,
                  scale, mean, args, disable_residual):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H = args.output_length
    act_win = {"lb": lb_acts, "fc": fc_mean, "gfc": gfc_acts}
    targets = _pick_sweep_targets(metas, act_win, args.n_sweeps, args.sweep_concepts)
    if not targets:
        print("  [2] no sweep targets found (check --sweep_concepts / concept names).")
        return
    sweep = [float(s) for s in args.sweep_vals.split(",") if s.strip()]
    cmap = plt.get_cmap("viridis")

    # Same clean vs polluted anchor windows for every concept -> comparable panels.
    lvl = (gt * scale + mean).mean(axis=1)
    clean_w, poll_w = int(np.argmin(lvl)), int(np.argmax(lvl))
    wins = [(clean_w, "clean"), (poll_w, "polluted")]

    for (layer, ti, tname) in targets:
        fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), squeeze=False)
        for ax, (w, tt) in zip(axes[0], wins):
            x, y = test_data[w]
            gt_r = y[:, 0].numpy() * scale + mean
            ax.plot(np.arange(H), gt_r, color="k", lw=2.6, label="ground truth", zorder=6)
            for si, sv in enumerate(sweep):
                yh = model(x.unsqueeze(0).to(device), disable_residual=disable_residual,
                           concept_override={layer: {ti: sv}})["y_hat"][0, :, 0]
                yh = yh.cpu().numpy() * scale + mean
                ax.plot(np.arange(H), yh, color=cmap(si / max(1, len(sweep) - 1)),
                        lw=1.7, label=f"p={sv:g}")
            ax.set_title(f"{tt} window  (GT mean {gt_r.mean():.0f} µg/m³)", fontsize=10)
            ax.set_xlabel("forecast hour"); ax.set_ylabel("PM2.5 (µg/m³)")
            ax.legend(fontsize=8, loc="best"); ax.grid(alpha=0.25)
        fig.suptitle(f"Counterfactual concept sweep — [{layer}] {tname}\n"
                     "(force the concept's activation 0 → 1 over the whole horizon)",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()
        safe = str(tname).replace("/", "_")
        fig.savefig(os.path.join(args.out_dir, f"case2_sweep_{layer}_{safe}.png"),
                    dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  [2] swept [{layer}] {tname}  -> case2_sweep_{layer}_{safe}.png")

    # ---- flip table: on the polluted window, flip its top-3 forecast concepts to 0 ----
    x0, y0 = test_data[poll_w]
    gt0 = y0[:, 0].numpy() * scale + mean
    base = model(x0.unsqueeze(0).to(device), disable_residual=disable_residual)["y_hat"][0, :, 0].cpu().numpy() * scale + mean
    base_mae = float(np.abs(gt0 - base).mean())
    top_fc = np.argsort(-fc_mean[poll_w])[:3] if fc_mean.shape[1] else []
    with open(os.path.join(args.out_dir, "case2_flip_table.csv"), "w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        wr.writerow(["window", "concept", "base_activation", "base_MAE",
                     "flipped_MAE(->0)", "delta_mean_forecast_ugm3"])
        for j in top_fc:
            yh = model(x0.unsqueeze(0).to(device), disable_residual=disable_residual,
                       concept_override={"fc": {int(j): 0.0}})["y_hat"][0, :, 0].cpu().numpy() * scale + mean
            wr.writerow([poll_w, metas["fc"][int(j)].get("name", f"fc{j}"),
                         f"{fc_mean[poll_w][int(j)]:.3f}", f"{base_mae:.2f}",
                         f"{float(np.abs(gt0 - yh).mean()):.2f}",
                         f"{float((yh - base).mean()):+.2f}"])
    print("  [2] flip table -> case2_flip_table.csv")


# ---------------------------------------------------------------------------
# #3  concept fidelity (precision@K)
# ---------------------------------------------------------------------------
def fidelity(lb_acts, fc_acts_full, gfc_acts, labels, metas, args, test_data, fc_mean):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = args.fidelity_topk
    rows = []  # (layer, name, pos_rate, precision@K)

    def _pk(acts_2d, labs_2d, meta, layer):
        # acts_2d, labs_2d: (M, C). For each concept, top-K by activation -> mean label.
        for i in range(acts_2d.shape[1]):
            a = acts_2d[:, i]; lab = labs_2d[:, i]
            top = np.argsort(-a)[:K]
            prec = float(lab[top].mean()) if len(top) else float("nan")
            rows.append((layer, meta[i].get("name", f"{layer}{i}"),
                         float(lab.mean()), prec, i))

    _pk(lb_acts, labels["lb"], metas["lb"], "lb")
    _pk(gfc_acts, labels["gfc"], metas["gfc"], "gfc")
    # forecast concepts: evaluate over (window, sub-window) pairs.
    if fc_acts_full.ndim == 3 and fc_acts_full.shape[2]:
        fa = fc_acts_full.reshape(-1, fc_acts_full.shape[2])
        fl = labels["fc"].reshape(-1, labels["fc"].shape[2])
        _pk(fa, fl, metas["fc"], "fc")

    with open(os.path.join(args.out_dir, "case3_fidelity.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["bottleneck", "concept", "positive_rate", f"precision@{K}"])
        for r in sorted(rows, key=lambda r: -r[3]):
            wr.writerow([r[0], r[1], f"{r[2]:.3f}", f"{r[3]:.3f}"])

    # bar chart
    rows_s = sorted(rows, key=lambda r: r[3])
    fig, ax = plt.subplots(figsize=(9, max(4, 0.24 * len(rows_s))))
    colmap = {"lb": C_LB, "fc": C_FC, "gfc": C_GFC}
    ax.barh(range(len(rows_s)), [r[3] for r in rows_s],
            color=[colmap[r[0]] for r in rows_s])
    ax.set_yticks(range(len(rows_s)))
    ax.set_yticklabels([f"[{r[0]}] {r[1]}" for r in rows_s], fontsize=6)
    ax.axvline(0.8, color="k", ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel(f"precision@{K}  (fraction of top-{K} activation windows where the LLM rule holds)")
    ax.set_xlim(0, 1)
    ax.set_title("Concept fidelity: do the highest-activation windows satisfy the concept's rule?")
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "case3_fidelity_bar.png"),
                                    dpi=140, bbox_inches="tight"); plt.close(fig)
    # ---- visual check: do the top-activation windows show the condition? ----
    act_win = {"lb": lb_acts, "fc": fc_mean, "gfc": gfc_acts}
    panel = sorted(rows, key=lambda r: -(r[3] - r[2]))[:max(0, args.fidelity_panels)]
    panel = [r for r in panel if act_win[r[0]].shape[1] > 0]
    if panel:
        feats = ["WSPM", "TEMP", "PRES", "DEWP", "PM10", "hour"]
        L = args.input_length
        N = act_win["lb"].shape[0]
        ref = np.arange(min(1000, N))

        def _feat_means(win_idx, use_lookback):
            vals = []
            for fn in feats:
                col = FEATCOL[fn]
                sl = slice(0, L) if use_lookback else slice(L, None)
                vals.append(np.mean([test_data[int(w)][0][sl, col].mean().item() for w in win_idx]))
            return np.array(vals)

        fig, axes = plt.subplots(1, len(panel), figsize=(4.2 * len(panel), 4.4), squeeze=False)
        for ax, r in zip(axes[0], panel):
            layer, name, i = r[0], r[1], r[4]
            top = np.argsort(-act_win[layer][:, i])[:K]
            use_lb = (layer == "lb")
            tm, gm = _feat_means(top, use_lb), _feat_means(ref, use_lb)
            yv = np.arange(len(feats)); wd = 0.4
            ax.barh(yv - wd / 2, gm, wd, color="#bbbbbb", label="global mean")
            ax.barh(yv + wd / 2, tm, wd, color=C_FC, label=f"top-{K} windows")
            ax.set_yticks(yv); ax.set_yticklabels(feats, fontsize=8); ax.invert_yaxis()
            ax.set_title(f"[{layer}] {name}\nprec@{K}={r[3]:.2f}", fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle("Concept fidelity (visual): raw features of each concept's top-activation windows\n"
                     "(lookback slice for lookback concepts, forecast slice otherwise)",
                     fontsize=12)
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "case3_fidelity_features.png"),
                                        dpi=140, bbox_inches="tight"); plt.close(fig)

    hi = sum(1 for r in rows if r[3] >= 0.8)
    print(f"  [3] fidelity: {hi}/{len(rows)} concepts have precision@{K} ≥ 0.80 "
          f"-> case3_fidelity.csv / _bar.png / _features.png")


# ---------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--concepts_path", type=str, required=True)
    p.add_argument("--segments_path", type=str, required=True)
    p.add_argument("--test_segments_path", type=str, default="",
                   help="Needed for #3 fidelity (real test labels).")
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--checkpoint_name", type=str, default="cbm.pth")
    p.add_argument("--out_dir", type=str, default=None)
    # architecture (MUST match training)
    p.add_argument("--input_length", type=int, default=96)
    p.add_argument("--output_length", type=int, default=96)
    p.add_argument("--input_dim", type=int, default=15)
    p.add_argument("--output_dim", type=int, default=1)
    p.add_argument("--num_layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--bottle_dim_p", type=float, default=0.1)
    p.add_argument("--n_subwindows", type=int, default=8)
    p.add_argument("--concept_emb_dim", type=int, default=16)
    p.add_argument("--static_concept_emb", action="store_true")
    p.add_argument("--residual_dim", type=int, default=0)
    p.add_argument("--decoder_hidden", type=int, default=512)
    p.add_argument("--decoder_depth", type=int, default=3)
    p.add_argument("--decoder_n_blocks", type=int, default=3)
    p.add_argument("--decoder_feedback", type=int, default=1)
    p.add_argument("--sw_id_dim", type=int, default=16)
    p.add_argument("--sw_gru_hidden", type=int, default=32)
    p.add_argument("--sw_summary_dim", type=int, default=64)
    p.add_argument("--attn_pool_hidden", type=int, default=32)
    p.add_argument("--randint_p", type=float, default=0.25)
    p.add_argument("--experiment_mode", type=str, default="future_aware",
                   choices=["future_aware", "lookback_only"])
    p.add_argument("--single_forecast_bottleneck", action="store_true")
    p.add_argument("--min_pos_rate", type=float, default=0.10)
    p.add_argument("--max_pos_rate", type=float, default=0.65)
    p.add_argument("--cap_lb_concepts", type=int, default=0)
    p.add_argument("--cap_fc_concepts", type=int, default=0)
    p.add_argument("--cap_gfc_concepts", type=int, default=0)
    p.add_argument("--cap_lb_sw_concepts", type=int, default=0)
    # case-study knobs
    p.add_argument("--disable_residual", action="store_true")
    p.add_argument("--features", type=str, default="TEMP,PRES,DEWP,RAIN,WSPM",
                   help="Covariates shown as THIN strips in #1/#4 (besides the "
                        "tall PM2.5 panel). Default: the 5 continuous drivers.")
    p.add_argument("--top_concepts", type=int, default=7)
    p.add_argument("--n_case1", type=int, default=3, help="# best-MAE windows for #1.")
    p.add_argument("--n_case4", type=int, default=3, help="# worst-MAE windows for #4.")
    p.add_argument("--min_window_gap", type=int, default=0,
                   help="Min index gap between chosen case windows so they don't "
                        "overlap in time. 0 => auto (input_length + output_length, "
                        "i.e. fully disjoint windows).")
    p.add_argument("--n_sweeps", type=int, default=4,
                   help="# of concept-sweep figures in #2 (one per important concept).")
    p.add_argument("--sweep_concepts", type=str, default="",
                   help="Comma list of concepts to sweep, 'layer:name' or 'name' "
                        "(layer defaults fc), e.g. 'fc:is_warming_block,gfc:is_forecast_very_humid'. "
                        "Empty => auto-pick one per keyword group (temperature/humidity/pollutant/wind).")
    p.add_argument("--sweep_vals", type=str, default="0,0.25,0.5,0.75,1.0")
    p.add_argument("--fidelity_topk", type=int, default=20)
    p.add_argument("--fidelity_panels", type=int, default=4,
                   help="# concepts (highest lift) to draw a raw-feature panel for in #3.")
    p.add_argument("--base_year", type=int, default=2013)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out_dir = args.out_dir or os.path.join(args.checkpoint_dir, "case_study")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    print(f"Using device: {device}\nOutput -> {args.out_dir}")

    print("\n── Building model + loading checkpoint ──")
    model, test_data, metas, labels = build(args, device)
    scale = float(test_data.ans_type.scale_[0]); mean = float(test_data.ans_type.mean_[0])
    dr = bool(args.disable_residual)

    print("── Collecting test forecasts + activations ──")
    yh, gt, lb_acts, fc_full, gfc_acts = collect(model, test_data, device, dr)
    fc_mean = fc_full.mean(axis=1) if fc_full.ndim == 3 and fc_full.shape[2] else np.zeros((len(gt), 0))
    per_mae = np.abs(gt - yh).mean(axis=1)                        # z-scored per-window MAE
    gap = args.min_window_gap if args.min_window_gap > 0 else (args.input_length + args.output_length)
    best = _select_nonoverlap(np.argsort(per_mae), args.n_case1, gap)
    worst = _select_nonoverlap(np.argsort(-per_mae), args.n_case4, gap)
    print(f"  non-overlapping (gap≥{gap}) best windows {best}; worst {worst}")

    print("── #1 Single-window walkthroughs (best) ──")
    for w in best:
        plot_window(int(w), test_data, yh, gt, lb_acts, fc_mean, gfc_acts, metas, scale, mean, args, "best")

    print("── #2 Concept probability sweeps (important concepts) ──")
    concept_sweep(model, test_data, device, metas, lb_acts, fc_mean, gfc_acts, gt, scale, mean, args, dr)

    if labels is not None:
        print("── #3 Concept fidelity (precision@K) ──")
        fidelity(lb_acts, fc_full, gfc_acts, labels, metas, args, test_data, fc_mean)
    else:
        print("── #3 skipped (pass --test_segments_path for fidelity) ──")

    print("── #4 Failure analysis (worst) ──")
    for w in worst:
        plot_window(int(w), test_data, yh, gt, lb_acts, fc_mean, gfc_acts, metas, scale, mean, args, "worst")

    print(f"\nDone. Figures/tables in {args.out_dir}")


if __name__ == "__main__":
    main()
