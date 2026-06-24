"""
Training entry point for the 1 + N Concept Bottleneck Model.

Loss
----

    loss = alpha_forecast        * L1(y_hat, y)
         + alpha_lookback        * BCE(lb_logits, lb_labels)
         + alpha_forecast_concept * BCE(fc_logits, fc_labels)

  - lb_logits has shape (B, K_lb); lb_labels matches.
  - fc_logits has shape (B, N, K_fc); fc_labels matches. BCEWithLogitsLoss
    averages elementwise so weighting by (1 / (N * K_fc)) is built in.

Concept labels are built by ``concept_labels.build_window_label_tables``
which loads the two-set concepts JSON, applies the rules to every Part-A
segment, and broadcasts to dataset windows.

Validation/test concept labels default to all zeros (the same caveat as
before; build val/test segment JSONs if you want real concept AUROC on
those splits). Forecast MAE on val/test is unaffected.

Run
---

    # PRSA / Beijing PM2.5 (currently active DATASET):
    PYTHONPATH=. python -u CBM/train_cbm.py \\
        --dataset            ./dataset/PRSA_data/Aotizhongxin.csv \\
        --concepts_path      ./CBM/output/concepts_validated_<TS>.json \\
        --segments_path      ./preprocessing/output/Aotizhongxin/Aotizhongxin_full_segments.json \\
        --val_segments_path  ./preprocessing/output/Aotizhongxin/Aotizhongxin_full_segments_vali.json \\
        --test_segments_path ./preprocessing/output/Aotizhongxin/Aotizhongxin_full_segments_test.json \\
        --epoch 30 --batch_size 16 \\
        --input_length 96 --output_length 24 \\
        --input_dim 15 --output_dim 1 \\
        --num_layer 4 --dropout 0.5 --embed_dim 128 \\
        --n_subwindows 4 --concept_emb_dim 16 \\
        --decoder_hidden 512 --decoder_depth 3 \\
        --residual_dim 16 --alpha_residual_reg 1e-3 \\
        --alpha_forecast 1.0 --alpha_lookback 1.0 --alpha_forecast_concept 1.0 \\
        --lr 3e-4 --weight_decay 1e-4 --patience 5 \\
        --case_studies 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parent.parent
_CBM_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p).resolve() != _CBM_DIR]
while str(_ROOT) in sys.path:
    sys.path.remove(str(_ROOT))
sys.path.insert(0, str(_ROOT))

from CBM.concept_bottleneck import ConceptBottleneckModel
from CBM.concept_labels import build_window_label_tables
from dataloader.dataloader import LOF_Dataset


# ---------------------------------------------------------------------------
# Dataset wrapper — adds per-window concept labels to (x, y) tuples
# ---------------------------------------------------------------------------

class CBMDataset(Dataset):
    def __init__(self, base: Dataset,
                 lb_labels: np.ndarray,
                 fc_labels: np.ndarray,
                 gfc_labels: np.ndarray):
        assert len(base) == len(lb_labels) == len(fc_labels) == len(gfc_labels), (
            f"len(base)={len(base)} vs lb={len(lb_labels)} vs "
            f"fc={len(fc_labels)} vs gfc={len(gfc_labels)}"
        )
        self.base       = base
        self.lb_labels  = torch.from_numpy(lb_labels).float()    # (N, K_lb)
        self.fc_labels  = torch.from_numpy(fc_labels).float()    # (N, K_sw, K_fc)
        self.gfc_labels = torch.from_numpy(gfc_labels).float()   # (N, K_gfc); K_gfc may be 0

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, self.lb_labels[idx], self.fc_labels[idx], self.gfc_labels[idx]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",        type=str, required=True)
    p.add_argument("--concepts_path",  type=str, required=True,
                   help="Two-set concepts JSON from CBM/concept_proposer/propose_concepts.py.")
    p.add_argument("--segments_path",  type=str, required=True,
                   help="Train segments JSON from preprocessing --split train.")
    p.add_argument("--val_segments_path",  type=str, default=None,
                   help="Optional. Validation segments JSON from preprocessing "
                        "--split vali. If provided, val concept labels are "
                        "real (not zeros) and val AUROC + accuracy are reported.")
    p.add_argument("--test_segments_path", type=str, default=None,
                   help="Optional. Test segments JSON from preprocessing "
                        "--split test. If provided, test concept labels are "
                        "real and a test-set per-concept accuracy + AUROC "
                        "table is printed at the end.")
    p.add_argument("--epoch",          type=int, required=True)
    p.add_argument("--batch_size",     type=int, default=16)
    p.add_argument("--input_length",   type=int, default=96 * 4)
    p.add_argument("--output_length",  type=int, default=96)
    p.add_argument("--input_dim",      type=int, default=23)
    p.add_argument("--output_dim",     type=int, default=1)
    p.add_argument("--num_layer",      type=int, default=4)
    p.add_argument("--dropout",        type=float, default=0.5)
    p.add_argument("--embed_dim",      type=int, default=128)
    p.add_argument("--bottle_dim_p",   type=float, default=0.1)

    p.add_argument("--n_subwindows",   type=int, default=4,
                   help="Forecast-side bottlenecks (must equal preprocessing's --n_subwindows).")
    p.add_argument("--concept_emb_dim", type=int, default=16,
                   help="Per-concept embedding dimensionality d in the full-CEM "
                        "design. Each concept owns two MLPs producing input-"
                        "conditional vectors in R^d; the decoder reads K*d dims "
                        "instead of D dims. Default 16 -- effective bottleneck "
                        "rank K*d (much wider than the old static rank K).")
    p.add_argument("--decoder_hidden", type=int, default=512,
                   help="Hidden width of the shared sub-decoder MLP (default 512).")
    p.add_argument("--decoder_depth",  type=int, default=3,
                   help="Number of Linear layers in the shared sub-decoder. "
                        "1 = single-Linear projection; 2 = 1 hidden layer; "
                        "3 = 2 hidden layers (default).")
    p.add_argument("--sw_id_dim",      type=int, default=16,
                   help="Width of the sub-window-id embedding fed to the shared sub-decoder.")
    p.add_argument("--sw_gru_hidden",  type=int, default=32,
                   help="Hidden width of the bidirectional GRU that refines the "
                        "per-sub-window forecast concept contexts (sequence-aware "
                        "decoder). Output per sub-window = 2 * sw_gru_hidden dims "
                        "concatenated into the decoder input. Set 0 to disable "
                        "this enhancement (legacy behavior). Automatically off "
                        "when --single_forecast_bottleneck is set (degenerate).")
    p.add_argument("--sw_summary_dim", type=int, default=64,
                   help="Width of the Linear projection (K_fc*d -> this) applied "
                        "before the BiGRU. Only used when --sw_gru_hidden > 0.")
    p.add_argument("--residual_dim",   type=int, default=16,
                   help="Width of the regularized residual channel that bypasses "
                        "the concept layer (hybrid CBM). 0 to disable. The decoder "
                        "input becomes concat(lb_ctx, fc_ctx_k, sw_emb, residual_k) "
                        "with the residual constrained by --alpha_residual_reg. "
                        "At test time, concept-only MAE is computed with residual "
                        "zeroed so you can see how much the residual is helping.")
    p.add_argument("--single_forecast_bottleneck", action="store_true",
                   help="ABLATION: collapse the N per-sub-window forecast "
                        "concept bottlenecks into a single GLOBAL forecast "
                        "bottleneck. Concept labels are then computed on the "
                        "whole forecast window (shape (W, K_fc)) instead of "
                        "per sub-window (shape (W, N, K_fc)). Use to test "
                        "whether the per-sub-window structure actually helps: "
                        "if test MAE goes up here, the N-sub-window design is "
                        "doing real work; if it stays the same or improves, "
                        "the sub-window split is unnecessary on this dataset.")

    p.add_argument("--lr",             type=float, default=3e-4,
                   help="Initial learning rate. Lowered from 1e-3 to stabilize "
                        "training in the 1+N architecture (concept and forecast "
                        "gradients pull harder against each other now).")
    p.add_argument("--weight_decay",   type=float, default=1e-4,
                   help="L2 weight decay on the Adam optimizer. Default 1e-4 "
                        "to dampen decoder overfit.")
    p.add_argument("--lr_decay",       type=float, default=0.5)
    p.add_argument("--lr_decay_step",  type=int,   default=999,
                   help="Multiply lr by --lr_decay every this many epochs. "
                        "Default 999 effectively disables the schedule and "
                        "lets early stopping drive termination.")
    p.add_argument("--patience",       type=int,   default=5,
                   help="Early-stopping patience: stop if val MAE has not "
                        "improved for this many epochs. Default 5. Set to 0 "
                        "to disable early stopping (run all --epoch epochs).")
    p.add_argument("--seed",           type=int,   default=0)

    p.add_argument("--alpha_forecast",         type=float, default=1.0,
                   help="Weight on L1 forecast loss. Default 1.0 — equal to the "
                        "two BCE weights. With K_lb + N*K_fc concept outputs (110 "
                        "by default) vs 96 forecast outputs, equal weighting is "
                        "already slightly concept-dominant; lowering this further "
                        "tends to leave the decoder undersupervised.")
    p.add_argument("--alpha_lookback",         type=float, default=1.0)
    p.add_argument("--alpha_forecast_concept", type=float, default=1.0)
    p.add_argument("--alpha_forecast_global_concept", type=float, default=1.0,
                   help="Weight on the global-forecast BCE term. Used only when "
                        "the concepts JSON includes a non-empty "
                        "'forecast_global_concepts' set (K_fc_global > 0).")
    p.add_argument("--alpha_boundary_smooth", type=float, default=0.01,
                   help="L1 penalty weight on the difference between adjacent "
                        "forecast timesteps at sub-window BOUNDARIES (positions "
                        "sw_len, 2*sw_len, ..., (N-1)*sw_len). Discourages the "
                        "discontinuities that the piecewise per-sub-window "
                        "decoder otherwise produces. Set 0 to disable.")
    p.add_argument("--min_pos_rate", type=float, default=0.08,
                   help="Drop concepts whose positive_rate is BELOW this value. "
                        "0.08 (8%%) filters concepts that fire so rarely that BCE "
                        "supervision degenerates into 'always predict False'. "
                        "Set 0 to keep all concepts (legacy behavior).")
    p.add_argument("--max_pos_rate", type=float, default=0.75,
                   help="Drop concepts whose positive_rate is ABOVE this value. "
                        "0.75 (75%%) filters concepts that fire so universally "
                        "that BCE degenerates into 'always predict True' with no "
                        "discriminative signal. Set 1 to keep all concepts.")
    p.add_argument("--alpha_residual_reg",     type=float, default=1e-3,
                   help="L2 penalty on the residual channel's magnitude. "
                        "Higher = decoder must rely more on concepts. Lower = "
                        "residual is free to carry most of the forecast signal. "
                        "Default 1e-3 is a starting point; tune to slide along "
                        "the interpretability/accuracy curve.")
    p.add_argument("--concept_warmup_epochs",  type=int,   default=0,
                   help="If > 0, train with alpha_forecast = 0 for this many "
                        "epochs before switching to the user-supplied value. "
                        "Lets the concept predictors fit cleanly before forecast "
                        "pressure is applied. Recommended: 3-5 if you suspect "
                        "concept hijacking; 0 for the standard run.")

    p.add_argument("--checkpoint_dir", type=str, default="./checkpoint/cbm")
    p.add_argument("--case_studies",   type=int, default=5)
    p.add_argument("--interp_cases",   type=int, default=6,
                   help="Number of GOOD (low-MAE) test windows to render as "
                        "per-case interpretability figures (continuous-feature "
                        "plots + active concepts in all three bottlenecks). "
                        "0 disables. Figures saved to "
                        "<checkpoint_dir>/interp_case_<i>_w<window>.png.")
    p.add_argument("--interp_top_k",   type=int, default=5,
                   help="Max number of active concepts to list per bottleneck "
                        "in each interpretability figure.")
    p.add_argument("--interp_base_year", type=int, default=2013,
                   help="Calendar base year for the PRSA 0-based 'year' column "
                        "(0 -> 2013 for the Aotizhongxin station).")
    return p.parse_args()


def set_seed(seed: int = 0):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed); random.seed(seed)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def _boundary_smoothness_loss(y_hat: torch.Tensor, sub_window_length: int,
                              n_subwindows: int) -> torch.Tensor:
    """Mean L1 difference between adjacent forecast timesteps at the
    sub-window boundaries (positions sw_len, 2*sw_len, ..., (N-1)*sw_len).

    y_hat shape: (B, T_out, O). Returns a scalar tensor on y_hat.device.
    Zero when n_subwindows <= 1 (no boundaries).
    """
    if n_subwindows <= 1 or sub_window_length <= 0:
        return y_hat.new_zeros(())
    diffs = []
    for k in range(1, n_subwindows):
        bd = k * sub_window_length
        diffs.append((y_hat[:, bd, :] - y_hat[:, bd - 1, :]).abs().mean())
    return torch.stack(diffs).mean()


def train_one_epoch(model, loader, optimizer, device,
                    alpha_forecast, alpha_lookback, alpha_forecast_concept,
                    alpha_forecast_global_concept, alpha_residual_reg,
                    alpha_boundary_smooth):
    model.train()
    cum_f, cum_lb, cum_fc, cum_gfc, cum_res, cum_sm, n = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,
    )
    l1_fn  = nn.L1Loss()
    bce_fn = nn.BCEWithLogitsLoss()
    sw_len = int(getattr(model, "sub_window_length", 0))
    n_sw   = int(getattr(model, "n_subwindows", 1))

    for x, y, lb_lab, fc_lab, gfc_lab in loader:
        x       = x.to(device)
        y       = y.to(device)
        lb_lab  = lb_lab.to(device)
        fc_lab  = fc_lab.to(device)
        gfc_lab = gfc_lab.to(device)
        out = model(x)

        loss_f   = l1_fn(out["y_hat"],     y)
        loss_lb  = bce_fn(out["lb_logits"], lb_lab)
        loss_fc  = bce_fn(out["fc_logits"], fc_lab)
        if out["gfc_logits"].numel() > 0:
            loss_gfc = bce_fn(out["gfc_logits"], gfc_lab)
        else:
            loss_gfc = torch.zeros((), device=device)
        if out["residual"].numel() > 0:
            loss_res = out["residual"].pow(2).mean()
        else:
            loss_res = torch.zeros((), device=device)
        # Boundary smoothness: small L1 penalty on |y_hat[bd] - y_hat[bd-1]|
        # at each sub-window seam. No effect when n_subwindows == 1.
        loss_sm = _boundary_smoothness_loss(out["y_hat"], sw_len, n_sw)

        loss = (alpha_forecast               * loss_f
                + alpha_lookback                * loss_lb
                + alpha_forecast_concept        * loss_fc
                + alpha_forecast_global_concept * loss_gfc
                + alpha_residual_reg            * loss_res
                + alpha_boundary_smooth         * loss_sm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cum_f   += loss_f.item()
        cum_lb  += loss_lb.item()
        cum_fc  += loss_fc.item()
        cum_gfc += loss_gfc.item()
        cum_res += loss_res.item()
        cum_sm  += loss_sm.item()
        n       += 1

    return (cum_f   / max(1, n), cum_lb / max(1, n),
            cum_fc  / max(1, n), cum_gfc / max(1, n),
            cum_res / max(1, n), cum_sm / max(1, n))


@torch.no_grad()
def _evaluate_concept_only(model, loader, device) -> tuple[float, float]:
    """Run forward with ``disable_residual=True`` and return (MSE, MAE).

    Used at the end of training to measure how much of the forecast
    accuracy is concept-driven vs residual-driven (interpretability
    attribution).
    """
    model.eval()
    mae_sum, mse_sum, denom = 0.0, 0.0, 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x = x.to(device); y = y.to(device)
        out = model(x, disable_residual=True)
        y_hat = out["y_hat"]
        mae_sum += (y_hat - y).abs().sum().item()
        mse_sum += ((y_hat - y) ** 2).sum().item()
        denom   += y.numel()
    return mse_sum / max(1, denom), mae_sum / max(1, denom)


@torch.no_grad()
def evaluate(model, loader, device):
    """Run forward over a dataset. Returns:
        (mae, mse,
         all lb_logits   (N, K_lb),
         all lb_labels   (N, K_lb),
         all fc_logits   (N, N_sw, K_fc),
         all fc_labels   (N, N_sw, K_fc),
         all y_hat       (N, T_out, O),
         all y_true      (N, T_out, O))
    """
    model.eval()
    all_lb_lg,  all_lb_la  = [], []
    all_fc_lg,  all_fc_la  = [], []
    all_gfc_lg, all_gfc_la = [], []
    all_yh, all_yt = [], []
    mae_sum, mse_sum, denom = 0.0, 0.0, 0

    for x, y, lb_lab, fc_lab, gfc_lab in loader:
        x = x.to(device); y = y.to(device)
        out = model(x)
        y_hat = out["y_hat"]
        mae_sum += (y_hat - y).abs().sum().item()
        mse_sum += ((y_hat - y) ** 2).sum().item()
        denom   += y.numel()
        all_lb_lg.append(out["lb_logits"].cpu());  all_lb_la.append(lb_lab)
        all_fc_lg.append(out["fc_logits"].cpu());  all_fc_la.append(fc_lab)
        all_gfc_lg.append(out["gfc_logits"].cpu()); all_gfc_la.append(gfc_lab)
        all_yh.append(y_hat.cpu());                all_yt.append(y.cpu())

    return (mse_sum / denom, mae_sum / denom,
            torch.cat(all_lb_lg),  torch.cat(all_lb_la),
            torch.cat(all_fc_lg),  torch.cat(all_fc_la),
            torch.cat(all_gfc_lg), torch.cat(all_gfc_la),
            torch.cat(all_yh),     torch.cat(all_yt))


def per_concept_auroc(logits_2d: torch.Tensor, labels_2d: torch.Tensor) -> list[float]:
    """AUROC per concept over a (N, K) tensor pair. NaN for degenerate labels."""
    from sklearn.metrics import roc_auc_score
    K = logits_2d.shape[1]
    L = labels_2d.numpy(); S = logits_2d.numpy()
    out = []
    for k in range(K):
        y_true = L[:, k]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            out.append(float("nan"))
        else:
            out.append(float(roc_auc_score(y_true, S[:, k])))
    return out


def per_concept_accuracy(logits_2d: torch.Tensor,
                         labels_2d: torch.Tensor,
                         threshold: float = 0.5) -> list[float]:
    """Threshold-0.5 binary accuracy per concept on (N, K) tensors.

    For each concept k, compute the fraction of inputs whose
    (sigmoid(logit_k) >= 0.5) prediction matches label_k. Unlike AUROC,
    accuracy is well-defined even when labels are all-zero or all-one --
    a concept that's True in 0% of inputs gets accuracy = fraction of
    inputs where the model also predicts False.
    """
    probs = torch.sigmoid(logits_2d)
    preds = (probs >= threshold).float()
    correct = (preds == labels_2d).float().mean(dim=0)
    return [float(a) for a in correct.tolist()]


def _print_lookback_table(title: str,
                          logits: torch.Tensor,
                          labels: torch.Tensor,
                          meta: list[dict]) -> None:
    """Print a (id, name, positive_rate_in_eval, accuracy, AUROC) table."""
    aurocs    = per_concept_auroc(logits, labels)
    accs      = per_concept_accuracy(logits, labels)
    pos_rates = labels.float().mean(dim=0).tolist()
    bar = "─" * 92
    print(f"\n  {title}")
    print(bar)
    print(f"   {'id':>3}  {'name':32s}  {'pos%':>6}  {'acc':>5}   {'AUROC':>5}")
    print(bar)
    for c, auc, acc, pr in zip(meta, aurocs, accs, pos_rates):
        auc_s = "  nan" if auc != auc else f"{auc:.3f}"
        print(f"   [{c['id']:>2}] {c['name']:32s}  "
              f"{pr*100:>5.1f}%  {acc*100:>4.1f}%   {auc_s}")
    print(bar)


def _print_forecast_table(title: str,
                          logits: torch.Tensor,
                          labels: torch.Tensor,
                          meta: list[dict],
                          n_subwindows: int) -> None:
    """Print TWO sub-tables for forecast concepts:
    one with per-sub-window AUROC, one with per-sub-window accuracy.
    Logits / labels have shape (N, n_subwindows, K_fc)."""
    print(f"\n  {title}")
    sws_hdr = "  ".join(f"sw{k}" for k in range(n_subwindows))
    bar = "─" * 100
    # AUROC sub-table
    print(bar)
    print(f"   {'id':>3}  {'name':32s}  AUROC per sub-window:  {sws_hdr}")
    print(bar)
    for j, c in enumerate(meta):
        per_sw = []
        for k in range(n_subwindows):
            aur = per_concept_auroc(logits[:, k, j:j+1], labels[:, k, j:j+1])[0]
            per_sw.append(f"{aur:.2f}" if aur == aur else " nan")
        print(f"   [{c['id']:>2}] {c['name']:32s}                         "
              + "  ".join(per_sw))
    print(bar)
    # Accuracy sub-table
    print(f"   {'id':>3}  {'name':32s}  Accuracy@0.5 per sub-window: {sws_hdr}")
    print(bar)
    for j, c in enumerate(meta):
        per_sw = []
        for k in range(n_subwindows):
            acc = per_concept_accuracy(logits[:, k, j:j+1], labels[:, k, j:j+1])[0]
            per_sw.append(f"{acc*100:.1f}")
        print(f"   [{c['id']:>2}] {c['name']:32s}                          "
              + "  ".join(per_sw))
    print(bar)


# ---------------------------------------------------------------------------
# Interpretability case figures
# ---------------------------------------------------------------------------

# Column layout of the (length, input_dim) window tensor produced by
# dataloader.LOF_Dataset for the PRSA schema. The first four columns are the
# calendar (used only to reconstruct timestamps), columns 4..13 are the raw
# continuous exogenous channels, and column 14 is the (z-scored, forecast-
# masked) PM2.5 target.
_PRSA_CAL_COLS   = {"year": 0, "month": 1, "day": 2, "hour": 3}
_PRSA_CONT_COLS  = [
    (4,  "PM10"), (5,  "SO2"),  (6,  "NO2"),  (7,  "CO"),   (8,  "O3"),
    (9,  "TEMP"), (10, "PRES"), (11, "DEWP"), (12, "RAIN"), (13, "WSPM"),
]
_PRSA_PM25_COL   = 14


def _season_from_month(m: int) -> str:
    return {12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn"}.get(int(m), "?")


def _window_timestamps(x: "torch.Tensor", base_year: int = 2013):
    """Reconstruct a per-row datetime list from the calendar columns of a
    single window tensor ``x`` of shape (length, input_dim).

    The PRSA CSV stores ``year`` as a 0-based offset (0 -> 2013, 1 -> 2014,
    ...). Returns a list[datetime] of length == x.shape[0]. Falls back to a
    plain hour index if any calendar value is malformed.
    """
    from datetime import datetime, timedelta
    yr = x[:, _PRSA_CAL_COLS["year"]].round().long().tolist()
    mo = x[:, _PRSA_CAL_COLS["month"]].round().long().tolist()
    dy = x[:, _PRSA_CAL_COLS["day"]].round().long().tolist()
    hr = x[:, _PRSA_CAL_COLS["hour"]].round().long().tolist()
    out = []
    for y, m, d, h in zip(yr, mo, dy, hr):
        try:
            out.append(datetime(base_year + int(y), int(m), int(d), int(h)))
        except Exception:
            # Malformed calendar value -> extrapolate hourly from last good ts.
            out.append((out[-1] + timedelta(hours=1)) if out else datetime(base_year, 1, 1))
    return out


def _fmt_ts(ts) -> str:
    """e.g. '2013-06-07 12:00 pm' (lowercase am/pm, no leading zero on hour)."""
    s = ts.strftime("%Y-%m-%d %I:%M %p").replace("AM", "am").replace("PM", "pm")
    # strip leading zero of the 12h hour for readability ("06:00" -> "6:00")
    date_part, time_part = s.split(" ", 1)
    hh, rest = time_part.split(":", 1)
    return f"{date_part} {int(hh)}:{rest}"


def _top_concepts(probs_1d, meta, top_k: int, thresh: float = 0.5):
    """Return a list of (prob, name, description) for the active concepts in a
    single bottleneck. 'Active' = prob >= thresh; if none clear the threshold
    we still surface the single highest-scoring concept so the panel is never
    empty. Capped at ``top_k`` entries, sorted by descending probability."""
    import torch
    if probs_1d.numel() == 0:
        return []
    order = torch.argsort(probs_1d, descending=True)
    rows = []
    for j in order.tolist():
        p = float(probs_1d[j])
        if p < thresh and rows:
            break
        cm = meta[j]
        rows.append((p, cm.get("name", f"c{j}"), str(cm.get("description", ""))))
        if len(rows) >= top_k:
            break
    return rows


def _save_interpretability_cases(
    out_dir: str,
    test_data,                 # LOF_Dataset (base) — gives x and inver_data()
    input_length: int,
    output_length: int,
    n_subwindows: int,
    t_yh,                      # (N, T_out, 1) z-scored predictions
    t_yt,                      # (N, T_out, 1) z-scored ground truth
    t_lb_lg,                   # (N, K_lb)
    t_fc_lg,                   # (N, N_sw, K_fc)  or (N, K_fc) if single
    t_gfc_lg,                  # (N, K_gfc)  (may be width 0)
    lb_meta, fc_meta, gfc_meta,
    n_cases: int = 6,
    top_k: int = 5,
    seed: int = 0,
    base_year: int = 2013,
    single_forecast_bottleneck: bool = False,
) -> None:
    """Save one figure per randomly-chosen GOOD (low-MAE) test window.

    Each figure shows, for the chosen window:
      * a title with the lookback time range, forecast time range, season,
        and the de-z-scored per-window MAE (µg/m³);
      * a grid of every continuous channel over the whole window with a red
        dashed line marking the lookback / forecast boundary;
      * the PM2.5 panel overlays forecast vs ground truth in the forecast
        region (de-z-scored with the dataset's StandardScaler);
      * a text panel listing the active concepts in all three bottlenecks
        (lookback / per-sub-window forecast / global forecast) with their
        predicted activation probabilities.

    "Good" windows are the lowest-MAE quartile; ``n_cases`` are sampled from
    that pool with a fixed seed so the selection is reproducible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    os.makedirs(out_dir, exist_ok=True)
    n_test = t_yh.shape[0]
    if n_test == 0:
        print("  [interp] no test windows — skipping.")
        return

    # ---- pick GOOD windows: lowest-MAE quartile, then random sample ----
    per_mae = (t_yh - t_yt).abs().mean(dim=(1, 2)).numpy()      # (N,)  z-scored
    n_good = max(1, n_test // 4)
    good_pool = np.argsort(per_mae)[:n_good]                    # best quartile
    rng = np.random.RandomState(seed)
    n_pick = min(n_cases, len(good_pool))
    chosen = rng.choice(good_pool, size=n_pick, replace=False)

    lb_acts  = torch.sigmoid(t_lb_lg)
    fc_acts  = torch.sigmoid(t_fc_lg)
    gfc_acts = torch.sigmoid(t_gfc_lg) if t_gfc_lg.shape[-1] > 0 else None

    sw_len = output_length // max(1, n_subwindows)

    for case_i, w in enumerate(chosen):
        w = int(w)
        x, _ = test_data[w]                      # x: (L+T, input_dim)
        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        L, T = input_length, output_length
        ts = _window_timestamps(x, base_year=base_year)

        lb_start, lb_end = ts[0], ts[L - 1]
        fc_start, fc_end = ts[L], ts[L + T - 1]
        season = _season_from_month(x[L, _PRSA_CAL_COLS["month"]].item())

        # de-z-scored series
        pm25_lb  = test_data.inver_data(x[:L, _PRSA_PM25_COL].numpy())
        truth_fc = test_data.inver_data(t_yt[w, :, 0].numpy())
        pred_fc  = test_data.inver_data(t_yh[w, :, 0].numpy())
        mae_raw  = float(np.abs(truth_fc - pred_fc).mean())

        # ---- figure layout: feature grid (3x4) + wide text panel ----
        fig = plt.figure(figsize=(20, 17))
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 1.55],
                              hspace=0.45, wspace=0.28)
        t_full = np.arange(L + T)
        t_fcst = np.arange(L, L + T)

        # continuous exogenous channels (cols 4..13) — full window
        for i, (col, name) in enumerate(_PRSA_CONT_COLS):
            ax = fig.add_subplot(gs[i // 4, i % 4])
            ax.plot(t_full, x[:, col].numpy(), color="#1f77b4", lw=1.1)
            ax.axvline(L, color="red", ls="--", lw=1.2, alpha=0.8)
            ax.set_title(name, fontsize=11, fontweight="bold")
            ax.tick_params(labelsize=8)
            ax.margins(x=0.01)

        # PM2.5 panel (slot index 10 -> row 2, col 2)
        ax_pm = fig.add_subplot(gs[2, 2])
        ax_pm.plot(np.arange(L), pm25_lb, color="#1f77b4", lw=1.1, label="PM2.5 lookback")
        ax_pm.plot(t_fcst, truth_fc, color="#2ca02c", lw=1.8, label="ground truth")
        ax_pm.plot(t_fcst, pred_fc, color="#d62728", lw=1.8, ls="--", label="forecast")
        ax_pm.axvline(L, color="red", ls="--", lw=1.2, alpha=0.8)
        ax_pm.set_title("PM2.5 (target)", fontsize=11, fontweight="bold")
        ax_pm.tick_params(labelsize=8)
        ax_pm.legend(fontsize=7, loc="upper left")
        ax_pm.margins(x=0.01)

        # spare slot (row 2, col 3) — quick forecast-vs-truth zoom
        ax_zoom = fig.add_subplot(gs[2, 3])
        ax_zoom.plot(np.arange(T), truth_fc, color="#2ca02c", lw=1.8, label="truth")
        ax_zoom.plot(np.arange(T), pred_fc, color="#d62728", lw=1.8, ls="--", label="forecast")
        ax_zoom.set_title("Forecast horizon (zoom)", fontsize=11, fontweight="bold")
        ax_zoom.set_xlabel("hours ahead", fontsize=8)
        ax_zoom.tick_params(labelsize=8)
        ax_zoom.legend(fontsize=7, loc="best")

        # ---- concept text panel ----
        ax_txt = fig.add_subplot(gs[3, :])
        ax_txt.axis("off")
        lines = []
        lines.append("ACTIVE CONCEPTS (predicted activation probability)")
        lines.append("=" * 110)

        lines.append("LOOKBACK bottleneck (whole 96-h history):")
        for p, nm, desc in _top_concepts(lb_acts[w], lb_meta, top_k):
            lines.append(f"   {p:5.2f}  {nm:32s} {desc[:60]}")
        if not lb_meta:
            lines.append("   (none)")
        lines.append("")

        lines.append(f"FORECAST sub-window bottleneck ({n_subwindows} x {sw_len}-h windows):")
        if single_forecast_bottleneck or fc_acts.dim() == 2:
            for p, nm, desc in _top_concepts(fc_acts[w], fc_meta, top_k):
                lines.append(f"   {p:5.2f}  {nm:32s} {desc[:60]}")
        else:
            for sw in range(fc_acts.shape[1]):
                sw_lo = ts[L + sw * sw_len]
                sw_hi = ts[min(L + (sw + 1) * sw_len - 1, L + T - 1)]
                lines.append(f"   sub-window {sw}  [{_fmt_ts(sw_lo)} → {_fmt_ts(sw_hi)}]:")
                rows = _top_concepts(fc_acts[w, sw], fc_meta, min(3, top_k))
                if not rows:
                    lines.append("      (none active)")
                for p, nm, desc in rows:
                    lines.append(f"      {p:5.2f}  {nm:32s} {desc[:52]}")
        lines.append("")

        if gfc_acts is not None:
            lines.append("GLOBAL-FORECAST bottleneck (whole 24-h forecast):")
            for p, nm, desc in _top_concepts(gfc_acts[w], gfc_meta, top_k):
                lines.append(f"   {p:5.2f}  {nm:32s} {desc[:60]}")

        ax_txt.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
                    family="monospace", fontsize=9, transform=ax_txt.transAxes)

        title = (
            f"Interpretability case {case_i + 1}  (test window {w})   "
            f"season: {season}   |   per-window MAE = {mae_raw:.1f} µg/m³\n"
            f"lookback window: {_fmt_ts(lb_start)} TO {_fmt_ts(lb_end)}   ;   "
            f"forecasting window: {_fmt_ts(fc_start)} TO {_fmt_ts(fc_end)}"
        )
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

        fpath = os.path.join(out_dir, f"interp_case_{case_i + 1}_w{w}.png")
        fig.savefig(fpath, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [interp] saved {fpath}  (MAE {mae_raw:.1f} µg/m³, {season})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["device"] = device
    print(f"Using device: {device}")
    set_seed(args.seed)

    # ----- Datasets -----
    print("\n── Loading data ──")
    train_data = LOF_Dataset(args.input_length, args.output_length, "train",
                              args.input_dim, args.dataset)
    val_data   = LOF_Dataset(args.input_length, args.output_length, "vali",
                              args.input_dim, args.dataset)
    test_data  = LOF_Dataset(args.input_length, args.output_length, "test",
                              args.input_dim, args.dataset)
    print(f"  train windows = {len(train_data)}")
    print(f"  vali  windows = {len(val_data)}")
    print(f"  test  windows = {len(test_data)}")

    # ----- Concept labels for TRAIN -----
    print("\n── Building concept labels (train) ──")
    if args.single_forecast_bottleneck:
        print("  [ABLATION] Single global forecast bottleneck -- "
              "labels computed on whole forecast window.")
    lb_train, fc_train, gfc_train, lb_meta, fc_meta, gfc_meta = build_window_label_tables(
        concepts_path=args.concepts_path,
        segments_path=args.segments_path,
        n_windows=len(train_data),
        only_ok=True,
        single_forecast_bottleneck=args.single_forecast_bottleneck,
        min_pos_rate=args.min_pos_rate,
        max_pos_rate=args.max_pos_rate,
    )
    K_lb  = lb_train.shape[1]
    K_gfc = gfc_train.shape[1]
    if args.single_forecast_bottleneck:
        K_sw = 0
        K_fc = fc_train.shape[1]
        print(f"  K_lookback = {K_lb}, K_forecast = {K_fc} (single bottleneck), "
              f"K_forecast_global = {K_gfc}")
    else:
        K_sw = fc_train.shape[1] if fc_train.ndim == 3 else 0
        K_fc = fc_train.shape[2] if fc_train.ndim == 3 else 0
        print(f"  K_lookback = {K_lb}, K_subwindows = {K_sw}, "
              f"K_forecast = {K_fc}, K_forecast_global = {K_gfc}")
        if K_sw != args.n_subwindows:
            print(f"  WARNING: --n_subwindows={args.n_subwindows} != K_sw={K_sw} from segments file. "
                  f"Using {K_sw} (taken from preprocessing).")
            args.n_subwindows = K_sw

    # ----- Build val/test concept labels if their segments JSONs were provided -----
    def _zero_fc(n_windows: int) -> np.ndarray:
        if args.single_forecast_bottleneck:
            return np.zeros((n_windows, K_fc), dtype=np.float32)
        return np.zeros((n_windows, K_sw, K_fc), dtype=np.float32)

    def _load_split_labels(name: str, path: str | None, n_windows: int):
        """Return (lb_labels, fc_labels, gfc_labels, is_real) for a split."""
        if path is None:
            print(f"  {name}: NO segments path supplied -> labels are placeholder zeros, "
                  "per-concept stats will not be reported for this split.")
            return (np.zeros((n_windows, K_lb), dtype=np.float32),
                    _zero_fc(n_windows),
                    np.zeros((n_windows, K_gfc), dtype=np.float32),
                    False)
        lb, fc, gfc, _, _, _ = build_window_label_tables(
            concepts_path=args.concepts_path,
            segments_path=path,
            n_windows=n_windows,
            only_ok=True,
            single_forecast_bottleneck=args.single_forecast_bottleneck,
            min_pos_rate=args.min_pos_rate,
            max_pos_rate=args.max_pos_rate,
        )
        # Shape match check.
        if args.single_forecast_bottleneck:
            if fc.shape[1] != K_fc:
                raise ValueError(
                    f"{name} segments at {path} produced fc-label shape {fc.shape}; "
                    f"train produced (W, {K_fc})."
                )
        else:
            if fc.shape[1] != K_sw or fc.shape[2] != K_fc:
                raise ValueError(
                    f"{name} segments at {path} produced fc-label shape {fc.shape}; "
                    f"train produced (W, {K_sw}, {K_fc}). n_subwindows mismatch?"
                )
        if gfc.shape[1] != K_gfc:
            raise ValueError(
                f"{name} segments at {path} produced gfc-label shape {gfc.shape}; "
                f"train produced (W, {K_gfc})."
            )
        print(f"  {name}: REAL concept labels loaded "
              f"(lb {lb.shape}, fc {fc.shape}, gfc {gfc.shape}).")
        return (lb.astype(np.float32),
                fc.astype(np.float32),
                gfc.astype(np.float32),
                True)

    print("\n── Building concept labels (val / test) ──")
    lb_val,  fc_val,  gfc_val,  val_labels_real  = _load_split_labels(
        "val",  args.val_segments_path,  len(val_data),
    )
    lb_test, fc_test, gfc_test, test_labels_real = _load_split_labels(
        "test", args.test_segments_path, len(test_data),
    )

    train_ds = CBMDataset(train_data, lb_train, fc_train, gfc_train)
    val_ds   = CBMDataset(val_data,   lb_val,   fc_val,   gfc_val)
    test_ds  = CBMDataset(test_data,  lb_test,  fc_test,  gfc_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # ----- Model -----
    print("\n── Building model ──")
    model = ConceptBottleneckModel(
        input_dim    = args.input_dim,
        mid_dim      = args.embed_dim,
        ext_dim      = args.input_dim - args.output_dim,
        output_dim   = args.output_dim,
        bottle_dim_p = args.bottle_dim_p,
        input_length = args.input_length,
        output_length= args.output_length,
        num_blocks   = args.num_layer,
        dropout      = args.dropout,
        n_lookback_concepts        = K_lb,
        n_forecast_concepts        = K_fc,
        n_global_forecast_concepts = K_gfc,
        n_subwindows               = args.n_subwindows,
        concept_emb_dim            = args.concept_emb_dim,
        decoder_hidden      = args.decoder_hidden,
        decoder_depth       = args.decoder_depth,
        sub_window_id_dim   = args.sw_id_dim,
        residual_dim        = args.residual_dim,
        sw_gru_hidden       = args.sw_gru_hidden,
        sw_summary_dim      = args.sw_summary_dim,
        single_forecast_bottleneck = args.single_forecast_bottleneck,
        use_rev_in   = 1,
        norm_type    = "B",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total trainable params: {n_params/1e6:.2f} M")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    # ----- Train loop -----
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "cbm.pth")
    best_val_mae    = float("inf")
    best_val_epoch  = -1
    epochs_no_improve = 0
    print("\n── Training ──")
    if args.concept_warmup_epochs > 0:
        print(f"  (concept warmup: alpha_forecast=0 for first "
              f"{args.concept_warmup_epochs} epochs)")
    if args.patience > 0:
        print(f"  (early stopping: stop if val MAE has not improved for "
              f"{args.patience} epochs)")
    for epoch in range(args.epoch):
        # Concept-only warmup: zero alpha_forecast while the concept predictors fit.
        eff_alpha_forecast = (0.0
                              if epoch < args.concept_warmup_epochs
                              else args.alpha_forecast)
        l_f, l_lb, l_fc, l_gfc, l_res, l_sm = train_one_epoch(
            model, train_loader, optimizer, device,
            eff_alpha_forecast, args.alpha_lookback, args.alpha_forecast_concept,
            args.alpha_forecast_global_concept, args.alpha_residual_reg,
            args.alpha_boundary_smooth,
        )
        val_mse, val_mae, *_ = evaluate(model, val_loader, device)
        improved = val_mae < best_val_mae
        marker = " *" if improved else ""
        gfc_str = f"  train_gfc_BCE={l_gfc:.4f}" if K_gfc > 0 else ""
        sm_str  = f"  train_sm={l_sm:.4f}"        if args.alpha_boundary_smooth > 0 else ""
        print(f"  [epoch {epoch+1:>3d}/{args.epoch}]  "
              f"train_L1={l_f:.4f}  "
              f"train_lb_BCE={l_lb:.4f}  "
              f"train_fc_BCE={l_fc:.4f}"
              f"{gfc_str}  "
              f"train_res={l_res:.4f}"
              f"{sm_str}  "
              f"val MAE={val_mae:.4f}{marker}")
        if improved:
            best_val_mae      = val_mae
            best_val_epoch    = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1
        # Optional scheduled lr decay (default off — lr_decay_step=999).
        if (epoch + 1) % args.lr_decay_step == 0:
            for g in optimizer.param_groups:
                g["lr"] *= args.lr_decay
        # Early stopping.
        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"  Early stop: no improvement for {args.patience} epochs. "
                  f"Best val MAE = {best_val_mae:.4f} at epoch {best_val_epoch}.")
            break

    print(f"\n  Best val MAE = {best_val_mae:.4f} achieved at epoch {best_val_epoch}.")

    # ----- Test -----
    print(f"\n── Loading best checkpoint ({ckpt_path}) ──")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("\n── Test forecast results ──")
    (test_mse, test_mae,
     t_lb_lg,  t_lb_la,
     t_fc_lg,  t_fc_la,
     t_gfc_lg, t_gfc_la,
     t_yh,     t_yt) = evaluate(model, test_loader, device)
    print(f"  test MSE      = {test_mse:.6f}  (full model)")
    print(f"  test MAE      = {test_mae:.6f}  (full model)")

    # Concept-only test: re-evaluate with residual zeroed. If the residual
    # was carrying real forecast signal, this MAE will be visibly higher.
    # A small gap means the model is genuinely concept-driven; a big gap
    # means the residual is doing most of the work.
    if args.residual_dim > 0:
        test_mse_co, test_mae_co = _evaluate_concept_only(model, test_loader, device)
        print(f"  test MAE      = {test_mae_co:.6f}  (CONCEPT-ONLY, residual disabled)")
        gap = test_mae_co - test_mae
        rel = (gap / max(1e-9, test_mae)) * 100
        print(f"  residual gain = {gap:+.4f}  ({rel:+.1f}% MAE improvement from residual)")
        attr = (test_mae / max(1e-9, test_mae_co)) * 100
        print(f"  -> concepts alone capture {attr:.1f}% of the concept-only "
              "MAE; residual fills the remaining detail.")

    # ----- Per-concept TEST stats (real labels if --test_segments_path was given) -----
    # Falls back to a TRAIN evaluation when the user didn't supply test segments,
    # noting that the table's labels are training-set labels and therefore
    # overstate generalization.
    if test_labels_real:
        print("\n── Per-concept TEST stats (against REAL test labels) ──")
        _print_lookback_table(
            "LOOKBACK concepts on TEST", t_lb_lg, t_lb_la, lb_meta,
        )
        if args.single_forecast_bottleneck:
            # Single global forecast bottleneck: report as a flat lookback-style
            # table (no per-sub-window grid).
            _print_lookback_table(
                "FORECAST concepts on TEST (single global bottleneck)",
                t_fc_lg, t_fc_la, fc_meta,
            )
        else:
            _print_forecast_table(
                "FORECAST concepts on TEST", t_fc_lg, t_fc_la, fc_meta, args.n_subwindows,
            )
        if K_gfc > 0:
            _print_lookback_table(
                "GLOBAL-FORECAST concepts on TEST (one rule per whole forecast window)",
                t_gfc_lg, t_gfc_la, gfc_meta,
            )
    else:
        print("\n── Per-concept stats: TEST labels not provided, falling back to TRAIN ──")
        print("    Pass --test_segments_path to get the REAL test-set tables.")
        tr_lb_lg,  tr_lb_la  = [], []
        tr_fc_lg,  tr_fc_la  = [], []
        tr_gfc_lg, tr_gfc_la = [], []
        model.eval()
        with torch.no_grad():
            for x, _, lb_lab, fc_lab, gfc_lab in train_loader:
                x = x.to(device)
                out = model(x)
                tr_lb_lg.append(out["lb_logits"].cpu());  tr_lb_la.append(lb_lab)
                tr_fc_lg.append(out["fc_logits"].cpu());  tr_fc_la.append(fc_lab)
                tr_gfc_lg.append(out["gfc_logits"].cpu()); tr_gfc_la.append(gfc_lab)
        tr_lb_lg  = torch.cat(tr_lb_lg);  tr_lb_la  = torch.cat(tr_lb_la)
        tr_fc_lg  = torch.cat(tr_fc_lg);  tr_fc_la  = torch.cat(tr_fc_la)
        tr_gfc_lg = torch.cat(tr_gfc_lg); tr_gfc_la = torch.cat(tr_gfc_la)
        _print_lookback_table(
            "LOOKBACK concepts on TRAIN (fallback)", tr_lb_lg, tr_lb_la, lb_meta,
        )
        if args.single_forecast_bottleneck:
            _print_lookback_table(
                "FORECAST concepts on TRAIN (fallback, single global bottleneck)",
                tr_fc_lg, tr_fc_la, fc_meta,
            )
        else:
            _print_forecast_table(
                "FORECAST concepts on TRAIN (fallback)", tr_fc_lg, tr_fc_la, fc_meta,
                args.n_subwindows,
            )
        if K_gfc > 0:
            _print_lookback_table(
                "GLOBAL-FORECAST concepts on TRAIN (fallback)",
                tr_gfc_lg, tr_gfc_la, gfc_meta,
            )

    # ----- Case studies -----
    if args.case_studies > 0:
        print(f"\n── Case studies: {args.case_studies} random test windows ──")
        t_lb_acts  = torch.sigmoid(t_lb_lg)
        t_fc_acts  = torch.sigmoid(t_fc_lg)
        t_gfc_acts = torch.sigmoid(t_gfc_lg) if K_gfc > 0 else None
        n_test = t_yh.shape[0]
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(n_test, min(args.case_studies, n_test), replace=False)

        for k_idx, w in enumerate(idx):
            print(f"\n  Case {k_idx+1} (test window {w}):")
            lb_top = torch.topk(t_lb_acts[w], k=min(5, K_lb))
            print(f"    top lookback concepts active:")
            for s, j in zip(lb_top.values, lb_top.indices):
                cm = lb_meta[j.item()]
                print(f"      {s.item():.3f}  {cm['name']}  ({cm['description'][:60]})")
            if t_gfc_acts is not None:
                gfc_top = torch.topk(t_gfc_acts[w], k=min(5, K_gfc))
                print(f"    top global-forecast concepts active:")
                for s, j in zip(gfc_top.values, gfc_top.indices):
                    cm = gfc_meta[j.item()]
                    print(f"      {s.item():.3f}  {cm['name']}  ({cm['description'][:60]})")
            if args.single_forecast_bottleneck:
                # Single global bottleneck: one set of activations to report.
                top = torch.topk(t_fc_acts[w], k=min(5, K_fc))
                names = ", ".join(
                    f"{s.item():.2f}*{fc_meta[j.item()]['name']}"
                    for s, j in zip(top.values, top.indices)
                )
                print(f"    top forecast concepts (single bottleneck): {names}")
            else:
                print(f"    top forecast concepts per sub-window:")
                for sw in range(args.n_subwindows):
                    top = torch.topk(t_fc_acts[w, sw], k=min(3, K_fc))
                    names = ", ".join(
                        f"{s.item():.2f}*{fc_meta[j.item()]['name']}"
                        for s, j in zip(top.values, top.indices)
                    )
                    print(f"      sw{sw}: {names}")
            mae_here = (t_yh[w] - t_yt[w]).abs().mean().item()
            print(f"    per-window MAE = {mae_here:.4f}")

    # ----- Interpretability case figures (good forecasts) -----
    if args.interp_cases > 0:
        print(f"\n── Interpretability figures: {args.interp_cases} good test windows ──")
        _save_interpretability_cases(
            out_dir       = args.checkpoint_dir,
            test_data     = test_data,
            input_length  = args.input_length,
            output_length = args.output_length,
            n_subwindows  = args.n_subwindows,
            t_yh          = t_yh,
            t_yt          = t_yt,
            t_lb_lg       = t_lb_lg,
            t_fc_lg       = t_fc_lg,
            t_gfc_lg      = t_gfc_lg,
            lb_meta       = lb_meta,
            fc_meta       = fc_meta,
            gfc_meta      = gfc_meta,
            n_cases       = args.interp_cases,
            top_k         = args.interp_top_k,
            seed          = args.seed,
            base_year     = args.interp_base_year,
            single_forecast_bottleneck = args.single_forecast_bottleneck,
        )


if __name__ == "__main__":
    main()
