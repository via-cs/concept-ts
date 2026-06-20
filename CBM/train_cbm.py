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
    def __init__(self, base: Dataset, lb_labels: np.ndarray, fc_labels: np.ndarray):
        assert len(base) == len(lb_labels) == len(fc_labels), (
            f"len(base)={len(base)} vs lb={len(lb_labels)} vs fc={len(fc_labels)}"
        )
        self.base      = base
        self.lb_labels = torch.from_numpy(lb_labels).float()    # (N, K_lb)
        self.fc_labels = torch.from_numpy(fc_labels).float()    # (N, K_sw, K_fc)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, self.lb_labels[idx], self.fc_labels[idx]


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
    p.add_argument("--lookback_subwindow_lengths", type=str, default="",
                   help="Comma-separated positive integers in CHRONOLOGICAL / "
                        "OLDEST-first order, giving the size (in timesteps) of "
                        "each lookback sub-window. Sum must equal --input_length. "
                        "Empty (default) = legacy single global lookback bottleneck. "
                        "Typical recency-weighted choice for a 96-hour lookback: "
                        "'48,24,16,8' -- four sub-windows with the most recent "
                        "8h getting a dedicated small bottleneck. MUST match the "
                        "value passed to preprocessing's --lookback_subwindow_lengths "
                        "for label shapes to align.")
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
    return p.parse_args()


def set_seed(seed: int = 0):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed); random.seed(seed)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device,
                    alpha_forecast, alpha_lookback, alpha_forecast_concept,
                    alpha_residual_reg):
    model.train()
    cum_f, cum_lb, cum_fc, cum_res, n = 0.0, 0.0, 0.0, 0.0, 0
    l1_fn  = nn.L1Loss()
    bce_fn = nn.BCEWithLogitsLoss()

    for x, y, lb_lab, fc_lab in loader:
        x      = x.to(device)
        y      = y.to(device)
        lb_lab = lb_lab.to(device)
        fc_lab = fc_lab.to(device)
        out = model(x)

        loss_f   = l1_fn(out["y_hat"],     y)
        loss_lb  = bce_fn(out["lb_logits"], lb_lab)
        loss_fc  = bce_fn(out["fc_logits"], fc_lab)
        # L2 penalty on residual norm. Mean over all (B, N, residual_dim)
        # elements -- if residual_dim==0 the tensor is empty and contributes 0.
        if out["residual"].numel() > 0:
            loss_res = out["residual"].pow(2).mean()
        else:
            loss_res = torch.zeros((), device=device)
        loss = (alpha_forecast        * loss_f
                + alpha_lookback         * loss_lb
                + alpha_forecast_concept * loss_fc
                + alpha_residual_reg     * loss_res)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cum_f   += loss_f.item()
        cum_lb  += loss_lb.item()
        cum_fc  += loss_fc.item()
        cum_res += loss_res.item()
        n       += 1

    return (cum_f / max(1, n), cum_lb / max(1, n),
            cum_fc / max(1, n), cum_res / max(1, n))


@torch.no_grad()
def _evaluate_concept_only(model, loader, device) -> tuple[float, float]:
    """Run forward with ``disable_residual=True`` and return (MSE, MAE).

    Used at the end of training to measure how much of the forecast
    accuracy is concept-driven vs residual-driven (interpretability
    attribution).
    """
    model.eval()
    mae_sum, mse_sum, denom = 0.0, 0.0, 0
    for x, y, _, _ in loader:
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
    all_lb_lg, all_lb_la = [], []
    all_fc_lg, all_fc_la = [], []
    all_yh, all_yt = [], []
    mae_sum, mse_sum, denom = 0.0, 0.0, 0

    for x, y, lb_lab, fc_lab in loader:
        x = x.to(device); y = y.to(device)
        out = model(x)
        y_hat = out["y_hat"]
        mae_sum += (y_hat - y).abs().sum().item()
        mse_sum += ((y_hat - y) ** 2).sum().item()
        denom   += y.numel()
        all_lb_lg.append(out["lb_logits"].cpu()); all_lb_la.append(lb_lab)
        all_fc_lg.append(out["fc_logits"].cpu()); all_fc_la.append(fc_lab)
        all_yh.append(y_hat.cpu());                all_yt.append(y.cpu())

    return (mse_sum / denom, mae_sum / denom,
            torch.cat(all_lb_lg), torch.cat(all_lb_la),
            torch.cat(all_fc_lg), torch.cat(all_fc_la),
            torch.cat(all_yh),    torch.cat(all_yt))


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
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["device"] = device
    print(f"Using device: {device}")
    set_seed(args.seed)

    # Parse the (possibly empty) lookback-subwindow-lengths string into a list.
    # Empty -> []. Non-empty -> list of positive ints; sum must equal input_length.
    if args.lookback_subwindow_lengths:
        lb_sw_lengths = [int(x) for x in args.lookback_subwindow_lengths.split(",") if x.strip()]
        if sum(lb_sw_lengths) != args.input_length:
            raise ValueError(
                f"sum(--lookback_subwindow_lengths) = {sum(lb_sw_lengths)} != "
                f"--input_length = {args.input_length}"
            )
    else:
        lb_sw_lengths = []
    args.lookback_subwindow_lengths = lb_sw_lengths
    args.n_lookback_subwindows = max(1, len(lb_sw_lengths))

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
    lb_train, fc_train, lb_meta, fc_meta = build_window_label_tables(
        concepts_path=args.concepts_path,
        segments_path=args.segments_path,
        n_windows=len(train_data),
        only_ok=True,
        single_forecast_bottleneck=args.single_forecast_bottleneck,
        n_lookback_subwindows=args.n_lookback_subwindows,
    )
    K_lb = lb_train.shape[-1]
    if args.single_forecast_bottleneck:
        K_sw = 0
        K_fc = fc_train.shape[1]
        print(f"  K_lookback = {K_lb}, K_forecast = {K_fc} (single bottleneck)")
    else:
        K_sw = fc_train.shape[1] if fc_train.ndim == 3 else 0
        K_fc = fc_train.shape[2] if fc_train.ndim == 3 else 0
        print(f"  K_lookback = {K_lb}, K_subwindows = {K_sw}, K_forecast = {K_fc}")
        if K_sw != args.n_subwindows:
            print(f"  WARNING: --n_subwindows={args.n_subwindows} != K_sw={K_sw} from segments file. "
                  f"Using {K_sw} (taken from preprocessing).")
            args.n_subwindows = K_sw

    # ----- Build val/test concept labels if their segments JSONs were provided -----
    def _zero_fc(n_windows: int) -> np.ndarray:
        if args.single_forecast_bottleneck:
            return np.zeros((n_windows, K_fc), dtype=np.float32)
        return np.zeros((n_windows, K_sw, K_fc), dtype=np.float32)

    def _zero_lb(n_windows: int) -> np.ndarray:
        if args.n_lookback_subwindows > 1:
            return np.zeros((n_windows, args.n_lookback_subwindows, K_lb),
                            dtype=np.float32)
        return np.zeros((n_windows, K_lb), dtype=np.float32)

    def _load_split_labels(name: str, path: str | None, n_windows: int):
        """Return (lb_labels, fc_labels, is_real) for a split."""
        if path is None:
            print(f"  {name}: NO segments path supplied -> labels are placeholder zeros, "
                  "per-concept stats will not be reported for this split.")
            return (_zero_lb(n_windows), _zero_fc(n_windows), False)
        lb, fc, _, _ = build_window_label_tables(
            concepts_path=args.concepts_path,
            segments_path=path,
            n_windows=n_windows,
            only_ok=True,
            single_forecast_bottleneck=args.single_forecast_bottleneck,
            n_lookback_subwindows=args.n_lookback_subwindows,
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
        print(f"  {name}: REAL concept labels loaded "
              f"(lb {lb.shape}, fc {fc.shape}).")
        return lb.astype(np.float32), fc.astype(np.float32), True

    print("\n── Building concept labels (val / test) ──")
    lb_val,  fc_val,  val_labels_real  = _load_split_labels(
        "val",  args.val_segments_path,  len(val_data),
    )
    lb_test, fc_test, test_labels_real = _load_split_labels(
        "test", args.test_segments_path, len(test_data),
    )

    train_ds = CBMDataset(train_data, lb_train, fc_train)
    val_ds   = CBMDataset(val_data,   lb_val,   fc_val)
    test_ds  = CBMDataset(test_data,  lb_test,  fc_test)

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
        n_lookback_concepts = K_lb,
        n_forecast_concepts = K_fc,
        n_subwindows        = args.n_subwindows,
        lookback_subwindow_lengths = args.lookback_subwindow_lengths,
        concept_emb_dim     = args.concept_emb_dim,
        decoder_hidden      = args.decoder_hidden,
        decoder_depth       = args.decoder_depth,
        sub_window_id_dim   = args.sw_id_dim,
        residual_dim        = args.residual_dim,
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
        l_f, l_lb, l_fc, l_res = train_one_epoch(
            model, train_loader, optimizer, device,
            eff_alpha_forecast, args.alpha_lookback, args.alpha_forecast_concept,
            args.alpha_residual_reg,
        )
        val_mse, val_mae, *_ = evaluate(model, val_loader, device)
        improved = val_mae < best_val_mae
        marker = " *" if improved else ""
        print(f"  [epoch {epoch+1:>3d}/{args.epoch}]  "
              f"train_L1={l_f:.4f}  "
              f"train_lb_BCE={l_lb:.4f}  "
              f"train_fc_BCE={l_fc:.4f}  "
              f"train_res={l_res:.4f}  "
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
    test_mse, test_mae, t_lb_lg, t_lb_la, t_fc_lg, t_fc_la, t_yh, t_yt = evaluate(
        model, test_loader, device,
    )
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
        if args.n_lookback_subwindows > 1:
            _print_forecast_table(
                "LOOKBACK concepts on TEST (per lookback sub-window)",
                t_lb_lg, t_lb_la, lb_meta, args.n_lookback_subwindows,
            )
        else:
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
    else:
        print("\n── Per-concept stats: TEST labels not provided, falling back to TRAIN ──")
        print("    Pass --test_segments_path to get the REAL test-set tables.")
        tr_lb_lg, tr_lb_la, tr_fc_lg, tr_fc_la = [], [], [], []
        model.eval()
        with torch.no_grad():
            for x, _, lb_lab, fc_lab in train_loader:
                x = x.to(device)
                out = model(x)
                tr_lb_lg.append(out["lb_logits"].cpu()); tr_lb_la.append(lb_lab)
                tr_fc_lg.append(out["fc_logits"].cpu()); tr_fc_la.append(fc_lab)
        tr_lb_lg = torch.cat(tr_lb_lg); tr_lb_la = torch.cat(tr_lb_la)
        tr_fc_lg = torch.cat(tr_fc_lg); tr_fc_la = torch.cat(tr_fc_la)
        if args.n_lookback_subwindows > 1:
            _print_forecast_table(
                "LOOKBACK concepts on TRAIN (fallback, per lookback sub-window)",
                tr_lb_lg, tr_lb_la, lb_meta, args.n_lookback_subwindows,
            )
        else:
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

    # ----- Case studies -----
    if args.case_studies > 0:
        print(f"\n── Case studies: {args.case_studies} random test windows ──")
        t_lb_acts = torch.sigmoid(t_lb_lg)
        t_fc_acts = torch.sigmoid(t_fc_lg)
        n_test = t_yh.shape[0]
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(n_test, min(args.case_studies, n_test), replace=False)

        for k_idx, w in enumerate(idx):
            print(f"\n  Case {k_idx+1} (test window {w}):")
            if args.n_lookback_subwindows > 1:
                print(f"    top lookback concepts per lookback sub-window:")
                for lbsw in range(args.n_lookback_subwindows):
                    top = torch.topk(t_lb_acts[w, lbsw], k=min(3, K_lb))
                    names = ", ".join(
                        f"{s.item():.2f}*{lb_meta[j.item()]['name']}"
                        for s, j in zip(top.values, top.indices)
                    )
                    print(f"      lb_sw{lbsw}: {names}")
            else:
                lb_top = torch.topk(t_lb_acts[w], k=min(5, K_lb))
                print(f"    top lookback concepts active:")
                for s, j in zip(lb_top.values, lb_top.indices):
                    cm = lb_meta[j.item()]
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


if __name__ == "__main__":
    main()
