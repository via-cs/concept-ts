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

    PYTHONPATH=. python -u CBM/train_cbm.py \\
        --dataset       ./dataset/LOF/RE.csv \\
        --concepts_path ./CBM/output/concepts_validated_<TS>.json \\
        --segments_path ./preprocessing/output/RE/RE_full_segments.json \\
        --epoch 30 --batch_size 16 \\
        --input_length 384 --output_length 96 \\
        --input_dim 23 --output_dim 1 \\
        --num_layer 4 --dropout 0.5 --embed_dim 128 \\
        --n_subwindows 4 \\
        --alpha_forecast 0.3 --alpha_lookback 1.0 --alpha_forecast_concept 1.0 \\
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
    p.add_argument("--segments_path",  type=str, required=True)
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
    p.add_argument("--decoder_hidden", type=int, default=256)
    p.add_argument("--sw_id_dim",      type=int, default=16,
                   help="Width of the sub-window-id embedding fed to the shared sub-decoder.")

    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--lr_decay",       type=float, default=0.5)
    p.add_argument("--lr_decay_step",  type=int,   default=5)
    p.add_argument("--seed",           type=int,   default=0)

    p.add_argument("--alpha_forecast",         type=float, default=0.3)
    p.add_argument("--alpha_lookback",         type=float, default=1.0)
    p.add_argument("--alpha_forecast_concept", type=float, default=1.0)

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
                    alpha_forecast, alpha_lookback, alpha_forecast_concept):
    model.train()
    cum_f, cum_lb, cum_fc, n = 0.0, 0.0, 0.0, 0
    l1_fn  = nn.L1Loss()
    bce_fn = nn.BCEWithLogitsLoss()

    for x, y, lb_lab, fc_lab in loader:
        x      = x.to(device)
        y      = y.to(device)
        lb_lab = lb_lab.to(device)
        fc_lab = fc_lab.to(device)
        out = model(x)

        loss_f  = l1_fn(out["y_hat"],     y)
        loss_lb = bce_fn(out["lb_logits"], lb_lab)
        loss_fc = bce_fn(out["fc_logits"], fc_lab)
        loss = (alpha_forecast        * loss_f
                + alpha_lookback         * loss_lb
                + alpha_forecast_concept * loss_fc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cum_f  += loss_f.item()
        cum_lb += loss_lb.item()
        cum_fc += loss_fc.item()
        n      += 1

    return cum_f / max(1, n), cum_lb / max(1, n), cum_fc / max(1, n)


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
    lb_train, fc_train, lb_meta, fc_meta = build_window_label_tables(
        concepts_path=args.concepts_path,
        segments_path=args.segments_path,
        n_windows=len(train_data),
        only_ok=True,
    )
    K_lb = lb_train.shape[1]
    K_sw = fc_train.shape[1] if fc_train.ndim == 3 else 0
    K_fc = fc_train.shape[2] if fc_train.ndim == 3 else 0
    print(f"  K_lookback = {K_lb}, K_subwindows = {K_sw}, K_forecast = {K_fc}")
    if K_sw != args.n_subwindows:
        print(f"  WARNING: --n_subwindows={args.n_subwindows} != K_sw={K_sw} from segments file. "
              f"Using {K_sw} (taken from preprocessing).")
        args.n_subwindows = K_sw

    # ----- Placeholder val/test labels (zeros) — for AUROC we still need shape match -----
    lb_val  = np.zeros((len(val_data),  K_lb),       dtype=np.float32)
    fc_val  = np.zeros((len(val_data),  K_sw, K_fc), dtype=np.float32)
    lb_test = np.zeros((len(test_data), K_lb),       dtype=np.float32)
    fc_test = np.zeros((len(test_data), K_sw, K_fc), dtype=np.float32)
    print("  (val/test concept labels initialized as zeros — build val/test segments JSONs"
          " for real concept AUROC on those splits.)")

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
        decoder_hidden      = args.decoder_hidden,
        sub_window_id_dim   = args.sw_id_dim,
        use_rev_in   = 1,
        norm_type    = "B",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total trainable params: {n_params/1e6:.2f} M")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ----- Train loop -----
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "cbm.pth")
    best_val_mae = float("inf")
    print("\n── Training ──")
    for epoch in range(args.epoch):
        l_f, l_lb, l_fc = train_one_epoch(
            model, train_loader, optimizer, device,
            args.alpha_forecast, args.alpha_lookback, args.alpha_forecast_concept,
        )
        val_mse, val_mae, *_ = evaluate(model, val_loader, device)
        print(f"  [epoch {epoch+1:>3d}/{args.epoch}]  "
              f"train_L1={l_f:.4f}  "
              f"train_lb_BCE={l_lb:.4f}  "
              f"train_fc_BCE={l_fc:.4f}  "
              f"val MAE={val_mae:.4f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), ckpt_path)
        if (epoch + 1) % args.lr_decay_step == 0:
            for g in optimizer.param_groups:
                g["lr"] *= args.lr_decay

    # ----- Test -----
    print(f"\n── Loading best checkpoint ({ckpt_path}) ──")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    print("\n── Test forecast results ──")
    test_mse, test_mae, t_lb_lg, _, t_fc_lg, _, t_yh, t_yt = evaluate(model, test_loader, device)
    print(f"  test MSE = {test_mse:.6f}")
    print(f"  test MAE = {test_mae:.6f}")

    # ----- Per-concept AUROC computed on TRAIN labels -----
    # (val/test labels are placeholder zeros; for real val/test AUROC, build
    # val/test segment JSONs and load them analogously.)
    print("\n── Per-concept AUROC (train labels) ──")
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

    lb_aurocs = per_concept_auroc(tr_lb_lg, tr_lb_la)
    print("\n  LOOKBACK concepts:")
    for c, auc in zip(lb_meta, lb_aurocs):
        flag = " " if (auc != auc or auc > 0.7) else "?"
        auc_s = "  nan " if auc != auc else f"{auc:.3f}"
        print(f"   {flag} [{c['id']:>2}] {c['name']:32s}  train AUROC = {auc_s}")

    print("\n  FORECAST concepts (AUROC per concept, per sub-window):")
    print(f"   {'name':32s}  " + "  ".join(f"sw{k}" for k in range(args.n_subwindows)))
    for j, c in enumerate(fc_meta):
        per_sw = []
        for k in range(args.n_subwindows):
            aur = per_concept_auroc(tr_fc_lg[:, k, j:j+1], tr_fc_la[:, k, j:j+1])[0]
            per_sw.append(f"{aur:.2f}" if aur == aur else " nan")
        print(f"   [{c['id']:>2}] {c['name']:32s}  " + "  ".join(per_sw))

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
            lb_top = torch.topk(t_lb_acts[w], k=min(5, K_lb))
            print(f"    top lookback concepts active:")
            for s, j in zip(lb_top.values, lb_top.indices):
                cm = lb_meta[j.item()]
                print(f"      {s.item():.3f}  {cm['name']}  ({cm['description'][:60]})")
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
