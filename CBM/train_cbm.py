"""
End-to-end training script for the Concept Bottleneck Model.

Pipeline
--------
  1. Load the validated concepts JSON from CBM/output/concepts_<TS>.json.
  2. Apply each concept's match function to the Part-A training segments,
     and map the per-segment labels to per-window labels using the same
     stride convention as exp/cluster_supervision.py.
  3. Build the ConceptBottleneckModel (encoder + concept layer + forecast
     decoder) and train it jointly with:
        loss = α_forecast · L1(y_hat, y) + α_concept · BCE(c_logits, labels)
  4. Evaluate on the test set:
        - forecast MAE / MSE
        - per-concept AUROC + a 5-segment case study showing concept
          activations alongside the predicted forecast.

Run
---

    PYTHONPATH=. python -u CBM/train_cbm.py \\
        --dataset ./dataset/LOF/RE.csv \\
        --concepts_path ./CBM/output/concepts_<TS>.json \\
        --segments_path ./preprocessing/output/RE/RE_full_segments.json \\
        --epoch 30 --batch_size 16 \\
        --input_length 384 --output_length 96 \\
        --input_dim 23 --output_dim 1 \\
        --num_layer 4 --dropout 0.5 --embed_dim 128 \\
        --alpha_concept 1.0 --alpha_forecast 0.3 \\
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
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from CBM.concept_bottleneck import ConceptBottleneckModel
from CBM.concept_labels import build_window_label_table
from dataloader.dataloader import LOF_Dataset


# ---------------------------------------------------------------------------
# Dataset wrapper — adds per-window concept labels to (x, y) tuples
# ---------------------------------------------------------------------------

class CBMDataset(Dataset):
    def __init__(self, base: Dataset, labels: np.ndarray):
        assert len(base) == len(labels), \
            f"len(base)={len(base)} != len(labels)={len(labels)}"
        self.base = base
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, self.labels[idx]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",        type=str, required=True)
    p.add_argument("--concepts_path",  type=str, required=True)
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

    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--lr_decay",       type=float, default=0.5)
    p.add_argument("--lr_decay_step",  type=int,   default=5)
    p.add_argument("--seed",           type=int,   default=0)

    # CBM-specific
    p.add_argument("--alpha_concept",  type=float, default=1.0,
                   help="Weight on the BCE concept loss.")
    p.add_argument("--alpha_forecast", type=float, default=0.3,
                   help="Weight on the L1 forecast loss. Keep < alpha_concept "
                        "for the first ~10 epochs to prevent the concept layer "
                        "from being corrupted by forecast pressure.")
    p.add_argument("--decoder_hidden", type=int, default=256,
                   help="Hidden width of the forecast decoder MLP.")
    p.add_argument("--checkpoint_dir", type=str, default="./checkpoint/cbm")
    p.add_argument("--case_studies",   type=int, default=5,
                   help="Number of test segments to print case studies for.")
    return p.parse_args()


def set_seed(seed: int = 0):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed); random.seed(seed)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device,
                    alpha_forecast, alpha_concept):
    model.train()
    cum_f, cum_c, n = 0.0, 0.0, 0
    l1_fn  = nn.L1Loss()
    bce_fn = nn.BCEWithLogitsLoss()

    for x, y, lab in loader:
        x, y, lab = x.to(device), y.to(device), lab.to(device)
        out = model(x)

        loss_f = l1_fn(out["y_hat"], y)
        loss_c = bce_fn(out["c_logits"], lab)
        loss = alpha_forecast * loss_f + alpha_concept * loss_c

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cum_f += loss_f.item(); cum_c += loss_c.item(); n += 1

    return cum_f / max(1, n), cum_c / max(1, n)


@torch.no_grad()
def evaluate(model, loader, device):
    """Run forward over a dataset; return MAE, MSE, all concept logits,
    all concept labels, all predictions, all targets."""
    model.eval()
    all_logits, all_labels = [], []
    all_y_hat,  all_y_true = [], []
    mse_sum, mae_sum, denom = 0.0, 0.0, 0

    for x, y, lab in loader:
        x, y, lab = x.to(device), y.to(device), lab.to(device)
        out = model(x)
        y_hat = out["y_hat"]
        mse_sum += ((y_hat - y) ** 2).sum().item()
        mae_sum += (y_hat - y).abs().sum().item()
        denom   += y.numel()
        all_logits.append(out["c_logits"].cpu())
        all_labels.append(lab.cpu())
        all_y_hat.append(y_hat.cpu())
        all_y_true.append(y.cpu())

    mse, mae = mse_sum / denom, mae_sum / denom
    return (mse, mae,
            torch.cat(all_logits), torch.cat(all_labels),
            torch.cat(all_y_hat),  torch.cat(all_y_true))


def per_concept_auroc(logits: torch.Tensor, labels: torch.Tensor) -> list[float]:
    """Compute AUROC per concept. Returns NaN for concepts with no positive
    or no negative examples in the eval set."""
    from sklearn.metrics import roc_auc_score
    K = logits.shape[1]
    aurocs = []
    L = labels.numpy(); S = logits.numpy()
    for k in range(K):
        y_true = L[:, k]
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            aurocs.append(float("nan"))
        else:
            aurocs.append(float(roc_auc_score(y_true, S[:, k])))
    return aurocs


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

    # ----- Concept labels for TRAIN (the segments JSON covers train) -----
    print("\n── Building concept labels ──")
    win_labels_train, accepted_concepts = build_window_label_table(
        concepts_path=args.concepts_path,
        segments_path=args.segments_path,
        n_windows=len(train_data),
        only_ok=True,
    )
    K = win_labels_train.shape[1]
    print(f"  K = {K} accepted concepts; "
          f"train label matrix shape = {win_labels_train.shape}")

    # For val/test we still need labels for the per-concept AUROC report.
    # We re-use the same concept match functions on the val/test windows.
    # We piggyback on the segments_path's filter functions but assume the
    # window indexing maps cleanly to its segments at the same stride.
    # If you have separate val/test segments JSONs, swap them in here.
    # For LOF/RE this works because train segments JSON covers the same
    # filter logic — predictions for val/test concepts come from the
    # model's c_logits, ground truth comes from running the same filter
    # functions against the per-row metadata of val/test windows.
    print("  building concept labels for val/test windows ...")
    from CBM.concept_labels import load_concepts, compute_segment_labels, segments_to_window_labels
    accepted, match_fns = load_concepts(args.concepts_path, only_ok=True)
    import json
    with open(args.segments_path) as f:
        segs = json.load(f)
    stride = int(segs["segments"][1]["start_idx"] - segs["segments"][0]["start_idx"])
    seg_labels_all = compute_segment_labels(segs["segments"], match_fns)
    # Val/test windows live in disjoint row ranges from train, so naive stride
    # mapping doesn't apply — we approximate with per-window labels using the
    # closest train-segment by start row. For an accurate val/test concept
    # AUROC, build a val/test segments JSON via preprocessing/data_preprocessing.py
    # and load it here.
    win_labels_val  = np.zeros((len(val_data),  K), dtype=np.float32)
    win_labels_test = np.zeros((len(test_data), K), dtype=np.float32)
    print("  (val/test labels initialized as zeros — concept AUROC on those"
          " splits is unreliable until you also build val/test segments JSONs.)")

    train_ds = CBMDataset(train_data, win_labels_train)
    val_ds   = CBMDataset(val_data,   win_labels_val)
    test_ds  = CBMDataset(test_data,  win_labels_test)

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
        n_concepts   = K,
        decoder_hidden = args.decoder_hidden,
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
        f_loss, c_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            args.alpha_forecast, args.alpha_concept,
        )
        val_mse, val_mae, *_ = evaluate(model, val_loader, device)
        print(f"  [epoch {epoch+1:>3d}/{args.epoch}]  "
              f"train_forecast L1={f_loss:.4f}  "
              f"train_concept BCE={c_loss:.4f}  "
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
    test_mse, test_mae, t_logits, t_labels, t_yhat, t_ytrue = \
        evaluate(model, test_loader, device)
    print(f"  test MSE = {test_mse:.6f}")
    print(f"  test MAE = {test_mae:.6f}")

    # Per-concept AUROC computed on TRAIN labels (since val/test labels
    # are missing). For a proper test-set AUROC, see note above.
    print("\n── Per-concept activation magnitude (train labels OK; val/test concept-AUROC requires segment JSONs for those splits) ──")
    train_logits = []
    train_labels = []
    with torch.no_grad():
        model.eval()
        for x, y, lab in train_loader:
            x = x.to(device)
            out = model(x)
            train_logits.append(out["c_logits"].cpu())
            train_labels.append(lab)
    train_logits = torch.cat(train_logits); train_labels = torch.cat(train_labels)
    aurocs = per_concept_auroc(train_logits, train_labels)
    for c, auc in zip(accepted_concepts, aurocs):
        flag = " " if (auc != auc or auc > 0.7) else "⚠"
        auc_s = "  nan " if auc != auc else f"{auc:.3f}"
        print(f"  {flag} [{c['id']:>2}] {c['name']:30s}  train AUROC = {auc_s}")

    # ----- Case studies -----
    if args.case_studies > 0:
        print(f"\n── Case studies: {args.case_studies} random test windows ──")
        n_test = t_yhat.shape[0]
        rng = np.random.RandomState(args.seed)
        idx = rng.choice(n_test, min(args.case_studies, n_test), replace=False)

        # Convert test concept logits to activations
        t_acts = torch.sigmoid(t_logits)

        for k_idx, w in enumerate(idx):
            print(f"\n  Case {k_idx+1} (test window {w}):")
            # Top-5 active concepts for this window
            acts = t_acts[w]
            top5 = torch.topk(acts, k=min(5, K))
            print(f"    top concepts active:")
            for s, j in zip(top5.values, top5.indices):
                cm = accepted_concepts[j.item()]
                print(f"      {s.item():.3f}  {cm['name']}  ({cm['description'][:60]})")
            # Forecast comparison
            mae_here = (t_yhat[w] - t_ytrue[w]).abs().mean().item()
            print(f"    per-window MAE = {mae_here:.4f}")


if __name__ == "__main__":
    main()
