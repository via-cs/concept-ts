"""
Part D: Prototype Initializer

Given N per-prototype instance JSON files (each produced by instance_finder.py)
and a *pretrained* ProtoTS encoder checkpoint, this script produces the
initialization tensors that the prototype-layer training loop can load:

    init_embedding.pt  shape (N+M, prototype_dim)       ← μ_init
    init_value.pt      shape (N+M, output_length, 1)    ← p_init
    anchor_mask.pt     shape (N+M,)                     ← 1 for LLM prototypes
                                                         0 for the M blanks

For each LLM-proposed prototype i (1 .. N):
    1. Read its top-K matching lookback/forecast windows from *_instances.json.
    2. Build LOF_Dataset-style input tensors (same format the encoder was
       trained on) and pass them through the encoder to get latents of
       shape (K, output_length, mid_dim).
    3. μ_i ← mean over (K instances × output_length timesteps) of the latents.
    4. p_i ← mean over K instances of the forecast-window target curve,
             expressed in the same scaled space the prototype layer operates in.

Then append M (default 2) "blank" prototypes: random μ (Kaiming uniform) and
random p (standard normal), mirroring the original ProtoTS initialization.

Usage:
    python LLM_prototype/prototype_initializer.py \
        --instances_dir LLM_prototype/output \
        --dataset dataset/LOF/RE.csv \
        --pretrain_checkpoint ./checkpoint/embedding_learning/RE_..._BDP0.1.pth \
        --output_dir LLM_prototype/output/init \
        --num_blank 2 \
        --top_k 20
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn

from dataloader.dataloader import LOF_Dataset
from model.embedding import Embedding_Learning


@torch.no_grad()
def encode_instances(model, dataset, start_indices, device):
    """Run the encoder on a list of instance windows.

    Returns a tensor of shape (K, output_length, mid_dim) in the encoder's
    internal latent space (rev_in normalized — same space as during training).
    """
    model.eval()
    latents = []
    for s in start_indices:
        if s < 0 or s >= len(dataset):
            print(f"  [warn] start_idx {s} out of range [0,{len(dataset)}); skipping")
            continue
        data, _ = dataset[s]
        data = data.unsqueeze(0).to(device)
        _, encoder = model(data)           # encoder: (1, output_length, mid_dim)
        latents.append(encoder.squeeze(0).cpu())
    if not latents:
        return None
    return torch.stack(latents, dim=0)    # (K, L, D)


def forecast_windows(dataset, start_indices, input_length, output_length):
    """Stack the scaled target curves of the K forecast windows.

    Uses dataset.output, which LOF_Dataset has already StandardScaler-fit on
    the training portion — this matches the convention of the existing
    ``--init_weight`` path in prototype_learning.py.

    Returns a tensor of shape (K_valid, output_length, 1).
    """
    scaled = np.asarray(dataset.output).reshape(-1, 1)
    out = []
    for s in start_indices:
        fs = s + input_length
        fe = fs + output_length
        if fe > len(scaled):
            continue
        out.append(scaled[fs:fe])
    if not out:
        return None
    return torch.from_numpy(np.stack(out, axis=0)).float()   # (K, L, 1)


def build_one_prototype_from_indices(
    name, start_indices, model, dataset, device,
    input_length, output_length, top_k=None,
):
    """Compute (μ_i, p_i) for a single prototype given its matching
    start_indices. This is the library entry point used by the
    orchestrator; the CLI path below wraps it with JSON loading.
    """
    if top_k is not None:
        start_indices = start_indices[:top_k]
    if not start_indices:
        return None

    latents = encode_instances(model, dataset, start_indices, device)
    if latents is None:
        return None

    mu = latents.mean(dim=(0, 1))                                # (D,)

    forecasts = forecast_windows(
        dataset, start_indices, input_length, output_length
    )
    if forecasts is None:
        return None
    p = forecasts.mean(dim=0)                                    # (L, 1)

    return {"name": name, "n_instances": latents.shape[0],
            "mu": mu, "p": p}


def build_one_prototype(
    instance_json_path, model, dataset, device,
    input_length, output_length, top_k=None,
):
    """CLI helper: read one *_instances.json and dispatch."""
    with open(instance_json_path) as f:
        info = json.load(f)

    selected = info["instance_finding"]["selected_instances"]
    start_indices = [inst["start_idx"] for inst in selected]
    name = info["prototype_spec"].get(
        "name", os.path.splitext(os.path.basename(instance_json_path))[0]
    )
    return build_one_prototype_from_indices(
        name, start_indices, model, dataset, device,
        input_length, output_length, top_k,
    )


def save_init_tensors(mu_list, p_list, names, output_dir,
                      num_blank, embed_dim, output_length, output_dim):
    """Assemble the (N+M) init tensors and write them to disk.

    Returns (mu_init, p_init, anchor_mask).
    """
    os.makedirs(output_dir, exist_ok=True)

    N = len(mu_list)
    if N == 0:
        raise RuntimeError("No prototypes could be initialized.")
    M = num_blank

    mu_init = torch.stack(mu_list, dim=0)        # (N, D)
    p_init  = torch.stack(p_list, dim=0)         # (N, L, 1)

    # Append M blank prototypes (Kaiming μ, N(0,1) p — matches Prototype_Layer).
    if M > 0:
        mu_blank = torch.empty(M, embed_dim)
        nn.init.kaiming_uniform_(mu_blank, nonlinearity="relu")
        p_blank = torch.randn(M, output_length, output_dim)
        mu_init = torch.cat([mu_init, mu_blank], dim=0)
        p_init  = torch.cat([p_init, p_blank], dim=0)

    anchor_mask = torch.zeros(N + M)
    anchor_mask[:N] = 1.0

    torch.save(mu_init,     os.path.join(output_dir, "init_embedding.pt"))
    torch.save(p_init,      os.path.join(output_dir, "init_value.pt"))
    torch.save(anchor_mask, os.path.join(output_dir, "anchor_mask.pt"))
    with open(os.path.join(output_dir, "prototype_names.json"), "w") as f:
        json.dump({"names": names, "num_blank": M, "total": N + M}, f, indent=2)

    return mu_init, p_init, anchor_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances_dir", type=str, required=True,
                        help="Directory containing *_instances.json files")
    parser.add_argument("--dataset", type=str, required=True,
                        help="CSV used during encoder training")
    parser.add_argument("--pretrain_checkpoint", type=str, required=True,
                        help="Pretrained encoder checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to dump init_embedding.pt / init_value.pt")
    parser.add_argument("--num_blank", type=int, default=2,
                        help="M — blank prototypes appended (random init)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Cap the per-prototype top-K; default uses every "
                             "instance saved by instance_finder.py")

    parser.add_argument("--input_length", type=int, default=96 * 4)
    parser.add_argument("--output_length", type=int, default=96)
    parser.add_argument("--input_dim", type=int, default=23)
    parser.add_argument("--output_dim", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bottle_dim_p", type=float, default=0.1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ["device"] = device
    os.makedirs(args.output_dir, exist_ok=True)

    # --------------------------------------------------------------
    # 1. Pretrained encoder
    # --------------------------------------------------------------
    model = Embedding_Learning(
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
    model.load_state_dict(torch.load(args.pretrain_checkpoint, map_location=device))
    model.eval()
    print(f"Loaded encoder from {args.pretrain_checkpoint}")

    # --------------------------------------------------------------
    # 2. Training dataset (instance_finder indices are relative to train split)
    # --------------------------------------------------------------
    train_dataset = LOF_Dataset(
        args.input_length, args.output_length,
        "train", args.input_dim, args.dataset,
    )
    print(f"Loaded {len(train_dataset)} training windows from {args.dataset}")

    # --------------------------------------------------------------
    # 3. Iterate LLM prototypes
    # --------------------------------------------------------------
    json_files = sorted(glob.glob(os.path.join(args.instances_dir, "*_instances.json")))
    if not json_files:
        raise FileNotFoundError(f"No *_instances.json found under {args.instances_dir}")

    mu_list, p_list, names = [], [], []
    for jf in json_files:
        result = build_one_prototype(
            jf, model, train_dataset, device,
            args.input_length, args.output_length, args.top_k,
        )
        if result is None:
            print(f"[skip] {os.path.basename(jf)} — no usable instances")
            continue
        mu_list.append(result["mu"])
        p_list.append(result["p"])
        names.append(result["name"])
        print(f"  ✓ {result['name']:30s}  n={result['n_instances']:3d}  "
              f"‖μ‖={result['mu'].norm():.3f}  mean(p)={result['p'].mean():+.3f}")

    mu_init, p_init, anchor_mask = save_init_tensors(
        mu_list, p_list, names, args.output_dir,
        num_blank=args.num_blank, embed_dim=args.embed_dim,
        output_length=args.output_length, output_dim=args.output_dim,
    )
    print("\n" + "=" * 60)
    print(f"Wrote {mu_init.shape[0]} prototype inits "
          f"({len(mu_list)} LLM-anchored + {args.num_blank} blank)")
    print(f"  init_embedding.pt   shape {tuple(mu_init.shape)}")
    print(f"  init_value.pt       shape {tuple(p_init.shape)}")
    print(f"  anchor_mask.pt      shape {tuple(anchor_mask.shape)}")
    print(f"Remember to pass --prototype_num {mu_init.shape[0]} "
          f"to prototype_learning.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
