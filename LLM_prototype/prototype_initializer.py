"""
Prototype Initializer (function-based, Mahalanobis-typicality version).

Given:
  - the final prototype JSON from run_loop.py (each prototype has a Python
    `select_segments` function),
  - the summary-vector artifacts (z-scored vectors + StandardScaler) from
    summary_vector.py,
  - the pretrained encoder checkpoint,

this script produces the three init tensors expected by
`exp/prototype_learning.py`:

    init_embedding.pt  shape (P, prototype_dim)        ← μ_init
    init_value.pt      shape (P, output_length, 1)     ← p_init
    anchor_mask.pt     shape (P,)                      ← 1.0 for LLM-anchored,
                                                         0.0 for blank

Per prototype, the steps are:
    1. Run the LLM-supplied function on the full segment list to obtain the
       matching index set M.
    2. Apply Mahalanobis-to-centroid in summary-vector space to pick the
       `typical_num` most typical members of M.
    3. For each of those typical segments, run the pretrained encoder over
       the full LOF_Dataset window and average over (instances × time) to
       get μ_i. Average the forecast-window targets to get p_i.

Optional `--num_blank` blank prototypes are appended (Kaiming μ, N(0,1) p).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataloader.dataloader import LOF_Dataset
from model.embedding import Embedding_Learning

# Local helpers from the proposing loop
sys.path.insert(0, os.path.join(_REPO_ROOT, "prototype_proposing_loop"))
from function_executor import execute_select_function
from summary_vector import load_artifacts


# ---------------------------------------------------------------------------
# Mahalanobis-to-centroid typicality
# ---------------------------------------------------------------------------

def mahalanobis_topk_typical(matched_indices: list, summary_vectors_z: np.ndarray,
                              typical_num: int) -> list:
    """Pick the `typical_num` indices closest to the matched-set centroid in
    Mahalanobis distance. If |M| <= typical_num, return M unchanged.

    Implementation notes:
      - vectors are already z-scored (per-dimension) by the StandardScaler
        from summary_vector.py, so the marginal scale problem is already
        handled.
      - we use Ledoit-Wolf shrinkage to stabilise the covariance estimate
        when |M| is small relative to D.
      - we use the pseudo-inverse for safety when the shrinkage matrix is
        still ill-conditioned.
    """
    from sklearn.covariance import LedoitWolf

    if len(matched_indices) == 0:
        return []
    if len(matched_indices) <= typical_num:
        return list(matched_indices)

    S = summary_vectors_z[matched_indices]                    # (|M|, D)
    mu = S.mean(axis=0)                                       # (D,)
    cov = LedoitWolf().fit(S).covariance_                     # (D, D)
    inv_cov = np.linalg.pinv(cov)                             # (D, D)
    diffs = S - mu                                            # (|M|, D)
    d2 = np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs)      # (|M|,)
    order = np.argsort(d2)                                    # ascending: closest first
    return [matched_indices[i] for i in order[:typical_num].tolist()]


# ---------------------------------------------------------------------------
# Encoder pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_segments(model: nn.Module, dataset: LOF_Dataset,
                     start_indices: list, device: str) -> torch.Tensor | None:
    """Run the encoder on a list of LOF_Dataset window indices.

    Each segment in the segment-bank JSON has a `start_idx` field — that's
    the row index in the original CSV. LOF_Dataset's own __getitem__ uses
    the same indexing convention, so we pass start_idx directly.

    The encoder now returns a time-averaged ``(B, mid_dim)`` tensor (the
    averaging happens inside ``ConvEmbedding_Learning.forward``), so each
    segment contributes a single ``(mid_dim,)`` vector here.
    """
    model.eval()
    latents = []
    for s in start_indices:
        if s < 0 or s >= len(dataset):
            print(f"  [warn] start_idx {s} out of range; skipping")
            continue
        data, _ = dataset[s]
        data = data.unsqueeze(0).to(device)
        _, encoder = model(data)             # (1, mid_dim) for ConvEmbedding,
                                              # (1, output_length, mid_dim) for MixerEmbedding
        if encoder.dim() == 3:
            encoder = encoder.mean(dim=1)    # average over time for Mixer encoder
        latents.append(encoder.squeeze(0).cpu())  # (mid_dim,)
    if not latents:
        return None
    return torch.stack(latents, dim=0)        # (K, mid_dim)


def forecast_curves(dataset: LOF_Dataset, start_indices: list,
                     input_length: int, output_length: int) -> torch.Tensor | None:
    """Stack the scaled target curves of K forecast windows. Uses
    dataset.output, which is already StandardScaler-fit."""
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
    return torch.from_numpy(np.stack(out, axis=0)).float()    # (K, L, 1)


# ---------------------------------------------------------------------------
# Per-prototype build
# ---------------------------------------------------------------------------

def build_one_prototype(spec: dict, segments: list,
                          summary_vectors_z: np.ndarray, dataset: LOF_Dataset,
                          model: nn.Module, device: str,
                          input_length: int, output_length: int,
                          typical_num: int) -> dict | None:
    """Compute (μ, p) for a single prototype spec."""
    name = spec.get("name", f"proto_{spec.get('prototype_id', '?')}")

    # 1. Run the LLM-supplied function
    exec_result = execute_select_function(spec.get("function", ""), segments)
    matched = exec_result["matched_indices"]
    if matched is None:
        print(f"  [skip] {name}: function execution failed -> {exec_result['error']}")
        return None
    if not matched:
        print(f"  [skip] {name}: function returned 0 segments")
        return None

    # 2. Mahalanobis-to-centroid typicality
    typical = mahalanobis_topk_typical(matched, summary_vectors_z, typical_num)
    if not typical:
        return None

    # Convert segment-bank indices (positions in segments[]) to actual
    # LOF_Dataset windowed-row indices (segments[i]["start_idx"]).
    start_indices = [segments[i]["start_idx"] for i in typical]

    # 3. Encode + average -> μ
    # The encoder already mean-pools over the time axis, so each row of
    # ``latents`` is one (mid_dim,) vector. We just average over instances.
    latents = encode_segments(model, dataset, start_indices, device)
    if latents is None:
        return None
    mu = latents.mean(dim=0)                                   # (D,)

    # 4. Average forecast curves -> p
    fcs = forecast_curves(dataset, start_indices, input_length, output_length)
    if fcs is None:
        return None
    p = fcs.mean(dim=0)                                        # (L, 1)

    return {
        "name": name,
        "n_matched": len(matched),
        "n_typical": latents.shape[0],
        "mu": mu, "p": p,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build (μ, p) init tensors from final prototype functions."
    )
    parser.add_argument("--final_prototypes", type=str, required=True,
                        help="final_prototypes_<timestamp>.json from run_loop.py")
    parser.add_argument("--segments_path", type=str, required=True,
                        help="Part-A full-segments JSON")
    parser.add_argument("--summary_dir", type=str, required=True,
                        help="Summary-vector artifacts dir from build_and_cluster()")
    parser.add_argument("--dataset", type=str, required=True,
                        help="CSV used during encoder training")
    parser.add_argument("--pretrain_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--typical_num", type=int, default=20,
                        help="K most typical segments per prototype")
    parser.add_argument("--num_blank", type=int, default=2)
    parser.add_argument("--random_value_init", action="store_true",
                        help="Skip p_init computation and use torch.randn for "
                             "prototype_value. Recommended whenever the model's "
                             "learned prototype_value is observed to differ "
                             "strongly from matched segments' averaged forecast "
                             "(typical for ProtoTS, where prototype_value acts as "
                             "a basis vector, not a literal forecast). Pair with "
                             "--anchor_lambda_val 0 in prototype_learning.py.")
    # encoder hyperparameters
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

    # ---- Load encoder ----
    model = Embedding_Learning(
        input_dim=args.input_dim, mid_dim=args.embed_dim,
        ext_dim=args.input_dim - args.output_dim, output_dim=args.output_dim,
        bottle_dim_p=args.bottle_dim_p,
        input_length=args.input_length, output_length=args.output_length,
        num_blocks=args.num_layer, dropout=args.dropout,
        norm_type="B", use_rev_in=1,
    ).to(device)
    model.load_state_dict(torch.load(args.pretrain_checkpoint, map_location=device))
    print(f"Loaded encoder      <- {args.pretrain_checkpoint}")

    # ---- Load training dataset (windowed) ----
    train_dataset = LOF_Dataset(args.input_length, args.output_length,
                                 "train", args.input_dim, args.dataset)
    print(f"Loaded {len(train_dataset)} training windows from {args.dataset}")

    # ---- Load Part-A segments ----
    with open(args.segments_path) as f:
        seg_data = json.load(f)
    segments = seg_data["segments"]
    print(f"Loaded {len(segments)} segment summaries")

    # ---- Load summary-vector artifacts (z-scored vectors used for typicality) ----
    summary = load_artifacts(args.summary_dir)
    summary_vectors_z = summary["vectors_z"]
    print(f"Loaded summary vectors: {summary_vectors_z.shape}")
    if summary_vectors_z.shape[0] != len(segments):
        raise RuntimeError(
            f"summary vectors ({summary_vectors_z.shape[0]}) do not match "
            f"segment count ({len(segments)}). Did you regenerate either?"
        )

    # ---- Load final prototypes ----
    with open(args.final_prototypes) as f:
        final = json.load(f)
    prototypes = final["prototypes"]
    print(f"Loaded {len(prototypes)} prototypes")

    # ---- Build μ, p per prototype ----
    mu_list, p_list, names = [], [], []
    for spec in prototypes:
        result = build_one_prototype(
            spec=spec, segments=segments,
            summary_vectors_z=summary_vectors_z,
            dataset=train_dataset, model=model, device=device,
            input_length=args.input_length,
            output_length=args.output_length,
            typical_num=args.typical_num,
        )
        if result is None:
            continue
        mu_list.append(result["mu"])
        p_list.append(result["p"])
        names.append(result["name"])
        print(f"  ✓ {result['name']:30s}  matched={result['n_matched']:4d}  "
              f"typical={result['n_typical']:3d}  ‖μ‖={result['mu'].norm():.3f}  "
              f"mean(p)={result['p'].mean():+.3f}")

    if not mu_list:
        raise RuntimeError("No prototypes could be initialised. Check function "
                           "execution and matching counts.")

    # ---- Stack + append blanks ----
    mu_init = torch.stack(mu_list, dim=0)                      # (N, D)
    N = mu_init.shape[0]

    if args.random_value_init:
        # Override the averaged-forecast p_init with random tensors. The
        # rationale is that prototype_value learns as a mixture-basis vector,
        # not as a literal forecast curve, so a "real" forecast init places
        # it in a coordinate system the model has to leave during training.
        p_init = torch.randn(N, args.output_length, args.output_dim)
        print(f"[--random_value_init] overrode p_init with torch.randn"
              f"({N}, {args.output_length}, {args.output_dim})")
    else:
        p_init = torch.stack(p_list, dim=0)                    # (N, L, 1)

    M = args.num_blank

    if M > 0:
        mu_blank = torch.empty(M, args.embed_dim)
        nn.init.kaiming_uniform_(mu_blank, nonlinearity="relu")
        p_blank = torch.randn(M, args.output_length, args.output_dim)
        mu_init = torch.cat([mu_init, mu_blank], dim=0)
        p_init  = torch.cat([p_init,  p_blank],  dim=0)

    anchor_mask = torch.zeros(N + M)
    anchor_mask[:N] = 1.0

    # ---- Save ----
    torch.save(mu_init,     os.path.join(args.output_dir, "init_embedding.pt"))
    torch.save(p_init,      os.path.join(args.output_dir, "init_value.pt"))
    torch.save(anchor_mask, os.path.join(args.output_dir, "anchor_mask.pt"))
    with open(os.path.join(args.output_dir, "prototype_names.json"), "w") as f:
        json.dump({"names": names, "num_anchored": N, "num_blank": M,
                   "total": N + M}, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Wrote {N + M} prototype inits ({N} LLM-anchored + {M} blank)")
    print(f"  init_embedding.pt   shape {tuple(mu_init.shape)}")
    print(f"  init_value.pt       shape {tuple(p_init.shape)}")
    print(f"  anchor_mask.pt      shape {tuple(anchor_mask.shape)}")
    print(f"\nRemember to pass --prototype_num {N + M} to prototype_learning.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
