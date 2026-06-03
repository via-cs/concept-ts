"""
DeepCluster / SwAV-style cluster-supervision loss for encoder pretraining.

The encoder is trained with two objectives in parallel:

    L_total = L_forecast + alpha * L_cluster

where ``L_cluster`` is a cross-entropy that pulls each segment's encoder
output toward a learnable centroid associated with its precomputed cluster.

Cluster assignments are produced once at training start by K-means on a
simple per-segment summary (mean of each input channel over the lookback
window). The centroids themselves are learnable nn.Parameters that drift
during training to match the encoder's output distribution per cluster.

After training, ``cluster_purity_probe`` reports intra-cluster vs
inter-cluster cosine similarity in encoder space — the diagnostic that
tells you whether the cluster loss actually shaped encoder geometry or
whether ``alpha`` was too small to bite.

Public surface
--------------
    compute_cluster_assignments(train_dataset, K, ...)
        K-means on per-window lookback means. Returns int array of length
        len(train_dataset).
    ClusteredDataset(base, assignments)
        Thin wrapper that adds the cluster index to each ``__getitem__``.
    ClusterLoss(K, embed_dim)
        Module with learnable centroids; forward computes cross-entropy.
    cluster_purity_probe(encoder_outputs, assignments, K, ...)
        Intra/inter cosine similarity by cluster.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Step 1: compute K-means cluster assignments from raw segment statistics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_cluster_assignments(
    train_dataset: Dataset,
    K: int,
    *,
    seed: int = 0,
    feature_strategy: str = "lookback_mean",
    verbose: bool = True,
) -> np.ndarray:
    """K-means on per-window features → returns one cluster index per window.

    Parameters
    ----------
    train_dataset : LOF_Dataset (already in 'train' mode)
    K             : number of clusters (typically matches prototype_num)
    feature_strategy : "lookback_mean" — average each channel over the
                       lookback portion of the window. Cheap, dataset-agnostic.

    Returns
    -------
    assignments : (N,) int array with values in [0, K).
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    n_windows    = len(train_dataset)
    in_dim       = train_dataset.in_dim
    input_length = train_dataset.input_length

    if verbose:
        print(f"[cluster_supervision] extracting per-window features for "
              f"{n_windows} windows, in_dim={in_dim}, lookback={input_length} …")

    feats = np.zeros((n_windows, in_dim), dtype=np.float32)
    for i in range(n_windows):
        x, _ = train_dataset[i]                        # x: (length, in_dim) tensor
        lookback = x[:input_length]                    # (input_length, in_dim)
        feats[i] = lookback.mean(dim=0).cpu().numpy()

    # Z-score per channel so K-means doesn't get dominated by one large-scale dim.
    feats_z = StandardScaler().fit_transform(feats)

    if verbose:
        print(f"[cluster_supervision] running K-means(K={K}) on "
              f"({n_windows}, {in_dim}) feature matrix …")
    km = KMeans(n_clusters=K, random_state=seed, n_init=10)
    assignments = km.fit_predict(feats_z).astype(np.int64)

    if verbose:
        counts = np.bincount(assignments, minlength=K)
        print(f"[cluster_supervision] cluster sizes: "
              f"{counts.tolist()}  "
              f"(min={counts.min()}, max={counts.max()}, "
              f"mean={counts.mean():.0f}, "
              f"imbalance={counts.max()/counts.min():.2f}x)")

    return assignments


# ---------------------------------------------------------------------------
# Step 1b: cluster assignments derived from LLM-supplied filter functions
# ---------------------------------------------------------------------------

def compute_cluster_assignments_from_llm(
    train_dataset,
    final_prototypes_path: str,
    segments_path: str,
    *,
    segment_stride: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Build per-window cluster assignments using the LLM-proposed
    ``select_segments`` filter functions from ``final_prototypes_*.json``.

    Steps:
        1. Load segments JSON (one entry per Part-A segment; carries
           ``start_idx`` and the stat dicts the filter functions reference).
        2. Load final_prototypes JSON (list of LLM-proposed prototypes,
           each with a ``function`` string and a ``name``).
        3. Execute each filter on the segment list → matching indices.
        4. Per-segment assignment: cluster k if exactly one filter matches;
           ``-1`` (ignored by the loss) if zero or multiple filters match.
        5. Map per-segment assignments to per-window assignments by floor-
           dividing each window's start index by ``segment_stride`` (inferred
           from the JSON if not supplied; for LOF this is typically 96).

    Returns
    -------
    assignments : (N_windows,) int64 array with values in {-1} ∪ [0, K).
    """
    import json
    import sys
    from pathlib import Path

    # The function_executor lives under prototype_proposing_loop/. Make sure
    # its parent is on sys.path even when cluster_supervision.py is invoked
    # as `python -m exp.cluster_supervision` from the project root.
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from prototype_proposing_loop.function_executor import execute_select_function

    # ----- load segments -----
    with open(segments_path) as f:
        segments_data = json.load(f)
    segments = segments_data["segments"]
    n_segments = len(segments)
    if verbose:
        print(f"[cluster_supervision/llm] loaded {n_segments} segments from {segments_path}")

    # Infer segment stride from the first two segments' start indices.
    if segment_stride is None:
        if n_segments < 2:
            raise ValueError(
                "Cannot infer segment_stride with <2 segments; pass --segment_stride explicitly."
            )
        segment_stride = int(segments[1]["start_idx"] - segments[0]["start_idx"])
    if verbose:
        print(f"[cluster_supervision/llm] using segment_stride={segment_stride}")

    # ----- load LLM prototype JSON -----
    with open(final_prototypes_path) as f:
        proto_data = json.load(f)
    if isinstance(proto_data, list):
        prototypes = proto_data
    elif "prototypes" in proto_data:
        prototypes = proto_data["prototypes"]
    elif "final_prototypes" in proto_data:
        prototypes = proto_data["final_prototypes"]
    else:
        raise ValueError(
            f"Cannot find prototype list inside {final_prototypes_path}. "
            f"Top-level keys: {list(proto_data.keys())}"
        )
    K = len(prototypes)
    if verbose:
        names = [p.get("name", f"proto_{i}") for i, p in enumerate(prototypes)]
        print(f"[cluster_supervision/llm] loaded {K} LLM prototypes: {names}")

    # ----- run each filter on the segment list -----
    seg_assignments = np.full(n_segments, -1, dtype=np.int64)
    ambiguous_count = 0
    for k, proto in enumerate(prototypes):
        code = proto.get("function", "")
        if not code.strip():
            if verbose:
                print(f"  [skip] prototype {k} ({proto.get('name', '?')}): empty function")
            continue
        result = execute_select_function(code, segments)
        matched = result.get("matched_indices") or []
        if result.get("error"):
            if verbose:
                print(f"  [warn] prototype {k} filter errored: "
                      f"{result['error'].splitlines()[0][:120]}")
            continue
        for seg_idx in matched:
            if seg_assignments[seg_idx] == -1:
                seg_assignments[seg_idx] = k
            else:
                # Multi-match — mark ambiguous (-1) so neither cluster claims it.
                seg_assignments[seg_idx] = -1
                ambiguous_count += 1

    if verbose:
        per_k = [int((seg_assignments == k).sum()) for k in range(K)]
        n_assigned   = int((seg_assignments >= 0).sum())
        n_unassigned = n_segments - n_assigned
        print(f"[cluster_supervision/llm] segment-level coverage: "
              f"{n_assigned}/{n_segments} assigned ({100*n_assigned/n_segments:.1f}%), "
              f"{n_unassigned} unassigned (no-match + {ambiguous_count} multi-match → -1)")
        print(f"  per-cluster segment counts: {per_k}")

    # ----- map segment-level → window-level assignments -----
    n_windows = len(train_dataset)
    window_assignments = np.full(n_windows, -1, dtype=np.int64)
    # Window w spans rows [w, w + length). We map it to segment ⌊w/stride⌋ —
    # the segment whose starting row is closest *at or below* w. Adjacent
    # windows therefore share assignment with the same segment until w
    # crosses the next stride boundary, which matches the non-overlapping
    # segment grid the preprocessing pipeline produces.
    for w in range(n_windows):
        seg_idx = w // segment_stride
        if seg_idx < n_segments:
            window_assignments[w] = seg_assignments[seg_idx]

    if verbose:
        n_assigned_w = int((window_assignments >= 0).sum())
        per_k_w = [int((window_assignments == k).sum()) for k in range(K)]
        print(f"[cluster_supervision/llm] window-level coverage: "
              f"{n_assigned_w}/{n_windows} ({100*n_assigned_w/n_windows:.1f}%) supervised")
        print(f"  per-cluster window counts: {per_k_w}")

    return window_assignments


# ---------------------------------------------------------------------------
# Step 2: dataset wrapper that returns (x, y, cluster_index) per sample
# ---------------------------------------------------------------------------

class ClusteredDataset(Dataset):
    """Wraps any (x, y)-returning dataset to also yield the precomputed
    cluster index. The base dataset's order must match ``assignments``."""

    def __init__(self, base: Dataset, assignments: np.ndarray):
        assert len(base) == len(assignments), (
            f"len(base)={len(base)} != len(assignments)={len(assignments)}"
        )
        self.base = base
        self.assignments = torch.from_numpy(assignments).long()

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        return x, y, self.assignments[idx]


# ---------------------------------------------------------------------------
# Step 3: the cluster-supervision loss module
# ---------------------------------------------------------------------------

class ClusterLoss(nn.Module):
    """Cross-entropy between encoder outputs and learnable cluster centroids.

    ``centroids`` is a (K, embed_dim) nn.Parameter, initialized with small
    random values and updated through backprop alongside the encoder.

    Forward expects the encoder output already reduced to (B, embed_dim).
    If your encoder returns (B, T, D), call ``encoder.mean(dim=1)`` before
    passing in (this mirrors what prototype_initializer.py does).
    """

    def __init__(self, K: int, embed_dim: int, init_scale: float = 0.1):
        super().__init__()
        self.K = K
        self.embed_dim = embed_dim
        self.centroids = nn.Parameter(torch.randn(K, embed_dim) * init_scale)

    def forward(self, encoder_output_2d: torch.Tensor,
                cluster_targets: torch.Tensor) -> torch.Tensor:
        assert encoder_output_2d.dim() == 2, (
            f"ClusterLoss expects 2D input (B, D); got {encoder_output_2d.dim()}D. "
            f"Reduce 3D encoder outputs with .mean(dim=1) before calling."
        )
        # Mask out samples with a -1 sentinel (LLM-filter unassigned / ambiguous).
        # Returns zero (with grad) if the entire batch is unassigned, so the
        # forecast loss still backpropagates normally.
        valid = cluster_targets >= 0
        if not valid.any():
            return encoder_output_2d.sum() * 0.0
        enc_v = encoder_output_2d[valid]
        tgt_v = cluster_targets[valid]
        logits = enc_v @ self.centroids.T                       # (B_valid, K)
        return F.cross_entropy(logits, tgt_v)


# ---------------------------------------------------------------------------
# Step 4: intra-vs-inter cosine probe — run after training to verify the
# cluster loss actually shaped encoder geometry
# ---------------------------------------------------------------------------

@torch.no_grad()
def cluster_purity_probe(
    encoder_outputs: torch.Tensor,
    assignments: np.ndarray,
    K: int,
    *,
    sample_per_cluster: int = 200,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """For each cluster, compute mean intra-cluster cosine (members vs
    members) and mean inter-cluster cosine (members vs non-members).
    Healthy clustering shows intra >> inter; if intra ≈ inter, the cluster
    loss did not shape geometry and ``alpha`` should be increased.

    Returns
    -------
    dict with keys:
        intra_per_cluster, inter_per_cluster   — (K,) lists
        intra_mean, inter_mean, gap            — scalars
        verdict                                — short str
    """
    enc = encoder_outputs.float()
    enc = enc / enc.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    assignments_t = torch.as_tensor(assignments, dtype=torch.long, device=enc.device)

    g = torch.Generator(device=enc.device).manual_seed(seed)

    intra_list, inter_list = [], []
    for k in range(K):
        in_idx  = (assignments_t == k).nonzero(as_tuple=True)[0]
        out_idx = (assignments_t != k).nonzero(as_tuple=True)[0]
        if in_idx.numel() < 2 or out_idx.numel() < 2:
            intra_list.append(float("nan"))
            inter_list.append(float("nan"))
            continue

        n_in  = min(sample_per_cluster, in_idx.numel())
        n_out = min(sample_per_cluster, out_idx.numel())
        in_sel  = in_idx[torch.randperm(in_idx.numel(),  generator=g)[:n_in]]
        out_sel = out_idx[torch.randperm(out_idx.numel(), generator=g)[:n_out]]

        E_in  = enc[in_sel]
        E_out = enc[out_sel]

        intra_mat = E_in  @ E_in.T          # (n_in, n_in)
        inter_mat = E_in  @ E_out.T         # (n_in, n_out)

        # Drop the diagonal of intra (self-cosine = 1).
        mask = ~torch.eye(n_in, dtype=torch.bool, device=enc.device)
        intra_list.append(intra_mat[mask].mean().item())
        inter_list.append(inter_mat.mean().item())

    intra_arr = np.array(intra_list, dtype=np.float64)
    inter_arr = np.array(inter_list, dtype=np.float64)

    finite_mask = np.isfinite(intra_arr) & np.isfinite(inter_arr)
    intra_mean = float(intra_arr[finite_mask].mean()) if finite_mask.any() else float("nan")
    inter_mean = float(inter_arr[finite_mask].mean()) if finite_mask.any() else float("nan")
    gap = intra_mean - inter_mean

    # Heuristic verdict — these thresholds are rough; tune per-dataset.
    if gap > 0.30:
        verdict = "STRONG clustering  (intra − inter > 0.30)"
    elif gap > 0.15:
        verdict = "moderate clustering"
    elif gap > 0.05:
        verdict = "weak clustering — consider raising alpha"
    else:
        verdict = "CLUSTERING DID NOT TAKE — alpha too small or feature space collapsed"

    result = {
        "intra_per_cluster": intra_list,
        "inter_per_cluster": inter_list,
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "gap":        gap,
        "verdict":    verdict,
    }

    if verbose:
        print()
        print("───────── cluster purity probe ─────────")
        print(f"  per-cluster intra cosine: "
              f"{[round(v, 3) for v in intra_list]}")
        print(f"  per-cluster inter cosine: "
              f"{[round(v, 3) for v in inter_list]}")
        print(f"  mean intra = {intra_mean:+.4f}")
        print(f"  mean inter = {inter_mean:+.4f}")
        print(f"  gap        = {gap:+.4f}   ← {verdict}")
        print("────────────────────────────────────────")

    return result
