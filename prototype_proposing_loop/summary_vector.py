"""
Per-segment summary vectors + clustering.

A summary vector is a fixed-length numerical fingerprint of a segment, built
deterministically from the precomputed statistics in each segment dict.
We use it for three things downstream:

  1. K-means clustering, so the LLM prompt can describe the dataset's
     natural regimes instead of just dumping global statistics.
  2. Mahalanobis-to-centroid typicality scoring, when picking top-K most
     typical segments inside each prototype's matching set.
  3. (Future) Cross-prototype similarity in feature space.

The vector concatenates the following blocks (default schema):
  Block A — LOOKBACK_CONTINUOUS    : per continuous channel × {mean, std, slope, range}
  Block B — LOOKBACK_CATEGORICAL   : per categorical channel × {mode}
  Block C — FORECAST_CONTINUOUS    : per continuous channel × {mean, std, slope, range}
  Block D — FORECAST_CATEGORICAL   : per categorical channel × {mode}
  Block E — TRANSITION             : forecast_mean − lookback_mean, key channels
  Block F — BOUNDARY_CALENDAR      : hour at forecast start + cyclic encodings

All blocks are computed from the segment dicts produced by
`preprocessing/data_preprocessing.py`, so this module never touches the raw
time series — segments come in, summary vectors come out.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Channel partitioning
# ---------------------------------------------------------------------------

def partition_channels(channel_meta: dict):
    """Split channel metadata into ordered (continuous, categorical) lists."""
    continuous = [c for c, m in channel_meta.items() if m["type"] == "continuous"]
    categorical = [c for c, m in channel_meta.items() if m["type"] == "discrete"]
    return continuous, categorical


# ---------------------------------------------------------------------------
# Per-segment vector
# ---------------------------------------------------------------------------

def build_summary_vector(segment: dict, continuous: list, categorical: list,
                          delta_channels: list = None) -> tuple[np.ndarray, list]:
    """Build the summary vector for a single segment.

    Returns
    -------
    vec : np.ndarray of shape (D,)
    dim_names : list of length D, parallel feature labels (handy for inspection)
    """
    if delta_channels is None:
        # Default: every continuous channel except the target gets a transition delta
        delta_channels = [c for c in continuous if c != "load"]

    parts: list[float] = []
    names: list[str] = []

    def _push(name: str, value):
        parts.append(float(value))
        names.append(name)

    lb = segment.get("lookback_stats", {})
    fc = segment.get("forecast_stats", {})

    # Block A — lookback continuous
    for ch in continuous:
        s = lb.get(ch, {})
        _push(f"lb_{ch}_mean",  s.get("mean", 0.0))
        _push(f"lb_{ch}_std",   s.get("std", 0.0))
        _push(f"lb_{ch}_slope", s.get("slope", 0.0))
        _push(f"lb_{ch}_range", s.get("range", 0.0))

    # Block B — lookback categorical
    for ch in categorical:
        s = lb.get(ch, {})
        _push(f"lb_{ch}_mode", s.get("mode", 0))

    # Block C — forecast continuous
    for ch in continuous:
        s = fc.get(ch, {})
        _push(f"fc_{ch}_mean",  s.get("mean", 0.0))
        _push(f"fc_{ch}_std",   s.get("std", 0.0))
        _push(f"fc_{ch}_slope", s.get("slope", 0.0))
        _push(f"fc_{ch}_range", s.get("range", 0.0))

    # Block D — forecast categorical
    for ch in categorical:
        s = fc.get(ch, {})
        _push(f"fc_{ch}_mode", s.get("mode", 0))

    # Block E — transition deltas (forecast_mean − lookback_mean)
    for ch in delta_channels:
        lb_mean = lb.get(ch, {}).get("mean", 0.0)
        fc_mean = fc.get(ch, {}).get("mean", 0.0)
        _push(f"delta_{ch}_mean", fc_mean - lb_mean)

    # Block F — boundary calendar
    fs = segment.get("time_context", {}).get("forecast_start", {})
    hour = float(fs.get("hour", 0))
    sm   = float(fs.get("solar_month", 1))
    sd   = float(fs.get("solar_day", 1))
    doy  = (sm - 1) * 30 + sd                     # rough day-of-year
    _push("hour", hour)
    _push("hour_sin", np.sin(2 * np.pi * hour / 24))
    _push("hour_cos", np.cos(2 * np.pi * hour / 24))
    _push("doy", doy)
    _push("doy_sin", np.sin(2 * np.pi * doy / 365))
    _push("doy_cos", np.cos(2 * np.pi * doy / 365))

    return np.asarray(parts, dtype=np.float64), names


def build_all_summary_vectors(segments: list, continuous: list,
                               categorical: list, delta_channels: list = None
                               ) -> tuple[np.ndarray, list]:
    """Stack summary vectors for every segment. Returns (N, D) and dim names."""
    rows = []
    dim_names = None
    for seg in segments:
        vec, names = build_summary_vector(seg, continuous, categorical, delta_channels)
        rows.append(vec)
        dim_names = names      # same for every segment
    return np.stack(rows, axis=0), dim_names


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

def standardize(vectors: np.ndarray, scaler: StandardScaler = None
                ) -> tuple[np.ndarray, StandardScaler]:
    """Z-score normalise with sklearn's StandardScaler. Pass an existing
    scaler to apply training-fit statistics; pass None to fit a new one."""
    if scaler is None:
        scaler = StandardScaler()
        return scaler.fit_transform(vectors), scaler
    return scaler.transform(vectors), scaler


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_segments(vectors_z: np.ndarray, n_clusters: int, random_state: int = 0
                     ) -> tuple[np.ndarray, KMeans]:
    """K-means clustering on standardised summary vectors."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(vectors_z)
    return labels, km


def summarize_cluster(segments: list, labels: np.ndarray, cluster_id: int,
                       continuous: list, categorical: list,
                       global_stats: dict = None) -> dict:
    """Build a compact, LLM-friendly summary of one cluster.

    Surfaces:
      - cluster size (count and fraction of dataset)
      - top distinguishing features (continuous channels whose cluster mean
        deviates most from the global mean, in std-units)
      - target signature (load mean, slope) over the forecast window
      - a representative segment_id (medoid in summary-vector space)
    """
    members = [seg for seg, lbl in zip(segments, labels) if lbl == cluster_id]
    n = len(members)
    if n == 0:
        return {"cluster_id": int(cluster_id), "n_segments": 0}

    sig = {}

    # Per-channel cluster means (continuous) — annotate deviation from global
    for ch in continuous:
        means = [m["lookback_stats"].get(ch, {}).get("mean") for m in members]
        means = [v for v in means if v is not None]
        if not means:
            continue
        cluster_mean = float(np.mean(means))
        entry = {"cluster_mean": round(cluster_mean, 2)}
        if global_stats and ch in global_stats:
            g = global_stats[ch]
            g_mean, g_std = g.get("mean"), g.get("std")
            if g_mean is not None and g_std and g_std > 1e-9:
                z = (cluster_mean - g_mean) / g_std
                entry["z_vs_global"] = round(float(z), 2)
        sig[ch] = entry

    # Per-channel cluster modes (categorical)
    for ch in categorical:
        modes = [m["lookback_stats"].get(ch, {}).get("mode") for m in members]
        modes = [v for v in modes if v is not None]
        if not modes:
            continue
        cnt = Counter(modes)
        top_val, top_cnt = cnt.most_common(1)[0]
        sig[ch] = {
            "dominant_value": int(top_val),
            "purity": round(top_cnt / len(modes), 3),
        }

    # Forecast signature (target only)
    fc_means = [m["forecast_stats"].get("load", {}).get("mean") for m in members]
    fc_means = [v for v in fc_means if v is not None]
    fc_slopes = [m["forecast_stats"].get("load", {}).get("slope") for m in members]
    fc_slopes = [v for v in fc_slopes if v is not None]
    fc_sig = {}
    if fc_means:
        fc_sig["mean"] = round(float(np.mean(fc_means)), 2)
        fc_sig["std"] = round(float(np.std(fc_means)), 2)
    if fc_slopes:
        fc_sig["slope_mean"] = round(float(np.mean(fc_slopes)), 4)

    # Pick top-3 most distinguishing continuous channels (largest |z_vs_global|)
    distinguishing = sorted(
        [(ch, abs(s.get("z_vs_global", 0.0))) for ch, s in sig.items()
         if "z_vs_global" in s],
        key=lambda x: -x[1],
    )[:3]
    distinguishing_str = ", ".join(
        f"{ch} (z={sig[ch]['z_vs_global']:+.2f})" for ch, _ in distinguishing
    )

    return {
        "cluster_id": int(cluster_id),
        "n_segments": int(n),
        "fraction": round(n / len(segments), 3),
        "distinguishing_features_top3": distinguishing_str,
        "feature_signatures": sig,
        "forecast_signature": fc_sig,
        "representative_segment_id": int(members[0]["segment_id"]),
    }


def summarize_all_clusters(segments: list, labels: np.ndarray, n_clusters: int,
                            continuous: list, categorical: list,
                            global_stats: dict = None) -> list:
    """Convenience: produce one summary dict per cluster, in cluster order."""
    return [
        summarize_cluster(segments, labels, k, continuous, categorical, global_stats)
        for k in range(n_clusters)
    ]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_artifacts(output_dir: str, vectors_raw: np.ndarray, vectors_z: np.ndarray,
                    scaler: StandardScaler, labels: np.ndarray, dim_names: list,
                    cluster_summaries: list):
    """Persist everything needed downstream (typicality + initializer)."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "summary_vectors_raw.npy"), vectors_raw)
    np.save(os.path.join(output_dir, "summary_vectors_z.npy"), vectors_z)
    np.save(os.path.join(output_dir, "cluster_labels.npy"), labels)
    np.save(os.path.join(output_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(output_dir, "scaler_scale.npy"), scaler.scale_)
    with open(os.path.join(output_dir, "dim_names.json"), "w") as f:
        json.dump(dim_names, f, indent=2)
    with open(os.path.join(output_dir, "cluster_summaries.json"), "w") as f:
        json.dump(cluster_summaries, f, indent=2)


def load_artifacts(output_dir: str) -> dict:
    """Counterpart to save_artifacts(). Returns a dict with all components."""
    out = {
        "vectors_raw": np.load(os.path.join(output_dir, "summary_vectors_raw.npy")),
        "vectors_z":   np.load(os.path.join(output_dir, "summary_vectors_z.npy")),
        "labels":      np.load(os.path.join(output_dir, "cluster_labels.npy")),
    }
    sc = StandardScaler()
    sc.mean_  = np.load(os.path.join(output_dir, "scaler_mean.npy"))
    sc.scale_ = np.load(os.path.join(output_dir, "scaler_scale.npy"))
    sc.var_   = sc.scale_ ** 2
    sc.n_features_in_ = sc.mean_.shape[0]
    out["scaler"] = sc
    with open(os.path.join(output_dir, "dim_names.json")) as f:
        out["dim_names"] = json.load(f)
    with open(os.path.join(output_dir, "cluster_summaries.json")) as f:
        out["cluster_summaries"] = json.load(f)
    return out


# ---------------------------------------------------------------------------
# One-shot driver: segments JSON → all artifacts on disk
# ---------------------------------------------------------------------------

def build_and_cluster(segments_path: str, output_dir: str, n_clusters: int,
                      channel_meta: dict, random_state: int = 0):
    """Load Part-A segments, build summary vectors, standardise, cluster,
    and write everything to ``output_dir``. Returns the in-memory bundle."""
    print(f"Loading segments from {segments_path}...")
    with open(segments_path) as f:
        seg_data = json.load(f)
    segments = seg_data["segments"]
    global_stats = seg_data.get("global_statistics", {})
    print(f"  Loaded {len(segments)} segments")

    continuous, categorical = partition_channels(channel_meta)
    print(f"  Channels: {len(continuous)} continuous, {len(categorical)} categorical")

    vectors_raw, dim_names = build_all_summary_vectors(segments, continuous, categorical)
    print(f"  Built summary vectors: {vectors_raw.shape}")

    vectors_z, scaler = standardize(vectors_raw)
    print(f"  Z-score normalised (per-feature)")

    labels, _ = cluster_segments(vectors_z, n_clusters, random_state=random_state)
    print(f"  K-means: {n_clusters} clusters")

    cluster_summaries = summarize_all_clusters(
        segments, labels, n_clusters, continuous, categorical, global_stats,
    )

    save_artifacts(output_dir, vectors_raw, vectors_z, scaler,
                   labels, dim_names, cluster_summaries)
    print(f"  Saved artifacts to {output_dir}/")

    return {
        "segments": segments,
        "global_stats": global_stats,
        "vectors_raw": vectors_raw,
        "vectors_z": vectors_z,
        "scaler": scaler,
        "labels": labels,
        "dim_names": dim_names,
        "cluster_summaries": cluster_summaries,
    }
