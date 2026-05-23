"""
case_studies.py — qualitative interpretability snapshot.

Picks K=5 test segments (4 stratified by LLM prototype via the saved filter
function, 1 stratified by metadata via a random workday filter), runs them
through the trained ProtoLLM, and emits:

  * a console table summarising metadata + top-3 activated layer-0 prototypes
  * one PNG per case study: input load curve, ground-truth forecast,
    predicted forecast, and the top-1 prototype's ``prototype_value`` curve
  * one HTML report (``case_studies.html``) tying it all together

Selection rule:
  * 4 cases via strategy (c): for each of the first ``--num_prototype_cases``
    LLM prototypes whose filter returns ≥1 test segment, pick one matching
    test segment at random (with ``--seed``).
  * 1 case via strategy (b): pick a random workday-and-not-holiday test
    segment via metadata, regardless of any LLM prototype's filter.

The HTML caption explicitly states how each case was sampled, so reviewers
can't accuse cherry-picking.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments._common import (
    build_test_segments,
    find_matching_test_segments,
    forward_capture,
    load_encoder_and_prototype,
    load_llm_prototypes,
    metadata_for_segment,
)
from dataloader.dataloader import LOF_Dataset


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--pretrain_checkpoint",  type=str, required=True)
    p.add_argument("--prototype_checkpoint", type=str, required=True)
    p.add_argument("--final_prototypes",     type=str, required=True)
    p.add_argument("--output_dir",           type=str, default="./experiments/output")

    p.add_argument("--num_prototype_cases", type=int, default=4,
                   help="Strategy (c): pick this many LLM prototypes and "
                        "sample one matching test segment for each.")
    p.add_argument("--num_random_cases",    type=int, default=1,
                   help="Strategy (b): random workday-and-not-holiday "
                        "metadata-stratified cases.")
    p.add_argument("--top_k", type=int, default=3,
                   help="How many top prototypes to print per case study.")
    p.add_argument("--seed", type=int, default=0)

    # Shape knobs (must match training).
    p.add_argument("--input_length",  type=int, default=384)
    p.add_argument("--output_length", type=int, default=96)
    p.add_argument("--input_dim",     type=int, default=23)
    p.add_argument("--output_dim",    type=int, default=1)
    p.add_argument("--embed_dim",     type=int, default=128)
    p.add_argument("--num_layer",     type=int, default=4)
    p.add_argument("--dropout",       type=float, default=0.5)
    p.add_argument("--bottle_dim_p",  type=float, default=0.1)
    p.add_argument("--prototype_num",       type=int, default=8)
    p.add_argument("--prototype_split_num", type=int, default=2)
    p.add_argument("--top_k_activate",      type=int, default=2)
    p.add_argument("--split_ratio",         type=float, default=0.4)

    p.add_argument("--segment_stride", type=int, default=96)
    return p.parse_args()


def _select_cases(llm_protos, test_segments, *, n_proto, n_random, rng):
    """Return list[(case_dict)] of length ≤ n_proto + n_random."""
    cases = []
    used_seg_idx = set()

    # ---- strategy (c) -------------------------------------------------
    picked_protos = 0
    for proto in llm_protos:
        if picked_protos >= n_proto:
            break
        matched, err = find_matching_test_segments(proto, test_segments)
        if err or not matched:
            continue
        # Filter out any segment we've already used in another case.
        candidates = [m for m in matched if m not in used_seg_idx]
        if not candidates:
            continue
        seg_idx = rng.choice(candidates)
        used_seg_idx.add(seg_idx)
        cases.append({
            "strategy":      "(c) stratified by LLM prototype filter",
            "target_proto":  proto["name"],
            "target_anchor": proto["anchor_idx"],
            "rubric":        proto.get("rubric", ""),
            "seg_idx":       seg_idx,
        })
        picked_protos += 1

    # ---- strategy (b) -------------------------------------------------
    workday_pool = [
        i for i, s in enumerate(test_segments)
        if i not in used_seg_idx
        and (s.get("cross_channel_stats", {}).get("uniform_day_type") == "workday")
        and not s.get("cross_channel_stats", {}).get("contains_holiday", False)
    ]
    rng.shuffle(workday_pool)
    for seg_idx in workday_pool[:n_random]:
        used_seg_idx.add(seg_idx)
        cases.append({
            "strategy":      "(b) random workday (metadata-stratified)",
            "target_proto":  None,
            "target_anchor": None,
            "rubric":        "",
            "seg_idx":       seg_idx,
        })

    return cases


def _plot_case(*, out_path, input_load_unnorm, true_forecast_unnorm,
               pred_forecast_unnorm, proto_value_curve, proto_name, title):
    """Plot the input window, true forecast, predicted forecast, and the top-1
    prototype's value curve on one figure. ``proto_value_curve`` may be ``None``
    if value-domain comparison is skipped."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    t_lb = list(range(len(input_load_unnorm)))
    t_fc = list(range(len(input_load_unnorm),
                      len(input_load_unnorm) + len(true_forecast_unnorm)))

    ax.plot(t_lb, input_load_unnorm, label="lookback (true load)",
            color="0.4", linewidth=1.2)
    ax.plot(t_fc, true_forecast_unnorm, label="ground-truth forecast",
            color="black", linewidth=2.0)
    ax.plot(t_fc, pred_forecast_unnorm, label="ProtoLLM prediction",
            color="tab:blue", linewidth=1.5, linestyle="--")
    if proto_value_curve is not None:
        ax.plot(t_fc, proto_value_curve,
                label=f"top-1 prototype value ({proto_name})",
                color="tab:red", linewidth=1.2, linestyle=":")

    ax.axvline(len(input_load_unnorm) - 0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("15-min step (lookback | forecast)")
    ax.set_ylabel("load")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(args.seed)
    print(f"device = {device}, seed = {args.seed}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Model + LLM specs + segments + dataset
    # ------------------------------------------------------------------
    llm_protos = load_llm_prototypes(args.final_prototypes)
    print(f"loaded {len(llm_protos)} LLM prototypes")

    encoder, prototype_model = load_encoder_and_prototype(
        pretrain_checkpoint=args.pretrain_checkpoint,
        prototype_checkpoint=args.prototype_checkpoint,
        prototype_num=args.prototype_num,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        embed_dim=args.embed_dim,
        input_length=args.input_length,
        output_length=args.output_length,
        num_layer=args.num_layer,
        dropout=args.dropout,
        bottle_dim_p=args.bottle_dim_p,
        prototype_split_num=args.prototype_split_num,
        top_k_activate=args.top_k_activate,
        split_ratio=args.split_ratio,
        device=device,
    )
    proto_values = prototype_model.prototype.prototype_value[0].detach().cpu()  # (P, 96, 1)
    P0 = proto_values.shape[0]

    test_segments = build_test_segments(
        csv_path=args.dataset,
        input_length=args.input_length,
        output_length=args.output_length,
        stride=args.segment_stride,
        split="test",
    )
    test_dataset = LOF_Dataset(
        args.input_length, args.output_length,
        "test", args.input_dim, args.dataset,
    )
    print(f"test segments = {len(test_segments)},  test windows = {len(test_dataset)}")

    # ------------------------------------------------------------------
    # Pick K case-study segments
    # ------------------------------------------------------------------
    cases = _select_cases(
        llm_protos, test_segments,
        n_proto=args.num_prototype_cases,
        n_random=args.num_random_cases,
        rng=rng,
    )
    print(f"selected {len(cases)} case studies")

    # Drop any that map to a dataset index out of bounds (rare).
    cases = [c for c in cases if c["seg_idx"] < len(test_segments)
             and int(test_segments[c["seg_idx"]]["start_idx"]) < len(test_dataset)]
    window_indices = [int(test_segments[c["seg_idx"]]["start_idx"]) for c in cases]

    cap = forward_capture(encoder, prototype_model, test_dataset,
                          window_indices, device, batch_size=len(window_indices) or 1)
    layer0 = cap["layer0_score"]        # (K, P0)
    pred   = cap["prediction"]          # (K, 96, 1) — normalized space
    target = cap["target"]              # (K, 96, 1) — normalized space
    inp    = cap["input_data"]          # (K, 480, in_dim)

    # ------------------------------------------------------------------
    # De-normalize for plotting (the dataset's scaler is the same train-fit
    # scaler used everywhere else).
    # ------------------------------------------------------------------
    scale = float(test_dataset.ans_type.scale_[0])
    mean  = float(test_dataset.ans_type.mean_[0])

    def un_norm(v):
        return (v * scale + mean).numpy()

    # The last channel in input_data is the (masked) load — use it for the
    # lookback portion of the plot.
    LOAD_COL = args.input_dim - 1

    # ------------------------------------------------------------------
    # Per-case rendering + HTML aggregation
    # ------------------------------------------------------------------
    name_by_anchor = {p["anchor_idx"]: p["name"] for p in llm_protos}
    rubric_by_anchor = {p["anchor_idx"]: p.get("rubric", "") for p in llm_protos}

    html_parts = ["<html><head><meta charset='utf-8'>"
                  "<title>ProtoLLM case studies</title>"
                  "<style>body{font-family:sans-serif;max-width:980px;margin:24px auto;}"
                  "table{border-collapse:collapse;}td,th{border:1px solid #ccc;padding:4px 8px;}"
                  ".rubric{color:#555;font-size:0.85em;}"
                  "h2{margin-top:32px;}img{width:100%;max-width:920px;}</style>"
                  "</head><body>"]
    html_parts.append(f"<h1>ProtoLLM case studies — {os.path.basename(args.dataset)}</h1>")
    html_parts.append(
        "<p>Each case study is one test segment. Top-3 layer-0 prototypes "
        "are the LLM-named prototypes that received the highest softmax "
        "score for this segment. Strategy (c) = test segment was sampled "
        "from the LLM prototype's own filter function; strategy (b) = "
        "test segment was sampled at random from workdays.</p>"
    )

    for case_no, (case, win_idx) in enumerate(zip(cases, window_indices), start=1):
        seg = test_segments[case["seg_idx"]]
        meta = metadata_for_segment(seg)

        scores = layer0[case_no - 1]                           # (P0,)
        top_k = torch.topk(scores, min(args.top_k, P0))
        top_names = [name_by_anchor.get(int(i), f"proto_{int(i)}")
                     for i in top_k.indices.tolist()]
        top_scores = [round(float(s), 4) for s in top_k.values.tolist()]

        # Plot
        plot_path = plots_dir / f"case_{case_no:02d}.png"
        top1_anchor = int(top_k.indices[0].item())
        top1_curve_unnorm = un_norm(proto_values[top1_anchor, :, 0])
        title = (f"Case {case_no} — {case['strategy']}"
                 + (f"  |  target prototype: {case['target_proto']}"
                    if case["target_proto"] else ""))
        _plot_case(
            out_path=plot_path,
            input_load_unnorm=un_norm(inp[case_no - 1, :, LOAD_COL]),
            true_forecast_unnorm=un_norm(target[case_no - 1, :, 0]),
            pred_forecast_unnorm=un_norm(pred[case_no - 1, :, 0]),
            proto_value_curve=top1_curve_unnorm,
            proto_name=top_names[0],
            title=title,
        )

        # Console
        print(f"\n=== Case {case_no} : window {win_idx} =====================================")
        print(f"  selection : {case['strategy']}")
        if case["target_proto"]:
            print(f"  target_proto : {case['target_proto']}")
        print(f"  metadata  : {meta}")
        print(f"  top-{args.top_k}:")
        for n, s in zip(top_names, top_scores):
            print(f"      {s:.4f}   {n}")

        # HTML
        html_parts.append(f"<h2>Case {case_no}</h2>")
        html_parts.append(f"<p><b>Selection:</b> {case['strategy']}"
                          + (f" — target prototype <code>{case['target_proto']}</code>"
                             if case["target_proto"] else "") + "</p>")
        html_parts.append("<table>")
        for k, v in meta.items():
            html_parts.append(f"<tr><th>{k}</th><td>{v}</td></tr>")
        html_parts.append("</table>")
        html_parts.append("<h3>Top prototypes (layer-0 softmax score):</h3>")
        html_parts.append("<table><tr><th>rank</th><th>score</th>"
                          "<th>prototype name</th><th>rubric</th></tr>")
        for rk, (n, s, anchor) in enumerate(
                zip(top_names, top_scores, top_k.indices.tolist()), start=1):
            rb = rubric_by_anchor.get(int(anchor), "")
            html_parts.append(
                f"<tr><td>{rk}</td><td>{s:.4f}</td>"
                f"<td><code>{n}</code></td>"
                f"<td class='rubric'>{rb}</td></tr>"
            )
        html_parts.append("</table>")
        html_parts.append(f"<p><img src='plots/{plot_path.name}'></p>")

    html_parts.append("</body></html>")
    html_path = out_dir / "case_studies.html"
    html_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"\nwrote {html_path}")
    print(f"      ({len(cases)} plots in {plots_dir})")

    # Also dump a JSON sidecar for downstream analysis.
    sidecar = []
    for case_no, (case, win_idx) in enumerate(zip(cases, window_indices), start=1):
        seg = test_segments[case["seg_idx"]]
        scores = layer0[case_no - 1]
        top_k = torch.topk(scores, min(args.top_k, P0))
        sidecar.append({
            "case_no":  case_no,
            "selection": case["strategy"],
            "target_prototype": case["target_proto"],
            "metadata": metadata_for_segment(seg),
            "top_k": [
                {"rank": rk,
                 "score": float(s),
                 "name": name_by_anchor.get(int(i), f"proto_{int(i)}"),
                 "anchor_idx": int(i)}
                for rk, (s, i) in enumerate(
                    zip(top_k.values.tolist(), top_k.indices.tolist()), start=1)
            ],
        })
    (out_dir / "case_studies.json").write_text(
        json.dumps(sidecar, indent=2), encoding="utf-8"
    )
    print(f"      sidecar: {out_dir / 'case_studies.json'}")


if __name__ == "__main__":
    main()
