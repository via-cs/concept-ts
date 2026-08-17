"""
Train every LOOKBACK-only baseline (Crossformer, DLinear, Informer,
iTransformer, LSTM, TimeXer) THREE times with different random seeds on one
dataset, then report each run's test MAE / RMSE and the mean +/- std across the
three runs.

Schema switch: this runner inherits the environment, so for LOF set
``PROTO_DATASET`` (or pass --proto_dataset), e.g.

    PROTO_DATASET=LOF python -u comparison_model_lookback/run_all_lookback.py \
        --dataset ./dataset/LOF/PC.csv --lookback_len 96 --pred_len 96 \
        --epoch 30 --batch_size 32

Options:
    --n_runs 3                 how many seeds per model (default 3)
    --seeds 2024,7,99          fix the seeds (overrides --n_runs); else drawn at random
    --models crossformer,lstm  run a subset
"""
import argparse
import os
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent          # repo root

MODELS = {
    "crossformer":  "comparison_model_lookback/train_crossformer.py",
    "dlinear":      "comparison_model_lookback/train_dlinear.py",
    "informer":     "comparison_model_lookback/train_informer.py",
    "itransformer": "comparison_model_lookback/train_itransformer.py",
    "lstm":         "comparison_model_lookback/train_lstm.py",
    "timexer":      "comparison_model_lookback/train_timexer.py",
}
DEFAULT_ORDER = ["crossformer", "dlinear", "informer", "itransformer", "lstm", "timexer"]


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Path to the dataset CSV.")
    p.add_argument("--models", default=",".join(DEFAULT_ORDER))
    p.add_argument("--n_runs", type=int, default=3, help="Seeds per model (default 3).")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated seeds (overrides --n_runs). "
                        "If omitted, --n_runs random seeds are drawn and reused "
                        "across all models so runs are paired.")
    p.add_argument("--lookback_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--checkpoint_root", default="./checkpoint/run_all_lookback")
    p.add_argument("--proto_dataset", default=None,
                   help="Optionally set PROTO_DATASET for all subprocesses (e.g. LOF).")
    return p.parse_args()


_RE_MAE_RAW = re.compile(r"test MAE\s*=\s*([-\d.]+)\s*\(RAW")
_RE_MAE_Z   = re.compile(r"test MAE\s*=\s*([-\d.]+)\s*\(z-scored\)")
_RE_RMSE    = re.compile(r"test MSE\s*=\s*[-\d.]+\s*\(RAW[^,]*,\s*RMSE\s*=\s*([-\d.]+)\)")


def parse_metrics(text):
    m = _RE_MAE_RAW.search(text); mae_raw = float(m.group(1)) if m else None
    m = _RE_MAE_Z.search(text);   mae_z   = float(m.group(1)) if m else None
    m = _RE_RMSE.search(text);    rmse    = float(m.group(1)) if m else None
    return {"mae_raw": mae_raw, "rmse_raw": rmse, "mae_z": mae_z}


def run_one(model, seed, args, env):
    ckpt = os.path.join(args.checkpoint_root, model, f"seed{seed}")
    cmd = [sys.executable, "-u", str(_ROOT / MODELS[model]),
           "--dataset", args.dataset,
           "--lookback_len", str(args.lookback_len),
           "--pred_len", str(args.pred_len),
           "--epoch", str(args.epoch),
           "--batch_size", str(args.batch_size),
           "--lr", str(args.lr),
           "--patience", str(args.patience),
           "--seed", str(seed),
           "--checkpoint_dir", ckpt]
    print("\n" + "=" * 78)
    print(f"▶ {model}  seed={seed}")
    print("=" * 78, flush=True)
    t0 = time.time()
    buf = []
    proc = subprocess.Popen(cmd, cwd=str(_ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line); buf.append(line)
    proc.wait()
    r = parse_metrics("".join(buf))
    r["secs"] = time.time() - t0
    r["status"] = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    r["seed"] = seed
    return r


def _mean_std(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return None, None
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return mean, std


def main():
    args = get_args()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    if args.proto_dataset:
        env["PROTO_DATASET"] = args.proto_dataset

    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in MODELS]
    if unknown:
        sys.exit(f"Unknown model(s): {unknown}. Choose from {list(MODELS)}.")

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [random.randrange(2**31) for _ in range(args.n_runs)]

    print(f"Dataset: {args.dataset}   schema={env.get('PROTO_DATASET','PRSA(default)')}")
    print(f"Models: {wanted}")
    print(f"Seeds (shared across models): {seeds}")

    results = {m: [] for m in wanted}          # model -> list of per-run dicts
    for m in wanted:
        for s in seeds:
            try:
                results[m].append(run_one(m, s, args, env))
            except Exception as e:
                results[m].append({"mae_raw": None, "rmse_raw": None, "mae_z": None,
                                   "secs": 0.0, "status": f"ERROR({e})", "seed": s})

    def f(x, p=3): return f"{x:.{p}f}" if isinstance(x, (int, float)) else "  —"

    # ---- per-run detail ----
    print("\n" + "#" * 86)
    print("# PER-RUN test error")
    print("#" * 86)
    h = f"{'model':13s} {'seed':>11} {'MAE(raw)':>11} {'RMSE(raw)':>11} {'MAE(z)':>9} {'time':>7} {'status':>9}"
    print(h); print("-" * len(h))
    for m in wanted:
        for r in results[m]:
            print(f"{m:13s} {r['seed']:>11} {f(r['mae_raw']):>11} {f(r['rmse_raw']):>11} "
                  f"{f(r['mae_z'],4):>9} {r['secs']:>6.0f}s {r['status']:>9}")

    # ---- mean +/- std summary ----
    print("\n" + "#" * 72)
    print(f"# SUMMARY over {len(seeds)} runs  (mean +/- std)")
    print("#" * 72)
    h2 = f"{'model':13s} {'MAE(raw) mean±std':>24} {'RMSE(raw) mean±std':>24}"
    print(h2); print("-" * len(h2))
    summary = {}
    for m in wanted:
        mae_m, mae_s = _mean_std([r["mae_raw"] for r in results[m]])
        rms_m, rms_s = _mean_std([r["rmse_raw"] for r in results[m]])
        summary[m] = (mae_m, mae_s, rms_m, rms_s)
        mae_str = f"{mae_m:.3f} ± {mae_s:.3f}" if mae_m is not None else "—"
        rms_str = f"{rms_m:.3f} ± {rms_s:.3f}" if rms_m is not None else "—"
        print(f"{m:13s} {mae_str:>24} {rms_str:>24}")

    # ---- CSV ----
    os.makedirs(args.checkpoint_root, exist_ok=True)
    csv_path = os.path.join(args.checkpoint_root, "summary.csv")
    with open(csv_path, "w") as fh:
        fh.write("model,seed,mae_raw,rmse_raw,mae_z,status\n")
        for m in wanted:
            for r in results[m]:
                fh.write(f"{m},{r['seed']},{r['mae_raw']},{r['rmse_raw']},{r['mae_z']},{r['status']}\n")
        fh.write("\nmodel,mae_raw_mean,mae_raw_std,rmse_raw_mean,rmse_raw_std\n")
        for m in wanted:
            a, b, c, d = summary[m]
            fh.write(f"{m},{a},{b},{c},{d}\n")
    print(f"\nSaved -> {csv_path}")


if __name__ == "__main__":
    main()
