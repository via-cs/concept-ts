"""
Run several comparison-model baselines end-to-end with ONE command and print a
summary table of their test MAE / RMSE (z-scored and raw units).

Covers: informer, tft, deepar, nhits (neural) and randomforest, xgboost,
lightgbm (tabular). Each baseline is launched as its own subprocess with the
right arguments (neural models get --epoch/--batch_size/--lr; tabular models
don't), its output is streamed live, and the final "test MAE / MSE / RMSE"
lines it prints are parsed into a summary.

Schema switch: this runner inherits the environment, so to run on LOF set
``PROTO_DATASET`` before calling it (or pass --proto_dataset), e.g.

    PROTO_DATASET=LOF python -u comparison_model/run_all.py \
        --dataset ./dataset/LOF/PC.csv --lookback_len 96 --pred_len 96 \
        --epoch 30 --batch_size 32

Run a subset with --models:

    python -u comparison_model/run_all.py --dataset ./dataset/PRSA_Data/Aotizhongxin.csv \
        --models informer,tft,lightgbm
"""
import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent          # repo root (has comparison_model/)

# model name -> (script relative to repo root, kind)  ; kind: "nn" | "tree"
MODELS = {
    "informer":     ("comparison_model/train_informer.py",      "nn"),
    "tft":          ("comparison_model/train_tft.py",           "nn"),
    "deepar":       ("comparison_model/train_deepar.py",        "nn"),
    "nhits":        ("comparison_model/train_nhits.py",         "nn"),
    "randomforest": ("comparison_model/train_random_forest.py", "tree"),
    "xgboost":      ("comparison_model/train_xgboost.py",       "tree"),
    "lightgbm":     ("comparison_model/train_lightgbm.py",      "tree"),
}
DEFAULT_ORDER = ["informer", "tft", "randomforest", "xgboost", "lightgbm", "deepar", "nhits"]


def get_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="Path to the dataset CSV.")
    p.add_argument("--models", default=",".join(DEFAULT_ORDER),
                   help="Comma-separated subset to run. Default: all 7.")
    p.add_argument("--lookback_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=96)
    # neural-only knobs (ignored for tabular models)
    p.add_argument("--epoch", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=None,
                   help="If omitted, each model draws its own random seed (printed).")
    p.add_argument("--checkpoint_root", default="./checkpoint/run_all",
                   help="Per-model checkpoints go in <root>/<model>/.")
    p.add_argument("--proto_dataset", default=None,
                   help="Optionally set PROTO_DATASET for all subprocesses "
                        "(e.g. LOF). If omitted, inherits the current env.")
    return p.parse_args()


def build_cmd(model, script, kind, args):
    ckpt = os.path.join(args.checkpoint_root, model)
    cmd = [sys.executable, "-u", str(_ROOT / script),
           "--dataset", args.dataset,
           "--lookback_len", str(args.lookback_len),
           "--pred_len", str(args.pred_len),
           "--checkpoint_dir", ckpt]
    if kind == "nn":
        cmd += ["--epoch", str(args.epoch),
                "--batch_size", str(args.batch_size),
                "--lr", str(args.lr),
                "--patience", str(args.patience)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    return cmd


# Regexes for the footer every train script prints.
_RE = {
    "mae_z":   re.compile(r"test MAE\s*=\s*([-\d.]+)\s*\(z-scored\)"),
    "mse_z":   re.compile(r"test MSE\s*=\s*([-\d.]+)\s*\(z-scored\)"),
    "mae_raw": re.compile(r"test MAE\s*=\s*([-\d.]+)\s*\(RAW"),
    "raw_mse_rmse": re.compile(r"test MSE\s*=\s*([-\d.]+)\s*\(RAW[^,]*,\s*RMSE\s*=\s*([-\d.]+)\)"),
}


def parse_metrics(text):
    out = {"mae_z": None, "rmse_z": None, "mae_raw": None, "rmse_raw": None, "seed": None}
    m = _RE["mae_z"].search(text);   out["mae_z"] = float(m.group(1)) if m else None
    m = _RE["mse_z"].search(text)
    out["rmse_z"] = float(m.group(1)) ** 0.5 if m else None
    m = _RE["mae_raw"].search(text); out["mae_raw"] = float(m.group(1)) if m else None
    m = _RE["raw_mse_rmse"].search(text)
    out["rmse_raw"] = float(m.group(2)) if m else None
    s = re.search(r"\[seed\] run seed = (\d+)", text)
    out["seed"] = int(s.group(1)) if s else None
    return out


def run_one(model, args, env):
    script, kind = MODELS[model]
    cmd = build_cmd(model, script, kind, args)
    print("\n" + "=" * 78)
    print(f"▶ {model}  ({kind})")
    print("  " + " ".join(cmd))
    print("=" * 78, flush=True)
    t0 = time.time()
    buf = []
    proc = subprocess.Popen(cmd, cwd=str(_ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        sys.stdout.write(line)          # live echo
        buf.append(line)
    proc.wait()
    dt = time.time() - t0
    text = "".join(buf)
    metrics = parse_metrics(text)
    metrics["status"] = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
    metrics["secs"] = dt
    return metrics


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

    print(f"Dataset: {args.dataset}   schema(PROTO_DATASET)={env.get('PROTO_DATASET','PRSA(default)')}")
    print(f"Running: {wanted}")
    results = {}
    for m in wanted:
        try:
            results[m] = run_one(m, args, env)
        except Exception as e:                      # keep going on failure
            results[m] = {"status": f"ERROR({e})", "mae_z": None, "rmse_z": None,
                          "mae_raw": None, "rmse_raw": None, "seed": None, "secs": 0.0}

    # ---- summary table ----
    print("\n" + "#" * 84)
    print("# SUMMARY — test error per model")
    print("#" * 84)
    hdr = f"{'model':14s} {'MAE(z)':>9} {'RMSE(z)':>9} {'MAE(raw)':>11} {'RMSE(raw)':>11} {'seed':>10} {'time':>7} {'status':>10}"
    print(hdr); print("-" * len(hdr))
    def f(x, p=4): return f"{x:.{p}f}" if isinstance(x, (int, float)) else "  —"
    for m in wanted:
        r = results[m]
        print(f"{m:14s} {f(r['mae_z']):>9} {f(r['rmse_z']):>9} "
              f"{f(r['mae_raw'],3):>11} {f(r['rmse_raw'],3):>11} "
              f"{str(r['seed'] if r['seed'] is not None else '—'):>10} "
              f"{r['secs']:>6.0f}s {r['status']:>10}")

    # ---- CSV ----
    os.makedirs(args.checkpoint_root, exist_ok=True)
    csv_path = os.path.join(args.checkpoint_root, "summary.csv")
    with open(csv_path, "w") as fh:
        fh.write("model,mae_z,rmse_z,mae_raw,rmse_raw,seed,secs,status\n")
        for m in wanted:
            r = results[m]
            fh.write(f"{m},{r['mae_z']},{r['rmse_z']},{r['mae_raw']},{r['rmse_raw']},"
                     f"{r['seed']},{r['secs']:.0f},{r['status']}\n")
    print(f"\nSaved summary -> {csv_path}")


if __name__ == "__main__":
    main()
