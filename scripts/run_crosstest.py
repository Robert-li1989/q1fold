#!/usr/bin/env python
"""
Cross-dataset multi-seed TEST runner
====================================

Takes checkpoints already trained by run_multiseed.py (one per seed) and tests
each on a DIFFERENT dataset, then aggregates the post-processed F1 as
mean +/- std. Use it for the standard generalization protocol: train on
RNAstralign, test on ArchiveII.

It does NOT train anything -- it only runs `python -m q1fold ... test ...`
against each existing seedN/weights.pmt, so it's fast.

Usage
-----
    python run_crosstest.py \
        --package-dir /home/.../q1fold \
        --config      /home/.../multiscale_357_attn1d_opm_dilated_v20_pairchannels128_reg_config.json \
        --weights-root /home/.../results/rnastralign_v20_multiseed \
        --test-file   /home/.../data/RNAstralign/ArchiveII_512.csv \
        --test-name   archiveII \
        --seeds 42 123 456 789 1011 \
        --device cuda --batch 4 --workers 4 --max-length 512 \
        --cuda-visible 3

--weights-root is the --out-root you gave run_multiseed.py; this script expects
<weights-root>/seed<seed>/weights.pmt for each seed. Results (per-seed metrics
CSV + logs) are written under <weights-root>/seed<seed>/ with a name derived
from --test-name, and the aggregate goes to
<weights-root>/crosstest_<test-name>_summary.csv.
"""

import argparse
import csv
import math
import os
import statistics
import subprocess
import sys


def run_cmd(cmd, env, log_path):
    with open(log_path, "w") as log:
        return subprocess.run(cmd, env=env, stdout=log,
                              stderr=subprocess.STDOUT).returncode


def parse_metrics_csv(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return None
    out = {}
    for k, v in zip(rows[0], rows[1]):
        try:
            out[k.strip()] = float(v)
        except ValueError:
            out[k.strip()] = v
    return out


def main():
    ap = argparse.ArgumentParser(description="Test existing per-seed checkpoints on a new dataset.")
    ap.add_argument("--package-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights-root", required=True,
                    help="the --out-root used by run_multiseed.py")
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--test-name", default="crosstest",
                    help="short label for output filenames (e.g. archiveII)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1011])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--cuda-visible", default=None)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    pkg_dir = os.path.abspath(args.package_dir)
    pkg_name = os.path.basename(pkg_dir.rstrip("/"))
    pkg_parent = os.path.dirname(pkg_dir)
    if not os.path.exists(os.path.join(pkg_dir, "__init__.py")):
        ap.error(f"{pkg_dir} has no __init__.py -- point --package-dir at the package folder.")

    env = dict(os.environ)
    env["PYTHONPATH"] = pkg_parent + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if args.cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible)

    per_seed = {}
    for seed in args.seeds:
        seed_dir = os.path.join(args.weights_root, f"seed{seed}")
        weights = os.path.join(seed_dir, "weights.pmt")
        out_csv = os.path.join(seed_dir, f"test_{args.test_name}_metrics.csv")

        if not os.path.exists(weights):
            print(f"[seed {seed}] MISSING weights at {weights} -- skipping")
            continue
        if args.skip_existing and os.path.exists(out_csv):
            print(f"[seed {seed}] {args.test_name} metrics exist -> reuse")
            m = parse_metrics_csv(out_csv)
            if m:
                per_seed[seed] = m
            continue

        cmd = [args.python, "-m", pkg_name,
               "-d", args.device, "-batch", str(args.batch),
               "-j", str(args.workers), "--max-length", str(args.max_length),
               "--config", args.config,
               "test", args.test_file,
               "-w", weights, "-o", out_csv]
        print(f"[seed {seed}] testing on {args.test_name} ...")
        rc = run_cmd(cmd, env,
                     os.path.join(args.weights_root,
                                  f"seed{seed}_test_{args.test_name}.log"))
        if rc != 0:
            print(f"[seed {seed}] TEST FAILED (rc={rc}); see "
                  f"seed{seed}_test_{args.test_name}.log")
            continue
        m = parse_metrics_csv(out_csv)
        if m is None:
            print(f"[seed {seed}] could not parse {out_csv}")
            continue
        per_seed[seed] = m
        print(f"[seed {seed}] {args.test_name} f1_post={m.get('f1_post')}")

    if not per_seed:
        print("\nNo successful seeds -- nothing to aggregate.")
        return

    def agg(metric):
        vals = [m[metric] for m in per_seed.values()
                if isinstance(m.get(metric), float)]
        if not vals:
            return None
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return vals, mean, std, min(vals), max(vals)

    print("\n" + "=" * 64)
    print(f"Cross-dataset summary [{args.test_name}]  "
          f"({len(per_seed)} seeds: {sorted(per_seed)})")
    print("=" * 64)
    summary = os.path.join(args.weights_root,
                           f"crosstest_{args.test_name}_summary.csv")
    with open(summary, "w") as fh:
        fh.write("metric,mean,std,min,max,n," +
                 ",".join(f"seed{s}" for s in sorted(per_seed)) + "\n")
        for metric in ("f1_post", "f1", "precision_post", "recall_post",
                       "precision", "recall"):
            a = agg(metric)
            if a is None:
                continue
            vals, mean, std, lo, hi = a
            ci = 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            print(f"  {metric:>16}: {mean:.4f} +/- {std:.4f}  "
                  f"(95% CI +/-{ci:.4f}, min {lo:.4f}, max {hi:.4f})")
            row = [f"{per_seed[s].get(metric, ''):.4f}"
                   if isinstance(per_seed[s].get(metric), float) else ""
                   for s in sorted(per_seed)]
            fh.write(f"{metric},{mean:.4f},{std:.4f},{lo:.4f},{hi:.4f},"
                     f"{len(vals)}," + ",".join(row) + "\n")
    print(f"\nSummary written to {summary}")


if __name__ == "__main__":
    main()
