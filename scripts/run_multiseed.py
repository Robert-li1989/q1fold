#!/usr/bin/env python
"""
Multi-seed runner for Q1Fold (v20)
==================================

Trains the SAME config across N seeds and reports post-processed F1 on the test
set as mean +/- std -- the honest headline number, with an error bar, for the
paper.

Why this is needed
------------------
The codebase hardcodes the global RNG seed to 42 in `q1fold/__init__.py main()`
(`tr.manual_seed(42); random.seed(42); np.random.seed(42)`), and there is no
`--seed` CLI flag. This runner sets each seed WITHOUT editing source, by
`sed`-patching a private copy of the package per seed (default) OR, if you
prefer, you can add a one-line env hook yourself (see NOTE at bottom).

What the seed does and does NOT vary
-----------------------------------
Changing the global seed varies the CLASSICAL layer initialization (attn1d,
OPM projections, relpos, 2D refiner) and the training data-shuffle order --
i.e. optimization + data-order variance, which is the dominant source of
run-to-run spread. It does NOT vary the quantum circuit's initial angles: the
extractor branches call `torch.manual_seed(seed)` with their own default
seed=42 at construction, independent of the global seed. Report the seed study
as "variance over classical init and data order" accordingly. (To also vary
quantum init, thread a seed into MultiScaleQuantumConv1D -- a code change this
runner deliberately avoids.)

Method (default, no source edits kept)
-------------------------------------
For each seed s:
  1. copy the package dir to a temp location,
  2. sed-replace the three `...seed(42)` lines in the copy's __init__.py with s,
  3. run `python -m q1fold ... train ...` and `... test ...` against the copy
     (via PYTHONPATH), writing weights + metrics into a per-seed output dir,
  4. parse the test metrics CSV.
Then aggregate f1_post (and f1) across seeds -> mean, std, min, max.

Usage
-----
    python run_multiseed.py \
        --package-dir q1fold \
        --config      /home/.../multiscale_357_attn1d_opm_dilated_v20_pairchannels128_reg_config.json \
        --train-file  /home/.../data/bpRNA-1m/TR0.csv \
        --valid-file  /home/.../data/bpRNA-1m/VL0.csv \
        --test-file   /home/.../data/bpRNA-1m/TS0.csv \
        --out-root    /home/.../results/v20_multiseed \
        --seeds 42 123 456 789 1011 \
        --epochs 50 --device cuda --batch 4 --workers 4 --max-length 512 \
        --cuda-visible 3

`--package-dir` is the directory that IS the importable `q1fold` package (the
folder containing its __init__.py). If unsure, it's the dir where model.py /
multiscale_quantum_conv1d.py live.
"""

import argparse
import csv
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile


SEED_LINE_RE = re.compile(
    r"(tr\.manual_seed\(|random\.seed\(|np\.random\.seed\()\s*42\s*\)"
)


def patch_seed_in_package(pkg_copy_dir, seed):
    """Replace the three hardcoded seed(42) calls in the copy's __init__.py."""
    init_path = os.path.join(pkg_copy_dir, "__init__.py")
    with open(init_path, "r") as fh:
        src = fh.read()
    patched, n = SEED_LINE_RE.subn(rf"\g<1>{seed})", src)
    if n == 0:
        raise RuntimeError(
            f"Could not find any `...seed(42)` line in {init_path}. "
            f"Check the package -- the seeding code may have moved."
        )
    with open(init_path, "w") as fh:
        fh.write(patched)
    return n


def run_cmd(cmd, env, log_path):
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    return proc.returncode


def parse_metrics_csv(path):
    """Read a one-row metrics CSV (header + values) into a dict of floats."""
    with open(path) as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return None
    header, values = rows[0], rows[1]
    out = {}
    for k, v in zip(header, values):
        try:
            out[k.strip()] = float(v)
        except ValueError:
            out[k.strip()] = v
    return out


def main():
    ap = argparse.ArgumentParser(description="Multi-seed train/test runner.")
    ap.add_argument("--package-dir", required=True,
                    help="directory that IS the q1fold package (contains its __init__.py)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-file", required=True)
    ap.add_argument("--valid-file", required=True)
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 1011])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--cuda-visible", default=None,
                    help="value for CUDA_VISIBLE_DEVICES (e.g. '3')")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a seed whose test metrics CSV already exists")
    args = ap.parse_args()

    pkg_dir = os.path.abspath(args.package_dir)
    pkg_name = os.path.basename(pkg_dir.rstrip("/"))
    if not os.path.exists(os.path.join(pkg_dir, "__init__.py")):
        ap.error(f"{pkg_dir} has no __init__.py -- point --package-dir at the "
                 f"package folder itself.")

    os.makedirs(args.out_root, exist_ok=True)
    per_seed = {}

    for seed in args.seeds:
        seed_out = os.path.join(args.out_root, f"seed{seed}")
        weights = os.path.join(seed_out, "weights.pmt")
        test_csv = os.path.join(seed_out, "test_metrics.csv")

        if args.skip_existing and os.path.exists(test_csv):
            print(f"[seed {seed}] test metrics exist -> skipping run")
            m = parse_metrics_csv(test_csv)
            if m:
                per_seed[seed] = m
            continue

        # The trainer creates -o itself and ERRORS if it already exists, so we
        # must NOT pre-create seed_out. If a stale dir is here (e.g. a failed
        # prior run), remove it so training can proceed.
        if os.path.exists(seed_out):
            print(f"[seed {seed}] removing stale output dir {seed_out}")
            shutil.rmtree(seed_out, ignore_errors=True)

        # --- build a seed-patched private copy of the package -------------- #
        tmp_parent = tempfile.mkdtemp(prefix=f"q1fold_seed{seed}_")
        pkg_copy = os.path.join(tmp_parent, pkg_name)
        shutil.copytree(pkg_dir, pkg_copy)
        n = patch_seed_in_package(pkg_copy, seed)
        print(f"[seed {seed}] patched {n} seed line(s); package copy at {pkg_copy}")

        env = dict(os.environ)
        # tmp_parent first on PYTHONPATH so `import q1fold` resolves to the copy
        env["PYTHONPATH"] = tmp_parent + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if args.cuda_visible is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible)

        common = [args.python, "-m", pkg_name,
                  "-d", args.device, "-batch", str(args.batch),
                  "-j", str(args.workers), "--max-length", str(args.max_length),
                  "--config", args.config]

        # --- train -------------------------------------------------------- #
        train_cmd = common + ["train", args.train_file,
                              "--valid-file", args.valid_file,
                              "-n", str(args.epochs),
                              "-o", seed_out]
        print(f"[seed {seed}] training ...")
        rc = run_cmd(train_cmd, env,
                     os.path.join(args.out_root, f"seed{seed}_train.log"))
        if rc != 0:
            print(f"[seed {seed}] TRAIN FAILED (rc={rc}); see "
                  f"seed{seed}_train.log")
            shutil.rmtree(tmp_parent, ignore_errors=True)
            continue

        # --- test --------------------------------------------------------- #
        test_cmd = common + ["test", args.test_file,
                             "-w", weights, "-o", test_csv]
        print(f"[seed {seed}] testing ...")
        rc = run_cmd(test_cmd, env,
                     os.path.join(args.out_root, f"seed{seed}_test.log"))
        shutil.rmtree(tmp_parent, ignore_errors=True)
        if rc != 0:
            print(f"[seed {seed}] TEST FAILED (rc={rc}); see "
                  f"seed{seed}_test.log")
            continue

        m = parse_metrics_csv(test_csv)
        if m is None:
            print(f"[seed {seed}] could not parse {test_csv}")
            continue
        per_seed[seed] = m
        fp = m.get("f1_post")
        print(f"[seed {seed}] f1_post={fp}")

    # ---- aggregate ------------------------------------------------------- #
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
    print(f"Multi-seed summary  ({len(per_seed)} seeds: "
          f"{sorted(per_seed)})")
    print("=" * 64)
    summary_path = os.path.join(args.out_root, "multiseed_summary.csv")
    with open(summary_path, "w") as fh:
        fh.write("metric,mean,std,min,max,n," +
                 ",".join(f"seed{s}" for s in sorted(per_seed)) + "\n")
        for metric in ("f1_post", "f1", "precision_post", "recall_post",
                       "precision", "recall"):
            a = agg(metric)
            if a is None:
                continue
            vals, mean, std, lo, hi = a
            # 95% CI half-width (normal approx) for convenience
            ci = 1.96 * std / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
            print(f"  {metric:>16}: {mean:.4f} +/- {std:.4f}  "
                  f"(95% CI +/-{ci:.4f}, min {lo:.4f}, max {hi:.4f})")
            per_seed_vals = [f"{per_seed[s].get(metric, ''):.4f}"
                             if isinstance(per_seed[s].get(metric), float) else ""
                             for s in sorted(per_seed)]
            fh.write(f"{metric},{mean:.4f},{std:.4f},{lo:.4f},{hi:.4f},"
                     f"{len(vals)}," + ",".join(per_seed_vals) + "\n")
    print(f"\nSummary written to {summary_path}")
    print("\nReport f1_post mean +/- std as the headline. This varies classical "
          "init + data order; quantum init is fixed (see module docstring).")


# NOTE: alternative without copying the package -- if you'd rather, add ONE line
# near the top of q1fold/__init__.py main(), BEFORE the hardcoded seeds:
#     _S = int(os.environ.get("Q1FOLD_SEED", "42"))
# then replace the three `...seed(42)` with `...seed(_S)`. This runner would
# then just set env["Q1FOLD_SEED"]=seed and skip the copy/patch. The copy
# approach is the default only because you said there's no seed hook and it
# needs zero permanent source edits.

if __name__ == "__main__":
    main()