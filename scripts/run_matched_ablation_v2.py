#!/usr/bin/env python3
"""
Q1Fold Table 2 -- matched control of the 1D extractor on bpRNA (journal version)
=================================================================================

Trains three variants of the FULL current Q1Fold pipeline (the config used for
Table 1: 1D attention -> OPM pairing -> rel-pos -> dilated 2D refiner -> loss),
changing ONLY the 1D extractor (`self.resnet1d`):

    variant       extractor_type          extractor params
    -----------   ---------------------   --------------------------------
    quantum       multiscale_quantum      69 variational + 702 BN = 771
    matched       matched_classical       69 feature-map + 702 BN = 771  (asserted equal)
    competitive   competitive_classical   6,777 + 702 BN = 7,479  (asserted <= 7,710)

Everything else is shared: same config, same optimizer group/LR/weight decay/
grad clip, same scheduler, same seeds, same splits, same post-processing and
the same model-selection rule (best validation F1 after post-processing).

Data: bpRNA-1m TR0 (train) / VL0 (model selection) / TS0 (reported).
Reported: Precision, Recall, F1 on TS0 after post-processing (raw also logged).

SETUP: none -- the q1fold package in this repository already includes the classical extractors.

QUICK CHECK (no training; builds each variant and prints/asserts param counts)
    python run_matched_ablation_v2.py --config <table1_config>.json --dry-run

RUN
    python run_matched_ablation_v2.py \\
        --config     multiscale_357_attn1d_opm_dilated_v20_pairchannels128_reg_config.json \\
        --train-file bpRNA_TR0.csv --valid-file bpRNA_VL0.csv --test-file bpRNA_TS0.csv \\
        --variants   quantum matched competitive \\
        --seeds      42 \\
        --out-dir    table2_bprna/ --gpu 0

    Re-running with the same --out-dir skips finished (variant, seed) runs, so
    variants can be split across GPUs (one --variants each) and seeds added later.

PARAMETER-BUDGET SWEEP (competitive extractor only, same architecture, wider)
    python run_matched_ablation_v2.py --config <...>.json \\
        --train-file ... --valid-file ... --test-file ... \\
        --budget-mults 10 33 100 333 1000 3333 10000 --seeds 42 \\
        --out-dir table2_bprna --gpu 0
    Budgets are multiples of the quantum extractor's 771 params; x10 is the
    existing 'competitive' run and is reused. Each multiple can also be run on
    its own GPU (e.g. --budget-mults 3333 --gpu 3). Then:
    python run_matched_ablation_v2.py --config <...>.json --summarise-only \\
        --out-dir table2_bprna --quantum-ref <P> <R> <F1>
    -> table2.md, scaling.csv, scaling.png/.pdf (F1 vs extractor params, log x).

OUTPUT
    table2_bprna/<variant>/seed<k>/{weights.pmt, train_log.csv, result.json}
    table2_bprna/table2_runs.csv      one row per (variant, seed)
    table2_bprna/table2_summary.csv   mean / std over seeds per variant
    table2_bprna/table2.md            paper-ready table
"""

import argparse
import json
import os
import random
import sys
import time
from copy import deepcopy
from datetime import datetime

# --gpu must be applied before torch initialises CUDA
if "--gpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[sys.argv.index("--gpu") + 1]

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Find the q1fold package: $Q1FOLD_PROJECT_DIR (path to the q1fold/ folder),
# or a q1fold/ folder next to this script, or this script sitting inside it.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_candidates = [SCRIPT_DIR, os.path.dirname(SCRIPT_DIR)]
if os.environ.get("Q1FOLD_PROJECT_DIR"):
    _candidates.insert(0, os.path.dirname(os.path.abspath(os.environ["Q1FOLD_PROJECT_DIR"])))
for _p in reversed(_candidates):
    sys.path.insert(0, _p)

from q1fold.model import Q1Fold                                          # noqa: E402
from q1fold.dataset import SeqDataset, pad_batch                          # noqa: E402
from q1fold.matched_extractors import param_breakdown, hidden_for_budget  # noqa: E402

# --------------------------------------------------------------------------- #
# Quantum-extractor seeding (runner-side; multiscale_quantum_conv1d.py untouched)
# --------------------------------------------------------------------------- #
# _QuantumBranch.__init__ calls torch.manual_seed(seed) / np.random.seed(seed)
# with a hard-coded base seed of 42. That (a) fixes the circuit's initial
# angles for every run and (b) RESETS the global RNG, so every module built
# after it (attention, pairing, 2D refiner) would get identical initial
# weights regardless of the run seed. The wrapper below gives the circuit the
# run seed (branches use seed, seed+1, seed+2 as before; seed 42 reproduces
# the original angles exactly) and restores the global RNG state afterwards,
# so the rest of the model is initialised from the run seed -- the same
# protocol as the classical variants.
_RUN_SEED = [42]
try:
    import q1fold.multiscale_quantum_conv1d as _mq

    if not getattr(_mq.MultiScaleQuantumConv1D, "_runner_seeded", False):
        _orig_mq_init = _mq.MultiScaleQuantumConv1D.__init__

        def _seeded_mq_init(self, *a, **k):
            k["seed"] = _RUN_SEED[0]
            state = _get_rng_state()
            try:
                _orig_mq_init(self, *a, **k)
            finally:
                _set_rng_state(state)

        _mq.MultiScaleQuantumConv1D.__init__ = _seeded_mq_init
        _mq.MultiScaleQuantumConv1D._runner_seeded = True
except ImportError:          # PennyLane missing: classical variants still run
    pass

QUANTUM_PARAMS = 771           # 69 variational + 702 BatchNorm affine (scales 3/5/7)
BASE_VARIANTS = ["quantum", "matched", "competitive"]


def variant_spec(name):
    """
    name -> (extractor_type, label, extra_cfg, budget or None)

      quantum          the Table 1 quantum extractor
      matched          parameter-matched classical (771)
      competitive      competitive classical, budget 10 x 771 (= competitive_x10)
      competitive_xM   same architecture, widened to fit M x 771 params
      plain_conv       Conv1d(4->36/105/210, k=3/5/7) + BN, linear      (9,465)
      plain_conv_relu  same, with ReLU after each conv                  (9,465)
    """
    if name == "quantum":
        return "multiscale_quantum", "Q1Fold (quantum)", {}, None
    if name == "matched":
        return "matched_classical", "Classical, parameter-matched", {}, None
    if name == "plain_conv":
        return "plain_conv", "Plain Conv1d ×3 (linear)", {"plain_conv_activation": None}, None
    if name == "plain_conv_relu":
        return ("plain_conv", "Plain Conv1d ×3 + ReLU",
                {"plain_conv_activation": "relu"}, None)
    if name == "competitive":
        name = "competitive_x10"
    if name.startswith("competitive_x"):
        mult = float(name[len("competitive_x"):])
        budget = int(round(mult * QUANTUM_PARAMS))
        hidden = hidden_for_budget(budget)
        label = f"Classical, competitive (≤{mult:g}× quantum)"
        return "competitive_classical", label, {"competitive_hidden": hidden}, budget
    raise ValueError(f"unknown variant {name!r}")


def mult_to_variant(m):
    """10 keeps the original 'competitive' name so its finished runs are reused."""
    return "competitive" if float(m) == 10 else f"competitive_x{m:g}"


# --------------------------------------------------------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path, args):
    cfg = {k: v for k, v in json.load(open(path)).items() if not k.startswith("_")}
    cfg.setdefault("max_len", args.max_length)
    cfg["batch_size"] = args.batch_size
    cfg["cache_path"] = args.cache_dir
    cfg["verbose"] = not args.quiet
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.patience is not None:
        cfg["patience"] = args.patience
    return cfg


def build(variant, cfg, device, train_len=1, seed=42):
    vcfg = deepcopy(cfg)
    ext_type, _, extra, _ = variant_spec(variant)
    vcfg.update(extra)
    vcfg["extractor_type"] = ext_type
    vcfg["extractor_seed"] = seed   # classical extractor init follows the run seed
    _RUN_SEED[0] = seed             # quantum extractor init follows the run seed
    vcfg["device"] = device
    return Q1Fold(train_len=train_len, **vcfg)


def model_params(net):
    ext = param_breakdown(net.resnet1d)
    ext["model_total"] = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return ext


def check_budgets(counts):
    """Fail loudly if the fairness constraints are violated."""
    if "quantum" in counts and "matched" in counts:
        q, m = counts["quantum"], counts["matched"]
        assert (q["feature_map"], q["batchnorm"]) == (m["feature_map"], m["batchnorm"]), \
            f"matched extractor {m} != quantum extractor {q}"
    for v, c in counts.items():
        budget = variant_spec(v)[3]
        if budget is not None:
            assert c["total"] <= budget, f"{v}: {c['total']} params > budget {budget}"
    if "quantum" in counts and counts["quantum"]["total"] != QUANTUM_PARAMS:
        print(f"  ! quantum extractor has {counts['quantum']['total']} params, not 771 -- "
              f"check multiscale_scales / multiscale_layers / position encoding in the config.")


def loader(path, cfg, args, training=False):
    return DataLoader(SeqDataset(path, training=training, **cfg),
                      batch_size=args.batch_size, shuffle=training,
                      num_workers=args.num_workers, collate_fn=pad_batch)


# --------------------------------------------------------------------------- #
def _get_rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _trim_log(logfile, last_epoch):
    """Drop log rows past the checkpointed epoch (written just before a crash)."""
    if os.path.exists(logfile):
        log = pd.read_csv(logfile)
        log[log.epoch <= last_epoch].to_csv(logfile, index=False)


def train_one(variant, seed, cfg, loaders, args):
    run_dir = os.path.join(args.out_dir, variant, f"seed{seed}")
    result_file = os.path.join(run_dir, "result.json")
    if os.path.exists(result_file):
        print(f"\n[skip] {variant} seed={seed}: {result_file} exists")
        return json.load(open(result_file))
    os.makedirs(run_dir, exist_ok=True)

    set_seed(seed)
    train_loader, valid_loader, test_loader = loaders
    net = build(variant, cfg, args.device, train_len=len(train_loader), seed=seed)
    set_seed(seed)   # re-seed after construction in case any module reset the global RNG
    params = model_params(net)
    print(f"\n{'=' * 72}\n  {variant_spec(variant)[1]}  |  seed {seed}  |  extractor params "
          f"{params['feature_map']} + {params['batchnorm']} BN = {params['total']}  |  "
          f"model total {params['model_total']:,}\n{'=' * 72}")

    weights = os.path.join(run_dir, "weights.pmt")
    logfile = os.path.join(run_dir, "train_log.csv")
    ckpt_file = os.path.join(run_dir, "last.pt")
    best_f1, best_epoch, bad, start_epoch, t_prev = -1.0, -1, 0, 0, 0.0
    patience = cfg.get("patience", 30)
    max_epochs = cfg.get("max_epochs", 1000)

    if os.path.exists(ckpt_file):
        # Resume an interrupted run from its last completed epoch.
        ck = torch.load(ckpt_file, map_location=args.device, weights_only=False)
        net.load_state_dict(ck["model"])
        net.optimizer.load_state_dict(ck["optimizer"])
        if getattr(net, "scheduler", None) is not None and ck.get("scheduler") is not None:
            net.scheduler.load_state_dict(ck["scheduler"])
        best_f1, best_epoch, bad = ck["best_f1"], ck["best_epoch"], ck["bad"]
        start_epoch, t_prev = ck["epoch"] + 1, ck["elapsed_s"]
        _set_rng_state(ck["rng"])
        _trim_log(logfile, ck["epoch"])
        print(f"  [resume] from epoch {start_epoch} (best valid f1_post {best_f1:.4f} "
              f"at epoch {best_epoch})")
    else:
        # No checkpoint: anything left in the folder is from a run that died
        # before checkpointing existed -- start it cleanly from epoch 0.
        for f in (logfile, weights):
            if os.path.exists(f):
                os.remove(f)
                print(f"  [restart] removed stale {os.path.basename(f)}")
    t0 = time.time() - t_prev

    stopped_early = False
    for epoch in range(start_epoch, max_epochs):
        if bad > patience:          # resumed after the early-stop epoch was reached
            stopped_early = True
            break
        net.current_epoch = epoch
        tr_m = net.fit(train_loader)
        va_m = net.test(valid_loader)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr_m.items()},
               **{f"valid_{k}": v for k, v in va_m.items()},
               "lr": net.optimizer.param_groups[0]["lr"], "elapsed_s": time.time() - t0}
        pd.DataFrame([row]).to_csv(logfile, mode="a", header=not os.path.exists(logfile),
                                   index=False)

        star = ""
        if va_m["f1_post"] > best_f1:
            best_f1, best_epoch, bad = va_m["f1_post"], epoch, 0
            torch.save(net.state_dict(), weights)
            star = "  * best"
        else:
            bad += 1
        print(f"  ep {epoch:3d}  train_f1 {tr_m['f1']:.4f}  valid_f1_post {va_m['f1_post']:.4f}  "
              f"P {va_m['precision_post']:.4f}  R {va_m['recall_post']:.4f}  "
              f"{time.time() - t0:7.0f}s{star}")

        # Checkpoint after every epoch (written atomically so a crash mid-write
        # can never leave a corrupt last.pt behind).
        tmp = ckpt_file + ".tmp"
        torch.save({
            "epoch": epoch, "model": net.state_dict(),
            "optimizer": net.optimizer.state_dict(),
            "scheduler": (net.scheduler.state_dict()
                          if getattr(net, "scheduler", None) is not None else None),
            "best_f1": best_f1, "best_epoch": best_epoch, "bad": bad,
            "elapsed_s": time.time() - t0, "rng": _get_rng_state(),
        }, tmp)
        os.replace(tmp, ckpt_file)

        if bad > patience:
            stopped_early = True
            break
    if stopped_early:
        print(f"  early stop (patience {patience})")

    net.load_state_dict(torch.load(weights, map_location=args.device, weights_only=True))
    te_m = net.test(test_loader)
    result = {
        "variant": variant, "seed": seed, "best_epoch": best_epoch,
        "valid_f1_post": best_f1,
        "test_precision": te_m["precision_post"], "test_recall": te_m["recall_post"],
        "test_f1": te_m["f1_post"],
        "test_precision_raw": te_m["precision"], "test_recall_raw": te_m["recall"],
        "test_f1_raw": te_m["f1"],
        "competitive_hidden": variant_spec(variant)[2].get("competitive_hidden"),
        "extractor_feature_params": params["feature_map"],
        "extractor_bn_params": params["batchnorm"],
        "extractor_total_params": params["total"],
        "model_total_params": params["model_total"],
        "train_time_s": time.time() - t0,
    }
    tmp = result_file + ".tmp"
    json.dump(result, open(tmp, "w"), indent=2)
    os.replace(tmp, result_file)
    if os.path.exists(ckpt_file):
        os.remove(ckpt_file)        # run finished; weights.pmt holds the best model
    print(f"  TS0  P {result['test_precision']:.4f}  R {result['test_recall']:.4f}  "
          f"F1 {result['test_f1']:.4f}")
    del net
    torch.cuda.empty_cache()
    return result


def _known_variant(name):
    try:
        variant_spec(name)
        return True
    except (ValueError, TypeError):
        return False


def summarise(out_dir, quantum_ref=None):
    rows = []
    for v in sorted(os.listdir(out_dir)):
        vdir = os.path.join(out_dir, v)
        if not (os.path.isdir(vdir) and _known_variant(v)):
            continue
        for s in sorted(os.listdir(vdir)):
            f = os.path.join(vdir, s, "result.json")
            if os.path.exists(f):
                rows.append(json.load(open(f)))
    if not rows:
        return
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(out_dir, "table2_runs.csv"), index=False)

    metrics = ["test_precision", "test_recall", "test_f1"]
    agg = runs.groupby("variant").agg(
        n_seeds=("seed", "count"),
        extractor_feature_params=("extractor_feature_params", "first"),
        extractor_bn_params=("extractor_bn_params", "first"),
        extractor_total_params=("extractor_total_params", "first"),
        model_total_params=("model_total_params", "first"),
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
    )
    order = {"quantum": 0, "matched": 1}
    agg = agg.loc[sorted(agg.index, key=lambda v: (order.get(v, 2), agg.loc[v, "extractor_total_params"]))]
    agg["x_quantum"] = agg.extractor_total_params / QUANTUM_PARAMS
    agg.to_csv(os.path.join(out_dir, "table2_summary.csv"))

    def cell(r, m):
        s = r[f"{m}_std"]
        return f"{r[f'{m}_mean']:.3f}" + (f" ± {s:.3f}" if r["n_seeds"] > 1 and pd.notna(s) else "")

    lines = ["| 1D extractor | Extractor params (feature + BN) | × quantum | P | R | F1 |",
             "|---|---|---|---|---|---|"]
    if quantum_ref is not None and "quantum" not in agg.index:
        qp, qr, qf = quantum_ref
        lines.append(f"| Q1Fold (quantum), from Table 1 | 69 + 702 = 771 | 1.0 | "
                     f"{qp:.3f} | {qr:.3f} | {qf:.3f} |")
    for v, r in agg.iterrows():
        lines.append(f"| {variant_spec(v)[1]} | {int(r.extractor_feature_params):,} + "
                     f"{int(r.extractor_bn_params):,} = {int(r.extractor_total_params):,} | "
                     f"{r.x_quantum:,.1f} | "
                     f"{cell(r, 'test_precision')} | {cell(r, 'test_recall')} | {cell(r, 'test_f1')} |")
    n = int(agg.n_seeds.max())
    lines.append(f"\nbpRNA-TS0, post-processed, model selected on VL0; "
                 f"{'mean ± std over ' + str(n) + ' seeds' if n > 1 else 'single seed'}.")
    md = "\n".join(lines)
    open(os.path.join(out_dir, "table2.md"), "w").write(md + "\n")
    print("\n" + md)
    write_scaling(out_dir, agg, quantum_ref)


def write_scaling(out_dir, agg, quantum_ref=None):
    """Parameter-scaling curve of the competitive extractor vs the quantum reference."""
    comp = agg[[v.startswith("competitive") for v in agg.index]]
    if len(comp) < 2:
        return
    cols = ["n_seeds", "extractor_total_params", "x_quantum",
            "test_precision_mean", "test_precision_std", "test_recall_mean", "test_recall_std",
            "test_f1_mean", "test_f1_std"]
    comp[cols].to_csv(os.path.join(out_dir, "scaling.csv"))

    q_f1 = None
    if "quantum" in agg.index:
        q_f1 = agg.loc["quantum", "test_f1_mean"]
    elif quantum_ref is not None:
        q_f1 = quantum_ref[2]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed: wrote scaling.csv only)")
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    yerr = comp.test_f1_std.fillna(0) if comp.n_seeds.max() > 1 else None
    ax.errorbar(comp.extractor_total_params, comp.test_f1_mean, yerr=yerr,
                marker="o", capsize=3, label="Classical, competitive")
    if "matched" in agg.index:
        m = agg.loc["matched"]
        ax.errorbar([m.extractor_total_params], [m.test_f1_mean],
                    yerr=[m.test_f1_std] if m.n_seeds > 1 else None,
                    marker="s", linestyle="none", capsize=3, label="Classical, parameter-matched")
    for v, mk in (("plain_conv", "^"), ("plain_conv_relu", "v")):
        if v in agg.index:
            r = agg.loc[v]
            ax.errorbar([r.extractor_total_params], [r.test_f1_mean],
                        yerr=[r.test_f1_std] if r.n_seeds > 1 else None,
                        marker=mk, linestyle="none", capsize=3, label=variant_spec(v)[1])
    if q_f1 is not None:
        ax.axhline(q_f1, linestyle="--", color="k", linewidth=1, label="Q1Fold (quantum)")
        ax.plot([QUANTUM_PARAMS], [q_f1], marker="*", markersize=12, color="k", linestyle="none")
    ax.set_xscale("log")
    ax.set_xlabel("1D extractor parameters (log scale)")
    ax.set_ylabel("bpRNA-TS0 F1")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "scaling.png"), dpi=200)
    fig.savefig(os.path.join(out_dir, "scaling.pdf"))
    print(f"  wrote {out_dir}/scaling.csv, scaling.png, scaling.pdf")


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="The Q1Fold config used for Table 1")
    p.add_argument("--train-file", help="bpRNA TR0 csv")
    p.add_argument("--valid-file", help="bpRNA VL0 csv (model selection)")
    p.add_argument("--test-file", help="bpRNA TS0 csv (reported)")
    p.add_argument("--variants", nargs="+", default=None,
                   help="quantum, matched, competitive, competitive_x<M>, "
                        "plain_conv, plain_conv_relu "
                        "(default: quantum matched competitive, unless --budget-mults is given)")
    p.add_argument("--budget-mults", nargs="+", type=float, default=None,
                   help="competitive-extractor budget sweep, as multiples of the quantum "
                        "extractor's 771 params, e.g. 10 33 100 333 1000 3333 10000")
    p.add_argument("--quantum-ref", nargs=3, type=float, metavar=("P", "R", "F1"), default=None,
                   help="Table 1 quantum P R F1, used in table2.md / scaling plot "
                        "when the quantum variant is not trained here")
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--out-dir", default="table2_bprna")
    p.add_argument("--device", default=None)
    p.add_argument("--gpu", default=None, help="GPU id; sets CUDA_VISIBLE_DEVICES")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=None, help="override config")
    p.add_argument("--patience", type=int, default=None, help="override config")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-dir", default=None,
                   help="optional SeqDataset cache, shared by all variants (same data)")
    p.add_argument("--dry-run", action="store_true", help="build variants, check params, exit")
    p.add_argument("--summarise-only", action="store_true",
                   help="skip training; rebuild table2.md and the scaling plot from finished runs")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    if a.gpu is not None:
        a.device = "cuda:0"
    elif a.device is None:
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    variants = list(a.variants or [])
    variants += [mult_to_variant(m) for m in (a.budget_mults or [])]
    if not variants:
        variants = list(BASE_VARIANTS)
    a.variants = list(dict.fromkeys(variants))      # de-duplicate, keep order
    for v in a.variants:
        try:
            variant_spec(v)
        except (ValueError, TypeError) as e:
            p.error(str(e))
    if not (a.dry_run or a.summarise_only) and not (a.train_file and a.valid_file and a.test_file):
        p.error("--train-file, --valid-file and --test-file are required unless --dry-run")
    return a


def main():
    args = parse_args()
    cfg = load_config(args.config, args)
    os.makedirs(args.out_dir, exist_ok=True)

    # Parameter audit -- always runs, before any training.
    # The quantum and matched extractors are always audited (to check they are
    # equal); training only happens for --variants.
    counts = {}
    for v in dict.fromkeys(["quantum", "matched"] + args.variants):
        set_seed(0)
        cfg_quiet = {**cfg, "verbose": False}
        counts[v] = model_params(build(v, cfg_quiet, "cpu"))
    print("\nExtractor parameter audit")
    for v, c in counts.items():
        budget = variant_spec(v)[3]
        hidden = variant_spec(v)[2].get("competitive_hidden")
        extra = f"  hidden {hidden:>5}  budget {budget:>10,}" if budget else ""
        print(f"  {v:18s} feature {c['feature_map']:>10,}  BN {c['batchnorm']:>4}  "
              f"extractor {c['total']:>10,} ({c['total'] / QUANTUM_PARAMS:7.1f}x)  "
              f"model {c['model_total']:>12,}{extra}")
    check_budgets(counts)
    print("  budgets OK: matched == quantum; every competitive variant within its budget")
    if args.dry_run:
        return
    if args.summarise_only:
        summarise(args.out_dir, quantum_ref=args.quantum_ref)
        return

    json.dump({**vars(args), "config": cfg, "param_audit": counts,
               "timestamp": str(datetime.now()), "torch": torch.__version__},
              open(os.path.join(args.out_dir, "run_config.json"), "w"), indent=2, default=str)

    loaders = (loader(args.train_file, cfg, args, training=True),
               loader(args.valid_file, cfg, args),
               loader(args.test_file, cfg, args))
    for seed in args.seeds:
        for v in args.variants:
            train_one(v, seed, cfg, loaders, args)
    summarise(args.out_dir, quantum_ref=args.quantum_ref)


if __name__ == "__main__":
    main()