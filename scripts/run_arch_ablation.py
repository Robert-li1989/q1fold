#!/usr/bin/env python3
"""
Q1Fold architectural ablations (single-variable, single seed) on bpRNA-1m
=========================================================================

Each variant = the V20 config (Q1Fold full model) with exactly ONE change.
Train on TR0, select the best epoch on VL0 (post-processed F1), report
Precision / Recall / F1 on TS0 after post-processing.

    variant            change from V20
    -----------------  ----------------------------------------------------------
    full               none (re-trained under the same epoch budget as the rest)
    k3                 multiscale_scales [3]           single scale
    k35                multiscale_scales [3, 5]        two scales
    single_obs         single-qubit X/Y/Z only, no two-qubit correlators
    no_attn1d          use_1d_attention false          extractor feeds pairing directly
    no_relpos          use_relpos_2d false             refiner without distance channels
    pair32             pair_channels 32                narrower pair representation
    pair64             pair_channels 64                narrower pair representation
    pair256            pair_channels 256               wider (optional, not in the default run)
    dil64              dilations [1,2,4,8,16,32,64]    deeper dilation ladder
    distweight         distance-weighted loss (v15 settings: linear, scale 2, ref 100, cap 4,
                       positives only)

NOTE  V20 already uses pair_channels=128, so the "Pair channels 128" row of the
draft table IS the full model. The default run uses 32 and 64 (both narrower);
add pair256 if you also want a wider point.

QUICK CHECK on CPU (~1 min: builds every variant, prints params, runs a tiny forward)
    python run_arch_ablation.py --repo-root /path/containing/q1fold \
        --config multiscale_357_attn1d_opm_dilated_v20_pairchannels128_reg_config.json --dry-run

QUICK CHECK on GPU (a few min; no results written). Per variant:
    tiny forward (shapes) -> worst-case train step at batch_size x max_length
    (peak GPU memory, catches OOM) -> fit + test on a few real TR0 / VL0
    sequences (data pipeline, metrics, post-processing) -> rough epoch time.
    python run_arch_ablation.py --repo-root ... --config ... --dry-run --gpu 0 \
        --train-file $D/TR0.csv --valid-file $D/VL0.csv [--check-samples 16]
    (--train-file / --valid-file are optional here; without them only the
     synthetic steps run.)

RUN
    D=data/bpRNA-1m
    python run_arch_ablation.py --repo-root /path/containing/q1fold \
        --config multiscale_357_attn1d_opm_dilated_v20_pairchannels128_reg_config.json \
        --train-file $D/TR0.csv --valid-file $D/VL0.csv --test-file $D/TS0.csv \
        --seed 42 --max-epochs 20 --out-dir arch_ablation --gpu 0

    Split across GPUs with the same --out-dir, e.g.
        --variants full k3 k35 single_obs --gpu 0
        --variants no_attn1d no_relpos pair32 --gpu 1
        --variants pair64 dil64 distweight --gpu 2
    Finished variants are skipped; interrupted ones resume from their last epoch.

SUMMARY ONLY
    python run_arch_ablation.py --repo-root ... --config ... --out-dir arch_ablation --summarise-only

OUTPUT
    arch_ablation/<variant>/seed<s>/{weights.pmt, train_log.csv, result.json}
    arch_ablation/ablation_runs.csv     one row per variant (P, R, F1, raw metrics, params, epoch)
    arch_ablation/ablation.md           readable table with dF1 vs full
    arch_ablation/ablation_table.tex    the LaTeX table with P / R / F1 filled in
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


def _add_repo_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [here, os.path.dirname(here)]
    if os.environ.get("Q1FOLD_PROJECT_DIR"):
        cands.insert(0, os.path.dirname(os.path.abspath(os.environ["Q1FOLD_PROJECT_DIR"])))
    if "--repo-root" in sys.argv:
        cands.insert(0, os.path.abspath(sys.argv[sys.argv.index("--repo-root") + 1]))
    for p in reversed(cands):
        sys.path.insert(0, p)


_add_repo_to_path()

import numpy as np                                                       # noqa: E402
import pandas as pd                                                      # noqa: E402
import torch                                                             # noqa: E402
import torch.nn as nn                                                    # noqa: E402
import torch.nn.functional as F                                          # noqa: E402
from torch.utils.data import DataLoader, Subset                          # noqa: E402

from q1fold.model import Q1Fold                                          # noqa: E402
from q1fold.dataset import SeqDataset, pad_batch                         # noqa: E402
from q1fold import multiscale_quantum_conv1d as msq                      # noqa: E402


# =========================================================================== #
# Variants: (table label, "change" text, config overrides, single_obs flag)
# =========================================================================== #
VARIANTS = {
    "full":       ("Q1Fold (full)",          "---",                                  {}, False),
    "k3":         ("$k=3$ only",             "single scale",
                   {"multiscale_scales": [3]}, False),
    "k35":        ("$k=3,5$",                "two scales",
                   {"multiscale_scales": [3, 5]}, False),
    "single_obs": ("Single-qubit obs.\\ only", "no two-qubit correlators",            {}, True),
    "no_attn1d":  ("No 1D attention",        "extractor feeds pairing directly",
                   {"use_1d_attention": False}, False),
    "no_relpos":  ("No relative position",   "2D refiner without distance channels",
                   {"use_relpos_2d": False}, False),
    "pair32":     ("Pair channels 32",       "narrower pair representation",
                   {"pair_channels": 32}, False),
    "pair64":     ("Pair channels 64",       "narrower pair representation",
                   {"pair_channels": 64}, False),
    "pair256":    ("Pair channels 256",      "wider pair representation",
                   {"pair_channels": 256}, False),
    "dil64":      ("Dilations to 64",        "deeper dilation ladder",
                   {"dilated_dilations": [1, 2, 4, 8, 16, 32, 64]}, False),
    "distweight": ("Distance-weighted loss", "long-range positives up-weighted",
                   {"loss_distance_weight": "linear", "loss_distance_scale": 2.0,
                    "loss_distance_ref": 100.0, "loss_distance_cap": 4.0,
                    "loss_distance_apply": "positive"}, False),
}
DEFAULT_VARIANTS = ["full", "k3", "k35", "single_obs", "no_attn1d", "no_relpos",
                    "pair32", "pair64", "dil64", "distweight"]
# Midrule positions in the LaTeX table (after these variants)
QUANTUM_BLOCK = {"full", "k3", "k35", "single_obs"}


# =========================================================================== #
# Single-qubit-observables ablation: patch MultiScaleQuantumConv1D so each
# branch measures only X/Y/Z per qubit (3k features) instead of 3k + 9*C(k,2).
# The QNode reads self.pair_list and the forward reads self.n_features at call
# time, so emptying pair_list after construction removes the correlators from
# the circuit; the per-branch BatchNorms and out_channels are rebuilt to match.
# =========================================================================== #
_SINGLE_OBS = {"on": False}
_orig_msq_init = msq.MultiScaleQuantumConv1D.__init__


def _patched_msq_init(self, *a, **kw):
    _orig_msq_init(self, *a, **kw)
    if not _SINGLE_OBS["on"]:
        return
    for br in self.branches:
        br.pair_list = []
        br.n_features = 3 * br.n_qubits
    self.bns = nn.ModuleList([nn.BatchNorm1d(br.n_features) for br in self.branches])
    self.out_channels = sum(br.n_features for br in self.branches)


msq.MultiScaleQuantumConv1D.__init__ = _patched_msq_init


# =========================================================================== #
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
    cfg["max_epochs"] = args.max_epochs
    cfg["patience"] = args.patience
    return cfg


def variant_cfg(name, cfg):
    vcfg = deepcopy(cfg)
    vcfg.update(deepcopy(VARIANTS[name][2]))
    return vcfg


def build(name, cfg, device, train_len=1):
    vcfg = variant_cfg(name, cfg)
    vcfg["device"] = device
    _SINGLE_OBS["on"] = VARIANTS[name][3]
    try:
        net = Q1Fold(train_len=train_len, **vcfg)
    finally:
        _SINGLE_OBS["on"] = False
    return net


def count_params(net):
    ext = sum(p.numel() for p in net.resnet1d.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_feat = getattr(net.resnet1d, "out_channels", None)
    return {"extractor_params": ext, "model_params": tot, "extractor_channels": n_feat}


def loader(path, cfg, args, training=False):
    return DataLoader(SeqDataset(path, training=training, **cfg),
                      batch_size=args.batch_size, shuffle=training,
                      num_workers=args.num_workers, collate_fn=pad_batch)


def smoke_forward(net, L=24):
    """Tiny CPU forward to make sure every variant's shapes line up."""
    idx = torch.randint(0, 4, (1, L))
    emb = torch.nn.functional.one_hot(idx, 4).permute(0, 2, 1).float()
    batch = {"embedding": emb,
             "interaction_prior": torch.rand(1, L, L),
             "canonical_mask": torch.ones(1, L, L)}
    net.eval()
    with torch.no_grad():
        y, y0 = net(batch)
    assert y.shape == (1, L, L), y.shape


def worst_case_step(net, B, L, n_steps=2):
    """
    Full train step (forward + loss + backward + optimizer) on a synthetic
    batch at the longest allowed length. Returns (seconds for the last step,
    peak GPU memory in GiB). The first step absorbs cuDNN / kernel warm-up.
    Note: this updates the throwaway model's weights; nothing is saved.
    """
    dev = net.device
    idx = torch.randint(0, 4, (B, L))
    emb = F.one_hot(idx, 4).permute(0, 2, 1).float()
    batch = {"embedding": emb, "interaction_prior": torch.rand(B, L, L),
             "canonical_mask": torch.ones(B, L, L)}
    y = torch.zeros(B, L, L, dtype=torch.long, device=dev)
    ar = torch.arange(L // 2)
    y[:, ar, L - 1 - ar] = 1                     # some long-range positives (exercises distweight)
    y[:, L - 1 - ar, ar] = 1
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    net.train()
    dt = 0.0
    for _ in range(n_steps):
        t = time.time()
        net.optimizer.zero_grad()
        out = net(batch)
        loss = net.loss_func(out, y)
        loss.backward()
        if net.grad_clip:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=net.grad_clip)
        net.optimizer.step()
        torch.cuda.synchronize()
        dt = time.time() - t
    assert torch.isfinite(loss), f"non-finite loss {loss.item()}"
    return dt, torch.cuda.max_memory_allocated() / 2 ** 30


def _subset_loader(path, cfg, args, training, n):
    ds = SeqDataset(path, training=training, **cfg)
    rng = np.random.RandomState(0)
    idx = sorted(rng.choice(len(ds), min(n, len(ds)), replace=False).tolist())
    dl = DataLoader(Subset(ds, idx), batch_size=args.batch_size, shuffle=False,
                    num_workers=0, collate_fn=pad_batch)
    return dl, len(idx), len(ds)


def dry_run(args, cfg):
    dev = args.device
    on_gpu = str(dev).startswith("cuda")
    if on_gpu and not torch.cuda.is_available():
        sys.exit("\nCUDA is not available in this shell (torch.cuda.is_available() is False).\n"
                 "Check nvidia-smi / CUDA_VISIBLE_DEVICES, or drop --gpu to check on CPU.")
    qcfg = {**cfg, "verbose": False}

    sub_tr = sub_va = None
    if args.train_file:
        sub_tr, k_tr, n_tr = _subset_loader(args.train_file, cfg, args, True, args.check_samples)
    if args.valid_file:
        sub_va, k_va, n_va = _subset_loader(args.valid_file, cfg, args, False, args.check_samples)

    if on_gpu:
        p = torch.cuda.get_device_properties(0)
        gpu_mem = p.total_memory / 2 ** 30
        print(f"\nGPU sanity check on {p.name} ({gpu_mem:.1f} GiB), "
              f"worst case batch {args.batch_size} x L {args.max_length}")
    else:
        print("\nSanity check on CPU (tiny forward only; pass --gpu for the full check)")
    if sub_tr is not None:
        print(f"  real-data check: {k_tr} of {n_tr:,} TR0 sequences"
              + (f", {k_va} of {n_va:,} VL0 sequences" if sub_va is not None else ""))

    failed = []
    for v in args.variants:
        set_seed(0)
        net = build(v, qcfg, dev, train_len=len(sub_tr) if sub_tr is not None else 1)
        pc = count_params(net)
        msg = (f"  {v:11s} ch {str(pc['extractor_channels']):>4}  ext {pc['extractor_params']:>5,}  "
               f"model {pc['model_params']:>10,}")
        try:
            smoke_forward(net)
            if on_gpu:
                dt, peak = worst_case_step(net, args.batch_size, args.max_length)
                warn = "  <-- tight" if peak > 0.9 * gpu_mem else ""
                msg += f" | worst step {dt:5.2f}s, peak {peak:5.1f} GiB{warn}"
            if sub_tr is not None:
                t = time.time()
                tr_m = net.fit(sub_tr)
                t_tr = time.time() - t
                assert np.isfinite(tr_m["loss"]), f"non-finite train loss {tr_m['loss']}"
                est = t_tr / k_tr * n_tr
                if sub_va is not None:
                    t = time.time()
                    va_m = net.test(sub_va)
                    est += (time.time() - t) / k_va * n_va
                    assert "f1_post" in va_m
                msg += f" | ~{est / 60:5.1f} min/epoch"
            msg += "  OK"
        except torch.cuda.OutOfMemoryError:
            msg += "  OOM"
            failed.append((v, "out of GPU memory"))
        except Exception as e:                                   # report and keep going
            msg += f"  FAILED: {type(e).__name__}: {e}"
            failed.append((v, repr(e)))
        print(msg, flush=True)
        del net
        if on_gpu:
            torch.cuda.empty_cache()

    if failed:
        print(f"\n  {len(failed)} variant(s) failed: " + ", ".join(v for v, _ in failed))
        sys.exit(1)
    print("\n  all variants passed"
          + ("; epoch times are rough (few sequences, first-batch overhead included)"
             if sub_tr is not None else ""))


# --------------------------------------------------------------------------- #
def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _set_rng_state(st):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def train_one(name, seed, cfg, loaders, args):
    run_dir = os.path.join(args.out_dir, name, f"seed{seed}")
    result_file = os.path.join(run_dir, "result.json")
    if os.path.exists(result_file):
        print(f"\n[skip] {name}: {result_file} exists")
        return json.load(open(result_file))
    os.makedirs(run_dir, exist_ok=True)

    train_loader, valid_loader, test_loader = loaders
    set_seed(seed)
    net = build(name, cfg, args.device, train_len=len(train_loader))
    set_seed(seed)          # quantum branches re-seed the global RNG in __init__
    pc = count_params(net)
    print(f"\n{'=' * 72}\n  {name}: {VARIANTS[name][0]}  |  seed {seed}  |  "
          f"extractor {pc['extractor_params']:,} params, {pc['extractor_channels']} ch  |  "
          f"model {pc['model_params']:,}\n{'=' * 72}")

    weights = os.path.join(run_dir, "weights.pmt")
    logfile = os.path.join(run_dir, "train_log.csv")
    ckpt = os.path.join(run_dir, "last.pt")
    best_f1, best_epoch, bad, start, t_prev = -1.0, -1, 0, 0, 0.0

    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=args.device, weights_only=False)
        net.load_state_dict(ck["model"])
        net.optimizer.load_state_dict(ck["optimizer"])
        if getattr(net, "scheduler", None) is not None and ck.get("scheduler") is not None:
            net.scheduler.load_state_dict(ck["scheduler"])
        best_f1, best_epoch, bad = ck["best_f1"], ck["best_epoch"], ck["bad"]
        start, t_prev = ck["epoch"] + 1, ck["elapsed_s"]
        _set_rng_state(ck["rng"])
        if os.path.exists(logfile):
            lg = pd.read_csv(logfile)
            lg[lg.epoch <= ck["epoch"]].to_csv(logfile, index=False)
        print(f"  [resume] epoch {start} (best valid f1_post {best_f1:.4f} @ {best_epoch})")
    else:
        for f in (logfile, weights):
            if os.path.exists(f):
                os.remove(f)
    t0 = time.time() - t_prev

    for epoch in range(start, args.max_epochs):
        if bad > args.patience:
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
        tmp = ckpt + ".tmp"
        torch.save({"epoch": epoch, "model": net.state_dict(),
                    "optimizer": net.optimizer.state_dict(),
                    "scheduler": (net.scheduler.state_dict()
                                  if getattr(net, "scheduler", None) is not None else None),
                    "best_f1": best_f1, "best_epoch": best_epoch, "bad": bad,
                    "elapsed_s": time.time() - t0, "rng": _rng_state()}, tmp)
        os.replace(tmp, ckpt)

    net.load_state_dict(torch.load(weights, map_location=args.device, weights_only=True))
    te = net.test(test_loader)
    result = {"variant": name, "label": VARIANTS[name][0], "seed": seed,
              "best_epoch": best_epoch, "valid_f1_post": best_f1,
              "test_precision": te["precision_post"], "test_recall": te["recall_post"],
              "test_f1": te["f1_post"],
              "test_precision_raw": te["precision"], "test_recall_raw": te["recall"],
              "test_f1_raw": te["f1"], "max_epochs": args.max_epochs,
              "overrides": VARIANTS[name][2], "single_qubit_obs_only": VARIANTS[name][3],
              **pc, "train_time_s": time.time() - t0}
    tmp = result_file + ".tmp"
    json.dump(result, open(tmp, "w"), indent=2)
    os.replace(tmp, result_file)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"  TS0  P {result['test_precision']:.4f}  R {result['test_recall']:.4f}  "
          f"F1 {result['test_f1']:.4f}")
    del net
    torch.cuda.empty_cache()
    return result


# =========================================================================== #
def summarise(out_dir, seed):
    rows = []
    for v in VARIANTS:
        f = os.path.join(out_dir, v, f"seed{seed}", "result.json")
        if os.path.exists(f):
            rows.append(json.load(open(f)))
    if not rows:
        print("  no finished runs yet")
        return
    order = list(VARIANTS)
    runs = pd.DataFrame(rows).sort_values("variant", key=lambda c: c.map(order.index))
    runs.to_csv(os.path.join(out_dir, "ablation_runs.csv"), index=False)
    full = runs[runs.variant == "full"]
    f_full = float(full.test_f1.iloc[0]) if len(full) else None

    md = ["| Variant | Change | Extractor ch | Model params | Best ep | P | R | F1 | ΔF1 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in runs.iterrows():
        d = f"{r.test_f1 - f_full:+.3f}" if f_full is not None and r.variant != "full" else "–"
        md.append(f"| {r.label} | {VARIANTS[r.variant][1]} | {r.extractor_channels} | "
                  f"{int(r.model_params):,} | {int(r.best_epoch)} | {r.test_precision:.3f} | "
                  f"{r.test_recall:.3f} | {r.test_f1:.3f} | {d} |")
    ep = int(runs.max_epochs.max())
    md.append(f"\nbpRNA-TS0, post-processed; best epoch selected on VL0; "
              f"{ep}-epoch budget; single seed ({seed}).")
    md = "\n".join(md)
    open(os.path.join(out_dir, "ablation.md"), "w").write(md + "\n")
    print("\n" + md)

    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{Single-variable ablations from the full model. Each row changes one",
           r"component; all other settings are held fixed. bpRNA-TS0, post-processed,",
           f"single seed, {ep}-epoch budget.}}",
           r"\label{tab:ablation}", r"\small", r"\begin{tabular}{llccc}", r"\toprule",
           r"Variant & Change from full model & P & R & F1 \\", r"\midrule"]
    prev_block = None
    for _, r in runs.iterrows():
        block = 0 if r.variant == "full" else (1 if r.variant in QUANTUM_BLOCK else 2)
        if prev_block is not None and block != prev_block:
            tex.append(r"\midrule")
        prev_block = block
        tex.append(f"{r.label} & {VARIANTS[r.variant][1]} & {r.test_precision:.3f} & "
                   f"{r.test_recall:.3f} & {r.test_f1:.3f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(out_dir, "ablation_table.tex"), "w").write("\n".join(tex) + "\n")
    print(f"\n  wrote {out_dir}/ablation_runs.csv, ablation.md, ablation_table.tex")


# =========================================================================== #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", default=None, help="folder that CONTAINS the q1fold/ package")
    p.add_argument("--config", required=True, help="V20 config json")
    p.add_argument("--train-file")
    p.add_argument("--valid-file")
    p.add_argument("--test-file")
    p.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                   help=f"any of: {' '.join(VARIANTS)}")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=20, help="same budget for every variant")
    p.add_argument("--patience", type=int, default=None,
                   help="early-stop patience (default = max-epochs, i.e. none)")
    p.add_argument("--out-dir", default="arch_ablation")
    p.add_argument("--device", default=None)
    p.add_argument("--gpu", default=None, help="GPU id; sets CUDA_VISIBLE_DEVICES")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--cache-dir", default=None, help="optional SeqDataset cache")
    p.add_argument("--dry-run", action="store_true",
                   help="sanity check, no results written: CPU tiny forward, or with --gpu a "
                        "worst-case train step + a few real batches per variant")
    p.add_argument("--check-samples", type=int, default=16,
                   help="real TR0 / VL0 sequences used per variant in --dry-run")
    p.add_argument("--summarise-only", action="store_true")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    if a.gpu is not None:
        a.device = "cuda:0"
    elif a.device is None:
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.patience is None:
        a.patience = a.max_epochs
    a.variants = list(dict.fromkeys(a.variants))
    bad = [v for v in a.variants if v not in VARIANTS]
    if bad:
        p.error(f"unknown variant(s) {bad}; choose from {list(VARIANTS)}")
    if not (a.dry_run or a.summarise_only) and not (a.train_file and a.valid_file and a.test_file):
        p.error("--train-file, --valid-file and --test-file are required")
    return a


def main():
    args = parse_args()
    cfg = load_config(args.config, args)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.summarise_only:
        summarise(args.out_dir, args.seed)
        return

    if args.dry_run:
        dry_run(args, cfg)
        return

    print("\nVariant audit (CPU build)")
    audit = {}
    for v in args.variants:
        set_seed(0)
        net = build(v, {**cfg, "verbose": False}, "cpu")
        audit[v] = count_params(net)
        print(f"  {v:11s} extractor ch {str(audit[v]['extractor_channels']):>4}  "
              f"extractor params {audit[v]['extractor_params']:>6,}  "
              f"model params {audit[v]['model_params']:>11,}   {VARIANTS[v][2] or ''}"
              f"{'single-qubit obs only' if VARIANTS[v][3] else ''}")
        del net

    json.dump({**vars(args), "config": cfg, "audit": audit,
               "variant_overrides": {v: VARIANTS[v][2] for v in args.variants},
               "timestamp": str(datetime.now()), "torch": torch.__version__},
              open(os.path.join(args.out_dir, f"run_config_{'_'.join(args.variants)}.json"), "w"),
              indent=2, default=str)

    loaders = (loader(args.train_file, cfg, args, training=True),
               loader(args.valid_file, cfg, args),
               loader(args.test_file, cfg, args))
    t = time.time()
    for v in args.variants:
        train_one(v, args.seed, cfg, loaders, args)
    print(f"\nFinished {len(args.variants)} variants in {(time.time() - t) / 3600:.1f} h")
    summarise(args.out_dir, args.seed)


if __name__ == "__main__":
    main()