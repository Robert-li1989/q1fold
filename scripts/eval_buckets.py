"""
Q1Fold bucketed evaluation
===========================

Splits precision / recall / F1 / average precision (AP) by base-pair distance
|i - j| and by sequence length, so you can see WHERE a change helped or hurt,
instead of only the aggregate number.

It reuses the training pipeline exactly:
  * data     : SeqDataset(test_file, **config) + pad_batch collate
  * model    : q1fold(weights=weights.pmt, **config)   (rebuild + load_state_dict)
  * post-proc: utils.postprocessing (max one-to-one pairing)
  * metric   : the same triangular sigmoid->threshold(0.5) on the upper
               triangle (offset=1) that contact_f1 uses.

Because it mirrors contact_f1, the OVERALL macro numbers it prints (mean over
sequences) should match the f1_post / precision_post / recall_post in your test
log. Treat that as a load-correctness sanity check: if they don't match, the
config/weights pairing is wrong.

Per-bucket P/R/F1 are MICRO-averaged (TP/FP/FN pooled across the whole test
set within a bucket). AP is computed on RAW sigmoid scores over canonical
candidate pairs only (the canonical mask the model itself uses), pooled across
sequences within a bucket.

Distance buckets
----------------
Default --dist-edges 13,31,51,101 gives the paper's buckets
    1-12, 13-30, 31-50, 51-100, 101+ (= >100)
Edges are EXCLUSIVE upper bounds: edges "12,30,50,100" would give 1-11, 12-29,
30-49, 50-99, 100+ instead.

Modes
-----
1) Evaluate one model (or two side by side)
    python eval_buckets.py --test-file TS0.csv \
        --config v20.json --weights runs/v20/weights.pmt --name quantum \
        --pr-curve --out bucket_report

    Rebuild a variant from a base config with --set KEY=VALUE (JSON-parsed),
    e.g. a Table 2 competitive classical extractor from the v20 config:
        --set extractor_type=competitive_classical --set competitive_hidden=37 \
        --set extractor_seed=42

    Two models: add --config2/--weights2/--name2 (and --set2 ...).
    dAP is always printed as (model 2 - model 1).

2) Sequence-level bootstrap (single or paired)
    add --bootstrap 2000 : per-bucket AP CIs; with two models also a PAIRED
    bootstrap CI on dAP (the same resampled sequences for both models).

3) Save per-sequence histograms for multi-seed aggregation
    add --save-hist hist/quantum_seed42.npz --run-seed 42
    (one file per model x seed; rnafold_probe.py --save-hist writes the same
    format, so RNAfold can be aggregated alongside the models)

4) Aggregate (no model loading, no GPU)
    python eval_buckets.py --aggregate hist/ --ref quantum \
        --bootstrap 2000 --out summary/
    Groups the .npz files by model name, then reports per bucket:
      * AP for every run (and its macro f1_post, for the sanity check)
      * AP mean +- std over seeds, with a hierarchical bootstrap CI
        (resample test sequences; within each replicate also resample seeds)
      * dAP = ref - group, with CI and the bootstrap fraction P(dAP <= 0)
    and writes per_run_ap.csv, group_ap.csv, delta_ap.csv and
    distance_buckets.pdf/.png.

If the package isn't importable from the current directory, pass
--repo-root /path/to/dir_containing_the_q1fold_package.
"""

import argparse
import glob
import json
import math
import os
import random
import sys
import warnings

import numpy as np

try:
    import torch as tr
except ImportError:      # --aggregate does not need torch
    tr = None


# --------------------------------------------------------------------------- #
# Bucket edges
# --------------------------------------------------------------------------- #

def build_buckets(edges, open_end=True, start=1):
    """
    Turn a sorted list of interior edges into (lo, hi, label) tuples with hi
    exclusive. e.g. edges=[13,31,51,101] ->
      [1,13) [13,31) [31,51) [51,101) [101, inf)
      labels 1-12, 13-30, 31-50, 51-100, 101+
    """
    bounds = [start] + list(edges)
    buckets = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        buckets.append((lo, hi, f"{lo}-{hi-1}"))
    if open_end:
        buckets.append((bounds[-1], np.iinfo(np.int64).max, f"{bounds[-1]}+"))
    return buckets


def bucket_index(value, buckets):
    for i, (lo, hi, _) in enumerate(buckets):
        if lo <= value < hi:
            return i
    return -1


# --------------------------------------------------------------------------- #
# Metric accumulation
# --------------------------------------------------------------------------- #

class Tally:
    """Micro TP/FP/FN accumulator with true/pred support counts."""
    def __init__(self, n):
        self.tp = np.zeros(n, dtype=np.int64)
        self.fp = np.zeros(n, dtype=np.int64)
        self.fn = np.zeros(n, dtype=np.int64)

    def add(self, idx, tp, fp, fn):
        if idx < 0:
            return
        self.tp[idx] += tp
        self.fp[idx] += fp
        self.fn[idx] += fn

    def prf(self, i):
        tp, fp, fn = int(self.tp[i]), int(self.fp[i]), int(self.fn[i])
        p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f = (2 * p * r / (p + r)) if (p == p and r == r and (p + r) > 0) else float("nan")
        return p, r, f, tp, fp, fn


# --------------------------------------------------------------------------- #
# Core evaluation for one model
# --------------------------------------------------------------------------- #

def evaluate(model, loader, postprocessing, dist_buckets, len_buckets, th=0.5,
             collect_pr=False, pr_bins=1000, keep_per_seq=False):
    model.eval()
    nb = len(dist_buckets)

    dist_post = Tally(nb)
    len_post = Tally(len(len_buckets))
    dist_raw = Tally(nb)
    len_raw = Tally(len(len_buckets))

    # PR-frontier histograms over RAW sigmoid probability (pos vs neg per
    # bucket). Sweeping a threshold on these reconstructs the full per-bucket
    # precision/recall curve without storing every pair.
    if collect_pr:
        d_pos = np.zeros((nb, pr_bins), dtype=np.int64)
        d_neg = np.zeros((nb, pr_bins), dtype=np.int64)
        l_pos = np.zeros((len(len_buckets), pr_bins), dtype=np.int64)
        l_neg = np.zeros((len(len_buckets), pr_bins), dtype=np.int64)
    else:
        d_pos = d_neg = l_pos = l_neg = None

    # per-sequence histograms (for the bootstrap / multi-seed aggregation)
    per_seq = ({"ids": [], "lengths": [], "pos": [], "neg": []}
               if (collect_pr and keep_per_seq) else None)
    canonical_masked = None

    # per-sequence macro accumulators (to reconcile with reported f1_post)
    macro = {"p": [], "r": [], "f": []}

    with tr.no_grad():
        for batch in loader:
            ref_all = batch["contact"]                    # [B, L, L] (-1 pad)
            lengths = batch["length"]
            ids = batch.get("id") or [None] * len(lengths)
            fwd_batch = {k: v for k, v in batch.items() if k != "contact"}

            out = model(fwd_batch)
            y_logits = (out[0] if isinstance(out, tuple) else out).cpu()

            canonical = batch.get("canonical_mask")
            canonical_masked = canonical is not None
            y_post = postprocessing(y_logits, canonical)  # [B, L, L]

            for b in range(y_logits.shape[0]):
                l = int(lengths[b])
                if l < 2:
                    continue
                ref = ref_all[b, :l, :l]
                ref_bin = (ref > 0.5)

                iu, ju = tr.triu_indices(l, l, offset=1)
                r = ref_bin[iu, ju].cpu().numpy().astype(bool)
                dist = (ju - iu).cpu().numpy()
                lb = bucket_index(l, len_buckets)

                # Canonical-candidate mask for the PR histograms. Non-canonical
                # positions' logits are zeroed in the forward pass (prob 0.5)
                # and would pile up as false positives at any threshold <= 0.5.
                # Excluding them makes AP the separability among canonical
                # candidates. Tallies are left untouched so they still
                # reconcile with the reported contact_f1 numbers.
                if collect_pr and canonical is not None:
                    cm = canonical[b, :l, :l]
                    keep = (cm[iu, ju] > 0.5).cpu().numpy().astype(bool)
                else:
                    keep = None

                for tag, logits, dtally, ltally in (
                    ("post", y_post[b, :l, :l], dist_post, len_post),
                    ("raw",  y_logits[b, :l, :l], dist_raw, len_raw),
                ):
                    prob = tr.sigmoid(logits)
                    prob_u = prob[iu, ju].cpu().numpy()
                    p = (prob_u > th)

                    tp = p & r
                    fp = p & ~r
                    fn = ~p & r

                    ltally.add(lb, int(tp.sum()), int(fp.sum()), int(fn.sum()))

                    for di, (lo, hi, _) in enumerate(dist_buckets):
                        m = (dist >= lo) & (dist < hi)
                        if not m.any():
                            continue
                        dtally.add(di, int((tp & m).sum()),
                                   int((fp & m).sum()),
                                   int((fn & m).sum()))

                    # PR histograms (RAW scores only)
                    if collect_pr and tag == "raw":
                        bins = np.clip((prob_u * pr_bins).astype(np.int64),
                                       0, pr_bins - 1)
                        sel = keep if keep is not None else np.ones_like(r, dtype=bool)
                        rp = r & sel               # true pairs, canonical
                        rn = (~r) & sel            # false pairs, canonical
                        if lb >= 0:
                            l_pos[lb] += np.bincount(bins[rp], minlength=pr_bins)
                            l_neg[lb] += np.bincount(bins[rn], minlength=pr_bins)
                        sp = np.zeros((nb, pr_bins), dtype=np.int64)
                        sn = np.zeros((nb, pr_bins), dtype=np.int64)
                        for di, (lo, hi, _) in enumerate(dist_buckets):
                            m = (dist >= lo) & (dist < hi)
                            if not m.any():
                                continue
                            sp[di] = np.bincount(bins[m & rp], minlength=pr_bins)
                            sn[di] = np.bincount(bins[m & rn], minlength=pr_bins)
                        d_pos += sp
                        d_neg += sn
                        if per_seq is not None:
                            sid = ids[b] if ids[b] is not None else f"seq{len(per_seq['ids'])}"
                            per_seq["ids"].append(str(sid))
                            per_seq["lengths"].append(l)
                            per_seq["pos"].append(sp.astype(np.uint32))
                            per_seq["neg"].append(sn.astype(np.uint32))

                    # macro reconciliation only from the post path
                    if tag == "post":
                        TP, FP, FN = int(tp.sum()), int(fp.sum()), int(fn.sum())
                        pp = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                        rr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                        ff = (2 * pp * rr / (pp + rr)) if (pp + rr) > 0 else 0.0
                        macro["p"].append(pp)
                        macro["r"].append(rr)
                        macro["f"].append(ff)

    macro_summary = {k: float(np.mean(v)) if v else float("nan")
                     for k, v in macro.items()}
    return {
        "dist_post": dist_post, "len_post": len_post,
        "dist_raw": dist_raw, "len_raw": len_raw,
        "macro_post": macro_summary,
        "canonical_masked": bool(canonical_masked),
        "per_seq": per_seq,
        "pr": None if not collect_pr else {
            "d_pos": d_pos, "d_neg": d_neg,
            "l_pos": l_pos, "l_neg": l_neg, "bins": pr_bins,
        },
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _fmt(x):
    return "  n/a " if (x != x) else f"{x:6.3f}"


def print_table(title, buckets, tallyA, nameA, tallyB=None, nameB=None):
    print(f"\n=== {title} ===")
    if tallyB is None:
        print(f"{'bucket':>10} | {'P':>6} {'R':>6} {'F1':>6} | "
              f"{'TP':>7} {'FP':>7} {'FN':>7}  (true)")
        for i, (_, _, lab) in enumerate(buckets):
            p, r, f, tp, fp, fn = tallyA.prf(i)
            print(f"{lab:>10} | {_fmt(p)} {_fmt(r)} {_fmt(f)} | "
                  f"{tp:7d} {fp:7d} {fn:7d}  (true={tp+fn})")
    else:
        print(f"{'bucket':>10} | {nameA:>20} | {nameB:>20} | {'dF1':>7}")
        print(f"{'':>10} | {'P':>6} {'R':>6} {'F1':>6} | "
              f"{'P':>6} {'R':>6} {'F1':>6} |")
        for i, (_, _, lab) in enumerate(buckets):
            pa, ra, fa, tpa, fpa, fna = tallyA.prf(i)
            pb, rb, fb, tpb, fpb, fnb = tallyB.prf(i)
            df = (fb - fa) if (fa == fa and fb == fb) else float("nan")
            df_s = "  n/a " if (df != df) else f"{df:+6.3f}"
            print(f"{lab:>10} | {_fmt(pa)} {_fmt(ra)} {_fmt(fa)} | "
                  f"{_fmt(pb)} {_fmt(rb)} {_fmt(fb)} | {df_s} "
                  f" (true={tpa+fna})")


def write_csv(path, buckets, tally, model_name, kind):
    rows = ["bucket,precision,recall,f1,tp,fp,fn,true,pred"]
    for i, (_, _, lab) in enumerate(buckets):
        p, r, f, tp, fp, fn = tally.prf(i)
        rows.append(f"{lab},{p:.4f},{r:.4f},{f:.4f},{tp},{fp},{fn},"
                    f"{tp+fn},{tp+fp}")
    with open(path, "w") as fh:
        fh.write(f"# model={model_name} split={kind}\n")
        fh.write("\n".join(rows) + "\n")


# --------------------------------------------------------------------------- #
# PR frontier (threshold sweep over RAW scores)
# --------------------------------------------------------------------------- #

def average_precision(pos_hist, neg_hist):
    """
    Area under the precision-recall curve (average precision) from pos/neg
    probability histograms. Sweeps threshold high->low; standard step integral
    AP = sum_n (R_n - R_{n-1}) * P_n.
    """
    return float(ap_batch(pos_hist, neg_hist))


def ap_batch(pos, neg):
    """
    Vectorised AP over the last axis. pos/neg: [..., bins] (counts or
    bootstrap-weighted counts). Returns [...]; NaN where a row has no positives.
    Identical to the step integral in average_precision().
    """
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    tp = np.cumsum(pos[..., ::-1], axis=-1)           # highest-prob bins first
    fp = np.cumsum(neg[..., ::-1], axis=-1)
    total = tp[..., -1]
    denom = tp + fp
    prec = np.where(denom > 0, tp / np.where(denom > 0, denom, 1.0), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = tp / total[..., None]
    drec = np.diff(rec, axis=-1, prepend=0.0)
    with np.errstate(invalid="ignore"):
        ap = np.sum(drec * prec, axis=-1)
    return np.where(total > 0, ap, np.nan)


def pr_at_thresholds(pos_hist, neg_hist, thresholds, pr_bins):
    """P/R/F1 at each probability threshold, from the histograms."""
    total_pos = int(pos_hist.sum())
    out = []
    for t in thresholds:
        thr_bin = min(max(int(math.ceil(t * pr_bins)), 0), pr_bins)
        tp = int(pos_hist[thr_bin:].sum())
        fp = int(neg_hist[thr_bin:].sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        r = tp / total_pos if total_pos > 0 else float("nan")
        f = (2 * p * r / (p + r)) if (p == p and r == r and (p + r) > 0) else float("nan")
        out.append((t, p, r, f))
    return out


def print_ap_table(title, buckets, prA, keys, nameA, prB=None, nameB=None):
    """keys: (pos_key, neg_key) e.g. ('d_pos','d_neg')."""
    posk, negk = keys
    print(f"\n=== {title} :: AP (area under RAW PR curve) ===")
    if prB is None:
        print(f"{'bucket':>10} | {'AP':>6} | {'#true':>8}")
        for i, (_, _, lab) in enumerate(buckets):
            ap = average_precision(prA[posk][i], prA[negk][i])
            ntrue = int(prA[posk][i].sum())
            print(f"{lab:>10} | {_fmt(ap)} | {ntrue:8d}")
    else:
        print(f"{'bucket':>10} | {nameA[:6]:>6} | {nameB[:6]:>6} | {'dAP':>7} | {'#true':>8}")
        for i, (_, _, lab) in enumerate(buckets):
            apa = average_precision(prA[posk][i], prA[negk][i])
            apb = average_precision(prB[posk][i], prB[negk][i])
            dap = apb - apa if (apa == apa and apb == apb) else float("nan")
            dap_s = "  n/a " if (dap != dap) else f"{dap:+6.3f}"
            ntrue = int(prA[posk][i].sum())
            print(f"{lab:>10} | {_fmt(apa)} | {_fmt(apb)} | {dap_s} | {ntrue:8d}")


def print_pr_sweep(title, buckets, pr, keys, name, thresholds, pr_bins):
    posk, negk = keys
    print(f"\n=== {title} :: [{name}] P/R by threshold (RAW) ===")
    header = f"{'bucket':>10} |"
    for t in thresholds:
        header += f"  P@{t:<4} R@{t:<4}|"
    print(header)
    for i, (_, _, lab) in enumerate(buckets):
        rowvals = pr_at_thresholds(pr[posk][i], pr[negk][i], thresholds, pr_bins)
        line = f"{lab:>10} |"
        for (_, p, r, _f) in rowvals:
            line += f" {_fmt(p)}{_fmt(r)}|"
        print(line)


# --------------------------------------------------------------------------- #
# Threshold tuning (RAW F1; tuned on VALIDATION, reported on TEST)
# --------------------------------------------------------------------------- #

def _f1_at_bin(pos_hist, neg_hist, thr_bin):
    total_pos = int(pos_hist.sum())
    tp = int(pos_hist[thr_bin:].sum())
    fp = int(neg_hist[thr_bin:].sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / total_pos if total_pos > 0 else 0.0
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return f, p, r


def best_threshold(pos_hist, neg_hist, pr_bins, grid):
    best_f, best_t = -1.0, 0.5
    for t in grid:
        tb = min(max(int(round(t * pr_bins)), 0), pr_bins)
        f, _, _ = _f1_at_bin(pos_hist, neg_hist, tb)
        if f > best_f:
            best_f, best_t = f, t
    return best_t, best_f


def _pooled(pr, posk, negk):
    return pr[posk].sum(axis=0), pr[negk].sum(axis=0)


def tune_and_report(val_pr, test_pr, dist_buckets, name, pr_bins, grid):
    """
    Pick the RAW decision threshold that maximizes F1 on the VALIDATION
    histograms, then report the resulting F1 on the TEST histograms -- both a
    single global threshold and a per-distance-bucket policy.
    """
    posk, negk = "d_pos", "d_neg"
    tb_def = int(round(0.5 * pr_bins))
    print(f"\n=== Threshold tuning [{name}]  (RAW F1; tuned on VAL -> reported on TEST) ===")

    vpos, vneg = _pooled(val_pr, posk, negk)
    g_t, g_valf = best_threshold(vpos, vneg, pr_bins, grid)
    tpos, tneg = _pooled(test_pr, posk, negk)
    f_def, p_def, r_def = _f1_at_bin(tpos, tneg, tb_def)
    f_glob, p_glob, r_glob = _f1_at_bin(tpos, tneg, int(round(g_t * pr_bins)))
    print(f"  global th*={g_t:.2f} (val F1={g_valf:.3f}) -> "
          f"TEST F1 {f_def:.3f} @0.50 (P={p_def:.3f} R={r_def:.3f})  ->  "
          f"{f_glob:.3f} @{g_t:.2f} (P={p_glob:.3f} R={r_glob:.3f})")

    print(f"  per-distance-bucket:")
    TP = FP = FN = 0
    for i, (_, _, lab) in enumerate(dist_buckets):
        bt, _ = best_threshold(val_pr[posk][i], val_pr[negk][i], pr_bins, grid)
        f0, _, _ = _f1_at_bin(test_pr[posk][i], test_pr[negk][i], tb_def)
        tb = int(round(bt * pr_bins))
        f1b, pb, rb = _f1_at_bin(test_pr[posk][i], test_pr[negk][i], tb)
        tp = int(test_pr[posk][i][tb:].sum())
        fp = int(test_pr[negk][i][tb:].sum())
        fn = int(test_pr[posk][i].sum()) - tp
        TP += tp; FP += fp; FN += fn
        print(f"    {lab:>8}: th*={bt:.2f}  TEST F1 {f0:.3f} @0.50 -> "
              f"{f1b:.3f} @{bt:.2f} (P={pb:.3f} R={rb:.3f})")
    f_pb = (2 * TP / (2 * TP + FP + FN)) if (2 * TP + FP + FN) > 0 else 0.0
    print(f"    pooled under per-bucket policy: TEST F1 {f_pb:.3f}  "
          f"(vs {f_def:.3f} @0.50 global)")


# --------------------------------------------------------------------------- #
# Per-sequence histogram files (shared format with rnafold_probe.py)
# --------------------------------------------------------------------------- #

def save_hist(path, per_seq, name, seed, dist_buckets, pr_bins,
              canonical_masked, macro, source="eval_buckets"):
    if per_seq is None or not per_seq["ids"]:
        raise ValueError("no per-sequence histograms collected")
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        name=np.array(name), seed=np.array(int(seed)),
        labels=np.array([b[2] for b in dist_buckets]),
        los=np.array([b[0] for b in dist_buckets], dtype=np.int64),
        his=np.array([b[1] for b in dist_buckets], dtype=np.int64),
        pr_bins=np.array(int(pr_bins)),
        ids=np.array(per_seq["ids"]),
        lengths=np.array(per_seq["lengths"], dtype=np.int64),
        d_pos=np.stack(per_seq["pos"]), d_neg=np.stack(per_seq["neg"]),
        canonical_masked=np.array(bool(canonical_masked)),
        macro_f1_post=np.array(float(macro.get("f", float("nan")))),
        macro_p_post=np.array(float(macro.get("p", float("nan")))),
        macro_r_post=np.array(float(macro.get("r", float("nan")))),
        source=np.array(source),
    )
    os.replace(tmp, path)
    print(f"[hist] wrote {path}  ({len(per_seq['ids'])} sequences)")


def load_hist(path):
    z = np.load(path, allow_pickle=False)
    return {
        "path": path, "name": str(z["name"]), "seed": int(z["seed"]),
        "labels": [str(x) for x in z["labels"]], "pr_bins": int(z["pr_bins"]),
        "ids": z["ids"].astype(str), "lengths": z["lengths"].astype(np.int64),
        "d_pos": z["d_pos"], "d_neg": z["d_neg"],
        "canonical_masked": bool(z["canonical_masked"]),
        "macro_f1_post": float(z["macro_f1_post"]),
        "source": str(z["source"]) if "source" in z.files else "?",
    }


def run_from_eval(res, name, seed, dist_buckets, pr_bins):
    ps = res["per_seq"]
    return {
        "path": "<memory>", "name": name, "seed": int(seed),
        "labels": [b[2] for b in dist_buckets], "pr_bins": pr_bins,
        "ids": np.array(ps["ids"]), "lengths": np.array(ps["lengths"]),
        "d_pos": np.stack(ps["pos"]), "d_neg": np.stack(ps["neg"]),
        "canonical_masked": res["canonical_masked"],
        "macro_f1_post": res["macro_post"]["f"], "source": "eval_buckets",
    }


def _unique_keys(ids):
    seen, out = {}, []
    for i in ids:
        k = seen.get(i, 0)
        out.append(i if k == 0 else f"{i}#{k}")
        seen[i] = k + 1
    return out


def align_runs(runs):
    """Restrict every run to the common set of test sequences, same order."""
    keysets = [_unique_keys(list(r["ids"])) for r in runs]
    common = set(keysets[0])
    for ks in keysets[1:]:
        common &= set(ks)
    order = [k for k in keysets[0] if k in common]
    if not order:
        raise SystemExit("ERROR: the runs share no sequence ids -- different test files?")
    for r, ks in zip(runs, keysets):
        pos = {k: i for i, k in enumerate(ks)}
        idx = np.array([pos[k] for k in order], dtype=np.int64)
        dropped = len(ks) - len(order)
        if dropped:
            print(f"  ! {r['name']} seed {r['seed']}: {dropped} sequences not shared "
                  f"by all runs were dropped ({os.path.basename(r['path'])})")
        r["d_pos"], r["d_neg"] = r["d_pos"][idx], r["d_neg"][idx]
        r["lengths"], r["ids"] = r["lengths"][idx], np.array(order)
    L0 = runs[0]["lengths"]
    for r in runs[1:]:
        if not np.array_equal(L0, r["lengths"]):
            raise SystemExit(f"ERROR: sequence lengths differ between runs "
                             f"({runs[0]['path']} vs {r['path']}): not the same test set.")
    return runs


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def bootstrap_weights(n, n_boot, rng):
    """[n_boot, n] multinomial resampling counts over test sequences."""
    W = np.empty((n_boot, n), dtype=np.float64)
    for b in range(n_boot):
        W[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return W


def run_point_ap(run):
    return ap_batch(run["d_pos"].sum(axis=0), run["d_neg"].sum(axis=0))


def run_boot_ap(run, W):
    n, nb, _ = run["d_pos"].shape
    out = np.full((W.shape[0], nb), np.nan)
    for b in range(nb):
        Hp, Hn = run["d_pos"][:, b, :], run["d_neg"][:, b, :]
        if Hp.sum() == 0:
            continue
        cols = np.flatnonzero(Hp.sum(axis=0) + Hn.sum(axis=0))  # skip empty bins
        bp = W @ Hp[:, cols].astype(np.float64)
        bn = W @ Hn[:, cols].astype(np.float64)
        out[:, b] = ap_batch(bp, bn)
    return out


def _multiplier(name):
    if name == "competitive":
        return 10.0
    if name.startswith("competitive_x"):
        try:
            return float(name[len("competitive_x"):])
        except ValueError:
            return 1e12
    return None


def group_order(names, ref):
    def key(g):
        if g == ref:
            return (0, 0.0, g)
        if g == "matched":
            return (1, 0.0, g)
        m = _multiplier(g)
        if m is not None:
            return (2, m, g)
        if g.lower().startswith("rnafold"):
            return (4, 0.0, g)
        return (3, 0.0, g)
    return sorted(names, key=key)


def pretty_name(g, custom=None):
    if custom and g in custom:
        return custom[g]
    if g == "quantum":
        return "Q1Fold (quantum)"
    if g == "matched":
        return "Classical, matched"
    m = _multiplier(g)
    if m is not None and m < 1e12:
        return f"Classical ×{m:g}"
    if g.lower().startswith("rnafold"):
        return "RNAfold (McCaskill)"
    return g


def bootstrap_summary(runs, ref, n_boot, ci=0.95, rng_seed=0, resample_seeds=True):
    labels = runs[0]["labels"]
    for r in runs[1:]:
        if r["labels"] != labels or r["pr_bins"] != runs[0]["pr_bins"]:
            raise SystemExit(f"ERROR: {r['path']} uses different buckets/bins "
                             f"({r['labels']}) than {runs[0]['path']} ({labels}).")
    if len({r["canonical_masked"] for r in runs}) > 1:
        print("  ! WARNING: some runs used the canonical mask for AP and some did "
              "not -- AP values are not comparable across them.")
    nb = len(labels)
    n = runs[0]["d_pos"].shape[0]
    rng = np.random.default_rng(rng_seed)
    W = bootstrap_weights(n, n_boot, rng) if n_boot > 0 else None

    groups = {}
    for r in runs:
        groups.setdefault(r["name"], []).append(r)
    for g in groups:
        groups[g].sort(key=lambda r: r["seed"])
    order = group_order(list(groups), ref)

    for r in runs:
        r["ap"] = run_point_ap(r)
        if W is not None:
            r["ap_boot"] = run_boot_ap(r, W)

    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    n_true = runs[0]["d_pos"].sum(axis=(0, 2)).astype(np.int64)
    gsum = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for g in order:
            rs = groups[g]
            P = np.stack([r["ap"] for r in rs])
            S = len(rs)
            entry = {"mean": P.mean(axis=0),
                     "std": P.std(axis=0, ddof=1) if S > 1 else np.full(nb, np.nan),
                     "n_seeds": S, "boot": None,
                     "lo": np.full(nb, np.nan), "hi": np.full(nb, np.nan)}
            if W is not None:
                Bk = np.stack([r["ap_boot"] for r in rs])          # [S, B, nb]
                if resample_seeds and S > 1:
                    sidx = rng.integers(0, S, (n_boot, S))
                    boot = Bk[sidx, np.arange(n_boot)[:, None], :].mean(axis=1)
                else:
                    boot = Bk.mean(axis=0)
                entry["boot"] = boot
                entry["lo"] = np.nanpercentile(boot, lo_q, axis=0)
                entry["hi"] = np.nanpercentile(boot, hi_q, axis=0)
            gsum[g] = entry

        deltas = {}
        if ref in gsum:
            for g in order:
                if g == ref:
                    continue
                d = {"delta": gsum[ref]["mean"] - gsum[g]["mean"],
                     "lo": np.full(nb, np.nan), "hi": np.full(nb, np.nan),
                     "p_le0": np.full(nb, np.nan)}
                if W is not None:
                    db = gsum[ref]["boot"] - gsum[g]["boot"]
                    d["lo"] = np.nanpercentile(db, lo_q, axis=0)
                    d["hi"] = np.nanpercentile(db, hi_q, axis=0)
                    valid = ~np.isnan(db)
                    d["p_le0"] = np.where(valid.any(axis=0),
                                          np.sum((db <= 0) & valid, axis=0)
                                          / np.maximum(valid.sum(axis=0), 1), np.nan)
                deltas[g] = d
        elif ref is not None:
            print(f"  ! reference group '{ref}' not found; groups are {order}. "
                  f"No dAP computed.")

    return {"labels": labels, "n_true": n_true, "n_seq": n, "n_boot": n_boot,
            "ci": ci, "order": order, "groups": groups, "gsum": gsum,
            "deltas": deltas, "ref": ref, "resample_seeds": resample_seeds}


def print_bootstrap_summary(S, custom=None):
    labels, order, gsum = S["labels"], S["order"], S["gsum"]
    cip = int(round(S["ci"] * 100))
    w = 9
    print(f"\n{'=' * 78}\nDistance-bucketed AP  |  {S['n_seq']} test sequences  |  "
          f"bootstrap B={S['n_boot']} ({cip}% CI"
          f"{', sequences + seeds' if S['resample_seeds'] else ', sequences'})\n{'=' * 78}")

    print("\n--- Per run: AP by |i-j|  (check f1_post against your test logs) ---")
    print(f"{'run':<26}{'f1_post':>8} |" + "".join(f"{l:>{w}}" for l in labels))
    for g in order:
        for r in S["groups"][g]:
            tag = f"{g}/seed{r['seed']}"
            f1 = r["macro_f1_post"]
            f1s = "    n/a" if f1 != f1 else f"{f1:8.3f}"
            print(f"{tag:<26}{f1s} |" + "".join(f"{_fmt(x):>{w}}" for x in r["ap"]))
    print(f"{'#true pairs (canonical)':<26}{'':>8} |" + "".join(f"{int(x):>{w}d}" for x in S["n_true"]))

    print(f"\n--- AP by |i-j|: mean over seeds (std over seeds) ---")
    print(f"{'group':<22}{'seeds':>6} |" + "".join(f"{l:>15}" for l in labels))
    for g in order:
        e = gsum[g]
        cells = []
        for m, s in zip(e["mean"], e["std"]):
            cells.append(f"{m:.3f}" + (f" ({s:.3f})" if s == s else "        "))
        print(f"{g:<22}{e['n_seeds']:>6} |" + "".join(f"{c:>15}" for c in cells))

    if S["n_boot"] > 0:
        print(f"\n--- AP by |i-j|: {cip}% bootstrap CI of the seed-averaged AP ---")
        print(f"{'group':<22}{'':>6} |" + "".join(f"{l:>15}" for l in labels))
        for g in order:
            e = gsum[g]
            cells = [f"[{lo:.3f},{hi:.3f}]" for lo, hi in zip(e["lo"], e["hi"])]
            print(f"{g:<22}{'':>6} |" + "".join(f"{c:>15}" for c in cells))

    if S["deltas"]:
        ref = S["ref"]
        print(f"\n--- dAP = {ref} - group  (positive = {ref} better) ---")
        print(f"{'group':<22}{'bucket':>9} | {'dAP':>7}  {cip}% CI{'':>11} {'P(dAP<=0)':>10}")
        for g, d in S["deltas"].items():
            for i, lab in enumerate(labels):
                dv = d["delta"][i]
                ci_s = (f"[{d['lo'][i]:+.3f},{d['hi'][i]:+.3f}]"
                        if d["lo"][i] == d["lo"][i] else "")
                p_s = f"{d['p_le0'][i]:.3f}" if d["p_le0"][i] == d["p_le0"][i] else ""
                dvs = "   n/a " if dv != dv else f"{dv:+.3f}"
                flag = ""
                if d["lo"][i] == d["lo"][i]:
                    if d["lo"][i] > 0:
                        flag = "  *"
                    elif d["hi"][i] < 0:
                        flag = "  (-)"
                print(f"{g:<22}{lab:>9} | {dvs:>7}  {ci_s:<18} {p_s:>10}{flag}")
            print()
        print("  *   CI excludes 0 in the ref's favour;  (-) CI excludes 0 against the ref")


def write_bootstrap_csvs(S, out_dir, prefix=""):
    os.makedirs(out_dir, exist_ok=True)
    labels = S["labels"]
    with open(os.path.join(out_dir, f"{prefix}per_run_ap.csv"), "w") as fh:
        fh.write("group,seed,f1_post,bucket,ap,n_true\n")
        for g in S["order"]:
            for r in S["groups"][g]:
                for i, lab in enumerate(labels):
                    fh.write(f"{g},{r['seed']},{r['macro_f1_post']:.4f},{lab},"
                             f"{r['ap'][i]:.5f},{int(S['n_true'][i])}\n")
    with open(os.path.join(out_dir, f"{prefix}group_ap.csv"), "w") as fh:
        fh.write("group,label,n_seeds,bucket,ap_mean,ap_std_seeds,ci_lo,ci_hi,n_true\n")
        for g in S["order"]:
            e = S["gsum"][g]
            for i, lab in enumerate(labels):
                fh.write(f"{g},{pretty_name(g)},{e['n_seeds']},{lab},{e['mean'][i]:.5f},"
                         f"{e['std'][i]:.5f},{e['lo'][i]:.5f},{e['hi'][i]:.5f},"
                         f"{int(S['n_true'][i])}\n")
    if S["deltas"]:
        with open(os.path.join(out_dir, f"{prefix}delta_ap.csv"), "w") as fh:
            fh.write(f"ref,group,bucket,dap_ref_minus_group,ci_lo,ci_hi,p_dap_le0\n")
            for g, d in S["deltas"].items():
                for i, lab in enumerate(labels):
                    fh.write(f"{S['ref']},{g},{lab},{d['delta'][i]:.5f},{d['lo'][i]:.5f},"
                             f"{d['hi'][i]:.5f},{d['p_le0'][i]:.4f}\n")
    print(f"\nCSVs written to {out_dir}/ ({prefix}per_run_ap.csv, {prefix}group_ap.csv"
          f"{', ' + prefix + 'delta_ap.csv' if S['deltas'] else ''})")


def plot_bootstrap_summary(S, out_dir, custom=None, prefix=""):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed: no figure)")
        return
    labels, order, gsum, ref = S["labels"], S["order"], S["gsum"], S["ref"]

    def tick(l):
        return f">{int(l[:-1]) - 1}" if l.endswith("+") else l.replace("-", "–")

    x = np.arange(len(labels))
    comp = [g for g in order if g != ref and not g.lower().startswith("rnafold")]
    cmap = plt.get_cmap("viridis")
    colors = {g: cmap(0.15 + 0.7 * k / max(len(comp) - 1, 1)) for k, g in enumerate(comp)}
    colors[ref] = "black"

    def style(g):
        if g == ref:
            return dict(color="black", marker="o", linestyle="-", linewidth=2.0, zorder=5)
        if g.lower().startswith("rnafold"):
            return dict(color="0.45", marker="s", linestyle="--", linewidth=1.4)
        return dict(color=colors[g], marker="^", linestyle="-", linewidth=1.3)

    has_delta = bool(S["deltas"])
    fig, axes = plt.subplots(1, 2 if has_delta else 1,
                             figsize=(10.5 if has_delta else 5.5, 3.9))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    for g in order:
        e, st = gsum[g], style(g)
        ax.plot(x, e["mean"], label=pretty_name(g, custom), markersize=5, **st)
        if np.isfinite(e["lo"]).any():
            ax.fill_between(x, e["lo"], e["hi"], color=st["color"], alpha=0.12, linewidth=0)
    ax.set_xticks(x)
    ax.set_xticklabels([tick(l) for l in labels])
    ax.set_xlabel("Base-pair separation |i − j| (nt)")
    ax.set_ylabel("Average precision")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("(a) AP by separation", fontsize=10, loc="left")

    if has_delta:
        ax = axes[1]
        gs = list(S["deltas"])
        K = len(gs)
        for k, g in enumerate(gs):
            d, st = S["deltas"][g], style(g)
            off = (k - (K - 1) / 2) * min(0.1, 0.6 / max(K, 1))
            yerr = None
            if np.isfinite(d["lo"]).any():
                yerr = np.vstack([d["delta"] - d["lo"], d["hi"] - d["delta"]])
            ax.errorbar(x + off, d["delta"], yerr=yerr, color=st["color"],
                        marker=st["marker"], linestyle="none", capsize=2.5,
                        markersize=5, label=f"vs {pretty_name(g, custom)}")
        ax.axhline(0, color="0.3", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([tick(l) for l in labels])
        ax.set_xlabel("Base-pair separation |i − j| (nt)")
        ax.set_ylabel(f"ΔAP ({pretty_name(ref, custom)} − other)")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("(b) Advantage of the reference", fontsize=10, loc="left")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{prefix}distance_buckets.{ext}"),
                    dpi=200 if ext == "png" else None)
    plt.close(fig)
    print(f"Figure written to {out_dir}/{prefix}distance_buckets.pdf/.png")


# --------------------------------------------------------------------------- #
# Model / loader construction
# --------------------------------------------------------------------------- #

def build_loader(pkg, config, data_file):
    from torch.utils.data import DataLoader
    return DataLoader(
        pkg.SeqDataset(data_file, **config),
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=0,
        collate_fn=pkg.pad_batch,
    )


def parse_sets(items, flag):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"{flag} expects KEY=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def make_config(config_path, device, batch_size, max_len, overrides=None, cache_dir=None):
    config = {
        "device": device,
        "batch_size": batch_size,
        "valid_split": 0.1,
        "max_len": max_len,
        "verbose": False,
        "cache_path": cache_dir,
    }
    config.update({k: v for k, v in json.load(open(config_path)).items()
                   if not k.startswith("_")})
    config.update(overrides or {})
    config["verbose"] = False
    config["cache_path"] = cache_dir
    return config


def build(pkg, config, test_file, weights):
    model = pkg.q1fold(weights=weights, **config)
    return model, build_loader(pkg, config, test_file)


# --------------------------------------------------------------------------- #
# Aggregate mode
# --------------------------------------------------------------------------- #

def aggregate(args):
    files = sorted(glob.glob(os.path.join(args.aggregate, "*.npz")))
    files = [f for f in files if not f.endswith(".tmp.npz")]
    if not files:
        raise SystemExit(f"no .npz files in {args.aggregate}")
    runs = [load_hist(f) for f in files]
    excl = set(args.exclude or [])
    runs = [r for r in runs if r["name"] not in excl]
    seen = {}
    for r in runs:
        k = (r["name"], r["seed"])
        if k in seen:
            raise SystemExit(f"ERROR: duplicate run {k}: {seen[k]} and {r['path']}")
        seen[k] = r["path"]
    print(f"[aggregate] {len(runs)} runs from {args.aggregate}: " +
          ", ".join(f"{g} x{sum(r['name'] == g for r in runs)}"
                    for g in dict.fromkeys(r["name"] for r in runs)))
    align_runs(runs)
    custom = parse_sets(args.label, "--label")
    S = bootstrap_summary(runs, args.ref, args.bootstrap, ci=args.ci,
                          rng_seed=args.boot_seed,
                          resample_seeds=not args.no_seed_resample)
    print_bootstrap_summary(S, custom)
    out = args.out or "bucket_summary"
    write_bootstrap_csvs(S, out)
    plot_bootstrap_summary(S, out, custom)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Bucketed eval by distance and length.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--test-file")
    ap.add_argument("--config")
    ap.add_argument("--weights")
    ap.add_argument("--name", default="modelA")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config entry for model 1 (JSON-parsed; repeatable)")
    ap.add_argument("--run-seed", type=int, default=-1,
                    help="training seed of model 1 (stored in --save-hist)")
    ap.add_argument("--config2", default=None)
    ap.add_argument("--weights2", default=None)
    ap.add_argument("--name2", default="modelB")
    ap.add_argument("--set2", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--run-seed2", type=int, default=-1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--cache-dir", default=None,
                    help="SeqDataset cache (pickled masks/priors per sequence id); "
                         "reuse across runs on the same test file to save time")
    ap.add_argument("--dist-edges", default="13,31,51,101",
                    help="interior (exclusive) edges for |i-j| buckets; default gives "
                         "1-12, 13-30, 31-50, 51-100, 101+")
    ap.add_argument("--len-edges", default="80,160,320",
                    help="interior edges for sequence-length buckets")
    ap.add_argument("--repo-root", default=None,
                    help="dir containing the q1fold package (added to sys.path)")
    ap.add_argument("--package", default="q1fold")
    ap.add_argument("--out", default=None, help="output dir for CSVs / figure")
    ap.add_argument("--raw", action="store_true",
                    help="also print/write the RAW (pre-postprocessing) tables")
    ap.add_argument("--pr-curve", action="store_true",
                    help="per-bucket AP over RAW scores plus a threshold sweep")
    ap.add_argument("--pr-thresholds", default="0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--pr-bins", type=int, default=1000)
    ap.add_argument("--val-file", default=None,
                    help="validation CSV for threshold tuning (must NOT be the test file)")
    ap.add_argument("--tune-threshold", action="store_true")
    ap.add_argument("--tune-grid", default="0.30,0.95,0.01")
    # new: per-sequence histograms, bootstrap, aggregation
    ap.add_argument("--save-hist", default=None, metavar="PATH.npz",
                    help="save model 1's per-sequence histograms for --aggregate")
    ap.add_argument("--save-hist2", default=None, metavar="PATH.npz")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="sequence-level bootstrap replicates for AP CIs (0 = off)")
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--no-seed-resample", action="store_true",
                    help="aggregate: resample sequences only, keep the seed set fixed")
    ap.add_argument("--aggregate", default=None, metavar="DIR",
                    help="aggregate every .npz in DIR (no model loading)")
    ap.add_argument("--ref", default="quantum",
                    help="aggregate: reference group for dAP = ref - group")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="aggregate: group names to leave out (e.g. RNAfold)")
    ap.add_argument("--label", action="append", default=[], metavar="NAME=Label",
                    help="aggregate: legend label for a group (repeatable)")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args)
        return

    for req in ("test_file", "config", "weights"):
        if getattr(args, req) is None:
            ap.error(f"--{req.replace('_', '-')} is required (unless --aggregate)")
    if tr is None:
        ap.error("PyTorch is required to evaluate models")

    if args.repo_root:
        sys.path.insert(0, args.repo_root)
    pkg = __import__(args.package)
    from importlib import import_module
    postprocessing = import_module(args.package + ".utils").postprocessing

    tr.manual_seed(42); random.seed(42); np.random.seed(42)

    dist_buckets = build_buckets([int(x) for x in args.dist_edges.split(",")], start=1)
    len_buckets = build_buckets([int(x) for x in args.len_edges.split(",")], start=1)

    if args.tune_threshold and not args.val_file:
        ap.error("--tune-threshold requires --val-file")
    if args.val_file and os.path.abspath(args.val_file) == os.path.abspath(args.test_file):
        print("WARNING: --val-file equals --test-file; thresholds would be "
              "tuned on the test set. Use a held-out validation file.")
    keep_seq = bool(args.save_hist or args.save_hist2 or args.bootstrap > 0)
    need_pr = args.pr_curve or args.tune_threshold or keep_seq

    # ---- Model A --------------------------------------------------------- #
    setsA = parse_sets(args.set, "--set")
    cfgA = make_config(args.config, args.device, args.batch_size, args.max_length,
                       setsA, args.cache_dir)
    if setsA:
        print(f"[{args.name}] config overrides: {setsA}")
    modelA, loader = build(pkg, cfgA, args.test_file, args.weights)
    print(f"[{args.name}] running bucketed eval ...")
    resA = evaluate(modelA, loader, postprocessing, dist_buckets, len_buckets,
                    collect_pr=need_pr, pr_bins=args.pr_bins, keep_per_seq=keep_seq)
    m = resA["macro_post"]
    print(f"[{args.name}] OVERALL macro (should match your f1_post log): "
          f"precision_post={m['p']:.3f}  recall_post={m['r']:.3f}  f1_post={m['f']:.3f}")

    resB = None
    if args.config2 and args.weights2:
        setsB = parse_sets(args.set2, "--set2")
        cfgB = make_config(args.config2, args.device, args.batch_size, args.max_length,
                           setsB, args.cache_dir)
        if setsB:
            print(f"[{args.name2}] config overrides: {setsB}")
        modelB, loaderB = build(pkg, cfgB, args.test_file, args.weights2)
        print(f"[{args.name2}] running bucketed eval ...")
        resB = evaluate(modelB, loaderB, postprocessing, dist_buckets, len_buckets,
                        collect_pr=need_pr, pr_bins=args.pr_bins, keep_per_seq=keep_seq)
        mb = resB["macro_post"]
        print(f"[{args.name2}] OVERALL macro (should match your f1_post log): "
              f"precision_post={mb['p']:.3f}  recall_post={mb['r']:.3f}  f1_post={mb['f']:.3f}")

    # ---- Tables (post-processed) ---------------------------------------- #
    if resB is None:
        print_table("By base-pair distance |i-j|  (post-processed)",
                    dist_buckets, resA["dist_post"], args.name)
        print_table("By sequence length  (post-processed)",
                    len_buckets, resA["len_post"], args.name)
    else:
        print_table("By base-pair distance |i-j|  (post-processed)",
                    dist_buckets, resA["dist_post"], args.name,
                    resB["dist_post"], args.name2)
        print_table("By sequence length  (post-processed)",
                    len_buckets, resA["len_post"], args.name,
                    resB["len_post"], args.name2)

    if args.raw:
        if resB is None:
            print_table("By base-pair distance |i-j|  (RAW, pre-postproc)",
                        dist_buckets, resA["dist_raw"], args.name)
            print_table("By sequence length  (RAW, pre-postproc)",
                        len_buckets, resA["len_raw"], args.name)
        else:
            print_table("By base-pair distance |i-j|  (RAW, pre-postproc)",
                        dist_buckets, resA["dist_raw"], args.name,
                        resB["dist_raw"], args.name2)
            print_table("By sequence length  (RAW, pre-postproc)",
                        len_buckets, resA["len_raw"], args.name,
                        resB["len_raw"], args.name2)

    # ---- Optional PR frontier ------------------------------------------- #
    if args.pr_curve:
        thresholds = [float(x) for x in args.pr_thresholds.split(",")]
        prA, prB = resA["pr"], (resB["pr"] if resB is not None else None)
        print_ap_table("By base-pair distance |i-j|", dist_buckets,
                       prA, ("d_pos", "d_neg"), args.name, prB, args.name2)
        print_ap_table("By sequence length", len_buckets,
                       prA, ("l_pos", "l_neg"), args.name, prB, args.name2)
        print_pr_sweep("By base-pair distance |i-j|", dist_buckets, prA,
                       ("d_pos", "d_neg"), args.name, thresholds, args.pr_bins)
        if prB is not None:
            print_pr_sweep("By base-pair distance |i-j|", dist_buckets, prB,
                           ("d_pos", "d_neg"), args.name2, thresholds, args.pr_bins)
        print_pr_sweep("By sequence length", len_buckets, prA,
                       ("l_pos", "l_neg"), args.name, thresholds, args.pr_bins)
        if prB is not None:
            print_pr_sweep("By sequence length", len_buckets, prB,
                           ("l_pos", "l_neg"), args.name2, thresholds, args.pr_bins)

    # ---- Optional threshold tuning -------------------------------------- #
    if args.tune_threshold:
        g0, g1, gs = [float(x) for x in args.tune_grid.split(",")]
        grid = np.round(np.arange(g0, g1 + 1e-9, gs), 4)
        print(f"\n[tuning] validation file: {args.val_file}")
        valA = evaluate(modelA, build_loader(pkg, cfgA, args.val_file), postprocessing,
                        dist_buckets, len_buckets, collect_pr=True, pr_bins=args.pr_bins)
        tune_and_report(valA["pr"], resA["pr"], dist_buckets,
                        args.name, args.pr_bins, grid)
        if resB is not None:
            valB = evaluate(modelB, build_loader(pkg, cfgB, args.val_file), postprocessing,
                            dist_buckets, len_buckets, collect_pr=True, pr_bins=args.pr_bins)
            tune_and_report(valB["pr"], resB["pr"], dist_buckets,
                            args.name2, args.pr_bins, grid)

    # ---- Per-sequence histograms ---------------------------------------- #
    if args.save_hist:
        save_hist(args.save_hist, resA["per_seq"], args.name, args.run_seed,
                  dist_buckets, args.pr_bins, resA["canonical_masked"], resA["macro_post"])
    if args.save_hist2 and resB is not None:
        save_hist(args.save_hist2, resB["per_seq"], args.name2, args.run_seed2,
                  dist_buckets, args.pr_bins, resB["canonical_masked"], resB["macro_post"])

    # ---- Optional bootstrap (single model, or paired dAP = model2 - model1)
    if args.bootstrap > 0:
        nameA, nameB = args.name, args.name2
        if resB is not None and nameA == nameB:
            nameB = nameB + "_2"
        runs = [run_from_eval(resA, nameA, args.run_seed, dist_buckets, args.pr_bins)]
        if resB is not None:
            runs.append(run_from_eval(resB, nameB, args.run_seed2, dist_buckets, args.pr_bins))
        align_runs(runs)
        ref = nameB if resB is not None else nameA
        S = bootstrap_summary(runs, ref, args.bootstrap, ci=args.ci, rng_seed=args.boot_seed)
        print_bootstrap_summary(S)
        if args.out:
            write_bootstrap_csvs(S, args.out, prefix="bootstrap_")
            plot_bootstrap_summary(S, args.out, prefix="bootstrap_")

    # ---- CSVs ------------------------------------------------------------ #
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        write_csv(os.path.join(args.out, f"{args.name}_distance.csv"),
                  dist_buckets, resA["dist_post"], args.name, "distance")
        write_csv(os.path.join(args.out, f"{args.name}_length.csv"),
                  len_buckets, resA["len_post"], args.name, "length")
        if resB is not None:
            write_csv(os.path.join(args.out, f"{args.name2}_distance.csv"),
                      dist_buckets, resB["dist_post"], args.name2, "distance")
            write_csv(os.path.join(args.out, f"{args.name2}_length.csv"),
                      len_buckets, resB["len_post"], args.name2, "length")
        if args.raw:
            write_csv(os.path.join(args.out, f"{args.name}_distance_raw.csv"),
                      dist_buckets, resA["dist_raw"], args.name, "distance_raw")
            write_csv(os.path.join(args.out, f"{args.name}_length_raw.csv"),
                      len_buckets, resA["len_raw"], args.name, "length_raw")
            if resB is not None:
                write_csv(os.path.join(args.out, f"{args.name2}_distance_raw.csv"),
                          dist_buckets, resB["dist_raw"], args.name2, "distance_raw")
                write_csv(os.path.join(args.out, f"{args.name2}_length_raw.csv"),
                          len_buckets, resB["len_raw"], args.name2, "length_raw")
        print(f"\nCSVs written to {args.out}/")


if __name__ == "__main__":
    main()