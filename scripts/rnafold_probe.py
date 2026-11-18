"""
RNAfold reference, bucketed by base-pair separation |i-j|
==========================================================

A thermodynamic folding method that never saw the training labels, scored
against the SAME test annotations, the SAME distance buckets, the SAME
canonical-candidate mask and the SAME sequence set as eval_buckets.py, so its
numbers sit directly next to the model tables.

Two readouts per distance bucket
--------------------------------
  * MFE structure      -> precision / recall / F1 (a single operating point,
                          comparable to the model's post-processed F1)
  * McCaskill base-pair -> average precision (AP): the base-pair probability
    probabilities         P(i,j) is used as a score, exactly like the model's
                          raw sigmoid, so AP is directly comparable to the
                          model AP that eval_buckets.py reports.

With --save-hist it writes per-sequence score histograms in the same .npz
format as eval_buckets.py --save-hist, so
    python eval_buckets.py --aggregate hist/ --ref quantum --bootstrap 2000
puts RNAfold on the same figure (with a paired bootstrap CI for
quantum - RNAfold).

Matching eval_buckets.py
------------------------
  * sequences : same CSV, same filter as SeqDataset (min_len <= L <= max_len),
                sequences with L < 2 skipped
  * truth     : 'base_pairs' column (1-based JSON list, as SeqDataset reads it),
                else the 'dotbracket' column
  * mask      : AP is computed over canonical candidates only, using the same
                rule as q1fold.utils.valid_mask (|i-j| > 3 and a pairable
                nucleotide pair, IUPAC codes included). Pass --repo-root so the
                q1fold rule is imported exactly; otherwise an A-U/G-C/G-U
                fallback is used and a warning is printed.
  * buckets   : default --dist-edges 13,31,51,101 -> 1-12, 13-30, 31-50,
                51-100, 101+ (exclusive upper edges, like eval_buckets.py)

Caveat to state in the paper: RNAfold cannot predict pseudoknots, while bpRNA
annotations include them, so part of any long-range drop for RNAfold is
pseudoknotted pairs it cannot represent. The script reports how many true
pairs per bucket are pseudoknotted (cross another pair) so you can quote it.

Requires ViennaRNA:  pip install ViennaRNA   (import RNA)

Usage
-----
    python rnafold_probe.py --test-file bpRNA_TS0.csv \
        --repo-root /path/to/dir_containing_q1fold \
        --workers 16 \
        --save-hist hist/RNAfold_seed0.npz \
        --out rnafold_report
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Buckets (identical scheme to eval_buckets.py)
# --------------------------------------------------------------------------- #

def build_buckets(edges, start=1):
    bounds = [start] + list(edges)
    buckets = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        buckets.append((lo, hi, f"{lo}-{hi-1}"))
    buckets.append((bounds[-1], np.iinfo(np.int64).max, f"{bounds[-1]}+"))
    return buckets


# --------------------------------------------------------------------------- #
# Structure parsing -> set of 0-indexed (i, j), i < j
# --------------------------------------------------------------------------- #

_OPEN = {"(": ")", "[": "]", "{": "}", "<": ">"}
_OPEN.update({chr(c): chr(c + 32) for c in range(ord("A"), ord("Z") + 1)})
_CLOSE = {v: k for k, v in _OPEN.items()}


def pairs_from_dotbracket(s):
    """Dot-bracket with (), [], {}, <> and Aa..Zz families -> 0-based pairs."""
    pairs = set()
    stacks = {k: [] for k in _OPEN}
    for k, ch in enumerate(s):
        if ch in _OPEN:
            stacks[ch].append(k)
        elif ch in _CLOSE:
            st = stacks[_CLOSE[ch]]
            if st:
                pairs.add((st.pop(), k))
    return pairs


def pairs_from_field(field, base=1):
    """'base_pairs' field: JSON [[i, j], ...] (1-based by default) or loose text."""
    try:
        lst = json.loads(field)
        out = set()
        for p in lst:
            i, j = sorted((int(p[0]) - base, int(p[1]) - base))
            if i != j:
                out.add((i, j))
        return out
    except (ValueError, TypeError, IndexError, KeyError):
        pass
    toks = (str(field).replace("[", " ").replace("]", " ").replace(",", " ")
            .replace(";", " ").replace("-", " ").split())
    nums = [int(t) for t in toks if t.lstrip("+").isdigit()]
    out = set()
    for k in range(0, len(nums) - 1, 2):
        i, j = sorted((nums[k] - base, nums[k + 1] - base))
        if i != j:
            out.add((i, j))
    return out


def pseudoknotted(pairs):
    """Pairs that cross at least one other pair (i < k < j < l)."""
    ps = sorted(pairs)
    pk = set()
    for a in range(len(ps)):
        i, j = ps[a]
        for b in range(a + 1, len(ps)):
            k, l = ps[b]
            if k >= j:
                break
            if l > j:
                pk.add((i, j))
                pk.add((k, l))
    return pk


# --------------------------------------------------------------------------- #
# Canonical-candidate rule (mirrors q1fold.utils.valid_mask)
# --------------------------------------------------------------------------- #

def _fallback_strength(pair):
    s = set(pair)
    if s == {"G", "C"}:
        return 3
    if s == {"A", "U"}:
        return 2
    if s == {"G", "U"}:
        return 0.8
    return 0


def build_pair_table(alphabet, strength_fn):
    """bool[a, b] = (a, b) may pair, symmetrised like valid_mask."""
    n = len(alphabet)
    T = np.zeros((n, n), dtype=bool)
    for x, a in enumerate(alphabet):
        for y, b in enumerate(alphabet):
            T[x, y] = (strength_fn([a, b]) > 0) or (strength_fn([b, a]) > 0)
    return T


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

_G = {}


def _init_worker(cfg):
    _G.update(cfg)
    import RNA  # noqa: F401  (fail early in each worker if missing)


def _process(task):
    idx, seq, true_pairs = task
    import RNA
    cfg = _G
    buckets, pr_bins = cfg["buckets"], cfg["pr_bins"]
    nb = len(buckets)
    L = len(seq)
    try:
        md = RNA.md()
        if cfg["temperature"] is not None:
            md.temperature = cfg["temperature"]
        fc = RNA.fold_compound(seq, md)
        ss, mfe = fc.mfe()
        pred = pairs_from_dotbracket(ss)

        sp = np.zeros((nb, pr_bins), dtype=np.uint32)
        sn = np.zeros((nb, pr_bins), dtype=np.uint32)
        if cfg["bpp"]:
            fc.exp_params_rescale(mfe)
            fc.pf()
            P = np.asarray(fc.bpp(), dtype=np.float64)[1:L + 1, 1:L + 1]
        else:
            P = None
    except Exception as e:        # noqa: BLE001
        return idx, None, repr(e)

    # --- MFE P/R/F1 per bucket (all pairs, like the model's post tables)
    los = np.array([b[0] for b in buckets])
    his = np.array([b[1] for b in buckets])

    def bidx(d):
        k = np.flatnonzero((los <= d) & (d < his))
        return int(k[0]) if k.size else -1

    tp = np.zeros(nb, np.int64); fp = np.zeros(nb, np.int64); fn = np.zeros(nb, np.int64)
    for (i, j) in pred:
        b = bidx(j - i)
        if b >= 0:
            if (i, j) in true_pairs:
                tp[b] += 1
            else:
                fp[b] += 1
    for (i, j) in true_pairs:
        if (i, j) not in pred:
            b = bidx(j - i)
            if b >= 0:
                fn[b] += 1
    TP, FP, FN = int(tp.sum()), int(fp.sum()), int(fn.sum())
    pp = TP / (TP + FP) if TP + FP else 0.0
    rr = TP / (TP + FN) if TP + FN else 0.0
    ff = 2 * pp * rr / (pp + rr) if pp + rr else 0.0

    # --- pseudoknot accounting of the truth
    pk_true = np.zeros(nb, np.int64)
    for (i, j) in pseudoknotted(true_pairs):
        b = bidx(j - i)
        if b >= 0:
            pk_true[b] += 1

    # --- McCaskill histograms over canonical candidates
    if P is not None:
        iu, ju = np.triu_indices(L, 1)
        dist = ju - iu
        truth = np.zeros((L, L), dtype=bool)
        for (i, j) in true_pairs:
            if 0 <= i < L and 0 <= j < L:
                truth[i, j] = True
        r = truth[iu, ju]
        if cfg["masked"]:
            codes = np.array([cfg["alpha_index"][c] for c in seq], dtype=np.int64)
            keep = cfg["pair_table"][codes[iu], codes[ju]] & (dist > 3)
        else:
            keep = np.ones_like(r)
        prob = P[iu, ju]
        bins = np.clip((prob * pr_bins).astype(np.int64), 0, pr_bins - 1)
        rp, rn = r & keep, (~r) & keep
        for b, (lo, hi, _) in enumerate(buckets):
            m = (dist >= lo) & (dist < hi)
            if m.any():
                sp[b] = np.bincount(bins[m & rp], minlength=pr_bins)
                sn[b] = np.bincount(bins[m & rn], minlength=pr_bins)

    return idx, {"tp": tp, "fp": fp, "fn": fn, "pk": pk_true,
                 "macro": (pp, rr, ff), "sp": sp, "sn": sn}, None


# --------------------------------------------------------------------------- #
# AP (same step integral as eval_buckets.average_precision)
# --------------------------------------------------------------------------- #

def average_precision(pos, neg):
    pos = np.asarray(pos, dtype=np.float64); neg = np.asarray(neg, dtype=np.float64)
    if pos.sum() == 0:
        return float("nan")
    tp = np.cumsum(pos[::-1]); fp = np.cumsum(neg[::-1])
    prec = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1e-300), 1.0)
    rec = tp / tp[-1]
    return float(np.sum(np.diff(rec, prepend=0.0) * prec))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f = (2 * p * r / (p + r)) if (p == p and r == r and (p + r) > 0) else float("nan")
    return p, r, f


def _fmt(x):
    return "  n/a " if (x != x) else f"{x:6.3f}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--seq-col", default="sequence")
    ap.add_argument("--pairs-col", default="base_pairs",
                    help="ground-truth base pairs (preferred if present, like SeqDataset)")
    ap.add_argument("--struct-col", default="dotbracket",
                    help="ground-truth dot-bracket (used if --pairs-col is absent)")
    ap.add_argument("--pairs-base", type=int, default=1,
                    help="index origin of --pairs-col (1 = bpseq/q1fold style)")
    ap.add_argument("--min-len", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=512,
                    help="same filter as SeqDataset max_len (eval_buckets --max-length)")
    ap.add_argument("--dist-edges", default="13,31,51,101")
    ap.add_argument("--pr-bins", type=int, default=1000)
    ap.add_argument("--no-canonical-mask", action="store_true",
                    help="score all pairs for AP (only if the model was evaluated "
                         "without use_canonical_mask)")
    ap.add_argument("--repo-root", default=None,
                    help="dir containing the q1fold package, to import its exact "
                         "canonical-pair rule")
    ap.add_argument("--package", default="q1fold")
    ap.add_argument("--no-bpp", action="store_true",
                    help="MFE only (no partition function, no AP)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="folding temperature in C (ViennaRNA default 37)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--max-seqs", type=int, default=0, help="quick run cap (0 = all)")
    ap.add_argument("--save-hist", default=None, metavar="PATH.npz",
                    help="per-sequence histograms in eval_buckets.py format")
    ap.add_argument("--name", default="RNAfold", help="group name in --save-hist")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    try:
        import RNA
    except ImportError:
        ap.error("ViennaRNA not installed. `pip install ViennaRNA` (import RNA).")

    buckets = build_buckets([int(x) for x in args.dist_edges.split(",")])
    nb = len(buckets)

    # ---- data: same filtering as SeqDataset --------------------------------
    df = pd.read_csv(args.test_file)
    n_all = len(df)
    df["_len"] = df[args.seq_col].astype(str).str.len()
    df = df[(df._len >= args.min_len) & (df._len <= args.max_len) & (df._len >= 2)]
    if args.max_seqs:
        df = df.head(args.max_seqs)
    print(f"{n_all} rows in {args.test_file}; {len(df)} kept "
          f"({args.min_len} <= len <= {args.max_len})")

    use_pairs = args.pairs_col in df.columns
    if not use_pairs and args.struct_col not in df.columns:
        ap.error(f"neither '{args.pairs_col}' nor '{args.struct_col}' column found "
                 f"(columns: {list(df.columns)})")

    # ---- canonical rule ----------------------------------------------------
    strength_fn, rule_src = _fallback_strength, "fallback A-U/G-C/G-U"
    if args.repo_root:
        sys.path.insert(0, args.repo_root)
    if not args.no_canonical_mask:
        try:
            from importlib import import_module
            utils = import_module(args.package + ".utils")
            strength_fn, rule_src = utils.pair_strength, f"{args.package}.utils.pair_strength"
        except Exception as e:     # noqa: BLE001
            print(f"  ! could not import {args.package}.utils ({e!r}); using the "
                  f"A-U/G-C/G-U fallback. Pass --repo-root to match the model's "
                  f"canonical mask exactly (matters only for IUPAC codes like N, R, Y).")

    tasks, ids, lengths = [], [], []
    alphabet = set()
    for k, rowd in enumerate(df.to_dict("records")):
        seq = str(rowd[args.seq_col]).strip().upper().replace("T", "U")
        if use_pairs:
            truth = pairs_from_field(rowd[args.pairs_col], base=args.pairs_base)
        else:
            truth = pairs_from_dotbracket(str(rowd[args.struct_col]).strip())
        alphabet |= set(seq)
        tasks.append((k, seq, truth))
        ids.append(str(rowd[args.id_col]) if args.id_col in rowd else f"seq{k}")
        lengths.append(len(seq))
    alphabet = sorted(alphabet)
    cfg = {
        "buckets": buckets, "pr_bins": args.pr_bins, "bpp": not args.no_bpp,
        "temperature": args.temperature, "masked": not args.no_canonical_mask,
        "alpha_index": {c: i for i, c in enumerate(alphabet)},
        "pair_table": build_pair_table(alphabet, strength_fn),
    }
    print(f"truth from '{args.pairs_col if use_pairs else args.struct_col}'; "
          f"canonical mask: {'off' if args.no_canonical_mask else rule_src}; "
          f"ViennaRNA {RNA.__version__}; {args.workers} workers")

    # ---- fold ----------------------------------------------------------------
    results = [None] * len(tasks)
    errors = []
    with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for n_done, (idx, res, err) in enumerate(
                pool.imap_unordered(_process, tasks, chunksize=4), 1):
            if err is not None:
                errors.append((ids[idx], err))
            results[idx] = res
            if n_done % 200 == 0:
                print(f"  ... {n_done}/{len(tasks)}", file=sys.stderr, flush=True)
    if errors:
        print(f"  ! {len(errors)} sequences failed in ViennaRNA and were skipped, "
              f"e.g. {errors[0]}")

    ok = [k for k, r in enumerate(results) if r is not None]
    tp = sum(results[k]["tp"] for k in ok); fp = sum(results[k]["fp"] for k in ok)
    fn = sum(results[k]["fn"] for k in ok); pk = sum(results[k]["pk"] for k in ok)
    d_pos = np.stack([results[k]["sp"] for k in ok])
    d_neg = np.stack([results[k]["sn"] for k in ok])
    macro = np.array([results[k]["macro"] for k in ok])
    mp, mr, mf = macro.mean(axis=0)

    # ---- report ----------------------------------------------------------------
    labels = [b[2] for b in buckets]
    aps = [average_precision(d_pos[:, b].sum(0), d_neg[:, b].sum(0)) for b in range(nb)]
    print(f"\nRNAfold vs annotation  ({len(ok)} sequences)")
    print(f"MFE macro over sequences: P={mp:.3f} R={mr:.3f} F1={mf:.3f}")
    print(f"\n{'bucket':>8} | {'P_mfe':>6} {'R_mfe':>6} {'F1_mfe':>6} | {'AP_bpp':>6} | "
          f"{'TP':>6} {'FP':>6} {'FN':>6} | {'#true':>6} {'#pk':>5} {'%pk':>5} | {'#true_can':>9}")
    rows = []
    for b, lab in enumerate(labels):
        p, r, f = prf(int(tp[b]), int(fp[b]), int(fn[b]))
        ntrue = int(tp[b] + fn[b])
        ncan = int(d_pos[:, b].sum())
        pct = 100.0 * pk[b] / ntrue if ntrue else float("nan")
        print(f"{lab:>8} | {_fmt(p)} {_fmt(r)} {_fmt(f)} | "
              f"{_fmt(aps[b]) if not args.no_bpp else '   -  '} | "
              f"{int(tp[b]):6d} {int(fp[b]):6d} {int(fn[b]):6d} | {ntrue:6d} "
              f"{int(pk[b]):5d} {pct:5.1f} | {ncan:9d}")
        rows.append((lab, p, r, f, aps[b], int(tp[b]), int(fp[b]), int(fn[b]),
                     ntrue, int(pk[b]), ncan))
    print("\n#true counts all annotated pairs (MFE tables); #true_can counts the canonical "
          "ones used for AP, which should equal eval_buckets' '#true pairs (canonical)'.")
    print("#pk = annotated pairs that are pseudoknotted (RNAfold cannot predict these).")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, "rnafold_distance.csv")
        with open(path, "w") as fh:
            fh.write("bucket,precision_mfe,recall_mfe,f1_mfe,ap_bpp,tp,fp,fn,"
                     "true,true_pseudoknotted,true_canonical\n")
            for (lab, p, r, f, a, t, fpp, fnn, nt, npk, nc) in rows:
                fh.write(f"{lab},{p:.4f},{r:.4f},{f:.4f},{a:.4f},{t},{fpp},{fnn},"
                         f"{nt},{npk},{nc}\n")
        print(f"\nCSV written to {path}")

    if args.save_hist:
        if args.no_bpp:
            print("  ! --save-hist needs the partition function; ignored with --no-bpp")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_hist)), exist_ok=True)
            tmp = args.save_hist + ".tmp.npz"
            np.savez_compressed(
                tmp,
                name=np.array(args.name), seed=np.array(0),
                labels=np.array(labels),
                los=np.array([b[0] for b in buckets], dtype=np.int64),
                his=np.array([b[1] for b in buckets], dtype=np.int64),
                pr_bins=np.array(int(args.pr_bins)),
                ids=np.array([ids[k] for k in ok]),
                lengths=np.array([lengths[k] for k in ok], dtype=np.int64),
                d_pos=d_pos, d_neg=d_neg,
                canonical_masked=np.array(not args.no_canonical_mask),
                macro_f1_post=np.array(float(mf)),   # MFE macro F1, for reference
                macro_p_post=np.array(float(mp)), macro_r_post=np.array(float(mr)),
                source=np.array("rnafold_probe"),
            )
            os.replace(tmp, args.save_hist)
            print(f"[hist] wrote {args.save_hist}  ({len(ok)} sequences)")


if __name__ == "__main__":
    main()