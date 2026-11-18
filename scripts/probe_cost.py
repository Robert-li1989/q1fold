"""
Q1Fold computational-cost probe  (fills every \\PH{} in Section "Computational cost")
=================================================================================

What it measures, for the v20 baseline config
(multiscale_357_attn1d_opm_dilated_v20_pairchannels128_reg_config.json):

  [1] Parameter budget by module          -> Table rows "Parameters"
  [2] Circuit resources per branch         -> qubits, 2q gates, depth, gate counts,
                                              Pauli measurement settings (QWC groups),
                                              and the true # of hardware circuit runs
  [3] Lookup-table extractor               -> builds the 4^3+4^5+4^7 = 17,472-entry
                                              table, checks it is numerically identical
                                              to the direct circuit simulation
  [4] Inference time per sequence          -> direct vs cached table (full model AND
                                              extractor-only), one-off table build time
  [5] Training time per epoch              -> full epoch, or N batches extrapolated
  [6] Peak GPU memory                      -> training and inference separately
  [7] F1 with the table                    -> simulated table; optional finite-shot
                                              perturbed tables; optional external
                                              (e.g. hardware-derived) table

It reuses the training pipeline exactly (SeqDataset + pad_batch, q1fold(**config),
model.test for f1_post), same convention as eval_buckets.py.

Typical use
-----------
python probe_cost.py \\
    --config multiscale_357_attn1d_opm_dilated_v20_pairchannels128_reg_config.json \\
    --weights runs/v20/weights.pmt \\
    --test-file data/TS0.csv \\
    --train-file data/TR0.csv \\
    --device cuda --batch-size 4 \\
    --n-infer 200 --train-batches 200 \\
    --shot-noise 1024,8192 \\
    --export-table q1fold_table_sim.npz \\
    --out cost_probe.json

Quick smoke test (CPU, minutes):
    ... --device cpu --n-infer 20 --train-batches 10 --skip-f1

Hardware hand-off:
    --export-table writes the RAW (pre-BatchNorm) expectation values, one array per
    branch, shape [4^k, n_features_k], rows indexed by the k-mer code described in
    kmer_code(). Fill the same file from hardware and pass it back with
    --table-file hw_table.npz to get the F1 for the hardware-derived table.

If the package isn't importable from the cwd, pass --repo-root /dir/containing/q1fold.
"""

import argparse
import copy
import itertools
import json
import math
import os
import random
import statistics
import sys
import time

import numpy as np
import torch as tr
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def sync(device):
    if str(device).startswith("cuda"):
        tr.cuda.synchronize()


def reset_peak(device):
    if str(device).startswith("cuda"):
        tr.cuda.empty_cache()
        tr.cuda.reset_peak_memory_stats()


def peak_mb(device):
    if str(device).startswith("cuda"):
        return tr.cuda.max_memory_allocated() / 2**20
    return float("nan")


def make_config(config_path, device, batch_size, max_len, cache_dir=None):
    config = {"device": device, "batch_size": batch_size, "valid_split": 0.1,
              "max_len": max_len, "verbose": False, "cache_path": cache_dir}
    config.update({k: v for k, v in json.load(open(config_path)).items()
                   if not k.startswith("_")})
    config["verbose"] = False
    config["cache_path"] = cache_dir
    return config


def build_loader(pkg, config, data_file, batch_size, shuffle=False, limit=None):
    from torch.utils.data import DataLoader, Subset
    ds = pkg.SeqDataset(data_file, **config)
    if limit is not None and limit < len(ds):
        ds = Subset(ds, list(range(limit)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      collate_fn=pkg.pad_batch)


def fmt_int(n):
    return f"{n:,}".replace(",", "{,}")


def fmt_s(x):
    if x != x:
        return "--"
    if x < 1e-2:
        return f"{x*1e3:.2f} ms"
    return f"{x:.3f} s"


# --------------------------------------------------------------------------- #
# [1] Parameter budget
# --------------------------------------------------------------------------- #

GROUPS = [
    ("Quantum extractor: variational", lambda n: n.startswith("resnet1d.branches")),
    ("Quantum extractor: normalization", lambda n: n.startswith("resnet1d.bns")),
    ("1D attention refiner", lambda n: n.startswith("attn1d")),
    ("Pair projections", lambda n: n.startswith(("pair_proj_a", "pair_proj_b", "pair_aux",
                                                 "convrank1", "convrank2"))),
    ("Relative-position embedding", lambda n: n.startswith("relpos")),
    ("2D dilated refiner", lambda n: n.startswith("resnet2d")),
    ("Output head", lambda n: n.startswith("conv2Dout")),
]


def param_budget(model):
    rows = {g: 0 for g, _ in GROUPS}
    detail = {g: [] for g, _ in GROUPS}
    other = []
    for name, p in model.named_parameters():
        for g, match in GROUPS:
            if match(name):
                rows[g] += p.numel()
                detail[g].append((name, p.numel()))
                break
        else:
            other.append((name, p.numel()))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # variational breakdown (theta vs embed offsets; pos_alpha only if PE on)
    var_detail = {}
    for name, n in detail["Quantum extractor: variational"]:
        key = name.split(".")[-1]
        var_detail[key] = var_detail.get(key, 0) + n
    return {"rows": rows, "total": total, "trainable": trainable,
            "other": other, "variational_breakdown": var_detail}


# --------------------------------------------------------------------------- #
# [2] Circuit resources
# --------------------------------------------------------------------------- #

def _specs_resources(qnode, *args):
    import pennylane as qml
    s = qml.specs(qnode)(*args)
    res = s.get("resources") if isinstance(s, dict) else getattr(s, "resources", None)
    if res is not None:
        depth = getattr(res, "depth", None)
        gate_types = dict(getattr(res, "gate_types", None) or getattr(res, "gate_counts", {}))
        gate_sizes = dict(getattr(res, "gate_sizes", {}))
        num_gates = getattr(res, "num_gates", None)
    else:   # older PennyLane
        depth = s.get("depth")
        gate_types = dict(s.get("gate_types", {}))
        gate_sizes = dict(s.get("gate_sizes", {}))
        num_gates = s.get("num_operations") or s.get("num_gates")
    return {"depth": depth, "gate_types": gate_types,
            "gate_sizes": {int(k): v for k, v in gate_sizes.items()},
            "num_gates": num_gates}


def _decomposed_depth(branch, args):
    """Depth after decomposing Rot -> RZ RY RZ (closer to a native gate set)."""
    import pennylane as qml
    try:
        tape = qml.workflow.construct_tape(branch._circuit)(*args)
    except Exception:
        try:
            branch._circuit.construct(args, {})
            tape = branch._circuit.qtape
        except Exception:
            return None
    ops = []
    for op in tape.operations:
        if op.name == "Rot":
            ops.extend(op.decomposition())
        else:
            ops.append(op)
    # greedy layer depth
    wire_layer = {}
    for op in ops:
        ws = list(op.wires)
        layer = max((wire_layer.get(w, 0) for w in ws), default=0) + 1
        for w in ws:
            wire_layer[w] = layer
    return max(wire_layer.values()) if wire_layer else 0


def measurement_groups(n_qubits):
    """# of qubit-wise-commuting measurement settings for 3k singles + 9*C(k,2) pairs."""
    import pennylane as qml
    P = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    obs = []
    for q in range(n_qubits):
        for a in "XYZ":
            obs.append(P[a](q))
    for q1, q2 in itertools.combinations(range(n_qubits), 2):
        for a in "XYZ":
            for b in "XYZ":
                obs.append(P[a](q1) @ P[b](q2))
    try:
        groups = qml.pauli.group_observables(obs, grouping_type="qwc", method="lf")
        return len(groups)
    except Exception:
        return None


def covering_settings(n_qubits, restarts=300, seed=0):
    """Smallest set of product-Pauli measurement bases (one of X/Y/Z per qubit) such
    that every qubit pair sees all 9 axis combinations = a strength-2 covering array
    CA(N; 2, k, 3). Every single-qubit X/Y/Z is then covered too. Randomised greedy
    with restarts; result is verified. Known optima: k=3 -> 9, k=5 -> 11, k=7 -> 12."""
    rng = random.Random(seed)
    pairs = list(itertools.combinations(range(n_qubits), 2))
    need = {(i, j, a, b) for i, j in pairs for a in range(3) for b in range(3)}
    all_rows = list(itertools.product(range(3), repeat=n_qubits))
    best = None
    for _ in range(restarts):
        unc, rows = set(need), []
        while unc:
            cand = rng.sample(all_rows, min(len(all_rows), 400))
            row = max(cand, key=lambda r: sum((i, j, r[i], r[j]) in unc for i, j in pairs))
            rows.append(row)
            unc -= {(i, j, row[i], row[j]) for i, j in pairs}
        if best is None or len(rows) < len(best):
            best = rows
    best = [list(r) for r in best]

    # Local search: try to fit the array into N-1 rows (single-cell flips).
    def uncovered(rows):
        cov = {(i, j, r[i], r[j]) for r in rows for i, j in pairs}
        return len(need) - len(cov)

    lower = 9
    while len(best) > lower:
        N = len(best) - 1
        found = None
        for _ in range(40):
            rows = [[rng.randrange(3) for _ in range(n_qubits)] for _ in range(N)]
            cost = uncovered(rows)
            for _ in range(4000):
                if cost == 0:
                    break
                r, c = rng.randrange(N), rng.randrange(n_qubits)
                old = rows[r][c]
                rows[r][c] = rng.choice([v for v in range(3) if v != old])
                new = uncovered(rows)
                if new <= cost:
                    cost = new
                else:
                    rows[r][c] = old
            if cost == 0:
                found = rows
                break
        if found is None:
            break
        best = found

    covered = {(i, j, r[i], r[j]) for r in best for i, j in pairs}
    assert covered == need
    return ["".join("XYZ"[a] for a in r) for r in best]


def circuit_resources(extractor, device):
    out = {}
    for branch in extractor.branches:
        k = branch.n_qubits
        nt = tr.zeros(1, k, dtype=tr.long, device=device)
        pos = tr.zeros(1, device=device)
        args = (branch.theta, branch.embed_weights, branch.pos_alpha, nt, pos)
        r = _specs_resources(branch._circuit, *args)
        r["two_qubit_gates"] = r["gate_sizes"].get(2, None)
        r["depth_rot_decomposed"] = _decomposed_depth(branch, args)
        r["n_qubits"] = k
        r["n_observables"] = branch.n_features
        r["qwc_groups_pennylane_greedy"] = measurement_groups(k)
        settings = covering_settings(k)
        r["qwc_measurement_settings"] = len(settings)
        r["measurement_bases"] = settings
        r["distinct_inputs"] = 4 ** k
        out[k] = r
    return out


# --------------------------------------------------------------------------- #
# [3] Lookup-table extractor
# --------------------------------------------------------------------------- #

ALPHA_TO_AGCU = (0, 2, 1, 3)   # same remap as _QuantumBranch._onehot_to_index


def kmer_code(digits, k):
    """digits: [..., k] in {0=A,1=G,2=C,3=U} (window order, left->right).
    code = sum_q digits[q] * 4^(k-1-q)."""
    w = 4 ** tr.arange(k - 1, -1, -1, device=digits.device, dtype=tr.long)
    return (digits * w).sum(-1)


def enumerate_kmers(k, device):
    codes = tr.arange(4 ** k, device=device, dtype=tr.long)
    digits = tr.stack([(codes // 4 ** (k - 1 - q)) % 4 for q in range(k)], dim=-1)
    return digits  # [4^k, k]; row i has kmer_code == i


@tr.no_grad()
def build_tables(extractor, device, chunk=4096):
    """Raw (pre-BN) circuit outputs for every k-mer. Returns list of [4^k, F_k]."""
    tables = []
    for branch in extractor.branches:
        k = branch.n_qubits
        digits = enumerate_kmers(k, device)
        parts = []
        for s in range(0, digits.shape[0], chunk):
            d = digits[s:s + chunk]
            pos = tr.zeros(d.shape[0], device=device)
            out = branch._circuit(branch.theta, branch.embed_weights,
                                  branch.pos_alpha, d, pos)
            if isinstance(out, (list, tuple)):
                out = tr.stack(out, dim=0).transpose(0, 1)
            elif out.ndim == 2:
                out = out.transpose(0, 1)
            parts.append(out.float())
        tables.append(tr.cat(parts, 0).contiguous())
    return tables


class CachedExtractor(nn.Module):
    """Drop-in for MultiScaleQuantumConv1D at inference: table lookup + same BNs."""

    def __init__(self, extractor, tables):
        super().__init__()
        self.ks = [b.n_qubits for b in extractor.branches]
        self.bns = extractor.bns
        self.tables = [t for t in tables]
        self.out_channels = extractor.out_channels
        self.register_buffer("remap", tr.tensor(ALPHA_TO_AGCU, dtype=tr.long))

    def to(self, *a, **kw):
        super().to(*a, **kw)
        self.tables = [t.to(*a, **kw) for t in self.tables]
        return self

    def forward(self, x):
        B, _, L = x.shape
        feats = []
        for k, table, bn in zip(self.ks, self.tables, self.bns):
            pad = (k - 1) // 2
            xp = F.pad(x, (pad, pad))
            win = xp.unfold(2, k, 1)                      # [B, 4, L, k]
            digits = self.remap[win.argmax(dim=1)]        # [B, L, k]  (zero pad -> A, as direct)
            code = kmer_code(digits, k)                   # [B, L]
            f = table[code]                               # [B, L, F]
            f = f.permute(0, 2, 1).contiguous()           # [B, F, L]
            feats.append(bn(f))
        return tr.cat(feats, dim=1)


def check_equivalence(model, extractor, cached, loader, device, n_batches):
    model.eval()
    max_abs, max_rel, n = 0.0, 0.0, 0
    with tr.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            x = batch["embedding"].to(device)
            a = extractor(x)
            b = cached(x)
            d = (a - b).abs()
            max_abs = max(max_abs, d.max().item())
            max_rel = max(max_rel, (d / a.abs().clamp_min(1e-6)).max().item())
            n += 1
    return {"batches": n, "max_abs_diff": max_abs, "max_rel_diff": max_rel}


# --------------------------------------------------------------------------- #
# [4] Inference timing
# --------------------------------------------------------------------------- #

@tr.no_grad()
def time_inference(model, loader, device, n_seq, warmup=3):
    """Per-sequence timing with batch_size=1 loader. Times the full forward and the
    extractor alone (model.resnet1d) separately."""
    model.eval()
    full, ext, lens = [], [], []
    for i, batch in enumerate(loader):
        if i >= n_seq + warmup:
            break
        fwd = {k: v for k, v in batch.items() if k != "contact"}
        x = fwd["embedding"].to(device)
        sync(device); t0 = time.perf_counter()
        model.resnet1d(x)
        sync(device); t1 = time.perf_counter()
        model(fwd)
        sync(device); t2 = time.perf_counter()
        if i >= warmup:
            ext.append(t1 - t0); full.append(t2 - t1); lens.append(int(batch["length"][0]))
    def summ(v):
        return {"mean": statistics.mean(v), "median": statistics.median(v),
                "std": statistics.pstdev(v), "n": len(v)} if v else {}
    return {"full": summ(full), "extractor": summ(ext),
            "mean_len": statistics.mean(lens) if lens else float("nan"),
            "max_len": max(lens) if lens else 0}


# --------------------------------------------------------------------------- #
# [5] Training timing
# --------------------------------------------------------------------------- #

def time_training(pkg, config, train_file, batch_size, device, n_batches):
    from torch.utils.data import DataLoader
    tr.manual_seed(0)
    model = pkg.q1fold(**config)            # fresh init; timing does not depend on weights
    model.verbose = False
    ds = pkg.SeqDataset(train_file, training=True, **config)
    full_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0,
                             collate_fn=pkg.pad_batch)
    n_total = len(full_loader)

    class _Capped:
        def __init__(self, it, n):
            self.it, self.n = it, n
        def __iter__(self):
            return itertools.islice(iter(self.it), self.n)
        def __len__(self):
            return self.n

    n_use = n_total if (n_batches is None or n_batches <= 0) else min(n_batches, n_total)
    # short warm-up (cuDNN autotune, allocator)
    model.fit(_Capped(full_loader, min(2, n_use)))
    reset_peak(device)
    sync(device); t0 = time.perf_counter()
    model.fit(_Capped(full_loader, n_use))
    sync(device); dt = time.perf_counter() - t0
    return {"batches_timed": n_use, "batches_per_epoch": n_total,
            "n_train_seqs": len(ds), "seconds_timed": dt,
            "seconds_per_epoch": dt * n_total / n_use,
            "extrapolated": n_use < n_total,
            "peak_mem_mb": peak_mb(device)}


# --------------------------------------------------------------------------- #
# [7] F1 with a given table
# --------------------------------------------------------------------------- #

def f1_with_tables(model, extractor, tables, loader, device):
    cached = CachedExtractor(extractor, tables).to(device)
    orig = model.resnet1d
    sched = model.scheduler
    model.resnet1d = cached
    model.scheduler = None                  # model.test steps the plateau scheduler
    try:
        m = model.test(loader)
    finally:
        model.resnet1d = orig
        model.scheduler = sched
    return {k: float(v) for k, v in m.items()}


def shot_noise_tables(tables, shots, seed=0):
    """Replace each exact <P> in [-1,1] by the mean of `shots` +-1 samples."""
    g = tr.Generator(device="cpu").manual_seed(seed)
    out = []
    for t in tables:
        p = ((1 + t.detach().cpu().double()) / 2).clamp(0, 1)
        k = _binom(p, shots, g)
        out.append((2 * k / shots - 1).float().to(t.device))
    return out


def _binom(p, n, g):
    # torch Binomial has no generator arg; use numpy for reproducibility
    rng = np.random.default_rng(int(tr.randint(0, 2**31 - 1, (1,), generator=g)))
    return tr.from_numpy(rng.binomial(n, p.numpy()).astype(np.float64))


def save_tables(path, extractor, tables, meta):
    arrs = {f"k{b.n_qubits}": t.detach().cpu().numpy() for b, t in zip(extractor.branches, tables)}
    np.savez_compressed(path, **arrs, meta=json.dumps(meta))
    print(f"[table] wrote {path}  ({', '.join(f'{k}:{v.shape}' for k, v in arrs.items())})")


def load_tables(path, extractor, device):
    z = np.load(path, allow_pickle=False)
    out = []
    for b in extractor.branches:
        a = z[f"k{b.n_qubits}"]
        assert a.shape == (4 ** b.n_qubits, b.n_features), \
            f"k={b.n_qubits}: expected {(4 ** b.n_qubits, b.n_features)}, got {a.shape}"
        out.append(tr.from_numpy(a).float().to(device))
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", default=None,
                    help="trained weights (needed for equivalence/F1; timing works without)")
    ap.add_argument("--test-file", required=True)
    ap.add_argument("--train-file", default=None, help="omit to skip training timing")
    ap.add_argument("--device", default="cuda" if tr.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="batch size for training timing and F1 eval (match your runs)")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--n-infer", type=int, default=100,
                    help="# test sequences for per-sequence inference timing (batch=1)")
    ap.add_argument("--train-batches", type=int, default=0,
                    help="time N training batches and extrapolate (0 = full epoch)")
    ap.add_argument("--verify-batches", type=int, default=10,
                    help="# test batches for direct-vs-table equivalence check")
    ap.add_argument("--skip-f1", action="store_true")
    ap.add_argument("--shot-noise", default="",
                    help="comma list of shot counts, e.g. 1024,8192 (finite-shot table F1)")
    ap.add_argument("--table-file", default=None,
                    help="external (e.g. hardware) RAW table .npz to evaluate")
    ap.add_argument("--export-table", default=None, help="write simulated RAW table .npz")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--package", default="q1fold")
    ap.add_argument("--out", default="cost_probe.json")
    args = ap.parse_args()

    if args.repo_root:
        sys.path.insert(0, args.repo_root)
    pkg = __import__(args.package)
    import pennylane as qml

    tr.manual_seed(42); random.seed(42); np.random.seed(42)
    dev = args.device
    config = make_config(args.config, dev, args.batch_size, args.max_length, args.cache_dir)
    if config.get("multiscale_use_position_encoding", False):
        raise SystemExit("multiscale_use_position_encoding=true: extractor is not "
                         "translation-equivariant, the table argument does not apply.")

    env = {"torch": tr.__version__, "pennylane": qml.__version__, "device": dev,
           "gpu": tr.cuda.get_device_name(0) if dev.startswith("cuda") else None,
           "cpu_threads": tr.get_num_threads(), "batch_size": args.batch_size}
    print(f"[env] {env}")
    report = {"config": os.path.basename(args.config), "env": env}

    model = pkg.q1fold(weights=args.weights, **config)
    model.verbose = False
    model.eval()
    extractor = model.resnet1d

    # ---- [1] parameters ------------------------------------------------- #
    pb = param_budget(model)
    report["params"] = pb
    print("\n=== [1] Parameter budget ===")
    for g, _ in GROUPS:
        print(f"  {g:<36}{pb['rows'][g]:>12,}")
    print(f"  {'Total':<36}{pb['total']:>12,}   (trainable {pb['trainable']:,})")
    print(f"  variational breakdown: {pb['variational_breakdown']}")
    if pb["other"]:
        print(f"  !! UNGROUPED params: {pb['other']}")
    ext_total = pb["rows"]["Quantum extractor: variational"] + \
        pb["rows"]["Quantum extractor: normalization"]
    print(f"  quantum extractor total = {ext_total:,}")

    # ---- [2] circuit resources ------------------------------------------ #
    print("\n=== [2] Circuit resources (per branch) ===")
    cr = circuit_resources(extractor, dev)
    report["circuit"] = {str(k): v for k, v in cr.items()}
    total_runs = 0
    for k, r in cr.items():
        runs = r["distinct_inputs"] * (r["qwc_measurement_settings"] or 1)
        total_runs += runs
        r["hardware_circuit_runs"] = runs
        print(f"  k={k}: qubits={r['n_qubits']}  depth={r['depth']} "
              f"(Rot->RZRYRZ: {r['depth_rot_decomposed']})  "
              f"2q gates={r['two_qubit_gates']}  gates={r['num_gates']}  "
              f"types={r['gate_types']}")
        print(f"        observables={r['n_observables']}  "
              f"QWC measurement settings={r['qwc_measurement_settings']}  "
              f"distinct inputs={r['distinct_inputs']:,}  -> circuit runs={runs:,}")
    n_inputs = sum(r["distinct_inputs"] for r in cr.values())
    report["distinct_inputs_total"] = n_inputs
    report["hardware_circuit_runs_total"] = total_runs
    print(f"  distinct inputs total = {n_inputs:,};  "
          f"circuit executions incl. measurement settings = {total_runs:,} (x shots each)")

    # ---- [3] table ------------------------------------------------------ #
    print("\n=== [3] Lookup table ===")
    sync(dev); t0 = time.perf_counter()
    tables = build_tables(extractor, dev)
    sync(dev); t_build = time.perf_counter() - t0
    tbytes = sum(t.numel() * t.element_size() for t in tables)
    report["table"] = {"build_seconds": t_build, "bytes": tbytes,
                       "shapes": [list(t.shape) for t in tables]}
    print(f"  built in {t_build:.2f} s; {tbytes/2**20:.2f} MiB; shapes "
          f"{[tuple(t.shape) for t in tables]}")
    if args.export_table:
        save_tables(args.export_table, extractor, tables,
                    {"config": os.path.basename(args.config),
                     "weights": args.weights, "kmer_code": "sum d_q*4^(k-1-q), d in A,G,C,U=0..3",
                     "values": "raw expectation values, pre-BatchNorm"})

    cached = CachedExtractor(extractor, tables).to(dev)
    eval_loader = build_loader(pkg, config, args.test_file, args.batch_size)
    eq = check_equivalence(model, extractor, cached, eval_loader, dev, args.verify_batches)
    report["equivalence"] = eq
    print(f"  direct vs table on {eq['batches']} batches: max|diff|={eq['max_abs_diff']:.2e}  "
          f"(max rel {eq['max_rel_diff']:.2e})  -> "
          f"{'OK' if eq['max_abs_diff'] < 1e-4 else 'CHECK!'}")

    # ---- [4] inference timing ------------------------------------------- #
    print(f"\n=== [4] Inference per sequence (batch=1, {args.n_infer} seqs) ===")
    one_loader = build_loader(pkg, config, args.test_file, 1)
    reset_peak(dev)
    inf_direct = time_inference(model, one_loader, dev, args.n_infer)
    inf_direct["peak_mem_mb"] = peak_mb(dev)
    orig = model.resnet1d
    model.resnet1d = cached
    reset_peak(dev)
    inf_cached = time_inference(model, one_loader, dev, args.n_infer)
    inf_cached["peak_mem_mb"] = peak_mb(dev)
    model.resnet1d = orig
    report["inference"] = {"direct": inf_direct, "cached": inf_cached}
    for tag, r in (("direct", inf_direct), ("cached", inf_cached)):
        print(f"  {tag:<7} full model {fmt_s(r['full']['mean'])} mean / "
              f"{fmt_s(r['full']['median'])} median;  extractor "
              f"{fmt_s(r['extractor']['mean'])} mean;  peak {r['peak_mem_mb']:.0f} MiB")
    sp = inf_direct["full"]["mean"] / inf_cached["full"]["mean"]
    share = inf_direct["extractor"]["mean"] / inf_direct["full"]["mean"]
    report["inference"]["speedup_full"] = sp
    report["inference"]["extractor_share_direct"] = share
    print(f"  mean length {inf_direct['mean_len']:.1f} nt (max {inf_direct['max_len']});  "
          f"extractor share of direct time = {100*share:.1f}%;  speed-up = {sp:.1f}x")
    print(f"  table build ({t_build:.2f} s) amortised after "
          f"~{t_build / max(inf_direct['full']['mean'] - inf_cached['full']['mean'], 1e-9):.1f} seqs")

    # ---- [5]/[6] training ----------------------------------------------- #
    if args.train_file:
        print(f"\n=== [5] Training time per epoch (batch={args.batch_size}) ===")
        del cached
        tr_res = time_training(pkg, config, args.train_file, args.batch_size, dev,
                               args.train_batches)
        report["training"] = tr_res
        print(f"  {tr_res['batches_timed']}/{tr_res['batches_per_epoch']} batches in "
              f"{tr_res['seconds_timed']:.1f} s -> {tr_res['seconds_per_epoch']/60:.1f} min/epoch "
              f"({'extrapolated' if tr_res['extrapolated'] else 'full epoch'}); "
              f"{tr_res['n_train_seqs']} train seqs; peak {tr_res['peak_mem_mb']:.0f} MiB")
        cached = CachedExtractor(extractor, tables).to(dev)

    # ---- [7] F1 --------------------------------------------------------- #
    if not args.skip_f1 and args.weights:
        print("\n=== [7] Test F1 with table-based extractor ===")
        res = {"simulated": f1_with_tables(model, extractor, tables, eval_loader, dev)}
        print(f"  simulated table      f1_post={res['simulated']['f1_post']:.4f}  "
              f"(should equal your logged test f1_post)")
        for s in [int(x) for x in args.shot_noise.split(",") if x.strip()]:
            nt = shot_noise_tables(tables, s)
            res[f"shots_{s}"] = f1_with_tables(model, extractor, nt, eval_loader, dev)
            print(f"  {s:>6} shots/setting  f1_post={res[f'shots_{s}']['f1_post']:.4f}")
        if args.table_file:
            ht = load_tables(args.table_file, extractor, dev)
            res["external"] = f1_with_tables(model, extractor, ht, eval_loader, dev)
            print(f"  external table       f1_post={res['external']['f1_post']:.4f}  "
                  f"({args.table_file})")
        report["f1"] = res

    # ---- LaTeX rows ----------------------------------------------------- #
    R = pb["rows"]
    c3, c5, c7 = (cr.get(3, {}), cr.get(5, {}), cr.get(7, {}))
    print("\n=== LaTeX (paste into Table~\\ref{tab:cost}) ===")
    lines = [
        f"Quantum extractor: variational      & {fmt_int(R['Quantum extractor: variational'])} \\\\",
        f"Quantum extractor: normalization    & {fmt_int(R['Quantum extractor: normalization'])} \\\\",
        f"1D attention refiner                & {fmt_int(R['1D attention refiner'])} \\\\",
        f"Pair projections                    & {fmt_int(R['Pair projections'])} \\\\",
        f"Relative-position embedding         & {fmt_int(R['Relative-position embedding'])} \\\\",
        f"2D dilated refiner                  & {fmt_int(R['2D dilated refiner'])} \\\\",
        f"Output head                         & {fmt_int(R['Output head'])} \\\\",
        "\\midrule",
        f"Total                               & {fmt_int(pb['total'])} \\\\",
        f"Two-qubit gates per branch          & {c3.get('two_qubit_gates')} / "
        f"{c5.get('two_qubit_gates')} / {c7.get('two_qubit_gates')} \\\\",
        f"Circuit depth                       & {c3.get('depth')} / {c5.get('depth')} / "
        f"{c7.get('depth')} \\\\",
        f"Measurement settings (QWC)          & {c3.get('qwc_measurement_settings')} / "
        f"{c5.get('qwc_measurement_settings')} / {c7.get('qwc_measurement_settings')} \\\\",
        f"Distinct circuit inputs             & {fmt_int(n_inputs)} \\\\",
    ]
    if "training" in report:
        lines.append(f"Training time per epoch             & "
                     f"{report['training']['seconds_per_epoch']/60:.1f} min \\\\")
    lines += [
        f"Inference, direct                   & {fmt_s(inf_direct['full']['mean'])} \\\\",
        f"Inference, cached table             & {fmt_s(inf_cached['full']['mean'])} \\\\",
    ]
    if "training" in report:
        lines.append(f"Peak GPU memory                     & "
                     f"{report['training']['peak_mem_mb']/1024:.1f} GB (train) / "
                     f"{inf_direct['peak_mem_mb']/1024:.2f} GB (infer) \\\\")
    print("\n".join(lines))
    report["latex_rows"] = lines

    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\nFull report -> {args.out}")


if __name__ == "__main__":
    main()
