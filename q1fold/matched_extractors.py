"""
Classical 1D extractors for the Table 2 matched control (journal version)
=========================================================================

Both classes are drop-in replacements for ``MultiScaleQuantumConv1D`` with
scales (3, 5, 7): same three parallel branches, same per-branch output width
(36 / 105 / 210 -> 351 channels), same receptive field per branch (exactly k
nucleotides), one BatchNorm1d per branch, no positional input. Only the
per-branch feature map changes.

Parameter budget of the quantum extractor (scales 3/5/7, 1 layer, no pos-enc)
------------------------------------------------------------------------------
    per branch k : theta [1, k, 3] = 3k   +   embed_weights [4, 2] = 8
    k = 3, 5, 7  : 17 + 23 + 29             =  69  variational
    BatchNorm    : 2 * (36 + 105 + 210)     = 702  affine
    total                                   = 771

1) MatchedClassicalMultiScale  -> exactly 69 + 702 = 771 parameters
-------------------------------------------------------------------
Each branch keeps the *identical* encoding and parameterisation as the quantum
branch -- per-nucleotide (RY, RZ) angles = biochemical base angle + learnable
offset (8 params), and one learnable Euler rotation per window position
(3k params) -- but computes the features classically on unit 3-vectors:

    v_i  = (sin a cos b, sin a sin b, cos a)        a, b = encoded angles
    u_i  = Rz(g) Ry(be) Rz(al) v_i                  (al, be, g) = theta[i]
    single-site features : u_i                       -> 3k
    pair features        : u_i (outer) u_j, i < j    -> 9 * C(k, 2)

This is exactly what the quantum circuit outputs if the CNOT ring is removed
(a product state, where <s_a s_b> = <s_a><s_b>). So the matched pair differs
in one thing only: entangling, non-factorisable correlations vs. their
classical factorised counterpart. Same inputs, same parameters, same readout
format, same normalisation.

2) CompetitiveClassicalMultiScale -> <= 10 x 771 = 7,710 parameters
-------------------------------------------------------------------
A free-form learned network per branch, receptive field still exactly k:

    Conv1d(4 -> h, kernel=k) -> GELU -> Conv1d(h -> h, 1) -> GELU
        -> Conv1d(h -> n_k, 1) -> GELU -> BatchNorm1d(n_k)

The second layer is 1x1 so the window stays k (a stacked kernel-k layer would
silently widen it to 2k-1 and break the comparison). The trailing GELU keeps
the n_k outputs from being a rank-h linear map, which the downstream linear
projection in the attention refiner would otherwise collapse. h = 14 gives
7,479 parameters (6,777 + 702 BN); h = 15 would exceed the budget (7,983).

Budget sweep: the same class with a larger `hidden` (see hidden_for_budget)
gives the scaling curve. Only the width changes; depth, kernel sizes,
receptive field, output channels and normalisation stay fixed.
"""

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

QUANTUM_EXTRACTOR_PARAMS = 69 + 702      # reference budget (scales 3/5/7)
COMPETITIVE_BUDGET = 10 * QUANTUM_EXTRACTOR_PARAMS   # 7,710


def branch_feature_count(k):
    """Quantum measurement count per branch: 3k + 9*C(k,2)."""
    return 3 * k + 9 * (k * (k - 1) // 2)


def _windows(x, k):
    """[B, 4, L] -> [B, L, 4, k] zero-padded sliding windows centred on each position."""
    pad = (k - 1) // 2
    w = F.pad(x, (pad, pad)).unfold(2, k, 1)          # [B, 4, L, k]
    return w.permute(0, 2, 1, 3)                       # [B, L, 4, k]


# =============================================================================
# 1) Parameter-matched classical extractor (exactly 771 params)
# =============================================================================

class _MatchedClassicalBranch(nn.Module):
    """Classical, unentangled counterpart of one k-qubit quantum branch (3k + 8 params)."""

    def __init__(self, window_size, seed=42):
        super().__init__()
        assert window_size % 2 == 1
        k = int(window_size)
        self.window_size = k

        # Same initialisation scales as _QuantumBranch.
        g = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(torch.randn(1, k, 3, generator=g) * 0.3)        # 3k
        self.embed_weights = nn.Parameter(torch.randn(4, 2, generator=g) * 0.1)   # 8
        # Frozen zero buffer, mirroring the quantum branch with position encoding
        # off (keeps the CLI's per-epoch norm logging working; not a parameter).
        self.register_buffer('pos_alpha', torch.zeros(k))

        # Same base-angle table and A/G/C/U index order as the quantum branch.
        self.register_buffer('base_angles', torch.tensor([
            [0.0,         0.0        ],   # A
            [math.pi / 2, 0.0        ],   # G
            [0.0,         math.pi / 2],   # C
            [math.pi / 2, math.pi / 2],   # U
        ], dtype=torch.float32))
        # alphabetical one-hot (A, C, G, U) -> (A, G, C, U)
        self.register_buffer('remap', torch.tensor([0, 2, 1, 3], dtype=torch.long))

        pairs = list(itertools.combinations(range(k), 2))
        self.register_buffer('pair_i', torch.tensor([p[0] for p in pairs], dtype=torch.long))
        self.register_buffer('pair_j', torch.tensor([p[1] for p in pairs], dtype=torch.long))
        self.n_features = 3 * k + 9 * len(pairs)

    @staticmethod
    def _rot_z(v, a):
        c, s = torch.cos(a), torch.sin(a)
        x, y, z = v.unbind(-1)
        return torch.stack([c * x - s * y, s * x + c * y, z], dim=-1)

    @staticmethod
    def _rot_y(v, a):
        c, s = torch.cos(a), torch.sin(a)
        x, y, z = v.unbind(-1)
        return torch.stack([c * x + s * z, y, -s * x + c * z], dim=-1)

    def forward(self, x):
        B, _, L = x.shape
        k = self.window_size
        w = _windows(x, k)                                    # [B, L, 4, k]
        idx = self.remap[w.argmax(dim=2)]                     # [B, L, k]  (zero-pad -> A, as in quantum)

        ang = self.base_angles + self.embed_weights           # [4, 2]
        a = ang[idx, 0]                                       # RY angle   [B, L, k]
        b = ang[idx, 1]                                       # RZ angle   [B, L, k]
        v = torch.stack([torch.sin(a) * torch.cos(b),
                         torch.sin(a) * torch.sin(b),
                         torch.cos(a)], dim=-1)                # [B, L, k, 3]

        # Rot(al, be, g) = RZ(g) RY(be) RZ(al), one per window position
        al, be, ga = self.theta[0].unbind(-1)                 # each [k]
        u = self._rot_z(v, al)
        u = self._rot_y(u, be)
        u = self._rot_z(u, ga)                                # [B, L, k, 3]

        single = u.reshape(B, L, 3 * k)
        pair = u[:, :, self.pair_i, :, None] * u[:, :, self.pair_j, None, :]   # [B, L, P, 3, 3]
        pair = pair.reshape(B, L, -1)
        out = torch.cat([single, pair], dim=-1)               # [B, L, n_features]
        return out.permute(0, 2, 1).contiguous()


class MatchedClassicalMultiScale(nn.Module):
    """Parameter-matched classical extractor: exactly 69 + 702 = 771 params for scales (3,5,7)."""

    def __init__(self, scales=(3, 5, 7), seed=42):
        super().__init__()
        self.scales = tuple(int(k) for k in scales)
        self.branches = nn.ModuleList([
            _MatchedClassicalBranch(k, seed=seed + i) for i, k in enumerate(self.scales)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(b.n_features) for b in self.branches])
        self.out_channels = sum(b.n_features for b in self.branches)

    def forward(self, x):
        return torch.cat([bn(br(x)) for br, bn in zip(self.branches, self.bns)], dim=1).float()

    def feature_breakdown(self):
        return {f"k={k}": b.n_features for k, b in zip(self.scales, self.branches)}


# =============================================================================
# 2) Competitive classical extractor (<= 7,710 params)
# =============================================================================

class CompetitiveClassicalMultiScale(nn.Module):
    """Higher-capacity classical extractor, capped at 10x the quantum extractor's parameters."""

    def __init__(self, scales=(3, 5, 7), hidden=14, seed=42, budget=None):
        super().__init__()
        # Seed only this module's init; fork_rng restores the global RNG
        # afterwards, so the run seed still drives the rest of the model,
        # data shuffling and dropout.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self._build(scales, hidden)
        n_params = sum(p.numel() for p in self.parameters())
        if budget is not None and n_params > budget:
            raise ValueError(f"CompetitiveClassicalMultiScale has {n_params} params "
                             f"> budget {budget}; lower `hidden`.")

    def _build(self, scales, hidden):
        self.scales = tuple(int(k) for k in scales)
        self.hidden = int(hidden)
        self.branches = nn.ModuleList()
        self._feats = []
        for k in self.scales:
            assert k % 2 == 1
            n = branch_feature_count(k)
            self._feats.append(n)
            self.branches.append(nn.Sequential(
                nn.Conv1d(4, hidden, k, padding=(k - 1) // 2),   # receptive field = k
                nn.GELU(),
                nn.Conv1d(hidden, hidden, 1),                    # 1x1: adds depth, not reach
                nn.GELU(),
                nn.Conv1d(hidden, n, 1),
                nn.GELU(),
            ))
        self.bns = nn.ModuleList([nn.BatchNorm1d(n) for n in self._feats])
        self.out_channels = sum(self._feats)

    def forward(self, x):
        return torch.cat([bn(br(x)) for br, bn in zip(self.branches, self.bns)], dim=1).float()

    def feature_breakdown(self):
        return {f"k={k}": n for k, n in zip(self.scales, self._feats)}


def competitive_param_count(hidden, scales=(3, 5, 7)):
    """Closed-form parameter count of CompetitiveClassicalMultiScale (incl. BN)."""
    h, total = int(hidden), 0
    for k in scales:
        n = branch_feature_count(k)
        total += (4 * h * k + h) + (h * h + h) + (h * n + n) + 2 * n
    return total


def hidden_for_budget(budget, scales=(3, 5, 7)):
    """Largest hidden width whose CompetitiveClassicalMultiScale fits within `budget` params."""
    if competitive_param_count(1, scales) > budget:
        raise ValueError(f"budget {budget} is below the smallest competitive extractor "
                         f"({competitive_param_count(1, scales)} params)")
    lo, hi = 1, 1
    while competitive_param_count(hi, scales) <= budget:
        hi *= 2
    while hi - lo > 1:                      # invariant: count(lo) <= budget < count(hi)
        mid = (lo + hi) // 2
        if competitive_param_count(mid, scales) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


def param_breakdown(module):
    """Split an extractor's trainable params into (feature map, BatchNorm affine)."""
    bn = sum(p.numel() for n, p in module.named_parameters() if n.startswith('bns.'))
    total = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"feature_map": total - bn, "batchnorm": bn, "total": total}


# =============================================================================
# 3) Plain parallel Conv1d extractor (linear, or with ReLU)
# =============================================================================

class PlainConvMultiScale(nn.Module):
    """
    The simplest classical drop-in: one Conv1d per scale, no hidden layer.

        branch k : Conv1d(4 -> n_k, kernel=k, padding=(k-1)//2) [-> ReLU] -> BatchNorm1d(n_k)
        n_k      : 36 / 105 / 210 for k = 3 / 5 / 7  (same as the quantum branches)

    Parameters (scales 3/5/7): 468 + 2,205 + 6,090 = 8,763 conv  +  702 BN  =  9,465,
    identical for both variants (ReLU has no parameters).

    activation=None  ("plain_conv"): fully linear. Together with the attention
        refiner's linear 1x1 input projection it is equivalent to a single
        kernel-7 Conv1d(4 -> d_model), i.e. a learned linear k-mer embedding;
        the 351 channels add parameters but no expressiveness.
    activation="relu" ("plain_conv_relu"): ReLU after each conv, so the 351
        channels are genuine nonlinear features of the k-mer window.
    """

    def __init__(self, scales=(3, 5, 7), activation=None, seed=42):
        super().__init__()
        assert activation in (None, "relu"), activation
        self.scales = tuple(int(k) for k in scales)
        self.activation = activation
        self._feats = [branch_feature_count(k) for k in self.scales]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.branches = nn.ModuleList()
            for k, n in zip(self.scales, self._feats):
                assert k % 2 == 1
                layers = [nn.Conv1d(4, n, k, padding=(k - 1) // 2)]
                if activation == "relu":
                    layers.append(nn.ReLU())
                self.branches.append(nn.Sequential(*layers))
            self.bns = nn.ModuleList([nn.BatchNorm1d(n) for n in self._feats])
        self.out_channels = sum(self._feats)

    def forward(self, x):
        return torch.cat([bn(br(x)) for br, bn in zip(self.branches, self.bns)], dim=1).float()

    def feature_breakdown(self):
        return {f"k={k}": n for k, n in zip(self.scales, self._feats)}
