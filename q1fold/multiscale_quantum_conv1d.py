"""
Q1Fold Multi-Scale Quantum 1D Feature Extraction Layer
========================================================

Three parallel variational quantum circuits operating on sliding windows of
size k = 3, 5, 7 (one qubit per nucleotide in each window).

Design choices (locked in for this experiment):
    - NO context qubits: qubit count == window size.
        * k=3  -> 3 qubits
        * k=5  -> 5 qubits
        * k=7  -> 7 qubits
    - Biochemically motivated base angles + learnable per-nucleotide offsets.
    - Single learnable position rotation per qubit (RZ(pos * alpha_q)) so that
      identical k-mers at different sequence positions produce different
      descriptors. This keeps the "qubits scale with window, not sequence
      length" claim intact while restoring positional sensitivity.
    - Variational layer: 1 layer of Rot(alpha, beta, gamma) on each qubit
      followed by a ring of CNOTs.
    - Measurements:
        (a) Single-qubit Paulis X, Y, Z on each qubit          -> 3n features
        (b) All two-axis correlators on ALL C(n, 2) pairs:     -> 9*C(n,2) features
            sigma_a (q_i) tensor sigma_b (q_j) for
            sigma_a, sigma_b in {X, Y, Z}.
      NO triplet measurements in this phase.

Feature counts per branch:
    k=3 -> 9 + 9*3   = 36
    k=5 -> 15 + 9*10 = 105
    k=7 -> 21 + 9*21 = 210
    Total when concatenated: 351 features per sequence position.

The three branches are concatenated along the channel axis, each followed by
its own BatchNorm1d (their feature distributions differ markedly).

Usage (drop-in for the legacy 6-qubit QuantumConv1D):
    extractor = MultiScaleQuantumConv1D(scales=(3, 5, 7))
    x: [B, 4, L] one-hot encoded RNA -> features: [B, 351, L]
"""

import warnings
warnings.filterwarnings("ignore", message=".*torch.load.*weights_only.*",
                        category=FutureWarning)

import itertools
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


# ---------------------------------------------------------------------------
# Single-scale branch
# ---------------------------------------------------------------------------

class _QuantumBranch(nn.Module):
    """
    A single k-mer VQC branch with k qubits (one per nucleotide in the window).

    Args:
        window_size (int): k (must be odd so the window centers on a position).
        n_layers (int): number of variational Rot+CNOT-ring layers (default 1).
        seed (int): RNG seed for parameter initialization.
    """

    # Biochemically motivated base angles for A, G, C, U.
    # Index order: A=0, G=1, C=2, U=3 (matches argmax over one-hot rows
    # ordered as A, C, G, U? -> see _onehot_index below).
    _BASE_ANGLES = {
        'A': (0.0, 0.0),                # purine, adenine
        'G': (math.pi / 2, 0.0),        # purine, guanine
        'C': (0.0, math.pi / 2),        # pyrimidine, cytosine
        'U': (math.pi / 2, math.pi / 2) # pyrimidine, uracil
    }

    def __init__(self, window_size, n_layers=1, seed=42,
                 use_position_encoding=True):
        super().__init__()
        assert window_size % 2 == 1, "window_size must be odd"
        self.window_size = int(window_size)
        self.n_qubits = int(window_size)
        self.n_layers = int(n_layers)
        self.use_position_encoding = bool(use_position_encoding)

        # ---- Trainable parameters ---------------------------------------- #
        # Variational rotation angles: [n_layers, n_qubits, 3] (Rot alpha,beta,gamma)
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.theta = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits, 3) * 0.3
        )
        # Learnable additive offsets on the (RY, RZ) encoding angles per
        # nucleotide type. Shape [4, 2]; row order A, G, C, U.
        self.embed_weights = nn.Parameter(torch.randn(4, 2) * 0.1)
        # Position-rotation scale per qubit. Kept as a parameter only when
        # use_position_encoding=True; otherwise it's a frozen zero buffer
        # so the position-injection RZ becomes a no-op AND there are no
        # gradients on it (gradient experiments showed pos_alpha actively
        # decaying toward zero -- the model preferred to ignore it).
        if self.use_position_encoding:
            self.pos_alpha = nn.Parameter(torch.ones(self.n_qubits) * (math.pi / 4))
        else:
            self.register_buffer('pos_alpha', torch.zeros(self.n_qubits))

        # ---- Fixed buffers (base angles, masks) -------------------------- #
        base_angles = torch.tensor([
            [0.0,          0.0         ],   # A
            [math.pi / 2,  0.0         ],   # G
            [0.0,          math.pi / 2 ],   # C
            [math.pi / 2,  math.pi / 2 ],   # U
        ], dtype=torch.float32)
        self.register_buffer('base_angles', base_angles)

        # ---- Measurement enumeration ------------------------------------- #
        # Pre-compute the list of two-qubit pairs and Pauli letters used in
        # the circuit closure (just to expose .n_features without running it).
        self.pair_list = list(itertools.combinations(range(self.n_qubits), 2))
        self.pauli_letters = ('X', 'Y', 'Z')
        self.n_features = (
            3 * self.n_qubits                       # single-qubit X, Y, Z
            + 9 * len(self.pair_list)               # all-axis two-qubit correlators
        )

        # ---- Build the QNode --------------------------------------------- #
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        def _pauli_op(letter, wire):
            if letter == 'X':
                return qml.PauliX(wire)
            if letter == 'Y':
                return qml.PauliY(wire)
            if letter == 'Z':
                return qml.PauliZ(wire)
            raise ValueError(letter)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(theta, embed_weights, pos_alpha,
                    nucleotide_indices, pos_batch):
            """
            Args
            ----
            theta : [n_layers, n_qubits, 3]
                Variational rotation angles.
            embed_weights : [4, 2]
                Learnable per-nucleotide additive offsets on (RY, RZ).
            pos_alpha : [n_qubits]
                Learnable per-qubit position-rotation scale.
            nucleotide_indices : [batch, n_qubits]  (long)
                Index in {0,1,2,3} = {A,G,C,U} for each window position.
            pos_batch : [batch]  (float)
                Normalized center position in [0, 1] for each window.
            """
            n = self.n_qubits

            # ---- Stage 1: nucleotide encoding (per qubit) ----------------
            for q in range(n):
                idx = nucleotide_indices[:, q]                          # [B]
                ry = self.base_angles[idx, 0] + embed_weights[idx, 0]   # [B]
                rz = self.base_angles[idx, 1] + embed_weights[idx, 1]   # [B]
                qml.RY(ry, wires=q)
                qml.RZ(rz, wires=q)

            # ---- Stage 2: position injection (single RZ per qubit) -------
            # Skipped entirely when use_position_encoding=False (pos_alpha
            # is a zero buffer in that case, and the RZ would be a no-op
            # anyway -- this guard saves the per-qubit gate cost).
            if self.use_position_encoding:
                for q in range(n):
                    qml.RZ(pos_batch * pos_alpha[q], wires=q)

            # ---- Stage 3: variational layers -----------------------------
            for layer_idx in range(self.n_layers):
                for q in range(n):
                    a, b, c = (theta[layer_idx, q, 0],
                               theta[layer_idx, q, 1],
                               theta[layer_idx, q, 2])
                    qml.Rot(a, b, c, wires=q)
                # Ring of CNOTs
                for q in range(n - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n - 1, 0])

            # ---- Stage 4: measurements -----------------------------------
            measurements = []
            # (a) Single-qubit X, Y, Z on every qubit
            for q in range(n):
                for letter in self.pauli_letters:
                    measurements.append(qml.expval(_pauli_op(letter, q)))
            # (b) All 9 two-axis correlators on every pair (q1 < q2)
            for q1, q2 in self.pair_list:
                for la in self.pauli_letters:
                    for lb in self.pauli_letters:
                        op = _pauli_op(la, q1) @ _pauli_op(lb, q2)
                        measurements.append(qml.expval(op))
            return measurements

        self._circuit = circuit

    # ----------------------------------------------------------------- #
    # Forward
    # ----------------------------------------------------------------- #

    @staticmethod
    def _onehot_to_index(windows):
        """
        windows : [BL, 4, k]  one-hot over (A, C, G, U)  -- *dataset order*

        Note: the project's OneHotEmbedding may use the order [A, C, G, U]
        (alphabetical). Our internal nucleotide index uses [A, G, C, U] to
        match the biochemical base-angle table. We remap here.

        Returns
        -------
        idx : [BL, k]  long  with values in {0=A, 1=G, 2=C, 3=U}
        """
        # argmax over channel axis -> [BL, k] with values in {0..3}
        argmax = windows.argmax(dim=1)
        # Project alphabetical (A=0, C=1, G=2, U=3) to (A=0, G=1, C=2, U=3).
        # remap: 0->0, 1->2, 2->1, 3->3
        remap = torch.tensor([0, 2, 1, 3], device=argmax.device, dtype=argmax.dtype)
        return remap[argmax]

    def forward(self, x):
        """
        x : [B, 4, L]  one-hot encoded RNA sequence (alphabetical A,C,G,U).
        Returns : [B, n_features, L]
        """
        B, _, L = x.shape
        k = self.window_size
        pad = (k - 1) // 2

        # Sliding windows: pad with zeros on both sides, then unfold.
        x_padded = F.pad(x, (pad, pad))                       # [B, 4, L+2*pad]
        windows = x_padded.unfold(2, k, 1)                    # [B, 4, L, k]
        windows = windows.permute(0, 2, 1, 3).reshape(-1, 4, k)  # [B*L, 4, k]

        # Convert one-hot to remapped nucleotide indices (A/G/C/U order).
        nt_idx = self._onehot_to_index(windows)               # [B*L, k]

        # Position encoding (normalized to [0,1]) -- broadcast along batch
        pos_codes = torch.linspace(0.0, 1.0, L, device=x.device)
        # repeat_interleave matches the .reshape(B*L, ...) layout we used:
        #   row r = sample b * L + position p, with p iterating fastest, so
        #   the correct pattern is positions tiled across each sample:
        # We arranged windows by permute(0, 2, 1, 3) then reshape(-1, 4, k),
        # so the row order is (b=0,p=0), (b=0,p=1), ..., (b=0,p=L-1),
        # (b=1,p=0), ... Hence pos_codes is repeated B times.
        pos_batch = pos_codes.repeat(B)                       # [B*L]

        # Run the circuit (batched via PennyLane's broadcasting).
        out = self._circuit(self.theta, self.embed_weights, self.pos_alpha,
                            nt_idx, pos_batch)

        # PennyLane returns a list of [B*L] tensors -> stack to [n_feat, B*L]
        if isinstance(out, (list, tuple)):
            out = torch.stack(out, dim=0).transpose(0, 1)     # [B*L, n_feat]
        elif out.ndim == 2:
            out = out.transpose(0, 1)                         # [B*L, n_feat]

        # Reshape back to [B, n_feat, L]
        out = out.view(B, L, self.n_features).permute(0, 2, 1).contiguous()
        return out.float()


# ---------------------------------------------------------------------------
# Multi-scale wrapper
# ---------------------------------------------------------------------------

class MultiScaleQuantumConv1D(nn.Module):
    """
    Concatenates the outputs of several _QuantumBranch instances along the
    channel axis. Each branch is followed by its own BatchNorm1d, since the
    feature distributions of the k=3 / k=5 / k=7 branches differ markedly
    (longer-range correlators tend to be smaller in absolute value).

    Args:
        scales (tuple[int]): odd window sizes, e.g. (3, 5, 7).
        n_layers (int): variational layer count per branch (shared).
        seed (int): base seed (each branch uses seed + i for diversity).

    Output channels: sum_k (3k + 9 * k*(k-1)/2).
        (3,)     -> 36
        (3,5)    -> 141
        (3,5,7)  -> 351
        (5,)     -> 105
        (7,)     -> 210
    """

    def __init__(self, scales=(3, 5, 7), n_layers=1, seed=42,
                 use_position_encoding=True):
        super().__init__()
        self.scales = tuple(int(k) for k in scales)
        self.use_position_encoding = bool(use_position_encoding)
        self.branches = nn.ModuleList([
            _QuantumBranch(window_size=k, n_layers=n_layers, seed=seed + i,
                           use_position_encoding=self.use_position_encoding)
            for i, k in enumerate(self.scales)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(branch.n_features) for branch in self.branches
        ])
        self.out_channels = sum(branch.n_features for branch in self.branches)

    def forward(self, x):
        feats = []
        for branch, bn in zip(self.branches, self.bns):
            f = branch(x)            # [B, n_feat_k, L]
            f = bn(f)
            feats.append(f)
        return torch.cat(feats, dim=1)  # [B, out_channels, L]

    # Convenience: report sizes for logging / sanity checks
    def feature_breakdown(self):
        return {
            f"k={k}": branch.n_features
            for k, branch in zip(self.scales, self.branches)
        }