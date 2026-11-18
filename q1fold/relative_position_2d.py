"""
Q1Fold 2D Relative-Position Encoding
====================================

Tier A position injection. Produces a [B, rp_channels, L, L] feature map that
encodes the relative offset (i - j) between sequence positions, to be
CONCATENATED as extra channels onto the 2D pair representation right before the
2D refiner (dilated ResNet / axial attention).

Rationale
---------
The model already has *absolute* position information: the 1D self-attention
refiner adds a sinusoidal positional encoding, and the (i, j) grid of the pair
map encodes absolute indices structurally. What is missing is an explicit,
unbounded notion of *relative distance* |i - j|, which is the quantity base
pairing actually depends on (loop length / span). The dilated 2D ResNet only
sees relative offset implicitly, through a bounded receptive field
(dilations 1..16 -> a few hundred nt at most). This module injects relative
distance directly, so long-range pairs get an explicit distance signal.

Bucketing
---------
Offsets are bucketed with a T5-style mixed linear/logarithmic scheme: exact
(per-offset) buckets for short distances, logarithmically-growing buckets for
long distances. This gives fine resolution where most pairs live (short range)
while still distinguishing long-range offsets in long RNAs, instead of dumping
everything past a fixed window into one "far" bucket.

Symmetry
--------
RNA secondary-structure contact maps are symmetric (pair (i, j) == (j, i)), and
the model symmetrizes its output. By default this module is symmetric too
(buckets |i - j|), so the injected prior does not fight the symmetrization. Set
symmetric=False for a signed (direction-aware) encoding if you want to let the
network distinguish 5'->3' orientation.

Capacity
--------
The only learnable parameters are the bucket embedding table:
    num_buckets * rp_channels  (e.g. 64 * 16 = 1024 params).
Deliberately tiny -- this targets generalization on long sequences without
adding the kind of capacity that would worsen the existing train/valid gap.

Usage
-----
    relpos = RelativePositionEncoding2D(rp_channels=16)
    rp = relpos(L, device=x1.device, batch_size=x1.shape[0])  # [B, 16, L, L]
    x1 = torch.cat([x1, rp], dim=1)   # add rp_channels to the 2D refiner input

These parameters are picked up by the *classical* optimizer automatically (the
module name does not match any of the quantum keywords in
Q1Fold._separate_parameters).
"""

import math

import torch
import torch.nn as nn


class RelativePositionEncoding2D(nn.Module):
    """
    Learned relative-position feature map for the 2D pair stack.

    Args:
        rp_channels (int): number of output channels (embedding dim per bucket).
        num_buckets (int): total number of relative-position buckets.
        max_distance (int): offsets beyond this are folded into the last
            (log) bucket. Set near your maximum sequence length.
        symmetric (bool): if True, bucket |i - j| (symmetric, default). If
            False, use signed offset (direction-aware), splitting buckets into
            negative/positive halves.
    """

    def __init__(self, rp_channels: int = 16, num_buckets: int = 64,
                 max_distance: int = 512, symmetric: bool = True):
        super().__init__()
        self.rp_channels = int(rp_channels)
        self.num_buckets = int(num_buckets)
        self.max_distance = int(max_distance)
        self.symmetric = bool(symmetric)

        self.emb = nn.Embedding(self.num_buckets, self.rp_channels)
        nn.init.normal_(self.emb.weight, std=0.02)

        # Cache of bucket-index maps keyed by (L, device-string). The indices
        # depend only on L, so we avoid recomputing them every forward; the
        # embedding lookup still runs each forward for gradients.
        self._bucket_cache = {}

    # ----------------------------------------------------------------- #
    @staticmethod
    def _bucketize(relative_position, num_buckets, max_distance, symmetric):
        """
        relative_position : [L, L] long tensor of (i - j)
        returns           : [L, L] long tensor of bucket indices in
                            [0, num_buckets)
        """
        ret = torch.zeros_like(relative_position)
        n = num_buckets
        if not symmetric:
            # Signed: lower half of buckets for negative offsets, upper half
            # for positive offsets.
            n = num_buckets // 2
            ret = ret + (relative_position > 0).long() * n
            rp = relative_position.abs()
        else:
            rp = relative_position.abs()

        max_exact = max(1, n // 2)
        is_small = rp < max_exact

        # Logarithmic spacing for the large-offset region.
        rp_large = max_exact + (
            torch.log(rp.float().clamp(min=1.0) / max_exact)
            / math.log(max_distance / max_exact)
            * (n - max_exact)
        ).long()
        rp_large = torch.minimum(
            rp_large, torch.full_like(rp_large, n - 1)
        )

        ret = ret + torch.where(is_small, rp, rp_large)
        return ret

    def _get_buckets(self, L, device):
        key = (L, str(device))
        buckets = self._bucket_cache.get(key)
        if buckets is None:
            idx = torch.arange(L, device=device)
            rel = idx[:, None] - idx[None, :]          # [L, L]  (i - j)
            buckets = self._bucketize(
                rel, self.num_buckets, self.max_distance, self.symmetric
            )                                          # [L, L] long
            self._bucket_cache[key] = buckets
        return buckets

    # ----------------------------------------------------------------- #
    def forward(self, L, device=None, batch_size=1):
        """
        Args:
            L (int): sequence length.
            device: target device (defaults to the embedding's device).
            batch_size (int): batch dimension to expand to.
        Returns:
            [batch_size, rp_channels, L, L] float tensor.
        """
        if device is None:
            device = self.emb.weight.device
        buckets = self._get_buckets(L, device)         # [L, L] long
        emb = self.emb(buckets)                         # [L, L, rp_channels]
        emb = emb.permute(2, 0, 1).contiguous()         # [rp_channels, L, L]
        emb = emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return emb
