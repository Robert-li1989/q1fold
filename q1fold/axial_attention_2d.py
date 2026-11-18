"""
Q1Fold Axial Attention 2D Refiner
==================================

Replacement for the 2D ResNet block in Q1Fold.model.Q1Fold.

The 2D pairing map produced by the outer product of the multi-scale quantum
features is refined here with axial attention -- self-attention applied along
the row dimension and the column dimension in turn -- which is the standard
inductive bias used for protein contact prediction (AlphaFold's Evoformer,
ESMFold's blocks).

Compared to the existing 2D ResNet (~5M parameters, local 5x5 kernels with
dilation), this module:
    * has ~10% the parameter count at d_model=128, n_blocks=2;
    * gives every (i, j) entry a global L-wide receptive field after one
      row+column pass (instead of needing many dilated convs to reach far);
    * naturally respects the symmetry of contact maps via shared row/column
      attention weights.

Interface (drop-in replacement for `self.resnet2d`):
    refiner = AxialAttention2DRefiner(in_channels=2,
                                      d_model=128,
                                      n_blocks=2,
                                      n_heads=8,
                                      out_channels=128)
    y_in : [B, in_channels, L, L]  -> y_out : [B, out_channels, L, L]

The downstream `conv2Dout` expects `out_channels` channels, so set
`filters_resnet2d=128` (or whatever d_model you chose) in your config to
match.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

class _AxialMHSA(nn.Module):
    """
    Multi-head self-attention applied along ONE spatial axis of a 4D feature
    map [B, C, L, L]. The choice of axis is determined by the caller via
    transposes (see `AxialBlock`); this layer itself is axis-agnostic and
    operates on tensors of shape [B*L, L, C].

    Weights are intentionally exposed so that the row and column attentions
    in `AxialBlock` can SHARE this single module, which:
        (a) halves parameter count vs. independent row/col attentions, and
        (b) bakes the contact-map transpose symmetry into the architecture.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Mem-efficient and Flash SDPA kernels on Ampere (RTX 3090) require
        # head_dim in {16, 32, 64, 128}. Other values silently fall back to
        # the math kernel, which materializes the L*L attention matrix and
        # OOMs at L=512. Fail loudly at construction instead.
        assert self.d_head in (8, 16, 32, 64, 128), (
            f"head_dim = d_model / n_heads = {d_model} / {n_heads} = {self.d_head}; "
            f"must be in {{8, 16, 32, 64, 128}} for memory-efficient SDPA on Ampere "
            f"(otherwise PyTorch falls back to the math kernel and the L*L attention "
            f"matrix is materialized -- OOM at L=512). "
            f"Try d_model=128 n_heads=8 (head_dim=16) or d_model=256 n_heads=8 "
            f"(head_dim=32)."
        )

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x):
        """
        x : [B*, L, d_model]    (B* is the batch dim with the orthogonal-axis
                                  positions already folded in)
        """
        Bx, L, _ = x.shape
        qkv = self.qkv(x)                                # [B*, L, 3d]
        q, k, v = qkv.chunk(3, dim=-1)                   # each [B*, L, d]
        # Reshape for multi-head: [B*, n_heads, L, d_head]
        q = q.view(Bx, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(Bx, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(Bx, L, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention with explicit kernel selection.
        #
        # On Ampere (RTX 3090), the math kernel materializes the full L*L
        # attention matrix (8.6 GB at L=512, B=2, n_heads=8 in fp32!) and
        # will silently OOM. We REQUIRE the memory-efficient or flash
        # kernel, which use O(B*L*d_head) memory per head.
        #
        # IMPORTANT: these kernels require head_dim to be a supported value
        # (16, 32, 64, 128). If d_model / n_heads is not one of these, the
        # call below will raise a RuntimeError -- by design, so we don't
        # silently fall back to the math kernel and OOM.
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                          SDPBackend.FLASH_ATTENTION]):
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        # attn : [B*, n_heads, L, d_head] -> [B*, L, d_model]
        attn = attn.transpose(1, 2).contiguous().view(Bx, L, self.d_model)

        return self.drop(self.out(attn))


class _AxialBlock(nn.Module):
    """
    One axial-attention block:
        x -> x + row_attn(LN(x))
          -> x + col_attn(LN(x))      (col_attn shares weights with row_attn)
          -> x + mlp   (LN(x))

    Inputs / outputs : [B, C=d_model, L, L]
    """

    def __init__(self, d_model: int, n_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # ONE shared MHSA used for both row and column attention. Sharing the
        # weights makes the block transpose-equivariant: f(x.T) == f(x).T.
        self.attn = _AxialMHSA(d_model, n_heads, dropout=dropout)

        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def _apply_attn_on_axis(self, x, axis: str):
        """
        x : [B, L, L, C]   (channels-last for LayerNorm + Linear)
        axis : 'row' attends across the SECOND L (columns at fixed row),
               'col' attends across the FIRST  L (rows at fixed column).
        """
        B, Lr, Lc, C = x.shape
        if axis == 'row':
            # Fold the row index into the batch: each row is a sequence of
            # length Lc; attend across columns.
            inp = x.reshape(B * Lr, Lc, C)
            out = self.attn(inp)
            return out.view(B, Lr, Lc, C)
        elif axis == 'col':
            # Fold the column index into the batch: each column is a
            # sequence of length Lr; attend across rows.
            inp = x.transpose(1, 2).reshape(B * Lc, Lr, C)
            out = self.attn(inp)
            return out.view(B, Lc, Lr, C).transpose(1, 2)
        else:
            raise ValueError(axis)

    def forward(self, x):
        # x in:  [B, C, L, L]  ->  channels-last for LN + Linear convenience
        x = x.permute(0, 2, 3, 1).contiguous()                    # [B, L, L, C]

        # Row attention with residual + pre-norm
        x = x + self._apply_attn_on_axis(self.norm1(x), 'row')
        # Column attention with residual + pre-norm (shared weights)
        x = x + self._apply_attn_on_axis(self.norm2(x), 'col')
        # MLP with residual + pre-norm
        x = x + self.mlp(self.norm3(x))

        # Back to channels-first
        return x.permute(0, 3, 1, 2).contiguous()                 # [B, C, L, L]


# --------------------------------------------------------------------------- #
# Top-level module
# --------------------------------------------------------------------------- #

class AxialAttention2DRefiner(nn.Module):
    """
    Drop-in replacement for the 2D ResNet that refines the L x L pairing map.

    Args
    ----
    in_channels : channels of the input feature map (mid_ch in the Q1Fold
        forward pass; typically 2 = raw outer product + interaction prior).
    d_model     : working channel width inside the attention blocks.
    n_blocks    : number of stacked axial blocks (each = row attn + col attn
                  with shared weights + MLP).
    n_heads     : attention heads per block (d_model must be divisible).
    out_channels: channels produced for the downstream `conv2Dout` layer.
                  Set this equal to filters_resnet2d in your Q1Fold config.
    dropout     : dropout applied to attention output and MLP.

    Parameter counts (rough):
        d_model=128, n_blocks=2, n_heads=8:
            attn (shared)  4 * 128^2 = 65K  per block (Q,K,V,O)
            mlp            2 * 4 * 128^2 = 131K per block
            norms          ~1K per block
            -> ~ 200K per block * 2 blocks = ~400K
            + in/out projections: ~ (2*128) + (128*128) = 16K + 16K = 32K
            ===========================================
            ~450K classical parameters total.
    """

    def __init__(self,
                 in_channels: int = 2,
                 d_model: int = 128,
                 n_blocks: int = 2,
                 n_heads: int = 8,
                 out_channels: int = 128,
                 dropout: float = 0.0):
        super().__init__()
        self.in_proj  = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.blocks   = nn.ModuleList([
            _AxialBlock(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Conv2d(d_model, out_channels, kernel_size=1)

        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.out_channels = out_channels

    def forward(self, x):
        """
        x : [B, in_channels, L, L]
        returns : [B, out_channels, L, L]
        """
        x = self.in_proj(x)                          # [B, d_model, L, L]
        for blk in self.blocks:
            x = blk(x)

        # Final LayerNorm before the output projection.
        x = x.permute(0, 2, 3, 1).contiguous()       # channels-last
        x = self.norm_out(x)
        x = x.permute(0, 3, 1, 2).contiguous()       # channels-first

        return self.out_proj(x)