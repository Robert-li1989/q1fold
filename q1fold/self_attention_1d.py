"""
Q1Fold 1D Self-Attention Refiner
=================================

This module operates on the per-position 1D quantum features [B, C, L]
BEFORE the rank-reduction + outer-product step. The motivation is that
the previous 2D axial attention received an already-compressed [B, 2, L, L]
input (the outer product of rank-64 projections of the 351 quantum
channels), so the attention had no quantum information left to attend over.

By putting attention on the 1D representation upstream of the bilinear
pair construction, we let the network refine the per-position quantum
features with global L-wide context BEFORE collapsing them into a 2D map.
This matches the design pattern used by AlphaFold's Evoformer and ESMFold,
where 1D / single representations are refined first and pair features are
constructed from already-attended single features.

Positional encoding toggle
---------------------------
This refiner injects a fixed sinusoidal positional encoding by default
(use_pos_enc=True). It can be disabled (use_pos_enc=False) so you can run a
clean ablation of where position information should live: with the PE off, the
attention block becomes permutation-equivariant and carries NO absolute
position, which lets you measure whether position injected elsewhere (e.g. in
the quantum extractor, or via a 2D relative-position encoding) actually
contributes on its own rather than being redundant with this PE.

Interface
---------
    refiner = SelfAttention1DRefiner(
        in_channels=351,        # multi-scale quantum output dim
        d_model=128,
        n_blocks=4,
        n_heads=8,
        use_pos_enc=True,       # set False to ablate the sinusoidal PE
    )
    x: [B, in_channels, L]  ->  z: [B, d_model, L]

The downstream pipeline (rank-reduction conv -> outer product -> 2D head)
operates on `z` instead of the raw quantum output.

Memory note
-----------
At L=512, B=2, d_model=128, n_heads=8 (head_dim=16) the SDPA call needs
only O(B * L * d) memory because we're doing 1D self-attention (not the
2D row-then-column variant). Peak working memory per block: ~10 MB. This
is roughly 1/L = 1/512 the memory cost of the 2D AxialAttention2DRefiner
on the same dimensions.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Single attention block
# --------------------------------------------------------------------------- #

class _MHSA1D(nn.Module):
    """
    Standard multi-head self-attention over a 1D sequence [B, L, d_model].

    Same SDPA kernel safety as the 2D axial module: head_dim must be in
    {16, 32, 64, 128} on Ampere so the memory-efficient or flash kernel is
    used. MATH fallback is disallowed.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head in (8, 16, 32, 64, 128), (
            f"head_dim = {self.d_head}; must be in {{8, 16, 32, 64, 128}} for "
            f"memory-efficient SDPA. Try d_model=128 n_heads=8 (head_dim=16)."
        )
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """ x : [B, L, d_model] """
        B, L, _ = x.shape
        qkv = self.qkv(x)                                    # [B, L, 3d]
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION,
                          SDPBackend.FLASH_ATTENTION]):
            attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

        attn = attn.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.drop(self.out(attn))


class _AttnBlock1D(nn.Module):
    """
    One transformer block: MHSA + MLP, pre-norm, with residual connections.
    """

    def __init__(self, d_model: int, n_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = _MHSA1D(d_model, n_heads, dropout=dropout)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, L, d_model]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Positional encoding (sinusoidal, fixed)
# --------------------------------------------------------------------------- #

class _SinusoidalPE(nn.Module):
    """
    Standard fixed sinusoidal positional encoding. Self-attention is
    permutation-equivariant, so we need to inject position info somehow.
    We use sinusoidal PEs (no extra parameters, generalizes to arbitrary
    lengths -- bpRNA sequences span ~30 to ~500+ nt).
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # buffer so it moves with .to(device) but is not a parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        """ x : [B, L, d_model] -> x + pe[:L] """
        L = x.shape[1]
        return x + self.pe[:L].unsqueeze(0)


# --------------------------------------------------------------------------- #
# Top-level 1D refiner
# --------------------------------------------------------------------------- #

class SelfAttention1DRefiner(nn.Module):
    """
    Refine per-position quantum features with N transformer blocks before
    the 2D pairing-map construction.

    Args
    ----
    in_channels  : input channels (e.g. 351 for multi-scale 3+5+7).
    d_model      : working width inside the attention blocks.
    n_blocks     : number of transformer blocks.
    n_heads      : attention heads. head_dim = d_model / n_heads must be
                   in {8, 16, 32, 64, 128}.
    out_channels : output channel count. Defaults to d_model.
    dropout      : dropout applied to attention output and MLP.
    use_pos_enc  : if True (default) add the fixed sinusoidal positional
                   encoding before the transformer blocks. Set False to ablate
                   absolute position from the attention stage.

    Parameter count (rough):
        in_channels=351, d_model=128, n_blocks=4, n_heads=8:
            in_proj    : 351 * 128 = 45K   (Conv1d, k=1)
            per block  : 4*d^2 (QKV+O) + 2*4*d^2 (MLP) ~= 200K
            4 blocks   : ~800K
            out_proj   : 128 * 128 = 16K
            =====================================================
            ~860K classical parameters total.
    """

    def __init__(self,
                 in_channels: int,
                 d_model: int = 128,
                 n_blocks: int = 4,
                 n_heads: int = 8,
                 out_channels: int = None,
                 dropout: float = 0.0,
                 use_pos_enc: bool = True,
                 max_len: int = 2048):
        super().__init__()
        if out_channels is None:
            out_channels = d_model
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.out_channels = out_channels
        self.use_pos_enc = bool(use_pos_enc)

        # 1x1 conv to project quantum features -> d_model.
        # Using Conv1d (rather than Linear) so the shape convention stays
        # [B, C, L] consistent with the upstream extractor.
        self.in_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)
        # Constructed unconditionally (it's a buffer, no params) so the
        # module's state_dict is identical whether or not the PE is applied;
        # application is gated in forward by use_pos_enc.
        self.pos_enc = _SinusoidalPE(d_model, max_len=max_len)

        self.blocks = nn.ModuleList([
            _AttnBlock1D(d_model=d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        # 1x1 conv to project back to out_channels for the downstream
        # rank-reduction step.
        self.out_proj = nn.Conv1d(d_model, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x : [B, in_channels, L]
        returns : [B, out_channels, L]
        """
        # in_proj: [B, in, L] -> [B, d_model, L]
        z = self.in_proj(x)
        # to channels-last for LN + Linear convenience: [B, L, d_model]
        z = z.transpose(1, 2)
        # inject sinusoidal positional info (ablatable)
        if self.use_pos_enc:
            z = self.pos_enc(z)
        # transformer blocks
        for blk in self.blocks:
            z = blk(z)
        z = self.norm_out(z)
        # back to channels-first and project
        z = z.transpose(1, 2)
        z = self.out_proj(z)
        return z                                # [B, out_channels, L]