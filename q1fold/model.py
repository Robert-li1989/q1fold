"""
Q1Fold Model Module

Hybrid quantum-classical convolutional neural network for RNA secondary structure prediction.
"""

from torch import nn
from torch.nn.functional import cross_entropy
import torch as tr
from tqdm import tqdm
import pandas as pd
import math

from .metrics import contact_f1
from .utils import mat2bp, postprocessing
from ._version import __version__
from .quantum_conv1d import QuantumConv1D
from .multiscale_quantum_conv1d import MultiScaleQuantumConv1D
from .axial_attention_2d import AxialAttention2DRefiner
from .self_attention_1d import SelfAttention1DRefiner
from .relative_position_2d import RelativePositionEncoding2D


# URL for pretrained weights (update this when you host your own weights)
Q1FOLD_WEIGHTS = None  # Set to URL when pretrained weights are available


def q1fold(pretrained=False, weights=None, **kwargs):
    """
    Q1Fold: A hybrid quantum-classical CNN for RNA secondary structure prediction
    
    Args:
        pretrained (bool): Use pretrained weights
        weights (str): Path to custom weights file
        **kwargs: Model hyperparameters
        
    Returns:
        Q1Fold model instance
    """
    model = Q1Fold(**kwargs)
    if pretrained:
        if Q1FOLD_WEIGHTS is None:
            raise ValueError("Pretrained weights are not available yet. Please provide custom weights.")
        print("Load pretrained weights...")
        model.load_state_dict(tr.hub.load_state_dict_from_url(Q1FOLD_WEIGHTS, map_location=tr.device(model.device)))
    else:
        if weights is not None:
            print(f"Load weights from {weights}")
            model.load_state_dict(tr.load(weights, map_location=tr.device(model.device), weights_only=True))
        else:
            print("No weights provided, using random initialization")
        
    return model


class Q1Fold(nn.Module):
    """
    Q1Fold: Hybrid Quantum-Classical CNN for RNA Secondary Structure Prediction
    
    Architecture:
        1. Quantum 1D feature extraction using 6-qubit VQC
        2. Classical 1D convolutions for feature refinement
        3. Matrix product for 2D pairing map construction
        4. 2D ResNet for contact map prediction
    
    Args:
        train_len: Number of training batches (for scheduler)
        embedding_dim: Input embedding dimension (default 4 for one-hot)
        device: Computing device ('cpu' or 'cuda')
        negative_weight: Weight for negative class in loss
        lr: Learning rate for classical parameters
        quantum_lr: Learning rate for quantum parameters
        lr_decay: Weight decay for AdamW
        loss_l1: L1 regularization coefficient
        loss_beta: Auxiliary loss coefficient
        scheduler: LR scheduler type ('plateau' or 'cycle')
        verbose: Whether to print progress
        interaction_prior: Type of interaction prior ('none' or 'probmat')
        output_th: Output threshold for binarization
    """
    
    def __init__(
        self,
        train_len=0,
        embedding_dim=4,
        device="cpu",
        negative_weight=0.1,
        lr=1e-4,
        quantum_lr=1e-3,            # DEPRECATED / inert (see note below)
        lr_decay=0,                 # DEPRECATED / inert
        classical_weight_decay=0.0,
        extractor_lr=None,          # LR for the swappable 1D extractor group
        weight_decay=None,          # unified weight decay for ALL parameters
        grad_clip=1.0,              # global grad-norm clip (applied to all params)
        loss_l1=0.0005,
        loss_beta=0.15,
        scheduler="plateau",
        scheduler_patience=5,
        verbose=True,
        interaction_prior="none",
        output_th=0.5,
        **kwargs
    ):
        # ------------------------------------------------------------------ #
        # Optimizer / training-recipe fairness note
        # ------------------------------------------------------------------ #
        # Earlier versions trained the *quantum* extractor with its own AdamW
        # (quantum_lr=2e-4, amsgrad, betas=(0.95,0.999)), gradient clipping,
        # and an adaptive L2 schedule, while *classical* extractors rode the
        # main optimizer (lr, weight_decay, no clip, no L2). That made the
        # extractor's training recipe depend on whether it was quantum or
        # classical -- a confound when comparing the two.
        #
        # The recipe is now a function of architectural ROLE, not of
        # quantum-vs-classical:
        #   * extractor group  (everything under `resnet1d`) -> lr=extractor_lr
        #   * main group       (everything else)            -> lr=lr
        # Both groups share one AdamW, one weight_decay, one global grad clip,
        # and one scheduler. `extractor_lr` defaults to `lr`, so by default the
        # whole model trains under a single identical recipe. The legacy
        # `quantum_lr` / `lr_decay` keys are accepted (for config compat) but
        # have NO effect; `classical_weight_decay` is honored as the default
        # for the unified `weight_decay` if the latter is not given.
        super().__init__()
        
        self.current_epoch = 0
        self.device = device
        self.class_weight = tr.tensor([negative_weight, 1.0]).float().to(device)
        self.loss_l1 = loss_l1
        self.loss_beta = loss_beta
        self.verbose = verbose
        self.config = kwargs
        self.output_th = output_th

        # ------------------------------------------------------------------ #
        # Distance-weighted loss (recall-targeting). OFF by default, so the
        # loss is bit-for-bit identical to the flat weighted cross-entropy
        # used by v9-v14. When `loss_distance_weight` names a mode, the loss
        # contribution of each pair position is multiplied by an increasing
        # function of |i - j|, applied to positives (default) so that MISSING
        # a long-range true pair costs more than missing a short-range one --
        # directly attacking the long-range false-negative deficit.
        #   loss_distance_weight : 'none' | 'linear' | 'sqrt' | 'log'
        #   loss_distance_scale  : strength (0 => neutral)
        #   loss_distance_ref    : reference distance for normalization
        #   loss_distance_cap    : max multiplier (0 => uncapped); a cap guards
        #                          against amplifying label noise on the very
        #                          longest (and least certain) bpRNA pairs
        #   loss_distance_apply  : 'positive' (default) | 'all'
        self.loss_distance_weight = kwargs.get('loss_distance_weight', 'none')
        self.loss_distance_scale = float(kwargs.get('loss_distance_scale', 0.0))
        self.loss_distance_ref = float(kwargs.get('loss_distance_ref', 100.0))
        self.loss_distance_cap = float(kwargs.get('loss_distance_cap', 0.0))
        self.loss_distance_apply = kwargs.get('loss_distance_apply', 'positive')
        self._dist_w_cache = {}

        mid_ch = 1
        self.interaction_prior = interaction_prior
        if interaction_prior != "none":
            mid_ch = 2

        # Define architecture
        self.build_graph(embedding_dim, mid_ch=mid_ch, **kwargs) 

        # Split parameters by architectural ROLE (extractor vs the rest).
        self.extractor_params = []
        self.main_params = []
        self._separate_parameters()

        # Resolve unified hyperparameters.
        if extractor_lr is None:
            extractor_lr = lr
        if weight_decay is None:
            weight_decay = classical_weight_decay
        self.extractor_lr = extractor_lr
        self.lr = lr
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

        # Whether the (architectural) extractor is a quantum circuit. This is
        # used ONLY for logging now -- it no longer changes the training
        # recipe in any way.
        self.has_quantum_params = any(
            n.endswith(('.theta', '.embed_weights', '.pos_alpha',
                        '.measurement_weights'))
            for n, _ in self.named_parameters()
        )
        if self.verbose:
            kind = "QUANTUM" if self.has_quantum_params else "classical"
            print(f"[Q1Fold] extractor is {kind}; unified recipe -> "
                  f"extractor_lr={extractor_lr}, main_lr={lr}, "
                  f"weight_decay={weight_decay}, grad_clip={grad_clip}")

        # Single optimizer, two parameter groups. The ONLY thing that differs
        # between groups is the learning rate, and the extractor group's LR is
        # identical regardless of whether the extractor is quantum or classical.
        param_groups = []
        if self.extractor_params:
            param_groups.append({"params": self.extractor_params,
                                  "lr": extractor_lr, "name": "extractor"})
        if self.main_params:
            param_groups.append({"params": self.main_params,
                                  "lr": lr, "name": "main"})
        self.optimizer = tr.optim.AdamW(param_groups, lr=lr,
                                        weight_decay=weight_decay)

        # Single LR scheduler over the whole optimizer.
        self.scheduler_name = scheduler
        if scheduler == "plateau":
            self.scheduler = tr.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=scheduler_patience
            )
        elif scheduler == "cycle":
            # Per-group max_lr keeps the extractor/main LR ratio intact.
            max_lrs = [g["lr"] for g in param_groups]
            self.scheduler = tr.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=max_lrs,
                steps_per_epoch=max(train_len, 1),
                epochs=self.config.get("max_epochs", 1000),
            )
        else:
            self.scheduler = None

        self.to(device)
  
    def build_graph(
        self,
        embedding_dim,
        kernel=3,
        filters=None,
        num_layers=0,
        dilation_resnet1d=3,
        resnet_bottleneck_factor=0.5,
        mid_ch=1,
        kernel_resnet2d=5,
        bottleneck1_resnet2d=256,
        bottleneck2_resnet2d=128,
        filters_resnet2d=256,
        rank=64,
        dilation_resnet2d=3,
        extractor_type='multiscale_quantum',
        multiscale_scales=(3, 5, 7),
        multiscale_layers=1,
        multiscale_use_position_encoding=False,
        use_1d_attention=False,
        attn1d_d_model=128,
        attn1d_n_blocks=4,
        attn1d_n_heads=8,
        attn1d_dropout=0.0,
        attn1d_use_position_encoding=True,
        pair_construction='scalar_bilinear',
        pair_channels=32,
        use_relpos_2d=False,
        relpos_channels=16,
        relpos_num_buckets=64,
        relpos_max_distance=512,
        relpos_symmetric=True,
        refiner_type='resnet2d',
        axial_d_model=96,
        axial_n_blocks=2,
        axial_n_heads=6,
        axial_dropout=0.0,
        dilated_filters=96,
        dilated_bottleneck=48,
        dilated_dilations=(1, 2, 4, 8, 16),
        dilated_kernel=5,
        **kwargs
    ):
        """Build the model architecture.

        `filters` is the number of channels feeding into the rank-reduction
        convs that build the 2D pairing map. If None (default), it is inferred
        from the chosen extractor's output channel count. This is required for
        `multiscale_quantum`, which can produce anywhere from 36 to 351
        channels depending on `multiscale_scales`.
        """
        pad = (kernel - 1) // 2

        self.use_restrictions = mid_ch != 1

        # ---- Pick feature extractor ------------------------------------- #
        if extractor_type == 'quantum':
            # Legacy 6-qubit, 52-feature module (kept for ablation baseline)
            n_feat = 52
            self.resnet1d = nn.Sequential(QuantumConv1D(), nn.BatchNorm1d(n_feat))

        elif extractor_type == 'multiscale_quantum':
            # New: 3-mer + 5-mer + 7-mer branches, biochemical encoding only
            # (no context qubits), single-q Paulis + all-axis pair correlators
            # (no triplet measurements in this phase).
            extractor = MultiScaleQuantumConv1D(
                scales=tuple(multiscale_scales),
                n_layers=multiscale_layers,
                use_position_encoding=multiscale_use_position_encoding,
            )
            n_feat = extractor.out_channels
            # MultiScaleQuantumConv1D already includes per-branch BN.
            self.resnet1d = extractor
            if self.verbose:
                print(f"[Q1Fold] multiscale extractor: scales={tuple(multiscale_scales)}, "
                      f"out_channels={n_feat}, "
                      f"use_position_encoding={multiscale_use_position_encoding}, "
                      f"breakdown={extractor.feature_breakdown()}")

        elif extractor_type == 'multiscale_classical':
            # Classical ablation counterpart to MultiScaleQuantumConv1D: same
            # k=3/5/7 parallel-branch structure and identical output-channel
            # contract (3k + 9*C(k,2) per branch -> 351 for (3,5,7)), but each
            # k-qubit VQC is replaced by a small Conv1d stack. Swapping ONLY
            # this isolates the quantum contribution; everything downstream
            # (attn1d, OPM pairing, dilated refiner, loss) is unchanged.
            from .classical_matched_baseline_v2 import MultiScaleClassicalBaseline
            extractor = MultiScaleClassicalBaseline(
                scales=tuple(multiscale_scales),
                hidden_channels=kwargs.get('multiscale_classical_hidden', 32),
                depth=kwargs.get('multiscale_classical_depth', 2),
            )
            n_feat = extractor.out_channels
            # Wrap in Sequential so parameter names become
            # `resnet1d.0.branches...`, which does NOT match the quantum
            # routing keywords ('resnet1d.branches', 'resnet1d.bns') in
            # _separate_parameters -> these params route to the classical
            # optimizer, as required for the ablation.
            self.resnet1d = nn.Sequential(extractor)
            if self.verbose:
                print(f"[Q1Fold] multiscale CLASSICAL extractor: "
                      f"scales={tuple(multiscale_scales)}, out_channels={n_feat}, "
                      f"breakdown={extractor.feature_breakdown()}")

        elif extractor_type == 'matched_classical':
            # Table 2: parameter-matched classical control (69 + 702 = 771).
            # Same encoding/parameterisation as the quantum branches, features
            # computed classically on a product state (no entanglement).
            from .matched_extractors import MatchedClassicalMultiScale
            extractor = MatchedClassicalMultiScale(
                scales=tuple(multiscale_scales),
                seed=kwargs.get('extractor_seed', 42),
            )
            n_feat = extractor.out_channels
            self.resnet1d = extractor
            if self.verbose:
                print(f"[Q1Fold] MATCHED classical extractor: out_channels={n_feat}, "
                      f"params={sum(p.numel() for p in extractor.parameters())}")

        elif extractor_type == 'competitive_classical':
            # Table 2: competitive classical control (<= 10 x 771 params).
            from .matched_extractors import CompetitiveClassicalMultiScale
            extractor = CompetitiveClassicalMultiScale(
                scales=tuple(multiscale_scales),
                hidden=kwargs.get('competitive_hidden', 14),
                seed=kwargs.get('extractor_seed', 42),
            )
            n_feat = extractor.out_channels
            self.resnet1d = extractor
            if self.verbose:
                print(f"[Q1Fold] COMPETITIVE classical extractor: out_channels={n_feat}, "
                      f"params={sum(p.numel() for p in extractor.parameters())}")

        elif extractor_type == 'plain_conv':
            # Plain parallel Conv1d(4 -> 36/105/210, k=3/5/7) + BN per branch,
            # optionally with ReLU (plain_conv_activation='relu').
            from .matched_extractors import PlainConvMultiScale
            extractor = PlainConvMultiScale(
                scales=tuple(multiscale_scales),
                activation=kwargs.get('plain_conv_activation', None),
                seed=kwargs.get('extractor_seed', 42),
            )
            n_feat = extractor.out_channels
            self.resnet1d = extractor
            if self.verbose:
                print(f"[Q1Fold] PLAIN CONV extractor "
                      f"(activation={kwargs.get('plain_conv_activation', None)}): "
                      f"out_channels={n_feat}, "
                      f"params={sum(p.numel() for p in extractor.parameters())}")

        elif extractor_type == 'classical_matched':
            from .classical_matched_baseline_v2 import ParameterMatchedCMB
            extractor = ParameterMatchedCMB()
            n_feat = getattr(extractor, 'out_channels', 52)
            self.resnet1d = nn.Sequential(extractor)

        elif extractor_type == 'classical_standard':
            from .classical_matched_baseline_v2 import ClassicalMatchedBaseline
            extractor = ClassicalMatchedBaseline()
            n_feat = getattr(extractor, 'out_channels', 52)
            self.resnet1d = nn.Sequential(extractor)

        else:
            raise ValueError(f"Unknown extractor_type: {extractor_type}")

        # Auto-infer downstream filter count if caller did not override.
        if filters is None:
            filters = n_feat



        # Additional 1D residual layers (if num_layers > 0)
        layers_1d = []
        for k in range(num_layers):
            layers_1d.append(
                ResidualLayer1D(
                    dilation_resnet1d,
                    resnet_bottleneck_factor,
                    filters,
                    kernel,
                )
            )
        if layers_1d:
            self.resnet1d = nn.Sequential(self.resnet1d, *layers_1d)

        # ---- Optional 1D self-attention refiner -------------------------- #
        # When enabled, the per-position quantum features [B, n_feat, L] are
        # refined by N transformer blocks BEFORE the rank-reduction +
        # outer-product step. This puts the attention upstream of the
        # bilinear-pair bottleneck, so it operates on the full multi-channel
        # quantum representation rather than on the already-compressed L*L
        # map. After this, `filters` becomes the refiner's output channels
        # so the downstream convrank layers see the refined representation.
        if use_1d_attention:
            self.attn1d = SelfAttention1DRefiner(
                in_channels=filters,
                d_model=attn1d_d_model,
                n_blocks=attn1d_n_blocks,
                n_heads=attn1d_n_heads,
                out_channels=attn1d_d_model,
                dropout=attn1d_dropout,
                use_pos_enc=attn1d_use_position_encoding,
            )
            filters = attn1d_d_model
            if self.verbose:
                n_params = sum(p.numel() for p in self.attn1d.parameters())
                print(f"[Q1Fold] 1D attention refiner: "
                      f"in={n_feat}, d_model={attn1d_d_model}, "
                      f"n_blocks={attn1d_n_blocks}, n_heads={attn1d_n_heads}, "
                      f"use_pos_enc={attn1d_use_position_encoding}, "
                      f"params={n_params:,}")
        else:
            self.attn1d = None

        # ---- Pair construction: scalar bilinear (legacy) vs OPM ---------- #
        # 'scalar_bilinear': two rank-`rank` projections -> outer product ->
        #     [B, L, L] (one scalar per (i,j)). Cheap, low-rank, but it's
        #     the model's bottleneck for complex pairwise interactions.
        # 'outer_product_mean': project to `pair_channels`, take element-wise
        #     outer product -> [B, pair_channels, L, L]. Many channels per
        #     pair position; matches AlphaFold/ESMFold-style pair reps.
        self.pair_construction = pair_construction
        self.pair_channels = pair_channels

        if pair_construction == 'scalar_bilinear':
            # Legacy: two rank-`rank` projections, outer product, scalar.
            self.convrank1 = nn.Conv1d(
                in_channels=filters,
                out_channels=rank,
                kernel_size=kernel,
                padding=pad,
                stride=1,
            )
            self.convrank2 = nn.Conv1d(
                in_channels=filters,
                out_channels=rank,
                kernel_size=kernel,
                padding=pad,
                stride=1,
            )
            # mid_ch already set by caller (1 or 2 depending on prior)
            pair_to_head_channels = mid_ch

        elif pair_construction == 'outer_product_mean':
            # OPM-style: two small projections to pair_channels, element-wise
            # outer product gives [B, pair_channels, L, L].
            self.pair_proj_a = nn.Conv1d(
                in_channels=filters,
                out_channels=pair_channels,
                kernel_size=kernel,
                padding=pad,
                stride=1,
            )
            self.pair_proj_b = nn.Conv1d(
                in_channels=filters,
                out_channels=pair_channels,
                kernel_size=kernel,
                padding=pad,
                stride=1,
            )
            # Auxiliary 1-channel projection so the aux loss on `y0` still
            # has a meaningful target (preserves loss_func interface).
            self.pair_aux = nn.Conv2d(
                in_channels=pair_channels,
                out_channels=1,
                kernel_size=1,
            )
            # 2D head receives pair_channels + (1 if probmat prior) channels
            if self.interaction_prior != "none":
                pair_to_head_channels = pair_channels + 1
            else:
                pair_to_head_channels = pair_channels
            # Override mid_ch so the 2D head's first conv has the right
            # in_channels.
            mid_ch = pair_to_head_channels

            if self.verbose:
                n_pair = (sum(p.numel() for p in self.pair_proj_a.parameters())
                        + sum(p.numel() for p in self.pair_proj_b.parameters())
                        + sum(p.numel() for p in self.pair_aux.parameters()))
                print(f"[Q1Fold] pair construction: outer_product_mean "
                      f"(channels={pair_channels}, 2D head in={mid_ch}, "
                      f"new params={n_pair:,})")

        else:
            raise ValueError(f"Unknown pair_construction: {pair_construction}")

        # ---- Optional 2D relative-position encoding (Tier A) ------------- #
        # Concatenate a learned relative-distance feature map onto the 2D pair
        # representation just before the refiner. This injects an explicit,
        # unbounded notion of |i - j| (the quantity base pairing depends on)
        # that the bounded-receptive-field 2D refiner only sees implicitly.
        # `mid_ch` is bumped here so the refiner's first conv accepts the extra
        # channels; the corresponding concat happens in forward().
        self.use_relpos_2d = bool(use_relpos_2d)
        self.relpos_channels = relpos_channels if self.use_relpos_2d else 0
        if self.use_relpos_2d:
            self.relpos = RelativePositionEncoding2D(
                rp_channels=relpos_channels,
                num_buckets=relpos_num_buckets,
                max_distance=relpos_max_distance,
                symmetric=relpos_symmetric,
            )
            mid_ch = mid_ch + relpos_channels
            if self.verbose:
                n_rp = sum(p.numel() for p in self.relpos.parameters())
                print(f"[Q1Fold] 2D relative-position encoding: "
                      f"channels={relpos_channels}, buckets={relpos_num_buckets}, "
                      f"max_distance={relpos_max_distance}, "
                      f"symmetric={relpos_symmetric}, 2D head in={mid_ch}, "
                      f"params={n_rp:,}")
        else:
            self.relpos = None

        # ---- Pick 2D refiner (ResNet vs. axial attention) ---------------- #
        if refiner_type == 'resnet2d':
            self.resnet2d = nn.Sequential(
                nn.Conv2d(in_channels=mid_ch, out_channels=filters_resnet2d, kernel_size=7, padding="same"),
                ResidualBlock2D(
                    filters_resnet2d,
                    bottleneck1_resnet2d,
                    kernel_resnet2d,
                    dilation_resnet2d,
                ),
                ResidualBlock2D(
                    filters_resnet2d,
                    bottleneck2_resnet2d,
                    kernel_resnet2d,
                    dilation_resnet2d,
                )
            )
            refiner_out_channels = filters_resnet2d
            if self.verbose:
                print(f"[Q1Fold] 2D refiner: ResNet2D (in={mid_ch}, "
                      f"filters={filters_resnet2d}, bottlenecks=[{bottleneck1_resnet2d}, "
                      f"{bottleneck2_resnet2d}])")

        elif refiner_type == 'dilated_resnet2d':
            # Dilation ladder: input conv -> N residual blocks with
            # exponentially increasing dilation. With dilations
            # [1,2,4,8,16] and kernel 5 (only the second conv of each block
            # is dilated), the receptive field is 1 + 6 (7x7 stem)
            # + sum(4*(1+d)) + 4 (5x5 head) = 155 positions, covering most
            # long-range RNA base pairs that the shallow ResNet (RF ~29)
            # could not see. This directly targets the low-recall failure
            # mode on long-range stems.
            blocks = [
                nn.Conv2d(in_channels=mid_ch, out_channels=dilated_filters,
                          kernel_size=7, padding="same")
            ]
            for d in dilated_dilations:
                blocks.append(
                    ResidualBlock2D(
                        dilated_filters,
                        dilated_bottleneck,
                        dilated_kernel,
                        d,
                    )
                )
            self.resnet2d = nn.Sequential(*blocks)
            refiner_out_channels = dilated_filters
            if self.verbose:
                n_params = sum(p.numel() for p in self.resnet2d.parameters())
                # Receptive field estimate
                rf = 1 + 6 + sum((dilated_kernel - 1) * (1 + d) for d in dilated_dilations) + (dilated_kernel - 1)   # 7x7 stem, blocks, 5x5 head
                print(f"[Q1Fold] 2D refiner: DilatedResNet2D "
                      f"(in={mid_ch}, filters={dilated_filters}, "
                      f"bottleneck={dilated_bottleneck}, "
                      f"dilations={tuple(dilated_dilations)}, "
                      f"kernel={dilated_kernel}, RF~{rf}, params={n_params:,})")

        elif refiner_type == 'axial':
            self.resnet2d = AxialAttention2DRefiner(
                in_channels=mid_ch,
                d_model=axial_d_model,
                n_blocks=axial_n_blocks,
                n_heads=axial_n_heads,
                out_channels=axial_d_model,
                dropout=axial_dropout,
            )
            refiner_out_channels = axial_d_model
            if self.verbose:
                n_params = sum(p.numel() for p in self.resnet2d.parameters())
                print(f"[Q1Fold] 2D refiner: AxialAttention "
                      f"(in={mid_ch}, d_model={axial_d_model}, "
                      f"n_blocks={axial_n_blocks}, n_heads={axial_n_heads}, "
                      f"params={n_params:,})")

        else:
            raise ValueError(f"Unknown refiner_type: {refiner_type}")

        # Output layer (reads from refiner_out_channels so this works for
        # both refiner_type='resnet2d' and refiner_type='axial')
        self.conv2Dout = nn.Conv2d(
            in_channels=refiner_out_channels,
            out_channels=1,
            kernel_size=kernel_resnet2d,
            padding="same",
        )

    def _separate_parameters(self):
        """
        Split parameters by architectural ROLE, not by quantum-vs-classical.

        Everything under `self.resnet1d` is the swappable 1D extractor (the
        only module that changes across the v9 / v9.1 / v9.2 configs) and goes
        into the extractor group; everything else (attention, pair builder, 2D
        refiner, output head) goes into the main group. This makes the training
        recipe depend on where a parameter sits, not on whether the extractor
        happens to be a quantum circuit -- removing the optimizer confound from
        the quantum-vs-classical comparison.

        Note: the relative-position embedding (`self.relpos.*`) does not start
        with "resnet1d", so it correctly routes into the main group.
        """
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("resnet1d"):
                self.extractor_params.append(param)
            else:
                self.main_params.append(param)

        if self.verbose:
            n_ext = sum(p.numel() for p in self.extractor_params)
            n_main = sum(p.numel() for p in self.main_params)
            print(f"Extractor parameters (resnet1d.*): {n_ext}")
            print(f"Main parameters (attn/pair/refiner/head): {n_main}")

    def forward(self, batch):
        """Forward pass."""
        x = batch["embedding"].to(self.device)
        batch_size = x.shape[0]
        L = x.shape[2]
        
        # 1D feature extraction
        y = self.resnet1d(x)

        # Optional 1D self-attention refinement (Option A architecture):
        # operate on the per-position quantum features BEFORE bilinear
        # pair construction, so attention sees the full multi-channel
        # representation instead of an already-compressed [B, 2, L, L] map.
        if self.attn1d is not None:
            y = self.attn1d(y)

        # ---- Pair construction --------------------------------------- #
        if self.pair_construction == 'scalar_bilinear':
            # Legacy: rank-reduce, outer product, symmetrize.
            ya = self.convrank1(y)
            ya = tr.transpose(ya, -1, -2)
            yb = self.convrank2(y)
            y = ya @ yb
            yt = tr.transpose(y, -1, -2)
            y = (y + yt) / 2
            y0 = y.view(-1, L, L)

            # Stack with prior if enabled -> [B, mid_ch, L, L]
            if self.interaction_prior != "none":
                prob_mat = batch["interaction_prior"].to(self.device)
                x1 = tr.zeros([batch_size, 2, L, L]).to(self.device)
                x1[:, 0, :, :] = y0
                x1[:, 1, :, :] = prob_mat
            else:
                x1 = y0.unsqueeze(1)

        elif self.pair_construction == 'outer_product_mean':
            # OPM: project to pair_channels, element-wise outer product gives
            # a multi-channel pair representation [B, C, L, L].
            ya = self.pair_proj_a(y)          # [B, C, L]
            yb = self.pair_proj_b(y)          # [B, C, L]
            # pair[b,c,i,j] = ya[b,c,i] * yb[b,c,j]
            pair = ya.unsqueeze(-1) * yb.unsqueeze(-2)   # [B, C, L, L]
            # Symmetrize for the auxiliary scalar (kept for compat with
            # the existing aux loss on y0). The full multi-channel `pair`
            # is NOT symmetrized -- the 2D head + final (y+yT)/2 enforce
            # output symmetry.
            y0 = self.pair_aux(pair).squeeze(1)    # [B, L, L]
            y0_t = tr.transpose(y0, -1, -2)
            y0 = (y0 + y0_t) / 2

            # Stack pair channels with prior if enabled -> [B, mid_ch, L, L]
            if self.interaction_prior != "none":
                prob_mat = batch["interaction_prior"].to(self.device)
                x1 = tr.cat([pair, prob_mat.unsqueeze(1)], dim=1)
            else:
                x1 = pair

        else:
            raise ValueError(f"Unknown pair_construction: {self.pair_construction}")

        # ---- Optional 2D relative-position channels (Tier A) ---------- #
        # Concatenate the learned relative-distance map onto the pair
        # representation. mid_ch was bumped by relpos_channels in build_graph
        # so the refiner's first conv already expects these extra channels.
        if self.relpos is not None:
            rp = self.relpos(L, device=x1.device, batch_size=batch_size)  # [B, C_rp, L, L]
            x1 = tr.cat([x1, rp], dim=1)

        # 2D ResNet
        y = self.resnet2d(x1)
        
        # Output
        y = self.conv2Dout(tr.relu(y)).squeeze(1)
        if batch["canonical_mask"] is not None:
            y = y.multiply(batch["canonical_mask"].to(self.device))
        yt = tr.transpose(y, -1, -2)
        y = (y + yt) / 2

        return y, y0

    def _dist_active(self):
        return self.loss_distance_weight not in (None, 'none', False, '', 0)

    def _distance_loss_weights(self, L, device):
        """
        Per-position multiplier of shape [L*L] as an increasing function of
        |i - j|. Cached per (L, device). Row-major flatten matches
        y.view(B, -1) (index k = i*L + j).
        """
        key = (L, str(device))
        w = self._dist_w_cache.get(key)
        if w is not None:
            return w
        idx = tr.arange(L, device=device)
        d = (idx[:, None] - idx[None, :]).abs().float()          # [L, L]
        ref = max(self.loss_distance_ref, 1.0)
        scale = self.loss_distance_scale
        mode = self.loss_distance_weight
        if mode == 'linear':
            mult = 1.0 + scale * (d / ref)
        elif mode == 'sqrt':
            mult = 1.0 + scale * tr.sqrt(d / ref)
        elif mode == 'log':
            mult = 1.0 + scale * tr.log1p(d) / math.log1p(ref)
        else:
            mult = tr.ones_like(d)
        if self.loss_distance_cap and self.loss_distance_cap > 0:
            mult = tr.clamp(mult, max=float(self.loss_distance_cap))
        w = mult.reshape(-1)                                     # [L*L]
        self._dist_w_cache[key] = w
        return w

    def _weighted_ce(self, logits2, target_flat, dist_w):
        """
        Distance-weighted 2-class cross-entropy.

        logits2     : [B, 2, N] stacked (-yhat, yhat) logits
        target_flat : [B, N] in {0, 1, -1(ignore)}
        dist_w      : [N] distance multiplier

        Reproduces PyTorch's class-weighted-mean normalization
        (sum(w * l) / sum(w)) and multiplies in the distance factor. With
        loss_distance_scale=0 (mult==1) this equals the original
        cross_entropy(..., weight=self.class_weight).
        """
        ce = cross_entropy(logits2, target_flat, ignore_index=-1,
                           reduction='none')                     # [B, N], 0 where ignored
        valid = (target_flat != -1)
        is_pos = (target_flat == 1)
        # per-position class weight
        cw = tr.where(is_pos, self.class_weight[1], self.class_weight[0])
        cw = cw * valid.to(cw.dtype)
        # distance multiplier
        dm = dist_w.unsqueeze(0)                                 # [1, N]
        if self.loss_distance_apply == 'all':
            factor = dm
        else:  # 'positive' -> only up-weight true long-range pairs
            factor = tr.where(is_pos, dm, tr.ones_like(dm))
        weight = cw * factor
        denom = weight.sum()
        if denom <= 0:
            return (ce * weight).sum()
        return (ce * weight).sum() / denom

    def loss_func(self, yhat, y, epoch=0):
        """Compute loss function."""
        L = y.shape[-1]
        y = y.view(y.shape[0], -1)
        yhat, y0 = yhat

        yhat = yhat.view(yhat.shape[0], -1)
        y0 = y0.view(y0.shape[0], -1)

        # L1 loss
        l1_loss = tr.mean(tr.relu(yhat[y != -1]))

        # NOTE: the previous adaptive L2 penalty applied ONLY to quantum
        # parameters (theta / embed_weights / pos_alpha), which differentially
        # regularized the quantum extractor and confounded the quantum-vs-
        # classical comparison. It has been removed; regularization is now the
        # single unified `weight_decay` applied to ALL parameters (including
        # the extractor) via AdamW, identical across configs.

        # Cross entropy loss (2-class stacked logits)
        yhat = yhat.unsqueeze(1)
        yhat = tr.cat((-yhat, yhat), dim=1)

        y0 = y0.unsqueeze(1)
        y0 = tr.cat((-y0, y0), dim=1)

        if not self._dist_active():
            # Original flat weighted cross-entropy (v9-v14 reproduce exactly).
            error_loss1 = cross_entropy(y0, y, ignore_index=-1, weight=self.class_weight)
            error_loss = cross_entropy(yhat, y, ignore_index=-1, weight=self.class_weight)
        else:
            # Distance-weighted cross-entropy (up-weights long-range positives).
            dist_w = self._distance_loss_weights(L, y.device)
            error_loss1 = self._weighted_ce(y0, y, dist_w)
            error_loss = self._weighted_ce(yhat, y, dist_w)

        loss = (
            error_loss
            + self.loss_beta * error_loss1
            + self.loss_l1 * l1_loss
        )
        return loss

    def fit(self, loader):
        """Train for one epoch."""
        self.train()
        metrics = {"loss": 0, "f1": 0, "precision": 0, "recall": 0}

        if self.verbose:
            loader = tqdm(loader)

        for batch in loader: 
            y = batch["contact"].to(self.device)
            batch.pop("contact")
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            y_pred = self(batch)
            
            loss = self.loss_func(y_pred, y, epoch=self.current_epoch)
            
            if isinstance(y_pred, tuple):
                y_pred = y_pred[0]

            # Get metrics
            f1, precision, recall = contact_f1(
                y.cpu(), y_pred.detach().cpu(), batch["length"], 
                method="triangular", return_all_metrics=True
            )

            metrics["loss"] += loss.item()
            metrics["f1"] += f1
            metrics["precision"] += precision
            metrics["recall"] += recall

            loss.backward()
            
            # Global gradient clipping over ALL parameters (identical recipe
            # for the extractor whether it is quantum or classical).
            if self.grad_clip is not None and self.grad_clip > 0:
                tr.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()

            # Update cycle scheduler (steps per batch)
            if self.scheduler_name == "cycle" and self.scheduler is not None:
                self.scheduler.step()

        for k in metrics:
            metrics[k] /= len(loader)

        return metrics

    def test(self, loader):
        """Evaluate on validation/test set."""
        self.eval()
        metrics = {
            "loss": 0, "f1": 0, "f1_post": 0, 
            "precision": 0, "recall": 0, 
            "precision_post": 0, "recall_post": 0
        }

        if self.verbose:
            loader = tqdm(loader)

        with tr.no_grad():
            for batch in loader:  
                y = batch["contact"].to(self.device)
                batch.pop("contact")
                lengths = batch["length"]
                
                y_pred = self(batch)
                loss = self.loss_func(y_pred, y, epoch=self.current_epoch)
                metrics["loss"] += loss.item()

                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

                y_pred_post = postprocessing(y_pred.cpu(), batch["canonical_mask"])

                # Raw metrics
                f1, precision, recall = contact_f1(
                    y.cpu(), y_pred.cpu(), lengths, th=self.output_th, 
                    reduce=True, method="triangular", return_all_metrics=True
                )
                
                # Post-processed metrics
                f1_post, precision_post, recall_post = contact_f1(
                    y.cpu(), y_pred_post.cpu(), lengths, th=self.output_th, 
                    reduce=True, method="triangular", return_all_metrics=True
                )

                metrics["f1"] += f1
                metrics["precision"] += precision
                metrics["recall"] += recall
                metrics["f1_post"] += f1_post
                metrics["precision_post"] += precision_post
                metrics["recall_post"] += recall_post

        for k in metrics:
            metrics[k] /= len(loader)

        # Update plateau scheduler
        if self.scheduler_name == "plateau" and self.scheduler is not None:
            self.scheduler.step(metrics["f1_post"])

        return metrics

    def pred(self, loader, logits=False):
        """Generate predictions."""
        self.eval()

        if self.verbose:
            loader = tqdm(loader)

        predictions, logits_list = [], [] 
        with tr.no_grad():
            for batch in loader: 
                lengths = batch["length"]
                seqid = batch["id"]
                sequences = batch["sequence"]

                y_pred = self(batch)
                
                if isinstance(y_pred, tuple):
                    y_pred = y_pred[0]

                y_pred_post = postprocessing(y_pred.cpu(), batch["canonical_mask"])

                for k in range(y_pred_post.shape[0]):
                    if logits:
                        logits_list.append(
                            (seqid[k],
                             y_pred[k, : lengths[k], : lengths[k]].squeeze().cpu(),
                             y_pred_post[k, : lengths[k], : lengths[k]].squeeze()
                            ))
                    predictions.append(
                        (seqid[k],
                        sequences[k],
                            mat2bp(
                                y_pred_post[k, : lengths[k], : lengths[k]].squeeze()
                            )                         
                        )
                    )
        predictions = pd.DataFrame(predictions, columns=["id", "sequence", "base_pairs"])

        return predictions, logits_list


class ResidualLayer1D(nn.Module):
    """1D Residual Layer with bottleneck architecture."""
    
    def __init__(
        self,
        dilation,
        resnet_bottleneck_factor,
        filters,
        kernel_size,
    ):
        super().__init__()

        num_bottleneck_units = math.floor(resnet_bottleneck_factor * filters)

        self.layer = nn.Sequential(
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Conv1d(
                filters,
                num_bottleneck_units,
                kernel_size,
                dilation=dilation,
                padding="same",
            ),
            nn.BatchNorm1d(num_bottleneck_units),
            nn.ReLU(),
            nn.Conv1d(num_bottleneck_units, filters, kernel_size=1, padding="same"),
        )

    def forward(self, x):
        return x + self.layer(x)


class ResidualBlock2D(nn.Module):
    """2D Residual Block with bottleneck architecture."""
    
    def __init__(self, filters, filters1, kernel_size, dilation):
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters1, kernel_size, padding="same"),
            nn.BatchNorm2d(filters1),
            nn.ReLU(),
            nn.Conv2d(
                filters1, filters, kernel_size, dilation=dilation, padding="same"
            ),
        )

    def forward(self, x):
        return self.layer(x) + x