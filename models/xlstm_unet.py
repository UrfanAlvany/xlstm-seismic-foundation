# xlstm_unet.py
# Official-style xLSTM U‑Net, patched to use TFLA Triton mLSTM kernels.
# Input to backbone:  [B, L, d_model]
# Output from backbone: ( [B, L, d_model], None )
# Safe defaults, BF16 autocast for trunk, FP32 reductions where it matters.

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---- Official xLSTM API (PyTorch) ----
try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
        FeedForwardConfig,
    )
except Exception as e:
    raise ImportError("Please install the official library: pip install xlstm") from e

# ---- TFLA kernels backend ----
_HAS_TFLA = False
try:
    from mlstm_kernels.torch.backend_module import mLSTMBackend, mLSTMBackendConfig
    _HAS_TFLA = True
except Exception:
    _HAS_TFLA = False

XLSTM_KERNEL_CALLS = {"count": 0}


# ------------------------------
# Safe per-sample standardization
# ------------------------------
class SafeStdNorm(nn.Module):
    """Per-sample, per-channel standardization with epsilon; works for [B,L,C] or [B,C,L]."""
    def __init__(self, eps: float = 1e-6, channels_last: bool = True):
        super().__init__()
        self.eps = eps
        self.channels_last = channels_last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"SafeStdNorm expects 3D tensors, got {x.shape}")
        if self.channels_last:  # [B, L, C]
            m = x.mean(1, keepdim=True)
            v = (x - m).pow(2).mean(1, keepdim=True)
        else:                   # [B, C, L]
            m = x.mean(-1, keepdim=True)
            v = (x - m).pow(2).mean(-1, keepdim=True)
        return (x - m) / (v.add(self.eps).sqrt())


# ------------------------------
# Config
# ------------------------------
@dataclass
class UNetConfig:
    # Core
    d_model: int = 32
    n_layers: int = 3
    pool: List[int] = None         # e.g., [4, 4]
    expand: int = 2
    dropout: float = 0.0
    max_seq_len: int = 4096

    # Toggle: U‑Net vs pure sequential mLSTM
    use_unet: bool = True          # False => single StageBlock at full resolution

    # xLSTM selection
    block_type: str = "mlstm"      # "mlstm" | "slstm" | "mixed"
    # For block_type=="mixed": per‑block schedule of types, e.g. ["mlstm","slstm",...]
    block_mix: Optional[List[str]] = None
    bidirectional: bool = True
    gradient_checkpointing: bool = False
    fuse_per_block: bool = True    # True = fuse after each block, False = once per stage

    # mLSTM params
    mlstm_num_heads: int = 2
    mlstm_conv1d_kernel_size: int = 4
    mlstm_qkv_proj_blocksize: int = 2
    mlstm_proj_factor: float = 1.6
    mlstm_backend: str = "chunkwise"  # "parallel" | "chunkwise"

    # sLSTM params
    slstm_num_heads: int = 4
    slstm_conv1d_kernel_size: int = 4
    slstm_bias_init: str = "powerlaw_blockdependent"

    # FeedForward (if used by stack)
    ff_proj_factor: float = 1.3
    ff_act_fn: str = "gelu"

    # TFLA kernels
    enable_tflakernels: bool = True
    chunk_size: int = 128
    chunkwise_kernel: str = "chunkwise--triton_xl_chunk"
    sequence_kernel: str = "native_sequence__triton"
    step_kernel: str = "triton"
    autocast_kernel_dtype: str = "bfloat16"  # "bfloat16" | "float16" | "float32"

    # Gate bias overrides (optional stability lever)
    override_gate_bias: bool = False
    igate_bias_init: float = -10.0
    fgate_bias_start: float = 3.0
    fgate_bias_end: float = 6.0

    # Stability
    gate_soft_cap: float = 15.0
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = True

    # Optional end head (not used in your pipeline)
    head_kernel_size: int = 5
    head_dropout: float = 0.10
    out_channels: int = 3  # P/S/noise

    def __post_init__(self):
        if self.pool is None:
            self.pool = [4, 4]


# ------------------------------
# Stage block with TFLA patching
# ------------------------------
class StageBlock(nn.Module):
    def __init__(self, d_model: int, cfg: UNetConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        self.drop = nn.Dropout(cfg.dropout)
        self.use_ckpt = cfg.gradient_checkpointing
        self.bidirectional = cfg.bidirectional
        self.fuse_per_block = cfg.fuse_per_block

        # Helpers to build one‑block stacks of requested type
        def _make_block_cfg(block_type: str):
            if block_type == "mlstm":
                layer_cfg = mLSTMLayerConfig(
                    embedding_dim=d_model,
                    num_heads=cfg.mlstm_num_heads,
                    conv1d_kernel_size=cfg.mlstm_conv1d_kernel_size,
                    qkv_proj_blocksize=cfg.mlstm_qkv_proj_blocksize,
                    proj_factor=cfg.mlstm_proj_factor,
                    dropout=cfg.dropout,
                    context_length=cfg.max_seq_len,
                )
                return (mLSTMBlockConfig(mlstm=layer_cfg), None)
            elif block_type == "slstm":
                layer_cfg = sLSTMLayerConfig(
                    embedding_dim=d_model,
                    num_heads=cfg.slstm_num_heads,
                    conv1d_kernel_size=cfg.slstm_conv1d_kernel_size,
                    dropout=cfg.dropout,
                    bias_init=cfg.slstm_bias_init,
                )
                ff_cfg = FeedForwardConfig(
                    proj_factor=cfg.ff_proj_factor,
                    act_fn=cfg.ff_act_fn,
                    dropout=cfg.dropout,
                    embedding_dim=d_model,
                )
                return (None, sLSTMBlockConfig(slstm=layer_cfg, feedforward=ff_cfg))
            else:
                raise ValueError(f"Invalid block_type in mix: {block_type}")

        def _stack_one(block_type: str):
            mlstm_cfg, slstm_cfg = _make_block_cfg(block_type)
            return xLSTMBlockStack(
                xLSTMBlockStackConfig(
                    mlstm_block=mlstm_cfg,
                    slstm_block=slstm_cfg,
                    context_length=cfg.max_seq_len,
                    num_blocks=1,
                    embedding_dim=d_model,
                    add_post_blocks_norm=True,
                    bias=False,
                    dropout=cfg.dropout,
                )
            )

        def _stack(num_blocks: int, block_type: str):
            # Build a multi‑block stack of the same type
            mlstm_cfg, slstm_cfg = _make_block_cfg(block_type)
            return xLSTMBlockStack(
                xLSTMBlockStackConfig(
                    mlstm_block=mlstm_cfg,
                    slstm_block=slstm_cfg,
                    context_length=cfg.max_seq_len,
                    num_blocks=num_blocks,
                    embedding_dim=d_model,
                    add_post_blocks_norm=True,
                    bias=False,
                    dropout=cfg.dropout,
                )
            )

        # Determine per‑block schedule for block_type=="mixed"
        if cfg.block_type == "mixed":
            mix = cfg.block_mix or ["mlstm", "slstm"]
            # Repeat or trim to exactly n_layers
            if len(mix) < cfg.n_layers:
                reps = (cfg.n_layers + len(mix) - 1) // len(mix)
                mix = (mix * reps)[: cfg.n_layers]
            else:
                mix = mix[: cfg.n_layers]
        else:
            mix = [cfg.block_type] * cfg.n_layers

        if not self.fuse_per_block:
            # Build single stacks containing all blocks (same type only)
            if cfg.block_type == "mixed":
                raise ValueError("fuse_per_block=False is incompatible with block_type='mixed' — set fuse_per_block=True")
            if self.bidirectional:
                self.fwd = _stack(cfg.n_layers, cfg.block_type)
                self.bwd = _stack(cfg.n_layers, cfg.block_type)
                self.fuse = nn.Linear(2 * d_model, d_model, bias=True)
            else:
                self.stack = _stack(cfg.n_layers, cfg.block_type)
        else:
            # Per‑block ModuleList; allows mixing per layer
            if self.bidirectional:
                self.blocks_fwd = nn.ModuleList([_stack_one(bt) for bt in mix])
                self.blocks_bwd = nn.ModuleList([_stack_one(bt) for bt in mix])
                self.fuses = nn.ModuleList([nn.Linear(2 * d_model, d_model, bias=True) for _ in mix])
            else:
                self.blocks = nn.ModuleList([_stack_one(bt) for bt in mix])

        # Patch to TFLA kernels if available/requested
        self._tflabackend = None
        self._tflabackend_cfg = None
        self._maybe_patch_mlstm_backend_for_tflakernels()

        # Fallback official chunkwise_simple (no TFLA)
        if (cfg.block_type in ("mlstm", "mixed")) and (cfg.mlstm_backend == "chunkwise") and not _HAS_TFLA:
            try:
                from xlstm.blocks.mlstm.backends import chunkwise_simple
                from xlstm.blocks.mlstm.cell import mLSTMCell
                def _chunk_call(**kw):
                    return chunkwise_simple(chunk_size=cfg.chunk_size, return_last_state=False, **kw)
                for m in self.modules():
                    if isinstance(m, mLSTMCell):
                        m.backend_fn = _chunk_call
            except Exception:
                pass

        # Optionally override gate biases across all mLSTM cells in this stage
        self._maybe_override_gate_bias()

    def _maybe_patch_mlstm_backend_for_tflakernels(self):
        if not (_HAS_TFLA and self.cfg.enable_tflakernels and self.cfg.block_type in ("mlstm", "mixed")):
            return

        # One backend instance per stage
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            self.cfg.autocast_kernel_dtype
        ]
        backend_config = mLSTMBackendConfig(
            chunkwise_kernel=self.cfg.chunkwise_kernel,
            sequence_kernel=self.cfg.sequence_kernel,
            step_kernel=self.cfg.step_kernel,
            chunk_size=self.cfg.chunk_size,
            return_last_states=False,
        )
        backend = mLSTMBackend(backend_config)
        self._tflabackend = backend
        self._tflabackend_cfg = backend_config

        def _soft_cap(x: Optional[torch.Tensor], cap: float) -> Optional[torch.Tensor]:
            if x is None:
                return x
            return cap * torch.tanh(x / cap)

        def _tflabackend_wrapper(**kw):
            # Map xLSTM cell args → backend and enforce contiguity
            q = kw.get("queries"); k = kw.get("keys"); v = kw.get("values")
            i = kw.get("igate_preact"); f = kw.get("fgate_preact")

            if i is not None and i.dim() == 4 and i.size(-1) == 1: i = i.squeeze(-1)
            if f is not None and f.dim() == 4 and f.size(-1) == 1: f = f.squeeze(-1)

            cap = float(self.cfg.gate_soft_cap)
            i = _soft_cap(i, cap) if i is not None else None
            f = _soft_cap(f, cap) if f is not None else None

            # IMPORTANT: TFLA kernels expect contiguous tiles for best stability
            if q is not None: q = q.contiguous()
            if k is not None: k = k.contiguous()
            if v is not None: v = v.contiguous()
            if i is not None: i = i.contiguous()
            if f is not None: f = f.contiguous()

            with torch.autocast(device_type="cuda", dtype=dtype):
                out = backend(q=q, k=k, v=v, i=i, f=f)
            # Ensure pure tensor output for downstream layers
            if isinstance(out, (tuple, list)):
                out = out[0]
            XLSTM_KERNEL_CALLS["count"] += 1
            return out

        # Patch every mLSTM cell in this stage
        from xlstm.blocks.mlstm.cell import mLSTMCell
        for m in self.modules():
            if isinstance(m, mLSTMCell):
                m.backend_fn = _tflabackend_wrapper

    def _maybe_override_gate_bias(self):
        if not (self.cfg.block_type == "mlstm" and bool(getattr(self.cfg, "override_gate_bias", False))):
            return
        try:
            from xlstm.blocks.mlstm.cell import mLSTMCell
        except Exception:
            return
        ig_init = float(getattr(self.cfg, "igate_bias_init", -10.0))
        fg_start = float(getattr(self.cfg, "fgate_bias_start", 3.0))
        fg_end = float(getattr(self.cfg, "fgate_bias_end", 6.0))
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, mLSTMCell):
                    # Input gate bias: constant negative value (e.g., -10)
                    if hasattr(m, "igate") and hasattr(m.igate, "bias") and m.igate.bias is not None:
                        m.igate.bias.fill_(ig_init)
                    # Forget gate bias: linspace over heads in [start, end]
                    if hasattr(m, "fgate") and hasattr(m.fgate, "bias") and m.fgate.bias is not None:
                        nh = m.fgate.bias.shape[0]
                        vals = torch.linspace(fg_start, fg_end, steps=nh, device=m.fgate.bias.device, dtype=m.fgate.bias.dtype)
                        m.fgate.bias.copy_(vals)

    def _run(self, fn, x: torch.Tensor) -> torch.Tensor:
        # Normalize inputs to tensors
        if isinstance(x, (tuple, list)):
            x = x[0]
        if self.use_ckpt and self.training:
            out = checkpoint(lambda t: fn(t), x, use_reentrant=False)
        else:
            out = fn(x)
        # Some upstream modules may return (y, state); we only need activations
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out

    def forward(self, x: torch.Tensor, state: Optional[Any] = None) -> Tuple[torch.Tensor, None]:
        # Be tolerant to callers that accidentally pass (x, state)
        if isinstance(x, (tuple, list)):
            x = x[0]
        # Optional runtime check (enable by exporting XLSTM_CHECK_FINITE=1)
        if torch.jit.is_scripting():
            check_finite = False
        else:
            # Guard against calling current_device() when CUDA is unavailable
            check_finite = bool(int(torch.cuda.is_available() and int(os.environ.get("XLSTM_CHECK_FINITE", "0"))))

        if not self.fuse_per_block:
            if hasattr(self, "stack"):
                y = self._run(self.stack, x)
            else:
                y_f = self._run(self.fwd, x)
                y_b = self._run(self.bwd, torch.flip(x, dims=[1])); y_b = torch.flip(y_b, dims=[1])
                y = self.fuse(torch.cat([y_f, y_b], dim=-1))
            y = self.drop(y)
            if check_finite:
                assert torch.isfinite(y).all(), "NaN/Inf detected after stage (no per-block fuse)."
            return y, None

        if not self.bidirectional:
            for blk in self.blocks:
                x = self._run(blk, x)
                x = self.drop(x)
                if check_finite:
                    assert torch.isfinite(x).all(), "NaN/Inf detected after block."
            return x, None

        # Per‑block fusion
        for fwd_blk, bwd_blk, fuse in zip(self.blocks_fwd, self.blocks_bwd, self.fuses):
            y_f = self._run(fwd_blk, x)
            y_b = self._run(bwd_blk, torch.flip(x, dims=[1])); y_b = torch.flip(y_b, dims=[1])
            x = self.drop(fuse(torch.cat([y_f, y_b], dim=-1)))
            if check_finite:
                assert torch.isfinite(x).all(), "NaN/Inf detected after bi‑dir fuse."
        return x, None


# ------------------------------
# U‑Net pooling helpers
# ------------------------------
class DownPool(nn.Module):
    """Fold `pool` steps and project to D*expand: (B,L,D) -> (B,L//pool,D*expand)"""
    def __init__(self, d_in: int, expand: int, pool: int):
        super().__init__()
        self.pool = pool
        self.proj = nn.Linear(d_in * pool, d_in * expand)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        B, L, D = x.shape
        if self.pool > 1:
            pad = (self.pool - (L % self.pool)) % self.pool
            if pad > 0:
                x = F.pad(x, (0, 0, 0, pad))
                L = x.shape[1]
            x = x.view(B, L // self.pool, D * self.pool).contiguous()
        return self.proj(x), None


class UpPool(nn.Module):
    """Project then unfold by `pool`, add skip: (B,L,D_in) -> (B,L*pool,D_in/expand)"""
    def __init__(self, d_in: int, expand: int, pool: int):
        super().__init__()
        self.pool = pool
        self.d_out = d_in // expand
        self.proj = nn.Linear(d_in, self.d_out * pool)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        B, L, D = x.shape
        x = self.proj(x).view(B, L * self.pool, self.d_out).contiguous()
        if skip is not None:
            if x.shape[1] > skip.shape[1]:
                x = x[:, : skip.shape[1], :]
            elif x.shape[1] < skip.shape[1]:
                x = F.pad(x, (0, 0, 0, skip.shape[1] - x.shape[1]))
            x = x + skip
        return x, None


# ------------------------------
# Backbone (U‑Net over xLSTM stacks) with toggle for pure sequential
# ------------------------------
class xLSTMUNetBackbone(nn.Module):
    def __init__(self, cfg: Optional[UNetConfig] = None, **kwargs):
        super().__init__()
        # Handle UNetConfig / Hydra kwargs
        if cfg is None:
            self.cfg = UNetConfig(**kwargs)
        elif isinstance(cfg, dict):
            self.cfg = UNetConfig(**cfg)
        elif isinstance(cfg, UNetConfig):
            self.cfg = cfg
        else:
            self.cfg = UNetConfig(**vars(cfg))

        self.d_model = self.cfg.d_model
        H = self.cfg.d_model

        self.is_unet = bool(getattr(self.cfg, "use_unet", True))

        if self.is_unet:
            # ----- U‑Net path -----
            enc_stages, downs = [], []
            for p in self.cfg.pool:
                enc_stages.append(StageBlock(H, self.cfg))
                downs.append(DownPool(H, self.cfg.expand, p))
                H *= self.cfg.expand

            center = [StageBlock(H, self.cfg)]

            up_stages, ups = [], []
            for p in reversed(self.cfg.pool):
                H //= self.cfg.expand
                ups.append(UpPool(H * self.cfg.expand, self.cfg.expand, p))
                up_stages.append(StageBlock(H, self.cfg))

            self.enc_stages = nn.ModuleList(enc_stages)
            self.downs = nn.ModuleList(downs)
            self.center = nn.ModuleList(center)
            self.up_stages = nn.ModuleList(up_stages)
            self.ups = nn.ModuleList(ups)
        else:
            # ----- Pure sequential path -----
            self.seq = StageBlock(H, self.cfg)  # depth = cfg.n_layers at original resolution

        # Final norm on backbone output (both modes)
        self.out_norm = nn.LayerNorm(self.cfg.d_model, eps=self.cfg.norm_eps)

        # Force FP32 reductions in xLSTM LayerNorm if present
        if self.cfg.norm_reduction_force_float32:
            self._force_fp32_reductions(eps=self.cfg.norm_eps)

        # Init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _force_fp32_reductions(self, eps: float):
        try:
            from xlstm.components.ln import LayerNorm as XLSTM_LN
        except Exception:
            XLSTM_LN = None
        if XLSTM_LN is None:
            return

        for m in self.modules():
            # Patch only the base LayerNorm, not MultiHeadLayerNorm
            if type(m) is XLSTM_LN:
                orig = m.forward  # capture per-instance

                def _fp32_forward(x, *args, _orig=orig, **kwargs):
                    # Always operate on a tensor; if tuple/list provided, take first tensor.
                    if isinstance(x, (tuple, list)):
                        x_tensor = x[0] if len(x) > 0 else x
                    else:
                        x_tensor = x
                    if not torch.is_tensor(x_tensor):
                        # Fallback to original if no tensor is found
                        return _orig(x, *args, **kwargs)
                    y = _orig(x_tensor.to(torch.float32), *args, **kwargs)
                    return y.to(x_tensor.dtype)

                m.forward = _fp32_forward

    def forward(self, x: torch.Tensor, state: Optional[Any] = None) -> Tuple[torch.Tensor, None]:
        # Convert input to float32; let autocast handle mixed precision
        x = x.float()

        # Global autocast around backbone computation to eliminate dtype conversion churn
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if not self.is_unet:
                x, _ = self.seq(x, state)
                x = self.out_norm(x)
                return x, None

            # U‑Net path
            skips: List[torch.Tensor] = []
            for stage, down in zip(self.enc_stages, self.downs):
                x, _ = stage(x, state)
                skips.append(x)
                x, _ = down(x)

            for block in self.center:
                x, _ = block(x, state)

            for stage, up in zip(self.up_stages, self.ups):
                skip = skips.pop()
                x, _ = up(x, skip)
                x, _ = stage(x, state)

            x = self.out_norm(x)
            return x, None


# ------------------------------
# (Optional) head; not used in your pipeline
# ------------------------------
class PhasePickHead(nn.Module):
    def __init__(self, d_model: int, out_channels: int, k: int = 5, pdrop: float = 0.10):
        super().__init__()
        pad = (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(pdrop),
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(pdrop),
        )
        self.linear = nn.Linear(d_model, out_channels, bias=True)
        self.conv = nn.Conv1d(out_channels, out_channels, kernel_size=k, padding=pad, bias=False)

    def forward(self, x_blh: torch.Tensor) -> torch.Tensor:
        x = x_blh.transpose(1, 2)
        x = self.net(x)
        x = x.transpose(1, 2)
        x = self.linear(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x


class PhasePickUNet(nn.Module):
    def __init__(self, cfg: Optional[UNetConfig] = None, **kwargs):
        super().__init__()
        if cfg is None:
            self.cfg = UNetConfig(**kwargs)
        elif isinstance(cfg, dict):
            self.cfg = UNetConfig(**cfg)
        elif isinstance(cfg, UNetConfig):
            self.cfg = cfg
        else:
            self.cfg = UNetConfig(**vars(cfg))

        self.input_norm = SafeStdNorm(eps=1e-6, channels_last=True)
        self.encoder = nn.Linear(3, self.cfg.d_model, bias=True)
        self.backbone = xLSTMUNetBackbone(self.cfg)
        self.head = PhasePickHead(self.cfg.d_model, self.cfg.out_channels,
                                  k=self.cfg.head_kernel_size, pdrop=self.cfg.head_dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x_blc: torch.Tensor) -> torch.Tensor:
        if x_blc.dim() != 3:
            raise ValueError(f"Expected [B, L, C], got {x_blc.shape}")
        x = x_blc if not (x_blc.size(1) == 3 and x_blc.size(-1) != 3) else x_blc.transpose(1, 2)

        with torch.autocast(device_type="cuda", enabled=False):
            x = self.input_norm(x.float())
            x = self.encoder(x)
        with torch.autocast(device_type="cuda", dtype=getattr(torch, self.cfg.autocast_kernel_dtype)):
            x, _ = self.backbone(x)
            logits = self.head(x)
        return logits

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PhasePickUNet":
        return PhasePickUNet(UNetConfig(**d))


# ------------------------------
# Utilities (kernel check)
# ------------------------------
def print_kernel_summary(model: nn.Module):
    """Summarize if/where TFLA kernels are active."""
    try:
        from xlstm.blocks.mlstm.cell import mLSTMCell
    except Exception:
        mLSTMCell = None

    print("\n" + "=" * 60)
    print("TFLA KERNEL VERIFICATION")
    print("=" * 60)

    n_cells = 0
    patched = 0
    stage_idx = -1
    for mod in model.modules():
        if isinstance(mod, StageBlock):
            stage_idx += 1
            cfg = getattr(mod, "_tflabackend_cfg", None)
            print(
                f"[Stage {stage_idx}]",
                "TFLA ACTIVE" if cfg is not None else "<no TFLA>",
                f"| chunkwise={getattr(cfg,'chunkwise_kernel',None)}",
                f"| seq={getattr(cfg,'sequence_kernel',None)}",
                f"| step={getattr(cfg,'step_kernel',None)}",
                f"| chunk_size={getattr(cfg,'chunk_size',None)}",
            )
        if mLSTMCell is not None and isinstance(mod, mLSTMCell):
            n_cells += 1
            fn = getattr(mod, "backend_fn", None)
            name = getattr(fn, "__name__", str(fn)) if hasattr(fn, "__name__") else str(fn)
            if "tflabackend" in name:
                patched += 1
    print(f"Summary: cells={n_cells}, patched={patched}, calls={XLSTM_KERNEL_CALLS['count']}")
    print("=" * 60 + "\n")


# ------------------------------
# Defaults / quick smoke test
# ------------------------------
CONFIG_DEFAULTS = dict(
    d_model=32, n_layers=3, expand=2, pool=[4, 4], dropout=0.0, max_seq_len=4096,
    use_unet=True,
    block_type="mlstm", bidirectional=True, gradient_checkpointing=False, fuse_per_block=True,
    mlstm_num_heads=2, mlstm_conv1d_kernel_size=4, mlstm_qkv_proj_blocksize=2, mlstm_proj_factor=1.6,
    mlstm_backend="parallel", ff_proj_factor=1.3, ff_act_fn="gelu",
    enable_tflakernels=True, chunk_size=256,
    chunkwise_kernel="chunkwise--triton_xl_chunk",
    sequence_kernel="native_sequence__triton",
    step_kernel="triton",
    autocast_kernel_dtype="bfloat16",
    gate_soft_cap=15.0, norm_eps=1e-6, norm_reduction_force_float32=True,
    head_kernel_size=5, head_dropout=0.10, out_channels=3,
)

# Backwards-compatibility alias (older code/tests import xLSTMUNet)
xLSTMUNet = xLSTMUNetBackbone

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = UNetConfig(**CONFIG_DEFAULTS)
    model = xLSTMUNetBackbone(cfg).to("cuda")
    x = torch.randn(2, 4096, cfg.d_model, device="cuda")
    with torch.no_grad():
        y, _ = model(x)
    print("Backbone output:", tuple(y.shape))
    print_kernel_summary(model) 
