import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from models.xlstm_unet import xLSTMUNetBackbone, UNetConfig
from models.quantizer import Wav2Vec2GumbelVectorQuantizer


class ContrastiveModel(nn.Module):
    def __init__(self,
                 model_backbone: str = 'xlstm_unet',
                 d_model: int = 512,
                 n_layers: int = 12,
                 d_state: int = 64,
                 d_conv: int = 7,
                 expand: int = 2,
                 headdim: int = 64,
                 use_mem_eff_path: bool = False,
                 pool: list = [],
                 complex_downward: bool = False,
                 final_projection_dim: int = 0,
                 num_negatives: int = 10,
                 temperature: float = 0.1,
                 pretraining: bool = True,
                 quantize: bool = False,
                 quantizer_args: Optional[dict] = None,
                 only_masked: bool = False,
                 **kwargs
                 ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.use_mem_eff_path = use_mem_eff_path
        self.final_projection_dim = final_projection_dim
        self.num_negatives = num_negatives
        self.temperature = temperature
        self.only_masked = only_masked
        
        self.pretraining = pretraining
        self.quantize = quantize

        if model_backbone == 'xlstm_unet':
            # Map wrapper args to UNetConfig
            unet_cfg = UNetConfig(
                d_model=d_model,
                n_layers=n_layers,
                pool=pool if pool else [4, 4],
                expand=expand,
                dropout=kwargs.get('dropout', 0.10),
                max_seq_len=kwargs.get('max_seq_len', 4096),
                use_unet=True,
                block_type='mlstm',
                bidirectional=kwargs.get('bidirectional', True),
                gradient_checkpointing=kwargs.get('gradient_checkpointing', False),
                fuse_per_block=True,
                mlstm_num_heads=kwargs.get('mlstm_num_heads', 4),
                mlstm_conv1d_kernel_size=kwargs.get('mlstm_conv1d_kernel_size', 4),
                mlstm_qkv_proj_blocksize=kwargs.get('mlstm_qkv_proj_blocksize', 4),
                mlstm_proj_factor=kwargs.get('mlstm_proj_factor', 1.6),
                ff_proj_factor=kwargs.get('ff_proj_factor', 1.6),
                ff_act_fn=kwargs.get('ff_act_fn', 'gelu'),
                enable_tflakernels=kwargs.get('enable_tflakernels', True),
                chunk_size=kwargs.get('chunk_size', 256),
                chunkwise_kernel=kwargs.get('chunkwise_kernel', 'chunkwise--triton_xl_chunk'),
                sequence_kernel=kwargs.get('sequence_kernel', 'native_sequence__triton'),
                step_kernel=kwargs.get('step_kernel', 'triton'),
                autocast_kernel_dtype=kwargs.get('autocast_kernel_dtype', 'bfloat16'),
            )
            self._net = xLSTMUNetBackbone(unet_cfg)
            print(f"Using xLSTMUNetBackbone with cfg={unet_cfg}")

        elif model_backbone == 'test':
            self._net = nn.Identity()
        else:
            print(f"Model backbone {model_backbone} not implemented for ContrastiveModel.")

        if final_projection_dim > 0:
            self.final_projection = nn.Linear(d_model, final_projection_dim)
            self.conv_feature_projection = nn.Linear(final_projection_dim, final_projection_dim)
        else:
            self.final_projection = nn.Linear(d_model, d_model)
            self.conv_feature_projection = nn.Linear(d_model, d_model)
        
        if self.quantize and quantizer_args is not None:
            self.quantizer = Wav2Vec2GumbelVectorQuantizer(
                num_codevector_groups=quantizer_args.get('num_codevector_groups', 2),
                num_codevectors_per_group=quantizer_args.get('num_codevectors_per_group', 320),
                conv_dim=quantizer_args.get('conv_dim', [d_model]),
                codevector_dim=quantizer_args.get('codevector_dim', d_model),
                scale_logits_in_quantization=quantizer_args.get('scale_logits_in_quantization', True)
            )
        else:
            self.quantizer = None

        
    def forward(self, x, state=None):
        if self.pretraining:
            # Accept encoders that return 3+ items; use the first three as (masked, target, mask)
            if isinstance(x, (tuple, list)):
                if len(x) >= 3:
                    x_masked, x_target, mask_indices = x[:3]
                else:
                    raise ValueError(f"ContrastiveModel expected 3 inputs, got {len(x)}")
            else:
                raise ValueError("ContrastiveModel expected a tuple (x_masked, x_target, mask_indices)")

            x_out, _ = self._net(x_masked)
            if self.only_masked:
                x_out = x_out[mask_indices].view(x_out.size(0), -1, x_out.size(-1))
            x_out = self.final_projection(x_out)

            if self.quantize:
                x_target, codevector_perplexity = self.quantizer(x_target, mask_time_indices=mask_indices)
                num_codevectors = self.quantizer.num_groups * self.quantizer.num_vars
                diversity_loss = (
                    (num_codevectors - codevector_perplexity) / num_codevectors
                ) #* mask_indices.sum()

            if self.only_masked:
               x_target = x_target[mask_indices].view(x_target.size(0), -1, x_target.size(-1))
               mask_indices = np.ones((x_target.size(0), x_target.size(1))).astype(bool)

            x_target = self.conv_feature_projection(x_target)

            if self.quantize:
                return (x_out, x_target, mask_indices, self.num_negatives, self.temperature, diversity_loss), None
            else:
                return (x_out, x_target, mask_indices, self.num_negatives, self.temperature), None
        else:
            x_out, _ = self._net(x)
            x_out = self.final_projection(x_out)
            return x_out, None

        
