# File: seismic_data_modeling/models/xlstm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    FeedForwardConfig,
)

class OfficialXLSTMModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float = 0.1,
        
        # Architecture configuration
        block_type: str = "slstm",           # "slstm", "mlstm", "mixed"
        bidirectional: bool = False,         # Enable bidirectional processing
        
        # sLSTM specific configurations
        slstm_num_heads: int = 4,
        slstm_conv1d_kernel_size: int = 4,
        slstm_bias_init: str = "powerlaw_blockdependent",
        slstm_backend: str = "cuda",
        
        # mLSTM specific configurations  
        mlstm_num_heads: int = 4,
        
        # FeedForward specific configurations
        ff_proj_factor: float = 1.3,
        ff_act_fn: str = "gelu",
        
        # Memory optimization
        gradient_checkpointing: bool = False,
        
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.block_type = block_type
        self.bidirectional = bidirectional
        self.gradient_checkpointing = gradient_checkpointing

        # Configure FeedForward Layer
        feedforward_cfg = FeedForwardConfig(
            embedding_dim=self.d_model,
            proj_factor=ff_proj_factor,
            act_fn=ff_act_fn,
        )

        # Configure block types based on architecture
        slstm_block_def = None
        mlstm_block_def = None
        slstm_at = None

        if block_type in ["slstm", "mixed"]:
            slstm_layer_cfg = sLSTMLayerConfig(
                embedding_dim=self.d_model,
                backend=slstm_backend,
                num_heads=slstm_num_heads,
                conv1d_kernel_size=slstm_conv1d_kernel_size,
                bias_init=slstm_bias_init,
            )
            slstm_block_def = sLSTMBlockConfig(
                slstm=slstm_layer_cfg,
                feedforward=feedforward_cfg,
            )

        if block_type in ["mlstm", "mixed"]:
            mlstm_layer_cfg = mLSTMLayerConfig(
                embedding_dim=self.d_model,
                num_heads=mlstm_num_heads,
            )
            mlstm_block_def = mLSTMBlockConfig(
                mlstm=mlstm_layer_cfg,
            )

        # Set block placement
        if block_type == "slstm":
            slstm_at = "all"
        elif block_type == "mlstm":
            slstm_at = []  # Empty list for mLSTM-only (no sLSTM blocks)
        elif block_type == "mixed":
            slstm_at = [i for i in range(0, n_layers, 2)]  # Alternate: sLSTM at even indices

        # Configure main xLSTMBlockStack
        stack_cfg = xLSTMBlockStackConfig(
            slstm_block=slstm_block_def,
            mlstm_block=mlstm_block_def,
            slstm_at=slstm_at,
            context_length=self.max_seq_len,
            num_blocks=self.n_layers,
            embedding_dim=self.d_model,
            add_post_blocks_norm=kwargs.get('add_post_blocks_norm', True),
            bias=kwargs.get('stack_bias', False),
            dropout=kwargs.get('stack_internal_dropout', 0.0),
        )

        if bidirectional:
            # Create two stacks for bidirectional processing
            self.forward_stack = xLSTMBlockStack(stack_cfg)
            self.backward_stack = xLSTMBlockStack(stack_cfg)
            self.combine_layer = nn.Linear(2 * d_model, d_model)
        else:
            self.xlstm_stack = xLSTMBlockStack(stack_cfg)
            
        self.dropout_layer = nn.Dropout(dropout)
        self.config = stack_cfg 

        self._validate_parameter_count()

    
    def _validate_parameter_count(self):
        """Validate and report actual parameter counts"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n🔍 PARAMETER VALIDATION")
        print(f"Total parameters: {total_params:,}")
        
        if self.bidirectional:
            forward_params = sum(p.numel() for p in self.forward_stack.parameters())
            backward_params = sum(p.numel() for p in self.backward_stack.parameters()) 
            combine_params = sum(p.numel() for p in self.combine_layer.parameters())
            print(f"Forward stack: {forward_params:,}")
            print(f"Backward stack: {backward_params:,}")
            print(f"Combine layer: {combine_params:,}")
            print(f"Sum check: {forward_params + backward_params + combine_params:,}")
            
            # Expected range validation
            expected_min = 540000
            expected_max = 570000
            if expected_min <= total_params <= expected_max:
                print(f"✅ Parameter count VALID for bidirectional sLSTM")
            else:
                print(f"⚠️  Parameter count UNEXPECTED! Expected: {expected_min:,}-{expected_max:,}")
        else:
            expected_min = 270000  
            expected_max = 285000
            if expected_min <= total_params <= expected_max:
                print(f"✅ Parameter count VALID for unidirectional sLSTM")
            else:
                print(f"⚠️  Parameter count UNEXPECTED! Expected: {expected_min:,}-{expected_max:,}")
        
        print(f"🔍 END VALIDATION\n")
        return total_params

    def forward(self, x, state=None):
        if self.bidirectional:
            if self.gradient_checkpointing and self.training:
                # Forward pass with gradient checkpointing
                forward_out = checkpoint(self.forward_stack, x, use_reentrant=False)
                # Backward pass (flip sequence) with gradient checkpointing
                # Clone input to avoid CUDAGraph tensor sharing conflicts
                x_flipped = torch.flip(x.clone(), [1])
                backward_out = checkpoint(self.backward_stack, x_flipped, use_reentrant=False)
                backward_out = torch.flip(backward_out, [1])  # Flip back
            else:
                # Normal forward pass
                forward_out = self.forward_stack(x)
                # Backward pass (flip sequence)
                # Clone input to avoid CUDAGraph tensor sharing conflicts
                x_flipped = torch.flip(x.clone(), [1])
                backward_out = self.backward_stack(x_flipped)
                backward_out = torch.flip(backward_out, [1])  # Flip back
            
            # Combine
            combined = torch.cat([forward_out, backward_out], dim=-1)
            out = self.combine_layer(combined)
        else:
            if self.gradient_checkpointing and self.training:
                out = checkpoint(self.xlstm_stack, x, use_reentrant=False)
            else:
                out = self.xlstm_stack(x)
        
        out = self.dropout_layer(out)
        return out, None

    def step(self, x_step, state):
        """
        Step function for autoregressive inference.
        Handles both unidirectional and bidirectional xLSTM models.
        """
        # x_step: [batch_size, 1, d_model] (input to encoder -> d_model)
        # state: previous state object from the xLSTM library.
        
        # The input `x_step` to this function is after your project's encoder, so it's already `d_model`.
        # It should be shaped [B, 1, D] for the stack's step, similar to how xLSTMLMModel handles it.
        if x_step.dim() == 2: # If input is [B, D]
            x_step = x_step.unsqueeze(1) # Make it [B, 1, D]

        if self.bidirectional:
            # Handle bidirectional step inference
            # For step-by-step inference, we need to process both directions
            # Note: This is complex for bidirectional as we need full sequence context
            # For now, we'll use forward direction only for step inference
            # This is a limitation but allows evaluation to work
            
            # Split state if needed (state could be tuple of (forward_state, backward_state))
            if state is not None and isinstance(state, tuple) and len(state) == 2:
                forward_state, backward_state = state
            else:
                forward_state = state
                backward_state = None
            
            # Process forward direction
            forward_output, new_forward_state = self.forward_stack.step(x_step, state=forward_state)
            
            # For bidirectional step inference, we can't properly process backward direction
            # without future context, so we'll approximate with forward direction only
            # This is a known limitation of bidirectional models in autoregressive inference
            backward_output = forward_output  # Approximation
            new_backward_state = backward_state  # Keep unchanged
            
            # Combine outputs using the combine layer
            combined = torch.cat([forward_output, backward_output], dim=-1)
            output_step = self.combine_layer(combined)
            
            # Return combined state
            new_state = (new_forward_state, new_backward_state)
        else:
            # Unidirectional processing (original logic)
            output_step, new_state = self.xlstm_stack.step(x_step, state=state)
        
        # The output_step from xlstm_block_stack.step will likely be [B, 1, D]
        return output_step.squeeze(1), new_state # Return [B, D] and new state

    def default_state(self, batch_size: int, device: torch.device):
        """
        Create default initial states for the xLSTMBlockStack.
        Handles both unidirectional and bidirectional models.
        """
        if self.bidirectional:
            # For bidirectional models, return tuple of (forward_state, backward_state)
            # According to xLSTMLMModel, initial state can be None.
            return (None, None)
        else:
            # For unidirectional models, return single state
            return None