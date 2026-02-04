"""Quantization module

Dimension keys:
  B: batch size
  L: sequence length
  C: codevector dimension
  D: feature dimension; = config.conv_dim[-1] in the quantization context
  G: number of codevector groups
  V: number of codevectors per group
  K = codevector_dim // G: codevector dimension devide by num of groups
  N = B * L: number of vectors to be quantized.

"""

import math
from typing import Optional, Tuple

import einops
import torch
from torch import nn
from torch.nn.functional import gumbel_softmax


class Wav2Vec2GumbelVectorQuantizer(nn.Module):
  """Vector quantization using gumbel softmax."""

  def __init__(self, num_codevector_groups, num_codevectors_per_group, conv_dim, codevector_dim, scale_logits_in_quantization=False):
    super().__init__()

    self.num_groups = num_codevector_groups # = G
    self.num_vars = num_codevectors_per_group # = V
    self.last_conv_dim = conv_dim[-1]

    if codevector_dim % self.num_groups != 0:
      raise ValueError(
          f"`config.codevector_dim {codevector_dim} must be divisible "
          f"by `config.num_codevector_groups` {self.num_groups}"
          "for concatenation"
      )

    # storage for codebook variables (codewords)
    # [1, G * V, codevector_dim // G]
    self.codevectors = nn.Parameter(
      torch.FloatTensor(
        1, self.num_groups * self.num_vars,
        codevector_dim // self.num_groups
      )
    )
    self.weight_proj = nn.Linear(
      in_features=conv_dim[-1],
      out_features=self.num_groups * self.num_vars
    )

    # can be decayed for training
    self.temperature = 2

    self.scale_logits_in_quantization = scale_logits_in_quantization

    # initialize codevectors
    self.weight_proj.weight.data.normal_(mean=0.0, std=1)
    self.weight_proj.bias.data.zero_()
    nn.init.uniform_(self.codevectors)

  @staticmethod
  def _compute_perplexity(
    probs: torch.Tensor,
    mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
    '''Compute perplexity of the code selection distribution.

    Args:
      probs: [N, G, V]. Here N = (B * L) is the amount of vectors that we want
        to quantize. For each vector idx n, group index g,
        probs[n, g, c] is the probabilites of selecting the c-th codevector
        in that group.

    Returns:
      perplexity: scalar. The perplexity of the code selection distribution.
    '''

    # avg_probs: [G, V]
    # It is the averaged probabilites over all sequences,
    # denoted by the \bar{p}_{gv} in the Wav2Vec2 paper.
    if mask is not None:
      B, L = mask.shape
      N, G, V = probs.shape
      assert N == B * L

      # mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
      mask_extended = einops.repeat(mask, 'b l -> (b l) g v', g=G, v=V)

      probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
      denom = mask.sum()
      if isinstance(denom, torch.Tensor):
        denom = denom.clamp_min(1)
      else:
        # Fallback in case mask is numpy; avoid divide-by-zero
        denom = max(1, int(denom))
      avg_probs = einops.reduce(probs, 's g v -> g v', 'sum') / denom
    else:
      avg_probs = einops.reduce(probs, 's g v -> g v', 'mean')

    plogp = avg_probs * torch.log(avg_probs + 1e-7)

    perplexity = torch.exp(
      -einops.reduce(plogp, 'g v -> g', 'sum')
    )

    perplexity = einops.reduce(perplexity, 'g ->', 'sum')
    return perplexity

  def forward(
    self,
    hidden_states: torch.Tensor,
    mask_time_indices: Optional[torch.Tensor],
    return_selected_codevector_indices: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    '''Forward pass.

    Args:
      hidden_states: [B, L, D]. The input hidden states.
      mask_time_indices: [B, L]. The mask for the input hidden states.

    Returns:
      codevectors
      perplexity

    '''


    batch_size, sequence_length, feature_dim = hidden_states.shape

    # project to codevector dim: [B, L, G * V]
    hidden_states = self.weight_proj(hidden_states)
    assert feature_dim == self.last_conv_dim

    if self.scale_logits_in_quantization:
      hidden_states = hidden_states / math.sqrt(self.last_conv_dim)



    # hidden_states: [B * L * G, V]
    hidden_states = einops.rearrange(
      hidden_states, 'b l (g v) -> (b l g) v',
      b=batch_size,
      l=sequence_length,
      g=self.num_groups
    )

    # Cache last perplexities for both masked-only and all-frames to aid logging
    self.last_perplexity_masked = None
    self.last_perplexity_all = None

    if self.training:
      # sample code vector probs via gumbel in differentiateable way
      # codevector_probs: [B * L * G, V]

      # split out the group variable
      hidden_states = einops.rearrange(
        hidden_states,
        '(b l g) v -> (b l) g v',
        b=batch_size,
        l=sequence_length,
        g=self.num_groups
      )

      # codevector_probs: [B * L, G, V]
      codevector_probs = gumbel_softmax(
          hidden_states.float(), tau=self.temperature, hard=True,
      ).type_as(hidden_states)


      # compute perplexity
      # codevector_soft_dist: [B * L, G, V]
      codevector_soft_dist = torch.softmax(hidden_states.float(), dim=-1)

      # Perplexity over masked positions (SeisLM default for loss)
      perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
      # Perplexity over all frames (for SeisLM-style logging comparison)
      perplexity_all = self._compute_perplexity(codevector_soft_dist, None)
    else:

      codevector_idx = hidden_states.argmax(dim=-1, keepdim=True)

      # codevector_probs: [B * L * G, V]
      # Each row of codevector_probs is a one-hot vector.
      # The the non-zero index of the i-th row is codevector_idx[i].
      codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
          -1, codevector_idx, 1.0
      )

      codevector_probs = einops.rearrange(
        codevector_probs, '(b l g) v -> (b l) g v',
        b=batch_size,
        l=sequence_length,
        g=self.num_groups
      )

      # Perplexity over masked positions (SeisLM default for loss)
      perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)
      # Perplexity over all frames (for SeisLM-style logging comparison)
      perplexity_all = self._compute_perplexity(codevector_probs, None)

    # codevector_probs: [B * L, G * V]
    # codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
    codevector_probs = einops.rearrange(
      codevector_probs, '(b l) g v -> (b l) (g v)',
      b=batch_size,
      l=sequence_length,
      g=self.num_groups
    )

    # use probs to retrieve codevectors
    # codevectors_per_group: [B * L, G * V, codevector_dim // G]
    codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors

    # [B * L, G, V, codevector_dim // G]
    codevectors = codevectors_per_group.view(
      batch_size * sequence_length, self.num_groups, self.num_vars, -1
    )
    # sum over code vectors within each group
    codevectors = einops.reduce(
      codevectors, '(b l) g v k -> b l (g k)', 'sum',
      b=batch_size,
      l=sequence_length,
      g=self.num_groups,
      v=self.num_vars,
    )

    # Expose last computed perplexities for downstream loggers
    try:
      self.last_perplexity_masked = perplexity.detach()
      self.last_perplexity_all = perplexity_all.detach()
    except Exception:
      pass

    if return_selected_codevector_indices:
      # selected_codevector_indices: [B, L, G]
      selected_codevector_indices = torch.argmax(
        einops.rearrange(
          codevector_probs, '(b l) (g v) -> b l g v',
          b=batch_size,
          l=sequence_length,
          g=self.num_groups
        ),
        dim=-1
      )
      return codevectors, perplexity, selected_codevector_indices
    else:
      return codevectors, perplexity
