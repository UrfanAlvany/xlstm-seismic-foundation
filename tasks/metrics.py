import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from functools import partial
import numpy as np
from typing import Tuple, Optional


def _unwrap_xy(x, y):
    """If x or y are (tensor, aux) tuples, unwrap the first tensor for generic metrics."""
    if isinstance(x, (tuple, list)):
        x = x[0]
    if isinstance(y, (tuple, list)):
        y = y[0]
    return x, y


def mse(output, target):
    output, target = _unwrap_xy(output, target)
    return F.mse_loss(output, target)


def mse_with_context(output, target, context_len):
    return F.mse_loss(output[:, context_len:], target[:, context_len:])


def log_mse(output, target):
    output, target = _unwrap_xy(output, target)
    return torch.log(F.mse_loss(output, target))


def mae(output, target):
    output, target = _unwrap_xy(output, target)
    return F.l1_loss(output, target)


def accuracy(output, target):
    output = output.view(-1, output.shape[-1])
    if target.numel() > output.shape[0]:
        # Mixup leads to this case: use argmax class
        target = target.argmax(dim=-1)
    target = target.view(-1)
    return torch.eq(torch.argmax(output, dim=-1), target).float().mean()

def cross_entropy(output, target):
    output = output.view(-1, output.shape[-1])
    target = target.view(-1)
    return F.cross_entropy(output, target)

def cross_entropy_with_context(output, target, context_len):
    output, target = _unwrap_xy(output, target)
    output = output[:, context_len:]
    target = target[:, context_len:]
    output = output.reshape(-1, output.shape[-1])
    target = target.reshape(-1)
    return F.cross_entropy(output, target)


def phase_pick_loss(y_pred, y_true, eps=1e-5):
    # vector cross entropy loss
    h = y_true * torch.log(F.softmax(y_pred, dim=-1) + eps)
    h = h.mean(1).sum(-1)  # Mean along sample dimension and sum along pick dimension
    h = h.mean()  # Mean over batch axis
    return -h


def masked_mse(output, target):
    """MSE computed only over masked positions.

    Accepts mask from either output or target if provided as a tuple: (tensor, mask).
    - output: Tensor | (Tensor, BoolTensor)
    - target: Tensor | (Tensor, BoolTensor)
    Mask shape should be broadcastable to output/target, typically [B, L, C] with dtype bool.
    """
    mask = None
    if isinstance(output, (tuple, list)) and len(output) >= 2:
        output, mask = output[0], output[1]
    if isinstance(target, (tuple, list)) and len(target) >= 2 and mask is None:
        target, mask = target[0], target[1]

    # Fallback: if no mask found, use standard MSE
    if mask is None:
        return F.mse_loss(output, target)

    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)
    mask = mask.to(device=output.device)
    if mask.dtype != torch.bool:
        mask = mask > 0.5

    # Ensure target is on same device
    target = target.to(device=output.device)

    # If mask has shape [B, L, C], apply elementwise selection
    # Avoid empty mask
    num = mask.sum().clamp_min(1)
    se = (output - target) ** 2
    masked_se = se[mask]
    if masked_se.numel() == 0:
        return F.mse_loss(output, target)
    return masked_se.mean()


def bidirAutoregLoss(x, y):
    x_ntk, x_ptk, x_tokens = x
    if x_tokens is None:
        diff = int(x_ntk.shape[1] - y.shape[1])
        next_token_loss = F.mse_loss(x_ntk[:, :-diff, :], y)
        prev_token_loss = F.mse_loss(x_ptk[:, diff:, :], y)
    else:
        diff = int(x_ntk.shape[1] - x_tokens.shape[1])
        next_token_loss = F.mse_loss(x_ntk[:, :-diff, :], x_tokens)
        prev_token_loss = F.mse_loss(x_ptk[:, diff:, :], x_tokens)
    return next_token_loss + prev_token_loss


#def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
def hierarchical_contrastive_loss(x: tuple, y=None):
    # hacky way to include the loss function in our training setup.
    z1, z2, alpha, temporal_unit = x
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d


def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def patch_huber_loss(x, y):

    x_next, x_prev, y_next, y_prev = x
    loss_next = F.huber_loss(x_next, y_next, delta=1.0)
    loss_prev = F.huber_loss(x_prev, y_prev, delta=1.0) 
    return loss_next + loss_prev

def compute_contrastive_logits(
    target_features: torch.Tensor,
    negative_features: torch.Tensor,
    predicted_features: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Compute logits for contrastive loss with numerically-stable cosine.

    Shapes (broadcasted):
      - predicted_features: [B, T, C]
      - target_features:    [1, B, T, C]
      - negative_features:  [K-1, B, T, C]
      - return logits:      [K, B, T]
    """
    # Stack positive and negatives along candidate dimension K
    target_features = torch.cat([target_features, negative_features], dim=0)

    # Numerically-stable cosine similarity via L2 normalization with eps
    pf = F.normalize(predicted_features.float(), dim=-1, eps=1e-6)
    tf = F.normalize(target_features.float(), dim=-1, eps=1e-6)

    # Broadcast multiply and sum over channel dim to get cosine similarities
    logits = (pf * tf).sum(dim=-1).type_as(target_features)

    # Apply temperature scaling
    logits = logits / temperature
    # Guard against NaNs/Infs from upstream numerics
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)
    return logits

def sample_negative_indices(
    features_shape: Tuple,
    num_negatives: int,
    mask_time_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    batch_size, sequence_length = features_shape

    # generate indices of the positive vectors themselves, repeat them `num_negatives` times
    sequence_length_range = np.arange(sequence_length)

    # get `num_negatives` random vector indices from the same utterance
    sampled_negative_indices = np.zeros(
        shape=(batch_size, sequence_length, num_negatives), dtype=np.int32
    )

    mask_time_indices = (
        mask_time_indices.astype(bool)
        if mask_time_indices is not None
        else np.ones(features_shape, dtype=bool)
    )

    for batch_idx in range(batch_size):
        high = mask_time_indices[batch_idx].sum() - 1
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        feature_indices = np.broadcast_to(
            np.arange(high + 1)[:, None], (high + 1, num_negatives)
        )
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        # avoid sampling the same positive vector, but keep the distribution uniform
        sampled_indices[sampled_indices >= feature_indices] += 1

        # remap to actual indices
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = (
            mapped_masked_indices[sampled_indices]
        )

        # correct for batch size
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices

def sampled_contrastive_loss(x, y, diversity_lambda: float | None = None):
    """Sampled contrastive (InfoNCE-style) loss with optional diversity regularizer.

    Expects x to be a tuple of:
      (x_model_out, x_conv_features, mask_indices, num_negatives, temperature)
    Optionally, when quantization is used, x may include a 6th element:
      diversity_loss (scalar tensor). If provided and diversity_lambda is set,
      the final loss is contrastive_loss + diversity_lambda * diversity_loss.
    """
    # Unpack mandatory items
    if isinstance(x, (tuple, list)) and len(x) >= 5:
        x_model_out, x_conv_features, mask_indices, num_negatives, temperature = x[:5]
        diversity_loss = x[5] if len(x) >= 6 else None
    else:
        raise ValueError("sampled_contrastive_loss expects a 5- or 6-tuple input.")
    '''
    # check for nan
    if torch.isnan(x_model_out).any():
        print('NaN in x_model_out')
    if torch.isnan(x_conv_features).any():
        print('NaN in x_conv_features')
    if torch.isnan(mask_indices).any():
        print('NaN in mask_indices')
    '''

    batch_size, sequence_length, hidden_size = x_conv_features.shape

    if isinstance(mask_indices, np.ndarray):
        mask_indices = torch.from_numpy(mask_indices).to(x_conv_features.device)

    # Early out if no masked positions in this batch (skip contribution)
    total_masked = int(mask_indices.sum().item()) if isinstance(mask_indices, torch.Tensor) else int(mask_indices.sum())
    if total_masked == 0:
        return x_model_out.new_tensor(0.0)
    
    # sample negative indices
    negative_sample_indices = sample_negative_indices(
        (batch_size, sequence_length),
        num_negatives=num_negatives,
        mask_time_indices=mask_indices.detach().cpu().numpy() if isinstance(mask_indices, torch.Tensor) else mask_indices
    )
    negative_sample_indices = torch.from_numpy(negative_sample_indices).to(x_conv_features.device)

    # for training, we sample negatives
    # 3. sample K negatives (distractors) quantized states for
    # contrastive loss if attention_mask is passed, make sure that padded
    # feature vectors cannot be sampled
    # sample negative quantized vectors BTC => (BxT)C
    negative_features = x_conv_features.view(-1, hidden_size)[
        negative_sample_indices.long().view(-1)
    ]

    negative_features = negative_features.view(
        batch_size, sequence_length, -1, hidden_size
    ).permute(2, 0, 1, 3)

    # 4. compute logits, corresponding to
    # `logs = sim(c_t, [q_t, \sim{q}_t]) / \kappa`
    # of equation (3) in https://arxiv.org/pdf/2006.11477.pdf
    logits = compute_contrastive_logits(
        x_conv_features[None, :],
        negative_features,
        x_model_out,
        temperature=temperature,
    )

    # 5. if a negative vector is identical to the positive
    # (i.e. when codebook utilization is low),
    # its cosine similarity will be masked
    neg_is_pos = (x_conv_features == negative_features).all(-1)

    if neg_is_pos.any():
        logits[1:][neg_is_pos] = float("-inf")

    # 6. compute contrastive loss \mathbf{L}_m = cross_entropy(logs) =
    # -log(exp(sim(c_t, q_t)/\kappa) / \sum_{\sim{q}} exp(sim(c_t, \sim{q})/\kappa))
    logits = logits.transpose(0, 2).reshape(-1, logits.size(0))
    target = (
        ((1 - mask_indices.long()) * -100)
        .transpose(  # type: ignore
            0, 1
        )
        .flatten()
    )

    # Ensure there is at least one finite logit per row; replace all -inf rows
    # with a finite baseline to avoid NaNs in CE
    finite_row = torch.isfinite(logits).any(dim=1)
    if not torch.all(finite_row):
        logits[~finite_row, 0] = 0.0
        logits[~finite_row, 1:] = -1e9

    contrastive_loss = nn.functional.cross_entropy(
        logits.float(), target, reduction="sum"
    )

    denom = mask_indices.sum().clamp_min(1)
    contrastive_loss = contrastive_loss / denom

    # Optional diversity penalty (codebook utilization)
    if (diversity_lambda is not None) and (diversity_loss is not None):
        contrastive_loss = contrastive_loss + float(diversity_lambda) * diversity_loss

    return contrastive_loss


metric_functions = {
    'mse': mse,
    'log_mse': log_mse,
    'mse-context': mse_with_context,
    'mae': mae,
    'accuracy': accuracy,
    'cross-entropy': cross_entropy,
    'cross-entropy-context': cross_entropy_with_context,
    'phase-pick': phase_pick_loss,
    'bidir-autoreg-loss': bidirAutoregLoss,
    'hierarchical-contrastive-loss': hierarchical_contrastive_loss,
    'patch-huber-loss': patch_huber_loss,
    'sampled-contrastive-loss': sampled_contrastive_loss,
    'masked_mse': masked_mse,
}
