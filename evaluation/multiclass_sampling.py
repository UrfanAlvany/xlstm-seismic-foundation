import torch
from torch.nn import functional as F


def greedy_prediction(pred: torch.Tensor):
    """
    Greedy prediction. Argmax(pred)
    :param pred: predictions [batch_size, num_classes]
    :return: greedy prediction [batch_size, 1]
    """
    return torch.argmax(pred, dim=-1).unsqueeze(1)


def multinomial_prediction(pred: torch.Tensor, temperature: float = 1.0):
    """
    Sample from predictions using softmax and multinomial distribution.
    :param pred: predictions [batch_size, num_classes]
    :param temperature: temperature for softmax. Default: 1.0
    :return: sampled prediction [batch_size, 1]
    """
    pred = pred / temperature
    return torch.multinomial(F.softmax(pred, dim=1), 1)


def top_k_prediction(pred: torch.Tensor, k: int, temperature: float = 1.0):
    """
    Sample from top k predictions
    :param pred: prediction before softmax [1, num_classes]
    :param k: number of classes to sample from
    :param temperature: temperature for softmax. Default: 1.0
    :return: sampled prediction [1, 1]
    """
    pred = pred / temperature
    pred = F.softmax(pred, dim=-1)
    top_k_prob, top_k_idx = torch.topk(pred, k)
    sample = torch.multinomial(top_k_prob, num_samples=1)
    return top_k_idx[:, int(sample)].unsqueeze(0)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf'), temperature: float = 1.0):
    """
    taken from: https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

        Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)

        cumulative_probs = torch.cumsum(F.softmax(sorted_logits / temperature, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Replace logits to be removed with -inf in the sorted_logits
        sorted_logits[sorted_indices_to_remove] = filter_value
        # Then reverse the sorting process by mapping back sorted_logits to their original position
        logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

    pred_token = torch.multinomial(F.softmax(logits, -1), 1)  # [BATCH_SIZE, 1]
    return pred_token
