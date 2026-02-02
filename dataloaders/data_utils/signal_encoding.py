import torch
import numpy as np
import matplotlib.pyplot as plt


def normalize_11(data: np.ndarray, d_min=None, d_max=None) -> np.ndarray:
    """
    Normalize numpy array to range [-1, 1]
    :param data: unnormalized data
    :return: normalized data
    """
    if d_min is None:
        d_min = np.min(data)
    if d_max is None:
        d_max = np.max(data)
    data = 2 * ((data - d_min) / (d_max - d_min)) - 1
    return data


def normalize_11_torch(data: torch.Tensor, d_min=None, d_max=None) -> torch.Tensor:
    """
    Normalize torch tensor to range [-1, 1]
    :param data: unnormalized data
    :return: normalized data as torch.Tensor
    """
    if d_min is None:
        d_min = torch.min(data)
    if d_max is None:
        d_max = torch.max(data)
    data = 2 * ((data - d_min) / (d_max - d_min)) - 1
    return data


def minmax_scale(tensor: torch.Tensor) -> torch.Tensor:
    """
    Scale tensor to range [-1, 1]. Each batch and each dimension is scaled separately
    :param tensor:
    :return:
    """
    min_val = torch.amin(tensor, dim=(1, 2), keepdim=True)
    max_val = torch.amax(tensor, dim=(1, 2), keepdim=True)
    return (tensor - min_val) / (max_val - min_val + 1e-6)


def quantize(signal: torch.Tensor, bits: int = 8, epsilon: float = 0.01) -> torch.Tensor:
    """
    Linear quantization of signal in [0, 1] to signal in [0, 2**bits -1]
    :param signal: signal in range [0, 1]
    :param bits: number of bits to quantize
    :param epsilon:
    :return:
    """
    quantization_levels = 2 ** bits
    signal *= quantization_levels - epsilon
    signal += epsilon / 2
    return signal.long()


def dequantize(signal: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Dequantize signal in [0, 2**bits] to [0, 1]
    :param signal: quantized signal
    :param bits: number of bits
    :return: dequantized signal
    """
    quantization_levels = 2 ** bits
    return signal.float() / (quantization_levels / 2) - 1


def mu_law_encode(signal: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Mu-law encoding the sigal. Signal is shifted to [0, 1]
    :param signal: input signal, should be in [-1, 1]
    :param bits: number of bits to quantize
    :return:
    """
    mu = torch.tensor(2 ** bits - 1)

    # this normalizes every sequence to [-1, 1]
    # signal = 2 * minmax_scale(signal) - 1

    numerator = torch.log1p(mu * torch.abs(signal + 1e-8))
    denominator = torch.log1p(mu)
    encoded = torch.sign(signal) * (numerator / denominator)

    return (encoded + 1) / 2  # * mu + 0.5


def mu_law_decode(encoded: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Perform inverse mu-law transformation. Gets a signal in [0, 1] from dequantize.
    Returns a signal in [-1, 1]
    """
    mu = 2 ** bits - 1

    # Invert the mu-law transformation
    x = torch.sign(encoded) * ((1 + mu) ** (torch.abs(encoded)) - 1) / mu
    return x


def quantize_encode(signal: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    Returns quantize(mu_law_encode(signal)). Output in range [0, 2**bits - 1]
    :param signal: Input signal in range [-1, 1]
    :param bits: number of bits to quantize
    :return: quantized signal [0, 2**bits - 1]
    """
    return quantize(mu_law_encode(signal, bits=bits), bits=bits)


def decode_dequantize(encoded: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """
    mu_law_decode(dequantize(encoded)). Output in range [-1, 1]
    :param encoded: encoded input signal in [0, 2**bits - 1]
    :param bits: number of bits to dequantize
    :return: dequantized, decoded signal [-1, 1]
    """
    return mu_law_decode(dequantize(encoded, bits=bits), bits=bits)


def mu_law_test_sine():
    t = torch.linspace(0, 1, 16_000)
    batch_size = 5
    sig = torch.zeros(batch_size, 16_000)
    for i in range(batch_size):
        sig[i] = 1 * torch.sin(2 * torch.pi * 20 * t) * 1.0 / (i ** 2 + 1) / (t + 1)

    # sig = normalize_11_torch(sig)

    fig, ax = plt.subplots(batch_size, 1, figsize=(10, 5))
    for i, a in enumerate(ax):
        a.plot(t, sig[i])
        a.set_xticks([])
    plt.suptitle('Input signal')
    plt.tight_layout()
    plt.show()

    sig = sig.unsqueeze(-1)
    print(f'sig.shape: {sig.shape}')

    encoded = quantize(mu_law_encode(sig, bits=8))
    fig, ax = plt.subplots(batch_size, 1, figsize=(10, 5))
    for i, a in enumerate(ax):
        a.plot(t, encoded.squeeze()[i])
        a.set_xticks([])
    plt.suptitle('Encoded signal')
    plt.tight_layout()
    plt.show()

    decoded = mu_law_decode(dequantize(encoded), bits=8)

    fig, ax = plt.subplots(batch_size, 1, figsize=(10, 5))
    for i, a in enumerate(ax):
        a.plot(t, decoded.squeeze()[i])
        a.set_xticks([])
    plt.suptitle('Decoded signal')
    plt.tight_layout()
    plt.show()

    mse = torch.nn.functional.mse_loss(decoded, sig)
    print(f'mse: {mse}')

    print(f'max quantization value: {torch.max(encoded)}, should be {255}')
    print(f'min quantization value: {torch.min(encoded)}, should be {0}')


if __name__ == '__main__':
    mu_law_test_sine()
