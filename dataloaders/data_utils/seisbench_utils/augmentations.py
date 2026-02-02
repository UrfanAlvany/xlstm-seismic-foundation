import torch
import numpy as np
from dataloaders.data_utils.signal_encoding import quantize_encode
import copy
import hashlib


class EvalPretrainMask:
    """
    Deterministic version of PretrainMask for dev/test.
    Uses a stable RNG derived from metadata (e.g., trace_name) + a user seed to
    produce the same masked positions per trace across runs.
    Output format matches PretrainMask: state_dict['X'] = ((x, masked_bool), meta)
    """
    def __init__(self, key=("X", "y"), p=0.75, seed=1234, block_lengths=(20, 50, 100)):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.p = float(p)
        self.seed = int(seed)
        self.block_lengths = tuple(block_lengths)

    def _rng_from_meta(self, meta_dict):
        key = None
        if isinstance(meta_dict, dict):
            for k in ("trace_name", "trace_id", "file_name", "id"):
                if k in meta_dict:
                    key = str(meta_dict[k]); break
        if key is None:
            key = str(meta_dict)
        h = hashlib.md5((key + f"::{self.seed}").encode("utf-8")).hexdigest()
        return np.random.RandomState(int(h[:8], 16))

    def _fill_mask(self, x_in, start_idx, block_length, rng):
        strategy = float(rng.rand())
        if strategy < 0.8:
            x_in[start_idx:start_idx + block_length, :] = 0
        elif strategy < 0.9:
            std = np.std(x_in[start_idx:start_idx + block_length], keepdims=True)
            x_in[start_idx:start_idx + block_length, :] = std * rng.randn(block_length, x_in.shape[1])
        # else: 10% leave as-is

    def __call__(self, state_dict):
        x, meta = state_dict[self.key[0]]
        y = x.copy()
        L, C = x.shape
        masked = np.zeros_like(x, dtype=np.float32)
        rng = self._rng_from_meta(meta)
        block_length = int(rng.choice(self.block_lengths))
        n_blocks = max(1, int(L * self.p / block_length))
        for _ in range(n_blocks):
            start = int(rng.randint(0, max(1, L - block_length + 1)))
            self._fill_mask(x, start, block_length, rng)
            masked[start:start + block_length, :] = 1.0
        state_dict[self.key[0]] = ((x, masked.astype(bool)), meta)
        state_dict[self.key[1]] = (y, meta)


class DeterministicCenterWindow:
    """
    Deterministic windowing to a fixed length by centering the crop/pad.
    Replaces SteeredWindow when no `_control_` is present.
    Operates before channel/sequence transpose (expects NCW-like X with
    shape (C, N) or (N,)).
    """
    def __init__(self, key="X", windowlen: int = 4096, strategy: str = "pad"):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.windowlen = int(windowlen)
        self.strategy = strategy

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]
        # x: (C, N) or (N,)
        if x is None:
            state_dict[self.key[1]] = (x, metadata)
            return
        if x.ndim == 2:
            N = x.shape[-1]
            if N >= self.windowlen:
                start = (N - self.windowlen) // 2
                end = start + self.windowlen
                xw = x[:, start:end]
            else:
                pad_left = (self.windowlen - N) // 2
                pad_right = self.windowlen - N - pad_left
                xw = np.pad(x, ((0, 0), (pad_left, pad_right)), mode="constant")
        elif x.ndim == 1:
            N = x.shape[0]
            if N >= self.windowlen:
                start = (N - self.windowlen) // 2
                end = start + self.windowlen
                xw = x[start:end]
            else:
                pad_left = (self.windowlen - N) // 2
                pad_right = self.windowlen - N - pad_left
                xw = np.pad(x, (pad_left, pad_right), mode="constant")
        else:
            # Unknown shape; no-op
            xw = x
        state_dict[self.key[1]] = (xw, metadata)


class DuplicateEvent:
    """
    Adds a rescaled version of the event to the empty part of the trace after the event.
    Event position and empty space are determined from a detection.
    Detections can be generated for example with :py:class:`~seisbench.generate.labeling.DetectionLabeller`.

    This implementation is modelled after the `implementation for EQTransformer <https://github.com/smousavi05/EQTransformer/blob/98676017f971efbb6f4475f42e415c3868d00c03/EQTransformer/core/EqT_utils.py#L255>`_.

    .. warning::
        This augmentation does **not** modify the metadata, as representing multiple picks of
        the same type is currently not supported. Workflows should therefore always first generate
        labels from metadata and then pass the labels in the key `label_keys`. These keys are automatically
        adjusted by addition of the labels.

    .. warning::
        This implementation currently has strict shape requirements:

        - (1, samples) for detection
        - (channels, samples) for data
        - (labels, samples) for labels

    :param inv_scale: The scale factor is defined by as 1/u, where u is uniform.
                      `inv_scale` defines the minimum and maximum values for u.
                      Defaults to (1, 10), e.g., scaling by factor 1 to 1/10.
    :param detection_key: Key to read detection from.
                          If key is a tuple, detection will be read from the first key and written to the second one.
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key
                to read from and the second one the key to write to.
    :param label_keys: Keys for the label columns.
                       Labels of the original and duplicate events will be added and capped at 1.
                       Note that this will lead to invalid noise traces.
                       Value can either be a single key specification or a list of key specifications.
                       Each key specification is either a string, for identical input and output keys,
                       or as a tuple of two strings, input and output keys.
                       Defaults to None.
    """

    def __init__(
            self, inv_scale=(1, 10), detection_key="detections", key="X", label_keys=None
    ):
        if isinstance(detection_key, str):
            self.detection_key = (detection_key, detection_key)
        else:
            self.detection_key = detection_key

        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

        # Single key
        if not isinstance(label_keys, list):
            if label_keys is None:
                label_keys = []
            else:
                label_keys = [label_keys]

        # Resolve identical input and output keys
        self.label_keys = []
        for key in label_keys:
            if isinstance(key, tuple):
                self.label_keys.append(key)
            else:
                self.label_keys.append((key, key))

        self.inv_scale = inv_scale

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]
        detection, _ = state_dict[self.detection_key[0]]
        detection_mask = detection[0] > 0.5

        if detection.shape[-1] != x.shape[-1]:
            raise ValueError("Number of samples in trace and detection disagree.")

        if self.key[0] != self.key[1]:
            # Ensure metadata is not modified inplace unless input and output key are anyhow identical
            metadata = copy.deepcopy(metadata)

        if detection_mask.any():
            n_samples = x.shape[-1]
            event_samples = np.arange(n_samples)[detection_mask]
            event_start, event_end = np.min(event_samples), np.max(event_samples) + 1

            if event_end + 20 < n_samples:
                second_start = np.random.randint(event_end + 20, n_samples)
                scale = 1 / np.random.uniform(*self.inv_scale)

                if self.key[0] != self.key[1]:
                    # Avoid inplace modification if input and output keys differ
                    x = x.copy()

                space = min(event_end - event_start, n_samples - second_start)
                x[:, second_start: second_start + space] += (
                        x[:, event_start: event_start + space] * scale
                )

                shift = second_start - event_start

                for label_key in self.label_keys + [self.detection_key]:
                    y, metadata = state_dict[label_key[0]]
                    if y.shape[-1] != n_samples:
                        raise ValueError(
                            f"Number of samples disagree between trace and label key '{label_key[0]}'."
                        )

                    if label_key[0] != label_key[1]:
                        metadata = copy.deepcopy(metadata)
                        y = y.copy()

                    y[:, shift:] += y[:, :-shift]
                    y = np.minimum(y, 1)
                    state_dict[label_key[1]] = (y, metadata)
        else:
            # Copy entries
            for label_key in self.label_keys + [self.detection_key]:
                y, metadata = state_dict[label_key[0]]
                if label_key[0] != label_key[1]:
                    metadata = copy.deepcopy(metadata)
                    y = y.copy()
                state_dict[label_key[1]] = (y, metadata)

        state_dict[self.key[1]] = (x, metadata)


class QuantizeAugmentation:
    def __init__(self, key='X', bits=8):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.bits = bits

    def __call__(self, state_dict):
        if self.bits > 0:
            x, metadata = state_dict[self.key[0]]
            x = quantize_encode(torch.from_numpy(x), bits=self.bits)
            x = x.numpy()
            state_dict[self.key[1]] = (x, metadata)


class FillMissingComponents:
    def __init__(self, key="X", ):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]

        if isinstance(x, list):
            x = [self._fill_missing_component(y) for y in x]
        else:
            x = self._fill_missing_component(x)

        state_dict[self.key[1]] = (x, metadata)

    def _fill_missing_component(self, x):
        std_devs = np.std(x, axis=1)

        # Find rows with zero standard deviation
        zero_std_rows = np.where(std_devs == 0)[0]
        non_zero_std_rows = np.where(std_devs != 0)[0]

        # If all rows or zero rows have zero standard deviation, return as is
        if len(zero_std_rows) == 0 or len(zero_std_rows) == x.shape[0]:
            return x

        # Otherwise, randomly replace zero-std rows with non-zero std rows
        for row_idx in zero_std_rows:
            replacement_row = np.random.choice(non_zero_std_rows)
            x[row_idx] = x[replacement_row]
        return x


class FilterZChannel:
    def __init__(self, key="X", ):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]
        # Keep channel dimension
        if x.ndim == 2:
            state_dict[self.key[1]] = (x[0:1, :], metadata)
        else:
            state_dict[self.key[1]] = (x, metadata)


class AutoregressiveShift:
    def __init__(self, key=('X', 'y'), ):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]
        state_dict[self.key[0]] = (x[:-1], metadata)
        state_dict[self.key[1]] = (x[1:], metadata)


class RandomMask:
    def __init__(self, key=('X', 'y'), p=0.5):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.p = p

    def __call__(self, state_dict):
        r = np.random.randint(0, 4)

        x, metadata = state_dict[self.key[0]]
        seq_len = x.shape[0]
        if r == 0:
            # zero out random elements with probability p. Channels are treated independently
            mask = np.random.choice([0, 1], size=x.shape, p=[self.p, 1.0 - self.p]).astype(np.float32)
            masked = x * mask
        elif r == 1:
            # zero out random samples with probability p. Zero out all three channels of the sample
            mask = np.random.choice([0, 1], size=seq_len, p=[self.p, 1.0 - self.p]).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)
            masked = x * mask
            mask = np.ones_like(x) * mask
        elif r == 2:
            # zero out a random block of the trace. Block length is 25% of trace length
            block_length = int(seq_len // 4)
            start_idx = np.random.randint(0, seq_len - block_length + 1)
            masked = x.copy()
            mask = np.ones_like(x)
            masked[start_idx:start_idx + block_length, :] = 0
            mask[start_idx:start_idx + block_length, :] = 0
        elif r == 3:
            # zero out three random blocks of 5% sequence length each
            block_length = int(seq_len * 0.05)
            masked = x.copy()
            mask = np.ones_like(x)
            for i in range(10):
                start_idx = np.random.randint(0, seq_len - block_length + 1)
                masked[start_idx:start_idx + block_length, :] = 0
                mask[start_idx:start_idx + block_length, :] = 0
        # state_dict['mask'] = np.invert(mask.astype(bool))
        state_dict[self.key[0]] = ((masked, np.invert(mask.astype(bool))), metadata)
        state_dict[self.key[1]] = (x, metadata)


class ChunkMask:
    def __init__(self, key=('X', 'y'), p=0.5):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.p = p

    def __call__(self, state_dict):
        # choose random masking task
        r = np.random.randint(0, 4)

        x, metadata = state_dict[self.key[0]]
        seq_len = x.shape[0]
        masked = x.copy()
        mask = np.ones_like(x)
        if r == 0:
            # Big Chunk
            # zero out a random block of the trace. Block length is approx 20% of trace length
            # the block is located in the middle 3/5 of the trace
            block_length = int(np.min((seq_len / 4 * np.abs(np.random.randn()), seq_len // 2)))
            # print(block_length)
            start_idx = np.random.randint(seq_len // 5, seq_len - block_length - seq_len // 5 + 1)
            masked[start_idx:start_idx + block_length, :] = 0
            mask[start_idx:start_idx + block_length, :] = 0
            '''
            elif r == 1:
                # Random Channel Chunk
                # zero out three random blocks of 10% sequence length each
                # choose a random channel to which the mask is applied
                for i in range(3):
                    block_length = int(seq_len * 0.1 * np.abs(np.random.randn()))
                    random_channel = np.random.randint(0, 3)
                    start_idx = np.random.randint(0, seq_len - block_length + 1)
                    masked[start_idx:start_idx + block_length, random_channel] = 0
                    mask[start_idx:start_idx + block_length, random_channel] = 0
            '''
        elif r == 1:
            # Small Chunks
            # zero out 5 random blocks of 4% sequence length each
            block_length = int(seq_len * 0.04)
            while np.sum(mask[:, 0]) / seq_len > 1 - self.p:
                start_idx = np.random.randint(0, seq_len - block_length + 1)
                masked[start_idx:start_idx + block_length, :] = 0
                mask[start_idx:start_idx + block_length, :] = 0
        elif r == 2 or r == 3:
            block_length = 20
            n_blocks = int(seq_len * self.p / block_length)
            for i in range(n_blocks):
                start_idx = np.random.randint(0, seq_len - block_length + 1)
                masked[start_idx:start_idx + block_length, :] = 0
                mask[start_idx:start_idx + block_length, :] = 0

        state_dict[self.key[0]] = ((masked, np.invert(mask.astype(bool))), metadata)
        state_dict[self.key[1]] = (x, metadata)


class BertStyleMask:
    def __init__(self, key=('X', 'y'), p=0.5):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.p = p

    def _fill_mask(self, x_in, start_idx, block_length, strategy):
        if strategy < 0.8:
            # zero out with 80% chance
            x_in[start_idx:start_idx + block_length, :] = 0
        elif strategy < 0.9:
            # random noise with 10% chance
            std = np.std(x_in[start_idx:start_idx + block_length], keepdims=True)
            x_in[start_idx:start_idx + block_length, :] = std * np.random.randn(block_length, 3)
        # do nothing with 10% chance

    def __call__(self, state_dict):
        # choose random masking task
        r = np.random.randint(0, 4)

        # choose random replacement strategy
        replacement_strat = np.random.random()

        x, metadata = state_dict[self.key[0]]
        y = x.copy()
        seq_len = x.shape[0]
        mask = np.ones_like(x)
        try:
            if r == 0:
                # Big Chunk
                # zero out a random block of the trace. Block length is self.p * seq_len
                # The center of the block is normally distributed around the center of the sequence
                block_length = int(np.abs(seq_len * self.p + seq_len * 0.01 * np.random.randn()))
                center = seq_len / 2 + seq_len / 4 * np.random.randn()
                center = np.max((block_length / 2 + 1, center))
                center = int(np.min((center, seq_len - (block_length / 2) - 1)))
                start_idx = int(center - block_length // 2)
                self._fill_mask(x, start_idx, block_length, replacement_strat)
                mask[start_idx:start_idx + block_length, :] = 0
            elif r == 1:
                # Small Chunks
                # zero out 5 random blocks of 4% sequence length each
                while np.sum(mask[:, 0]) / seq_len > 1 - self.p:
                    block_length = int(np.abs(seq_len * 0.04 + seq_len * 0.01 * np.random.randn()))
                    start_idx = np.random.randint(0, seq_len - block_length + 1)
                    self._fill_mask(x, start_idx, block_length, replacement_strat)
                    mask[start_idx:start_idx + block_length, :] = 0
            elif r == 2 or r == 3:
                block_length = 20
                n_blocks = int(seq_len * 2 * self.p / block_length)
                for i in range(n_blocks):
                    start_idx = np.random.randint(0, seq_len - block_length + 1)
                    self._fill_mask(x, start_idx, block_length, replacement_strat)
                    mask[start_idx:start_idx + block_length, :] = 0
        except:
            print('could not generate mask')

        state_dict[self.key[0]] = ((x, np.invert(mask.astype(bool))), metadata)
        state_dict[self.key[1]] = (y, metadata)


class TransposeLabels:
    def __init__(self, key=('X', 'y'), ):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        # x, meta_x = state_dict[self.key[0]]
        y, meta_y = state_dict[self.key[1]]
        # state_dict[self.key[0]] = (np.transpose(x, (0, 2, 1)), meta_x)
        state_dict[self.key[1]] = (y.T, meta_y)


class TransposeSeqChannels:
    def __init__(self, key=('X', 'y'), ):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        x, meta_x = state_dict[self.key[0]]
        state_dict[self.key[0]] = (np.transpose(x, (1, 0)), meta_x)


class SquashAugmentation:
    def __init__(self, key=('X', 'y'), squash_func: str = 'sqrt', alpha: float = 1.0):
        """
        Squash the trace with a squashing function. The squashing function is either 'sqrt' or 'log'
        :param key:
        :param squash_func: one of ['sqrt', 'log']
        :param alpha: trace is multiplied by alpha before being squashed (squash_func(alpha * x) * sign(x))
        """
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.squash_func = squash_func
        self.alpha = alpha

    def __call__(self, state_dict):
        """
        Squash the trace with a squashing function. The state_dict['X'] contains the input sequence of
        dimensions [channels, seq_len]. The state_dict['y'] contains the ground truth. For pre-training,
        this is also a sequence of dimensions [channels, seq_len].
        :param state_dict:
        :return:
        """
        x, metadata = state_dict[self.key[0]]

        # remove mean
        x = x - np.mean(x, axis=-1, keepdims=True)

        # squash
        if self.squash_func == 'sqrt':
            x = np.sqrt(self.alpha * np.abs(x)) * np.sign(x)
        elif self.squash_func == 'log':
            # print('log')
            x = np.log(1.0 + self.alpha * np.abs(x)) * np.sign(x)
            if np.sum(np.isnan(x)) > 0:
                print('WARNING: Nan')
                # print(self.squash_func)

        state_dict[self.key[0]] = (x, metadata)


class BrainMask:
    def __init__(self, key=('X', 'y'), p1=0.5, p2=0.5):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.p1 = p1
        self.p2 = p2

    def _fill_mask(self, x_in, start_idx, block_length, strategy):
        if strategy < 0.8:
            # zero out with 80% chance
            x_in[start_idx:start_idx + block_length, :] = 0
        elif strategy < 0.9:
            # random noise with 10% chance
            std = np.std(x_in[start_idx:start_idx + block_length], keepdims=True)
            x_in[start_idx:start_idx + block_length, :] = std * np.random.randn(block_length, 3)
        # do nothing with 10% chance

    def __call__(self, state_dict):
        # choose random masking task
        r = np.random.randint(0, 2)

        # choose random replacement strategy

        x, metadata = state_dict[self.key[0]]
        y = x.copy()
        seq_len = x.shape[0]
        mask = np.ones_like(x)
        try:
            if r == 0:
                # mask self.p percent of sequence with blocks of 20 samples
                block_length = 20
                n_blocks = int(seq_len * self.p1 / block_length)
                for i in range(n_blocks):
                    replacement_strat = np.random.random()
                    start_idx = np.random.randint(0, seq_len - block_length + 1)
                    self._fill_mask(x, start_idx, block_length, replacement_strat)
                    mask[start_idx:start_idx + block_length, :] = 0
            elif r == 1:
                # future masking
                block_length = int(seq_len * self.p2)
                x[-block_length:, :] = 0
                mask[-block_length:, :] = 0
        except:
            print('could not generate mask')

        state_dict[self.key[0]] = ((x, np.invert(mask.astype(bool))), metadata)
        state_dict[self.key[1]] = (y, metadata)

class PretrainMask:
    def __init__(self, key=('X', 'y'), p=0.75):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key
        self.p = p

    def _fill_mask(self, x_in, start_idx, block_length, strategy):
        if strategy < 0.8:
            # zero out with 80% chance
            x_in[start_idx:start_idx + block_length, :] = 0
        elif strategy < 0.9:
            # random noise with 10% chance
            std = np.std(x_in[start_idx:start_idx + block_length], keepdims=True)
            x_in[start_idx:start_idx + block_length, :] = std * np.random.randn(block_length, 3)
        # do nothing with 10% chance

    def __call__(self, state_dict):
        # choose random masking task
        r = np.random.randint(0, 4)

        x, metadata = state_dict[self.key[0]]
        y = x.copy()
        seq_len = x.shape[0]
        mask = np.ones_like(x)
        try:
            if r == 0 or r == 1:
                # mask self.p percent of sequence with blocks of 20 samples
                block_length = 20
            elif r == 2:
                block_length = 50
            else:
                block_length = 100
            n_blocks = int(seq_len * self.p / block_length)
            for i in range(n_blocks):
                replacement_strat = np.random.random()
                start_idx = np.random.randint(0, seq_len - block_length + 1)
                self._fill_mask(x, start_idx, block_length, replacement_strat)
                mask[start_idx:start_idx + block_length, :] = 0

        except:
            print('could not generate mask')

        state_dict[self.key[0]] = ((x, np.invert(mask.astype(bool))), metadata)
        state_dict[self.key[1]] = (y, metadata)


class RMSNormAugmentation:
    def __init__(self, key=('X', 'y'), ):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        x, meta_x = state_dict[self.key[0]]
        rms = np.sqrt(np.mean(x ** 2))
        x = x / rms
        state_dict[self.key[0]] = (x, meta_x)


class CopyXY:
    def __init__(self, key=('X', 'y'), ):
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        x, meta_x = state_dict[self.key[0]]
        state_dict[self.key[1]] = (x, meta_x)
