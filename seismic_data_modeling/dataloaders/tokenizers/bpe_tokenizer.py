import torch
from collections import defaultdict
import glob
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataloaders.data_utils.costa_rica_utils import get_metadata
import dataloaders.data_utils.signal_encoding as enc
from scipy.signal import decimate
import traceback
from tqdm import tqdm


class BPETokenizer():
    def __init__(self):
        self.vocab_size = 512
        self.vocab = []
        self.merges = {}

    def train(self, vocab_size: int, initial_vocab_size: int, words: torch.Tensor) -> None:
        self.vocab_size = vocab_size
        self.vocab = torch.arange(0, initial_vocab_size).tolist()
        word_freqs = defaultdict(int)

        for w in words:
            word_freqs[w] += 1

        splits = {word: [c.item() for c in word] for word in word_freqs.keys()}

        def compute_pair_freqs(w_splits):
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                split = w_splits[word]
                if len(split) == 1:
                    continue
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_freqs[pair] += freq
            return pair_freqs

        def merge_pair(a, b, w_splits):
            for word in word_freqs:
                split = w_splits[word]
                if len(split) == 1:
                    continue

                i = 0
                while i < len(split) - 1:
                    if split[i] == a and split[i + 1] == b:
                        if isinstance(a, tuple) and isinstance(b, tuple):
                            split = split[:i] + [a + b] + split[i + 2:]
                        elif isinstance(a, tuple):
                            split = split[:i] + [a + (b,)] + split[i + 2:]
                        elif isinstance(b, tuple):
                            split = split[:i] + [(a,) + b] + split[i + 2:]
                        else:
                            split = split[:i] + [(a, b)] + split[i + 2:]

                    i += 1
                w_splits[word] = split
            return w_splits

        # main loop to compute BPE vocabulary and merge rules
        print(f'Start training')
        print(f'Target vocab size: {vocab_size}\n')
        p_bar = tqdm(total=vocab_size, initial=len(self.vocab))
        p_bar.set_description('Learning BPE')
        while len(self.vocab) < vocab_size:
            # print(f'vocab size: {len(self.vocab)} / {vocab_size}')

            # compute pair frequencies
            pair_freqs = compute_pair_freqs(splits)

            # find best pair (pair with the highest frequency)
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq

            # merge the best pair in the word splits
            splits = merge_pair(*best_pair, splits)

            # add best pair to merge rules and vocabulary
            if isinstance(best_pair[0], tuple) and isinstance(best_pair[1], tuple):
                self.merges[best_pair] = best_pair[0] + best_pair[1]
                self.vocab.append(best_pair[0] + best_pair[1])
            elif isinstance(best_pair[0], tuple):
                self.merges[best_pair] = best_pair[0] + (best_pair[1],)
                self.vocab.append(best_pair[0] + (best_pair[1],))
            elif isinstance(best_pair[1], tuple):
                self.merges[best_pair] = (best_pair[0],) + best_pair[1]
                self.vocab.append((best_pair[0],) + best_pair[1])
            else:
                self.merges[best_pair] = (best_pair[0], best_pair[1])
                self.vocab.append((best_pair[0], best_pair[1]))

            p_bar.update()
        p_bar.close()
        print('\n*** DONE ***')

    def get_vocab(self):
        return self.vocab

    def get_vocab_size(self):
        return self.vocab_size

    def get_merges(self):
        return self.merges

    def encode(self, sequence: torch.Tensor) -> tuple[list, torch.Tensor]:
        """
        Encode a sequence of integers with BPE. Returns the tokens in a list and the IDs as a torch tensor
        :param sequence: Input sequence of integers as a torch.Tensor [seq_len] or [batch_size, seq_len]
        :return: tokens (list), IDs (torch.Tensor)
        """
        if sequence.dim() == 1:
            sequences = [sequence.tolist()]
        else:
            sequences = sequence.tolist()

        for pair, merge in self.merges.items():
            for idx, split in enumerate(sequences):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split[i:i + 2] = [merge]
                    else:
                        i += 1
                sequences[idx] = split

        # tokens = sum(sequences, [])
        ids = [torch.tensor([self.vocab.index(t) for t in tokens]).long() for tokens in sequences]
        if len(ids) == 1:
            ids = ids[0]
        return sequences, ids

    def _flatten_list(self, lst: list) -> list:
        # Initialize an empty list to store the integers
        flat_list = []

        # Iterate over each element in the list
        for item in lst:
            if isinstance(item, int):  # If the item is an integer, append it to the flat list
                flat_list.append(item)
            elif isinstance(item, tuple):  # If the item is a tuple, recursively flatten it
                flat_list.extend(self._flatten_list(item))
            else:
                raise ValueError("The list contains unsupported elements; only integers and tuples are allowed.")

        return flat_list

    def decode(self, ids: list | torch.Tensor) -> torch.Tensor:
        if isinstance(ids, torch.Tensor):
            ids = ids.squeeze().tolist()

        if not isinstance(ids[0], list):
            ids = [ids]

        all_sequences_decoded = []
        for seq in ids:
            tokens = [self.vocab[i] for i in seq]
            decoded = self._flatten_list(tokens)
            all_sequences_decoded.append(torch.tensor(decoded))
        return all_sequences_decoded

    def decode_to_tokens(self, ids: list | torch.Tensor) -> list:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        if not isinstance(ids[0], list):
            ids = [ids]

        tokens_all = []
        for seq_ids in ids:
            tokens = [self.vocab[i] for i in seq_ids]
            tokens_all.append(tokens)
        return tokens_all

    def load_from_file(self, file_path: str) -> None:
        try:
            vocab, merges = torch.load(file_path)
        except Exception as e:
            traceback.print_exc()

        self.vocab, self.merges = vocab, merges
        self.vocab_size = len(vocab)

    def get_available_vocabularies(self) -> list:
        """
        Returns a list of available vocabularies
        :return: list of available vocabularies
        """
        # Get absolute path of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the absolute path to the 'vocabularies' subdirectory
        vocab_dir = os.path.join(current_dir, 'vocabularies')
        file_list = glob.glob(os.path.join(vocab_dir, '*'))
        # print('Available vocabularies:')
        # print(file_list)
        return file_list

    def load_vocab(self, name: str):
        available_vocabs = self.get_available_vocabularies()
        name = name.split('/')[-1]
        names = [v.split('/')[-1] for v in available_vocabs]

        if name not in names:
            print(f'Vocabulary {name} not found')
        else:
            idx = names.index(name)
            self.load_from_file(available_vocabs[idx])

    def save_vocab(self, name: str):
        # Get absolute path of this file
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the absolute path to the 'vocabularies' subdirectory
        save_dir = os.path.join(current_dir, 'vocabularies/' + name)
        torch.save((self.vocab, self.merges), save_dir)
        print(f'Saved vocab and merges to: {save_dir}')


def get_test_data_5_days():
    data_path = '../data/costa_rica/small_subset'
    file_list = glob.glob(os.path.join(data_path, '*.pt'))
    print(f'number of files: {len(file_list)}')
    file_day_list = [(file, int(get_metadata(file.split('/')[-1])['day'])) for file in file_list]
    sorted_file_list = [f for f, d in sorted(file_day_list, key=lambda x: x[1])]

    full_data = []
    for i, f in enumerate(sorted_file_list):
        data = torch.load(f)
        full_data.append(data)
        if i > 5:
            break
    full_data_torch = torch.cat(full_data)
    full_data_torch = torch.from_numpy(decimate(full_data_torch.numpy(), q=100).copy())

    data_sqrt = torch.sqrt(torch.abs(full_data_torch)) * torch.sign(full_data_torch)
    data_sqrt_norm = enc.normalize_11_torch(data_sqrt, d_min=-np.sqrt(783285), d_max=np.sqrt(783285))
    data_encoded = enc.quantize_encode(data_sqrt_norm, bits=8)

    plt.plot(data_encoded)
    plt.show()

    data_stacked = torch.reshape(data_encoded, [-1, 1200])
    return data_stacked


def train_test():
    print('Loading data...')
    data = get_test_data_5_days()
    print(f'data shape: {data.shape}\n')

    print('Test training')
    tokenizer = BPETokenizer()
    tokenizer.train(vocab_size=512, initial_vocab_size=256, words=data)

    print(f'\nvocab size: {tokenizer.get_vocab_size()}')

    print('\nVocab:')
    print(tokenizer.get_vocab())

    print('\nMerges:')
    print(tokenizer.get_merges())

    tokenizer.save_vocab('test_vocab')


def load_vocab_test():
    tokenizer = BPETokenizer()
    vocab_list = tokenizer.get_available_vocabularies()
    tokenizer.load_vocab(vocab_list[0])
    print(tokenizer.get_vocab())


def decode_test():
    tokenizer = BPETokenizer()
    tokenizer.load_vocab('bpe_vocab_4096_d100_train')
    vocab_size = tokenizer.get_vocab_size()
    batch_size = 16
    seq_len = 1024
    x_ids = torch.load('../data/costa_rica/bpe_small/cr_small_bpe_encoded_ids.pt')[:batch_size * seq_len].reshape(
        batch_size, seq_len).unsqueeze(-1)
    x_decoded_batch = tokenizer.decode(x_ids)
    x_decoded_single_tensor = tokenizer.decode(x_ids[0])
    x_decoded_list = tokenizer.decode([2, 500, 2000, 3000, 4000, 1234, 2345])
    print('done')


def encode_test():
    tokenizer = BPETokenizer()
    tokenizer.load_vocab('test_vocab')
    vocab_size = tokenizer.get_vocab_size()
    batch_size = 5
    seq_len = 1024
    x = get_test_data_5_days()
    x_encoded_batch = tokenizer.encode(x[:batch_size])
    x_encoded_single_tensor = tokenizer.encode(x[0])
    print(x.shape)


if __name__ == '__main__':
    # train_test()
    # load_vocab_test()
    # decode_test()
    encode_test()
