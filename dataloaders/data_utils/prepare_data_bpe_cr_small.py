import glob
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataloaders.data_utils.costa_rica_utils import get_metadata
import dataloaders.data_utils.signal_encoding as enc
from scipy.signal import decimate
from dataloaders.tokenizers.bpe_tokenizer import BPETokenizer
import time


BITS = 8
VOCAB_SIZE = 4096
DOWNSAMPLE = 100
NUM_TEST_FILES = 3


# load data files from small subset
data_path = '../data/costa_rica/small_subset'
file_list = glob.glob(os.path.join(data_path, '*.pt'))
file_day_list = [(file, int(get_metadata(file.split('/')[-1])['day'])) for file in file_list]
sorted_file_list = [f for f, d in sorted(file_day_list, key=lambda x: x[1])]

print(f'number of files: {len(file_list)}')
print('\nFile list: ')
print(sorted_file_list, '\n')

# stack data into one tensor
full_data = []
for f in sorted_file_list:
    data = torch.load(f)
    full_data.append(data)
full_data_torch = torch.cat(full_data)

# downsample data
full_data_torch = torch.from_numpy(decimate(full_data_torch.numpy(), q=DOWNSAMPLE).copy())

print(f'shape of full downsampled data: {full_data_torch.shape}')

# squeeze, normalize and quantize data
data_sqrt = torch.sqrt(torch.abs(full_data_torch)) * torch.sign(full_data_torch)
d_min = torch.min(data_sqrt)
d_max = torch.max(data_sqrt)
d_max = max(abs(d_min), abs(d_max))
d_min = -d_max
print(f'd_min: {d_min}, d_max: {d_max}')
data_sqrt_norm = enc.normalize_11_torch(data_sqrt, d_min=d_min, d_max=d_max)
data_encoded = enc.quantize_encode(data_sqrt_norm, bits=BITS)

# train tokenizer
tokenizer = BPETokenizer()
train_data = data_encoded[:-NUM_TEST_FILES*int(8640000//DOWNSAMPLE)]
train_data_stacked = torch.reshape(train_data, [-1, 1200])
print(f'train data shape: {train_data_stacked.shape}')
tokenizer.train(vocab_size=VOCAB_SIZE, initial_vocab_size=2**BITS, words=train_data_stacked)
tokenizer.save_vocab('bpe_vocab_4096_d100_train')

print('\n***\nSTART TOKENIZING\n***\n')

start_time = time.time()
tokens, ids = tokenizer.encode(data_encoded)
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

torch.save(ids, '../data/costa_rica/bpe_small/cr_small_bpe_encoded_ids.pt')
torch.save(tokens, '../data/costa_rica/bpe_small/cr_small_bpe_encoded_tokens.pt')
