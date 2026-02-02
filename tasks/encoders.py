import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataloaders.data_utils.signal_encoding import quantize_encode
from utils.config_utils import instantiate
from dataloaders.base import SequenceDataset
from tasks.positional_encoding import PositionalEncoding

# S4/Sashimi imports - kept for legacy encoder support but models removed
try:
    from models.sashimi.s4_standalone import LinearActivation, S4Block as S4
    from models.sashimi.sashimi_standalone import UpPool, FFBlock, ResidualBlock
except ImportError:
    LinearActivation = S4 = UpPool = FFBlock = ResidualBlock = None

from omegaconf import OmegaConf



class Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return x


class DummyEncoder(Encoder):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

    def forward(self, x):
        return x


class LinearEncoder(Encoder):
    def __init__(self, in_features, out_features, num_classes=None):
        super().__init__(in_features=in_features, out_features=out_features)
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight._no_weight_decay = True

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        return self.linear(x)



class LayerNormLinearEncoder(Encoder):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.input_norm = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.linear(x)
        return x


class MLPEncoder(Encoder):
    def __init__(self, in_features, out_features, hidden_features=256):
        super().__init__(in_features, out_features)
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


class EmbeddingEncoder(Encoder):
    def __init__(self, in_features, out_features, num_classes=256):
        super().__init__(in_features, out_features)
        self.embedding = nn.Embedding(num_classes, out_features)

    def forward(self, x):
        x = x.squeeze(-1)
        x = self.embedding(x)
        return x


class SigEncoder(nn.Module):
    def __init__(
            self,
            d_model=64,
            seq_len=1024,
            latent_dim=64,
            regression=False,
            vocab_size=256,
            nhead=4,
            dim_feedforward=128,
            num_layers=2
    ):
        super(SigEncoder, self).__init__()
        self.regression = regression

        if self.regression:
            self.embedding = nn.Linear(1, d_model)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model=d_model, max_len=seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )
        self.dim_reduction = nn.Linear(in_features=d_model, out_features=1)
        self.out_proj = nn.Linear(in_features=seq_len, out_features=latent_dim)

    def forward(self, x, state=None):
        # x: [batch_size, seq_len, 1]
        if self.regression:
            x = self.embedding(x)
        else:
            x = self.embedding(x.squeeze())
        # x: [batch_size, seq_len, d_model]
        x = self.pe(x.transpose(0, 1)).transpose(0, 1)
        # x: [batch_size, seq_len, d_model]
        x = self.encoder(x)
        # x: [batch_size, seq_len, d_model]
        x = F.relu(self.dim_reduction(x)).squeeze(-1)
        # x: [batch_size, seq_len]
        x = self.out_proj(x)
        # x: [batch_size, latent_dim]
        return x


def s4_block(dim, bidirectional=False, dropout=0.0, **s4_args):
    layer = S4(
        d_model=dim,
        d_state=64,
        bidirectional=bidirectional,
        dropout=dropout,
        transposed=True,
        **s4_args,
    )
    return ResidualBlock(
        d_model=dim,
        layer=layer,
        dropout=dropout,
    )


def ff_block(dim, ff=2, dropout=0.0):
    layer = FFBlock(
        d_model=dim,
        expand=ff,
        dropout=dropout,
    )
    return ResidualBlock(
        d_model=dim,
        layer=layer,
        dropout=dropout,
    )


class S4Encoder(nn.Module):
    def __init__(self, d_model, n_blocks, bidirectional=False):
        super(S4Encoder, self).__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.bidirectional = bidirectional

        # self.input_linear = nn.Linear(1, d_model)
        self.input_linear = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=5, stride=1, padding=2)

        blocks = []
        for i in range(n_blocks):
            blocks.append(s4_block(dim=d_model))
            blocks.append(ff_block(dim=d_model))

        self.blocks = nn.ModuleList(blocks)

    def _forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_linear(x)
        for block in self.blocks:
            x, _ = block(x)
        return x[:, :, -1]

    def forward(self, x, state=None):
        if self.bidirectional:
            x_rev = torch.flip(x, dims=[1])
            out_forward = self._forward(x)
            out_rev = self._forward(x_rev)
            return out_forward + out_rev
        else:
            return self._forward(x)


class S4ClassEncoder(nn.Module):
    def __init__(self, d_model, n_blocks, num_classes, bidirectional=False):
        super(S4ClassEncoder, self).__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.bidirectional = bidirectional
        self.num_classes = num_classes

        # self.input_linear = nn.Linear(1, d_model)
        self.input_linear = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=5, stride=1, padding=2)
        self.output_linear = nn.Linear(in_features=d_model, out_features=num_classes)

        blocks = []
        for i in range(n_blocks):
            blocks.append(s4_block(dim=d_model))
            blocks.append(ff_block(dim=d_model))

        self.blocks = nn.ModuleList(blocks)

    def _forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_linear(x)
        for block in self.blocks:
            x, _ = block(x)
        x = x[:, :, -1]
        x = self.output_linear(x)
        x = torch.argmax(x, dim=-1).unsqueeze(-1)
        # x = torch.multinomial(F.softmax(x, dim=-1), 1)
        return x.long()

    def forward(self, x, state=None):
        if self.bidirectional:
            x_rev = torch.flip(x, dims=[1])
            out_forward = self._forward(x)
            out_rev = self._forward(x_rev)
            return out_forward + out_rev
        else:
            return self._forward(x)


class DownPoolEncoder(nn.Module):
    def __init__(self, hidden_dim, pool=16, n_blocks=0, remove_mean=False):
        super(DownPoolEncoder, self).__init__()
        self.pool = pool
        self.n_blocks = n_blocks
        self.remove_mean = remove_mean

        self.linear1 = nn.Linear(pool, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

        # self.out_proj = nn.Linear(64 * pool, out_channels)

        if self.n_blocks > 0:
            blocks = []
            for i in range(n_blocks):
                blocks.append(s4_block(dim=hidden_dim))
                blocks.append(ff_block(dim=hidden_dim))

            self.blocks = nn.ModuleList(blocks)

    def forward(self, x, state=None):
        if self.remove_mean:
            means = torch.mean(x, dim=1, keepdim=True)
            x = x - means
        x = x.reshape(x.shape[0], -1, self.pool)
        x = F.relu(self.linear1(x))
        if self.n_blocks > 0:
            x = x.transpose(1, 2)
            for block in self.blocks:
                x, _ = block(x)
            x = x.transpose(1, 2)
        x = self.linear2(x)
        # x = self.out_proj(x.flatten(1, -1))
        if self.remove_mean:
            x[:, 0] = means[:, 0]
        return x.squeeze(-1)

    def prepare_data(self, x, y):
        batch_size = x.shape[0]
        sample_len = x.shape[1]
        # TODO make sample len variable
        x = x.reshape(-1, 1024, x.shape[-1])
        y = y.reshape(-1, 1024, y.shape[-1])
        x = self.forward(x)
        y = self.forward(y)
        return x.reshape(batch_size, -1, 64), y.reshape(batch_size, -1, 64)


class DownPoolClassEncoder(nn.Module):
    def __init__(self, hidden_dim, slice_length=1024, pool=16, num_classes=256, remove_mean=False):
        super(DownPoolClassEncoder, self).__init__()
        self.pool = pool
        self.remove_mean = remove_mean
        self.num_classes = num_classes
        self.bits = int(torch.log2(torch.tensor(num_classes)).item())

        self.linear1 = nn.Linear(pool, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.out_proj = nn.Linear(slice_length // pool, num_classes)

    def forward(self, x, state=None):
        if self.remove_mean:
            means = torch.mean(x, dim=1, keepdim=True)
            x = x - means
            # means_encoded = quantize_encode(means, bits=self.bits)
        x = x.reshape(x.shape[0], -1, self.pool)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        if self.remove_mean:
            x[:, 0] = means[:, 0]
            # x = x + means_encoded
        x = self.out_proj(x.flatten(1, -1))
        x = torch.argmax(x, dim=-1)
        # x = torch.clamp(0.5 * x + 0.5 * means_encoded.squeeze(), 0, self.num_classes - 1).long()
        return x.unsqueeze(-1)


class ConditionalLinearEncoder(nn.Module):
    def __init__(self, in_features, out_features, num_classes):
        super(ConditionalLinearEncoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_classes = num_classes
        self.linear1 = nn.Linear(in_features, out_features)
        self.embedding = nn.Embedding(num_classes, out_features)

    def forward(self, x):
        samples, labels = x
        samples = self.linear1(samples)
        embeddings = self.embedding(labels.squeeze(-1)).unsqueeze(1)
        return samples + embeddings


class ConvNetEncoder(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=5, dim=128, bias=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
        )
        self.conv2 = nn.Conv1d(
            in_channels=dim,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.transpose(1, 2)


class Conv1dSubampling(nn.Module):
    """
    Convolutional 1d subsampling with padding to control sequence length reduction.
    Args:
        in_channels (int): Number of channels in the input (e.g., n_mels for spectrogram)
        out_channels (int): Number of channels produced by the convolution (typically model dimension)
        reduce_time_layers (int): Number of halving conv layers to apply (default is 2 for 1/4 reduction)

    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing sequence of inputs

    Returns:
        - **outputs** (batch, time, dim): Tensor produced by the convolution
    """

    def __init__(self, in_channels: int, out_channels: int, reduce_time_layers: int = 2) -> None:
        super(Conv1dSubampling, self).__init__()

        # First, reduce the time_length
        time_reduce_layers = []
        for _ in range(reduce_time_layers):
            time_reduce_layers.extend([
                nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
                nn.GELU()
            ])
        self.time_reduce = nn.Sequential(*time_reduce_layers)

        # Then, mix the model_dim
        self.dim_mix = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)  # Change shape to (batch_size, dim, time)
        tokens = self.time_reduce(inputs)
        outputs = self.dim_mix(tokens)
        outputs = outputs.permute(0, 2, 1)  # Revert shape to (batch_size, time, dim)
        return outputs, tokens.permute(0, 2, 1)


class BidirAutoregEncoder(nn.Module):
    def __init__(self, in_features, out_features, dropout: float = 0.0, mask: float = 0.0, pretraining: bool = True):
        super(BidirAutoregEncoder, self).__init__()
        self.mask = mask
        self.num_zero_elements = 10
        self.pretraining = pretraining
        self.conv_subsampling = Conv1dSubampling(
            in_channels=in_features,
            out_channels=out_features,
            reduce_time_layers=2
        )
        self.input_projection = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x, x_tokens = self.conv_subsampling(x)
        x = self.input_projection(x)
        if self.mask > 0:
            batch_size, seq_len, channels = x.shape
            num_masking_points = int(seq_len * self.mask)
            for batch_idx in range(batch_size):
                mask_indices = torch.randperm(seq_len - self.num_zero_elements - 1)[:num_masking_points]
                for start_idx in mask_indices:
                    end_idx = min(start_idx + self.num_zero_elements, seq_len)
                    x[batch_idx, start_idx:end_idx, :] = 0

        # Return tuple for pretraining, single tensor for finetuning/evaluation
        if self.pretraining:
            return (x, x_tokens)
        else:
            return x
    

from models.ts2vec_utils import take_per_row, get_crop_indices, compute_mask_indices, compute_mask_indices_fairseq


class ConvDownLayer(nn.Module):
    def __init__(self, in_feature, out_features, kernel_size=3, stride=2, dilation=1):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_feature,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
        )
        self.ln = nn.LayerNorm(out_features)

    def forward(self, x):
        # x: [batch_size, seq_len, in_features]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = F.gelu(x)
        return x

class ConvDownEncoder(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 n_layers=2,
                 kernel_size=3, 
                 dim=256, 
                 stride=2,
                 temporal_unit=0,
                 mask_prob=0.65,
                 mask_length=10,
                 mask_channel_prob=0.0,
                 mask_channel_length=1,
                 pretraining=True
                 ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.dim = dim
        self.stride = stride
        self.temporal_unit = temporal_unit
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_length = mask_channel_length
        self.n_layers = n_layers
        
        self.pretraining = pretraining

        # setup convolutional layers and layer normalization
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                f_in = in_features
            else:
                f_in = dim

            # convolutional layer
            conv_layer = ConvDownLayer(
                in_feature=f_in,
                out_features=dim,
                kernel_size=kernel_size,
                stride=stride
            )
            self.layers.append(conv_layer)

        # learnable mask embedding
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(dim).uniform_()
        )

        # final projection layer
        self.final_projection = nn.Linear(dim, out_features)
    
    def apply_mask(self, x):
        # x: [B, rand_crop_len, self.dim]
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                self.mask_prob,
                self.mask_length,
                min_masks=2,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                self.mask_channel_prob,
                self.mask_channel_length,
                min_masks=0
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices
    
    def _forward(self, x):
        # x: [batch_size, seq_len, in_features]
        # forward pass without the final projection layer
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        # x: [batch_size, seq_len, in_features]

        if self.pretraining:
            crop_l, crop_left, crop_right, crop_eleft, crop_eright, crop_offset = get_crop_indices(
                input_shape=x.shape,
                temporal_unit=self.temporal_unit,
                downsample_rate=self.stride ** self.n_layers
            )

            # process the overlapping crops
            out1 = self._forward(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
            out2 = self._forward(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))

            # mask the crops
            out1, mask_1_indices = self.apply_mask(out1)
            out2, mask_2_indices = self.apply_mask(out2)

            # final projection
            out1 = self.final_projection(out1)
            out2 = self.final_projection(out2)

            return (out1, out2, crop_l, self.temporal_unit)
        else:
            x = self._forward(x)
            x = self.final_projection(x)
            return x
        
class ConvDownEncoderContrastive(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 n_layers=2,
                 kernel_size=3, 
                 dim=256, 
                 stride=2,
                 dilation=1,
                 mask_prob=0.65,
                 mask_length=10,
                 mask_channel_prob=0.0,
                 mask_channel_length=1,
                 pretraining=True,
                 bert_style=False,
                 same_mask_length=False,
                 ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.dim = dim
        self.stride = stride
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_length = mask_channel_length
        self.n_layers = n_layers
        self.dilation = dilation
        self.bert_style = bert_style
        self.same_mask_length = same_mask_length

        self.rng = np.random.default_rng(42)

        if self.bert_style:
            if not same_mask_length:
                print("WARNING: BERT-style masking requires same_mask_length=True. Setting it to True.")
            self.same_mask_length = True

        self.pretraining = pretraining

        # setup convolutional layers and layer normalization
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                f_in = in_features
            else:
                f_in = dim

            # convolutional layer
            conv_layer = ConvDownLayer(
                in_feature=f_in,
                out_features=dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation if i > 0 else 1,
            )
            self.layers.append(conv_layer)

        # learnable mask embedding
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(out_features).uniform_()
        )

        # final projection layer
        self.final_norm = nn.LayerNorm(dim)
        self.final_projection = nn.Linear(dim, out_features)

    def split_mask_segments_bool(self, mask_indices, split=(0.1, 0.1, 0.8)):
        """
        Splits continuous masked segments into three groups and returns three boolean masks.
        Args:
            mask_indices: np.ndarray of shape (B, L), boolean mask.
            split: tuple of three floats summing to 1.0, e.g. (0.1, 0.1, 0.8)
        Returns:
            Three np.ndarrays of shape (B, L), boolean masks for each group.
        """
        B, L = mask_indices.shape
        all_segments = []

        if isinstance(mask_indices, torch.Tensor):
            mask_indices = mask_indices.cpu().numpy()

        for b in range(B):
            mask = mask_indices[b]
            in_segment = False
            seg_start = None
            for i, val in enumerate(mask):
                if val and not in_segment:
                    in_segment = True
                    seg_start = i
                elif not val and in_segment:
                    in_segment = False
                    all_segments.append((b, seg_start, i))  # [start, end)
            if in_segment:
                all_segments.append((b, seg_start, L))

        # Randomize segment order
        self.rng.shuffle(all_segments)

        n = len(all_segments)
        n1 = int(n * split[0])
        n2 = int(n * split[1])
        # The rest goes to the last group
        n3 = n - n1 - n2

        group1 = all_segments[:n1]
        group2 = all_segments[n1:n1+n2]
        group3 = all_segments[n1+n2:]

        mask1 = np.zeros_like(mask_indices, dtype=bool)
        mask2 = np.zeros_like(mask_indices, dtype=bool)
        mask3 = np.zeros_like(mask_indices, dtype=bool)

        for b, start, end in group1:
            mask1[b, start:end] = True
        for b, start, end in group2:
            mask2[b, start:end] = True
        for b, start, end in group3:
            mask3[b, start:end] = True

        return mask1, mask2, mask3
    
    def apply_mask(self, x):
        # x: [B, rand_crop_len, self.dim]
        B, T, C = x.shape
        if self.mask_prob > 0:
            if self.same_mask_length:
                mask_indices = compute_mask_indices_fairseq(
                    shape=(B, T),
                    padding_mask=None,
                    mask_prob=self.mask_prob,
                    mask_length=self.mask_length,
                    require_same_masks=True,
                )
            else:
                mask_indices = compute_mask_indices(
                    (B, T),
                    self.mask_prob,
                    self.mask_length,
                    min_masks=2,
                )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            if self.bert_style:
                # we do not alter mask_1, replace mask_2 with noise and mask_3 with mask_emb
                mask_1, mask_2, mask_3 = self.split_mask_segments_bool(mask_indices, split=(0.1, 0.1, 0.8))
                std = torch.std(x[mask_2], dim=0, keepdim=True) # compute std for noise, one entry per channel
                x[mask_2] = (std * torch.randn_like(x[mask_2])).to(x.dtype).to(x.device)  # replace with noise
                x[mask_3] = self.mask_emb.to(x.dtype)
            else:
                x[mask_indices] = self.mask_emb.to(x.dtype)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            if self.same_mask_length:
                mask_channel_indices = compute_mask_indices_fairseq(
                    shape=(B, C),
                    padding_mask=None,
                    mask_prob=self.mask_channel_prob,
                    mask_length=self.mask_channel_length,
                    require_same_masks=True,
                )
            else:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    min_masks=0
                )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices
    
    def _forward(self, x):
        # x: [batch_size, seq_len, in_features]
        # forward pass without the final projection layer
        for layer in self.layers:
            x = layer(x)
        return x

    def forward(self, x):
        # x: [batch_size, seq_len, in_features]

        if self.pretraining:
            x = self._forward(x)
            x_normalized = self.final_norm(x)
            x_projected = self.final_projection(x_normalized)
            # apply mask
            x_projected, mask_indices = self.apply_mask(x_projected)

            return (x_projected, x_normalized, mask_indices)
        else:
            x = self._forward(x)
            x = self.final_norm(x)
            x = self.final_projection(x)
            return x


class PatchEncoder(nn.Module):
    def __init__(self, in_features, out_features, patch_size, n_patches):
        super(PatchEncoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.training_patches = n_patches + 1

        self.pretraining = True

        # Linear projection layer
        self.conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
    
    def forward(self, x):
        """
        :param x: Input tensor of shape (B, L, C)
        :return: Output tensor of shape (B, n_patches, out_features)
        """

        B, L, C = x.shape

        if self.pretraining:
            # random crop if the length of the sequence is larger than the number of patches times the patch size
            if L > self.training_patches * self.patch_size:
                random_start = torch.randint(low=0, high=L - self.training_patches * self.patch_size + 1, size=(1,)).item()
                x = x[:, random_start:random_start + self.training_patches * self.patch_size, :]
            
            # ground truth for later
            y_next = x[:, self.patch_size:, :]  # ground truth for next patch
            y_prev = x[:, :-self.patch_size, :]  # ground truth for previous patch

            x = x.transpose(1, 2)  # Change shape to (B, C, L)
            x = self.conv(x)  # Apply convolution
            x = x.transpose(1, 2)  # Change shape back to (B, n_patches, out_features)

            x_in_next = x[:, :-1, :]  # Input for next patch
            x_in_prev = x[:, 1:, :]  # Input for previous patch

            return x_in_next, x_in_prev, y_next, y_prev
        else:
            x = x.transpose(1, 2)
            x = self.conv(x)
            x = x.transpose(1, 2)
            return x

        

enc_registry = {
    'dummy': DummyEncoder,
    'linear': LinearEncoder,
    'layernorm': LayerNormLinearEncoder,
    'mlp': MLPEncoder,
    'embedding': EmbeddingEncoder,
    'transformer': SigEncoder,
    's4-encoder': S4Encoder,
    'pool': DownPoolEncoder,
    'learned-classes': DownPoolClassEncoder,
    's4-class': S4ClassEncoder,
    'conditional-linear': ConditionalLinearEncoder,
    'convnet-encoder': ConvNetEncoder,
    'bidir-autoreg-encoder': BidirAutoregEncoder,
    'conv-down-encoder': ConvDownEncoder,
    'patch-encoder': PatchEncoder,
    'conv-down-encoder-contrastive': ConvDownEncoderContrastive,
}

pretrain_encoders = ['transformer, s4-encoder', 'pool', 'learned-classes', 's4-class']


def instantiate_encoder(encoder, dataset: SequenceDataset = None, model=None):
    if encoder is None:
        return None

    if encoder._name_ in pretrain_encoders:
        obj = instantiate(enc_registry, encoder)
        return obj

    if dataset is None:
        print('Please specify dataset to instantiate encoder')
        return None

    if model is None:
        print('Please specify model to instantiate encoder')
        return None

    in_features = dataset.d_data
    out_features = model.d_model

    if dataset.num_classes is not None:
        obj = instantiate(
            enc_registry,
            encoder,
            in_features=in_features,
            out_features=out_features,
            num_classes=dataset.num_classes
        )
    else:
        obj = instantiate(
            enc_registry,
            encoder,
            in_features=in_features,
            out_features=out_features
        )

    return obj


def instantiate_encoder_simple(encoder, d_data, d_model):
    obj = instantiate(enc_registry, encoder, in_features=d_data, out_features=d_model)
    return obj


def load_encoder_from_file(encoder_file, dataset: SequenceDataset = None, model=None):
    encoder_state_dict, hparams = torch.load(encoder_file, weights_only=False)
    enc_config = OmegaConf.create(hparams['encoder'])
    encoder = instantiate_encoder(enc_config, dataset=dataset, model=model)
    encoder.load_state_dict(encoder_state_dict)

    # freeze parameters
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder, hparams
