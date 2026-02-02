import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config_utils import instantiate
from dataloaders.base import SequenceDataset
from tasks.positional_encoding import PositionalEncoding

# S4/Sashimi imports - kept for legacy decoder support but models removed
try:
    from models.sashimi.s4_standalone import LinearActivation, S4Block as S4
    from models.sashimi.sashimi_standalone import UpPool, FFBlock, ResidualBlock
except ImportError:
    LinearActivation = S4 = UpPool = FFBlock = ResidualBlock = None
from einops import rearrange

from omegaconf import OmegaConf


class Decoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Decoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, state=None):
        return x


class DummyDecoder(Decoder):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)

    def forward(self, x, state=None):
        return x


class LinearDecoder(Decoder):
    def __init__(self, in_features, out_features, num_classes=None):
        super().__init__(in_features, out_features)
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight._no_weight_decay = True

    def forward(self, x, state=None):
        return self.linear(x)


class SigDecoder(nn.Module):
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
        super(SigDecoder, self).__init__()
        self.regression = regression
        self.seq_proj = nn.Linear(in_features=latent_dim, out_features=seq_len)
        self.dim_proj = nn.Linear(in_features=1, out_features=d_model)
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
        if self.regression:
            self.dim_reduction = nn.Linear(in_features=d_model, out_features=1)
        else:
            self.dim_reduction = nn.Linear(in_features=d_model, out_features=vocab_size)

    def forward(self, x, state=None):
        # x: [batch_size, latent_dim]
        x = F.relu(self.seq_proj(x))
        # x: [batch_size, seq_len]
        x = F.relu(self.dim_proj(x.unsqueeze(-1)))
        # x: [batch_size, seq_len, d_model]
        x = self.pe(x.transpose(0, 1)).transpose(0, 1)
        # x: [batch_size, seq_len, d_model]
        x = self.encoder(x)
        # x: [batch_size, seq_len, d_model]
        x = self.dim_reduction(x)
        # x: [batch_size, seq_len, 1]
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


class S4Decoder(nn.Module):
    def __init__(self, d_model, n_blocks, bidirectional=False, add_mean=False):
        super(S4Decoder, self).__init__()
        self.d_model = d_model
        self.n_blocks = n_blocks
        self.bidirectional = bidirectional
        self.add_mean = add_mean

        # if self.add_mean:
        #    self.in_proj = nn.Linear(in_features=d_model, out_features=d_model)

        self.conv_t = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=16,
            stride=16,
            padding=0,
        )
        self.out_proj = nn.Linear(d_model, 1)

        blocks = []
        for i in range(n_blocks):
            blocks.append(s4_block(dim=d_model))
            blocks.append(ff_block(dim=d_model))
        self.blocks = nn.ModuleList(blocks)

    def _forward(self, x, state=None):
        if x.dim == 3:
            x = x.squeeze(-1)
        if self.add_mean:
            means = x[:, 0]
            # x = self.in_proj(x)
            x[:, 1:] = x[:, 1:] + means.unsqueeze(-1)
        x = self.conv_t(x.unsqueeze(1))
        for block in self.blocks:
            x, _ = block(x)
        x = self.out_proj(x.transpose(1, 2))
        # if self.add_mean:
        #    x = x + means.unsqueeze(-1).unsqueeze(-1)
        return x

    def forward(self, x, state=None):
        if self.bidirectional:
            x_rev = torch.flip(x, dims=[1])
            out_forward = self._forward(x)
            out_rev = self._forward(x_rev)
            return out_forward + out_rev
        else:
            return self._forward(x)


class UpPoolDecoder(nn.Module):
    def __init__(self, d_model, pool):
        super(UpPoolDecoder, self).__init__()
        self.d_model = d_model
        self.pool = pool
        self.conv_t = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=self.pool,
            stride=self.pool,
            padding=0,
        )
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x, state=None):
        if x.dim == 3:
            x = x.squeeze(-1)
        x = self.conv_t(x.unsqueeze(1))
        x = self.out_proj(x.transpose(1, 2))
        return x


class EmbeddingDecoder(nn.Module):
    def __init__(self, num_classes, output_dim, d_model=64, n_blocks=0):
        super(EmbeddingDecoder, self).__init__()
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.d_model = d_model
        self.n_blocks = n_blocks

        self.embedding = nn.Embedding(self.num_classes, self.output_dim)

        self.conv_t = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=16,
            stride=16,
            padding=0,
        )
        self.out_proj = nn.Linear(d_model, 1)

        blocks = []
        for i in range(n_blocks):
            blocks.append(s4_block(dim=d_model))
            blocks.append(ff_block(dim=d_model))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, state=None):
        if self.n_blocks == 0:
            x = self.embedding(x).transpose(1, 2)
            return x
        else:
            x = self.embedding(x)
            x = self.conv_t(x)
            for block in self.blocks:
                x, _ = block(x)
            x = self.out_proj(x.transpose(1, 2))

        return x


class PhasePickDecoder(nn.Module):
    def __init__(self, d_model, output_dim=3, convolutional=False, kernel_size=33, dropout=0.0):
        """
        Decoder for phase picking tasks
        :param d_model: dimension of model backbone
        :param output_dim: dimension of output, for phase picking this is usually 3
        :param convolutional: If true, use a Conv1d with kernel size 'kernel_size'. Else, use a linear layer.
        :param kernel_size: Size of the convolutional kernel. Must be uneven!
        :param dropout: input dropout rate
        """
        super(PhasePickDecoder, self).__init__()
        self.convolutional = convolutional

        if self.convolutional:
            assert kernel_size % 2 == 1, 'Kernel size must be uneven'

            self.conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=output_dim,
                kernel_size=kernel_size,
                stride=1,
                padding=int(kernel_size // 2),
            )
        else:
            self.linear = nn.Linear(d_model, output_dim)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, state=None):
        x = self.dropout(x)
        if self.convolutional:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)
        else:
            return self.linear(x)


class LargePhasePickDecoder(nn.Module):
    def __init__(self, d_model, output_dim=3, dropout=0.0, kernel_size_1=129, kernel_size_2=5):
        super(LargePhasePickDecoder, self).__init__()
        self.d_model = d_model

        self.net = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size_1, stride=1,
                      padding=kernel_size_1 // 2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size_2, stride=1,
                      padding=kernel_size_2 // 2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(in_channels=d_model, out_channels=output_dim, kernel_size=kernel_size_2, stride=1,
                      padding=kernel_size_2 // 2, bias=False)
        )

    def forward(self, x, state=None):
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.transpose(1, 2)
        return x


class SequenceClassifier(nn.Module):
    def __init__(self, in_features, out_features, num_classes, mode='avg', kernel_size=3, dropout=0.0, manual_input_dim=None):
        super(SequenceClassifier, self).__init__()
        self.mode = mode

        # Optionally project input features to a target width (e.g., 128 -> 256)
        self.in_features = in_features
        self.target_in_features = manual_input_dim if manual_input_dim is not None else in_features
        # Keep a pre-projection module to remain compatible with older checkpoints (may be unused)
        self.pre_proj = None
        if in_features != self.target_in_features:
            self.pre_proj = nn.Linear(in_features, self.target_in_features)

        # SeisLM-style two-stage conv reduction when target width differs from backbone width
        self.use_seislm_head = (self.target_in_features != self.in_features and mode == 'double-conv')

        self.linear = nn.Linear(self.target_in_features, num_classes)
        if mode == 'avg-mlp':
            self.lin1 = nn.Linear(self.target_in_features, 2 * self.target_in_features)
            self.lin2 = nn.Linear(2 * self.target_in_features, num_classes)

        if mode == 'double-conv':
            padding = int(kernel_size//2)
            if self.use_seislm_head:
                # conv(128->128, stride=2) then conv(128->256, stride=2)
                self.conv1 = nn.Sequential(
                    nn.Conv1d(in_channels=self.in_features, out_channels=self.in_features, kernel_size=kernel_size, stride=2,
                              padding=padding, bias=False),
                    nn.BatchNorm1d(self.in_features),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                )
                self.conv2 = nn.Sequential(
                    nn.Conv1d(in_channels=self.in_features, out_channels=self.target_in_features, kernel_size=kernel_size, stride=2,
                              padding=padding, bias=False),
                    nn.BatchNorm1d(self.target_in_features),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                )
            else:
                # Legacy path: operate fully at target width
                self.net = nn.Sequential(
                    nn.Conv1d(in_channels=self.target_in_features, out_channels=self.target_in_features, kernel_size=kernel_size, stride=1,
                              padding=padding, bias=False),
                    nn.BatchNorm1d(self.target_in_features),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    nn.Conv1d(in_channels=self.target_in_features, out_channels=self.target_in_features, kernel_size=kernel_size, stride=1,
                              padding=padding, bias=False),
                    nn.BatchNorm1d(self.target_in_features),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                )

    def forward(self, x, state=None):
        if self.mode == 'avg':
            x = torch.mean(x, dim=1)
            x = self.linear(x)
        elif self.mode == 'last':
            x = x[:, -1, :]
            x = self.linear(x)
        elif self.mode == 'avg-mlp':
            x = torch.mean(x, dim=1)
            x = F.gelu(self.lin1(x))
            x = self.lin2(x)
        elif self.mode == 'double-conv':
            if self.use_seislm_head:
                # SeisLM-style: conv1 keeps channels, conv2 reduces to target width (both stride=2)
                # Input expected shape [B, L, D_in]
                x_t = x.transpose(1, 2)
                x_t = self.conv1(x_t)
                x_t = self.conv2(x_t)
                x_f = x_t.transpose(1, 2)
                # Masked mean over time to ignore padded frames (zeros across channels)
                # Build mask at input resolution, then downsample with nearest to match x_f length
                mask_in = (x.abs().sum(dim=-1) > 0).float().unsqueeze(1)  # [B,1,L]
                with torch.no_grad():
                    mask_ds = torch.nn.functional.interpolate(mask_in, size=x_f.size(1), mode='nearest').squeeze(1)  # [B,L']
                denom = mask_ds.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1]
                x_sum = (x_f * mask_ds.unsqueeze(-1)).sum(dim=1)          # [B, C]
                x_mean = x_sum / denom                                    # [B, C]
                x = self.linear(x_mean)
            else:
                # Legacy: convs at target width, stride=1
                if x.size(-1) != self.target_in_features:
                    if self.pre_proj is not None and self.pre_proj.weight.device != x.device:
                        self.pre_proj = self.pre_proj.to(x.device)
                    x = self.pre_proj(x) if self.pre_proj is not None else x
                x = x.transpose(1, 2)
                x = self.net(x).transpose(1, 2)
                x = torch.mean(x, dim=1)
                x = self.linear(x)
        else:
            print(f'Unknown mode: {self.mode}')
        return x


class CausalDecoder(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=16):
        super(CausalDecoder, self).__init__()
        self.kernel_size = kernel_size

        # Define 1D convolution with 1 input channel, 1 output channel, and kernel_size
        self.conv = nn.Conv1d(
            in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=kernel_size - 1,
            bias=False
        )

    def forward(self, x, state=None):
        # x: [B, L, D_in] then output: [B, L, D_out] (for training)
        # at inference time: x is [B, D_in], state [B, L, D_in], output [B, D_out]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add the channel dimension: [batch_size, 1, seq_len]

        if state is not None:
            x = torch.cat((state, x), dim=1)

        # Apply the causal convolution
        x = x.transpose(1, 2)
        out = self.conv(x)[:, :, :-(self.kernel_size - 1)]
        out = out.transpose(1, 2)

        if state is not None:
            out = out[:, -1, :]
        # Remove the extra channel dimension: [batch_size, seq_len]
        return out.squeeze(1)


class ConvNetDecoder(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=5, dim=128):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        self.conv2 = nn.Conv1d(
            in_channels=dim,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

    def forward(self, x, state=None):
        x = x.transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x.transpose(1, 2)


class Conv1dUpsampling(nn.Module):
    def __init__(self, hidden_dim: int, reduce_time_layers: int = 2):
        super(Conv1dUpsampling, self).__init__()

        # Upsample only in the time dimension, increase time dimensions of the hidden_states tensor
        layers = []
        for _ in range(reduce_time_layers):
            layers.extend([
                nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.GELU()
            ])
        self.time_upsample = nn.Sequential(*layers)

        # Reduce the potential effects of padded artifacts introduced by the upsampling
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, dim, time)
        x = self.time_upsample(x)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # Revert shape to (batch_size, time, dim)
        return x


class BidirAutoregDecoder(nn.Module):
    def __init__(self, in_features, out_features, upsample=False):
        super(BidirAutoregDecoder, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.upSample = Conv1dUpsampling(hidden_dim=in_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, state=None):
        x_ntk, x_ptk, x_tokens = x
        if self.upsample:
            x_tokens = None
            x_ntk = self.upSample(x_ntk)
            x_ptk = self.upSample(x_ptk)
        out_ntk = self.linear(x_ntk)
        out_ptk = self.linear(x_ptk)
        return (out_ntk, out_ptk, x_tokens)


class BidirPhasePickDecoder(nn.Module):
    def __init__(self, in_features, out_features, upsample=True, kernel_size=33):
        super(BidirPhasePickDecoder, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.upSample = Conv1dUpsampling(hidden_dim=in_features)

        self.linear = nn.Linear(in_features, in_features)
        self.conv_1 = nn.Conv1d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.out_conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=1,
            padding=int(kernel_size // 2),
        )

    def forward(self, x, state=None):
        x_ntk, x_ptk, _ = x
        if self.upsample:
            # x_ntk = self.upSample(x_ntk)
            out = self.upSample(x_ptk)
        # out = torch.cat([x_ntk, x_ptk], dim=-1)
        out = F.gelu(self.conv_1(out.transpose(1, 2)))
        out = self.out_conv(out).transpose(1, 2)[:, 4:-4, :]
        return out


class BidirPhasePickDecoderSmall(nn.Module):
    def __init__(self, in_features, out_features, upsample=True, kernel_size=33):
        super(BidirPhasePickDecoderSmall, self).__init__()
        self.upsample = upsample

        self.out_conv = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=kernel_size,
            stride=1,
            padding=int(kernel_size // 2),
        )
        if self.upsample:
            self.upSample = Conv1dUpsampling(hidden_dim=out_features)
        self.out_proj = nn.Conv1d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x, state=None):
        if len(x) == 2:
            x_ptk = x[0]
        elif len(x) == 3:
            x_ptk = x[1]
        else:
            x_ptk = x
        out = self.out_conv(x_ptk.transpose(1, 2)).transpose(1, 2)
        if self.upsample:
            out = self.upSample(out)
        out = self.out_proj(out.transpose(1, 2)).transpose(1, 2)
        return out


class UpsamplingDecoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(UpsamplingDecoder, self).__init__()
        self.upSample = Conv1dUpsampling(hidden_dim=in_features)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, state=None):
        out = self.upSample(x)
        out = self.linear(out)
        return out


class UpPool(nn.Module):
    def __init__(self, d_input, expand, pool, transposed=True):
        """
        if transposed is True, the input is [batch_size, H, seq_len]
        else: [batch_size, seq_len, H]
        """
        super().__init__()
        self.d_output = d_input // expand
        self.pool = pool
        self.transposed = transposed

        self.linear = LinearActivation(
            d_input,
            self.d_output * pool,
            transposed=True,
        )

    def forward(self, x, skip=None):
        if not self.transposed:
            x = x.transpose(1, 2)
        x = self.linear(x)

        x = F.pad(x[..., :-1], (1, 0))  # Shift to ensure causality
        x = rearrange(x, '... (h s) l -> ... h (l s)', s=self.pool)

        if skip is not None:
            x = x + skip
        if not self.transposed:
            x = x.transpose(1, 2)
        return x, None

    def step(self, x, state, **kwargs):
        """
        x: (..., H)
        """
        assert len(state) > 0
        y, state = state[0], state[1:]
        if len(state) == 0:
            assert x is not None
            squeeze_idx = -1 if self.transposed else 1
            x = x.unsqueeze(-1)
            x = self.linear(x)
            x = x.squeeze(-1)
            x = rearrange(x, '... (h s) -> ... h s', s=self.pool)
            state = list(torch.unbind(x, dim=-1))
        else:
            assert x is None
        return y, state

    def default_state(self, *batch_shape, device=None):
        state = torch.zeros(batch_shape + (self.d_output, self.pool), device=device)  # (batch, h, s)
        state = list(torch.unbind(state, dim=-1))  # List of (..., H)
        return state


class CausalUpsamplingDecoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(CausalUpsamplingDecoder, self).__init__()
        self.upSample = UpPool(
            d_input=in_features,
            expand=4,
            pool=4,
            transposed=False,
        )
        self.linear = nn.Linear(in_features // 4, out_features)

    def forward(self, x, state=None):
        x, _ = self.upSample(x)
        x = self.linear(x)
        return x


class CausalBidirAutoregDecoder(nn.Module):
    def __init__(self, in_features, out_features, upsample=False):
        super(CausalBidirAutoregDecoder, self).__init__()
        self.upsample = upsample
        if self.upsample:
            self.upPool = UpPool(
                d_input=in_features,
                expand=4,
                pool=4,
                transposed=False,
            )
            self.linear = nn.Linear(in_features // 4, out_features)
        else:
            self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, state=None):
        x_ntk, x_ptk, x_tokens = x
        if self.upsample:
            x_tokens = None
            x_ntk, _ = self.upPool(x_ntk)
            x_ptk, _ = self.upPool(x_ptk)
        out_ntk = self.linear(x_ntk)
        out_ptk = self.linear(x_ptk)
        return (out_ntk, out_ptk, x_tokens)


class SanityCheckPhasePicker(nn.Module):
    def __init__(self, in_features, out_features, upsample: bool = False, output_len: int = 4096):
        super().__init__()
        self.upsample = upsample
        self.output_len = output_len
        hidden_dim = 64

        self.linear1 = nn.Linear(in_features, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, out_features)

        self.conv = nn.Conv1d(
            in_channels=out_features,
            out_channels=out_features,
            kernel_size=7,
            stride=1,
            padding=3
        )

    def forward(self, x, state=None):
        if isinstance(x, tuple) or isinstance(x, list):
            if len(x) == 3:
                x = x[1]
            elif len(x) == 2:
                x = x[0]
        seq_len = x.shape[1]

        x = x.transpose(1, 2)
        if self.upsample:
            x = F.interpolate(x, size=4 * seq_len, mode='linear').transpose(1, 2)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        len_diff = x.shape[1] - self.output_len
        if len_diff > 0:
            x = x[:, len_diff // 2: - len_diff // 2, :]
        return x


class DoubleConvPhasePickDecoder(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, upsample=False, upsample_by=4, output_len=4096, dropout=0.0, manual_input_dim=None, upsample_last=False):
        super(DoubleConvPhasePickDecoder, self).__init__()
        self.upsample = upsample
        self.output_len = output_len
        self.upsample_by = upsample_by
        self.upsample_last = upsample_last

        if manual_input_dim is not None:
            in_features = manual_input_dim
        
        assert kernel_size % 2 == 1, 'Kernel size must be uneven'
        padding = int(kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=1,
                      padding=padding, bias=False),
            nn.BatchNorm1d(in_features),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=1,
                      padding=padding, bias=False),
            nn.BatchNorm1d(in_features),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.linear = nn.Linear(in_features, out_features)
        self.conv = nn.Conv1d(in_channels=out_features, out_channels=out_features, kernel_size=kernel_size, stride=1,
                              padding=padding, bias=False)

    def forward(self, x, state=None):
        if isinstance(x, tuple) or isinstance(x, list):
            if len(x) == 3:
                x = x[1]
            elif len(x) == 2:
                x = x[0]
        x = x.transpose(1, 2)

        if self.upsample and not self.upsample_last:
            x = F.interpolate(x, self.upsample_by * x.shape[-1], mode='linear')
        
        x = self.net(x).transpose(1, 2)
        
        if self.upsample and self.upsample_last:
            x = x.transpose(1, 2)
            x = F.interpolate(x, self.upsample_by * x.shape[-1], mode='linear')
            x = x.transpose(1, 2)
        
        x = self.linear(x).transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        if self.output_len > 0:
            len_diff = x.shape[1] - self.output_len
            if len_diff > 0:
                x = x[:, len_diff // 2: - len_diff // 2, :]
        return x


class DoubleConvPhasePickConcatDecoder(nn.Module):
    """
    Phase-pick head that concatenates upsampled features with raw input, as in SeisLM.

    - Expects a tuple (x_feat, x_raw) where:
        x_feat: [B, Lf, C_feat] feature sequence from backbone (pre-classifier)
        x_raw:  [B, L_raw, C_raw] raw waveform (e.g., 3 components)
    - Upsamples features to target length (output_len or upsample_by * Lf),
      optionally aligns raw to the same length, concatenates on channels, and
      applies two conv layers (kernel=3, stride=1, GELU, Dropout) that maintain
      C_feat channels. Then maps to out_features (e.g., 3 classes) and applies
      a final smoothing conv like other phase-pick decoders.
    """

    def __init__(self, in_features, out_features, kernel_size=3, upsample=True,
                 upsample_by=4, output_len=4096, dropout=0.2, manual_input_dim=None):
        super(DoubleConvPhasePickConcatDecoder, self).__init__()
        self.expects_raw = True
        self.upsample = upsample
        self.upsample_by = upsample_by
        self.output_len = output_len

        # Feature channels (C_feat) to maintain through the conv stack
        c_feat = manual_input_dim if manual_input_dim is not None else in_features
        c_raw = 3  # Z, N, E

        assert kernel_size % 2 == 1, 'Kernel size must be uneven'
        padding = int(kernel_size // 2)

        # Two conv layers that keep channel dimension at c_feat (SeisLM-style)
        self.conv1 = nn.Conv1d(
            in_channels=c_feat + c_raw, out_channels=c_feat,
            kernel_size=kernel_size, stride=1, padding=padding, bias=False
        )
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv1d(
            in_channels=c_feat, out_channels=c_feat,
            kernel_size=kernel_size, stride=1, padding=padding, bias=False
        )
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Map per-timestep features to class logits
        self.to_logits = nn.Linear(c_feat, out_features)
        # Optional smoothing conv in class space
        self.smooth = nn.Conv1d(
            in_channels=out_features, out_channels=out_features,
            kernel_size=kernel_size, stride=1, padding=padding, bias=False
        )

    def forward(self, x, state=None):
        # Expect (features, raw)
        if not (isinstance(x, (tuple, list)) and len(x) == 2):
            # Fallback to legacy behaviors if needed
            raise ValueError('DoubleConvPhasePickConcatDecoder expects (features, raw) tuple')

        x_feat, x_raw = x  # [B, Lf, C_feat], [B, L_raw, C_raw]

        # Move to (B, C, L)
        x_feat = x_feat.transpose(1, 2)  # [B, C_feat, Lf]
        x_raw = x_raw.transpose(1, 2)    # [B, C_raw, L_raw]

        # Determine target length
        target_len = x_feat.shape[-1]
        if self.upsample:
            target_len = self.upsample_by * target_len

        if self.output_len and self.output_len > 0:
            target_len = self.output_len

        # Upsample features to target_len if needed
        if x_feat.shape[-1] != target_len:
            x_feat = F.interpolate(x_feat, size=target_len, mode='linear')

        # Align raw to target_len if needed
        if x_raw.shape[-1] != target_len:
            x_raw = F.interpolate(x_raw, size=target_len, mode='linear')

        # Concatenate on channels: [B, C_feat+C_raw, L]
        x_cat = torch.cat([x_feat, x_raw], dim=1)

        # Two conv layers maintaining c_feat channels
        x_cat = self.conv1(x_cat)
        x_cat = self.act1(x_cat)
        x_cat = self.drop1(x_cat)

        x_cat = self.conv2(x_cat)
        x_cat = self.act2(x_cat)
        x_cat = self.drop2(x_cat)

        # Map to logits per time step: (B, C_feat, L) -> (B, L, C_feat) -> linear -> (B, L, n_classes)
        x_cat = x_cat.transpose(1, 2)
        x_cat = self.to_logits(x_cat)
        # Smoothing conv in class space
        x_cat = x_cat.transpose(1, 2)  # (B, n_classes, L)
        x_cat = self.smooth(x_cat)
        x_cat = x_cat.transpose(1, 2)

        # Crop center if length mismatch exists
        if self.output_len and self.output_len > 0:
            len_diff = x_cat.shape[1] - self.output_len
            if len_diff > 0:
                x_cat = x_cat[:, len_diff // 2: - len_diff // 2, :]
        return x_cat

from einops import rearrange

class PatchDecoder(nn.Module):
    def __init__(self, in_features, out_features, patch_size):
        """
        convert x (B, n_patches, in_features) to (B, n_patches * patch_size, out_features)
        """
        super(PatchDecoder, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.patch_size = patch_size

        self.pretraining = True

        # Linear projection layer
        self.projection = nn.Linear(
            in_features=in_features,
            out_features=out_features * patch_size,
            bias=False
        )

    def forward(self, x, state=None):
        """
        :param x: Input tensor of shape (B, n_patches, in_features)
        :return: Output tensor of shape (B, n_patches * patch_size, out_features)
        """
        if self.pretraining:
            x_next, x_prev, y_next, y_prev = x
            
            x_next = self.projection(x_next)  # Apply linear projection
            x_next = rearrange(x_next, 'b n (p c) -> b (n p) c', p=self.patch_size, c=self.out_features)
            x_prev = self.projection(x_prev)  # Apply linear projection
            x_prev = rearrange(x_prev, 'b n (p c) -> b (n p) c', p=self.patch_size, c=self.out_features)
            return x_next, x_prev, y_next, y_prev  # Return the projected patches
        else:
            x_next, x_prev = x
            x_next = self.projection(x_next)  # Apply linear projection
            x_next = rearrange(x_next, 'b n (p c) -> b (n p) c', p=self.patch_size, c=self.out_features)
            x_prev = self.projection(x_prev)  # Apply linear projection
            x_prev = rearrange(x_prev, 'b n (p c) -> b (n p) c', p=self.patch_size, c=self.out_features)
            
            return x_next, x_prev


class PatchPhasePickDecoder(nn.Module):
    def __init__(self, in_features, out_features, patch_size, kernel_size=3, temporal_kernel_size=9, dropout=0.0):
        super(PatchPhasePickDecoder, self).__init__()

        self.patch_size = patch_size

        self.temporal_shift = nn.Conv1d(
            in_channels=in_features,
            out_channels=in_features,
            kernel_size=temporal_kernel_size,
            padding=temporal_kernel_size // 2,
        )

        padding = int(kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=1,
                      padding=padding, bias=False),
            nn.BatchNorm1d(in_features),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, stride=1,
                      padding=padding, bias=False),
            nn.BatchNorm1d(in_features),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.linear = nn.Linear(in_features, out_features)
        self.conv = nn.Conv1d(
            in_channels=out_features, 
            out_channels=out_features, 
            kernel_size=kernel_size, 
            stride=1,
            padding=padding, 
            bias=False
            )

    def forward(self, x, state=None):
        x_forward, x_backward = x

        # align forward and backward information
        x_backward = self.temporal_shift(x_backward.transpose(1, 2)).transpose(1, 2)

        # combine forward and backward information
        x_in = x_forward + x_backward
        x_in = x_in.transpose(1, 2)
        # interpolate to match output sequence length
        x_in = F.interpolate(x_in, self.patch_size * x_in.shape[-1], mode='linear')
        
        # apply the network
        x_in = self.net(x_in).transpose(1, 2)
        x_in = self.linear(x_in).transpose(1, 2)
        x_in = self.conv(x_in).transpose(1, 2)

        return x_in


dec_registry = {
    'dummy': DummyDecoder,
    'linear': LinearDecoder,
    'transformer': SigDecoder,
    's4-decoder': S4Decoder,
    'pool': UpPoolDecoder,
    'embedding': EmbeddingDecoder,
    'phase-pick': PhasePickDecoder,
    'sequence-classifier': SequenceClassifier,
    'causal-decoder': CausalDecoder,
    'convnet-decoder': ConvNetDecoder,
    'bidir-autoreg-decoder': BidirAutoregDecoder,
    'bidir-phasepick-decoder': BidirPhasePickDecoder,
    'bidir-phasepick-decoder-small': BidirPhasePickDecoderSmall,
    'upsampling-decoder': UpsamplingDecoder,
    'causal-upsampling-decoder': CausalUpsamplingDecoder,
    'causal-bidir-autoreg-decoder': CausalBidirAutoregDecoder,
    'sanity-check-decoder': SanityCheckPhasePicker,
    'double-conv-phase-pick': DoubleConvPhasePickDecoder,
    'double-conv-phase-pick-concat': DoubleConvPhasePickConcatDecoder,
    'large-phase-pick-decoder': LargePhasePickDecoder,
    'patch-decoder': PatchDecoder,
    'patch-phase-pick-decoder': PatchPhasePickDecoder,
}

pretrain_decoders = ['transformer', 's4-decoder', 'pool', 'embedding']
phasepick_decoders = ['phase-pick', 'large-phase-pick-decoder',]


def instantiate_decoder(decoder, dataset: SequenceDataset = None, model: nn.Module = None):
    if decoder is None:
        return None

    if decoder._name_ in pretrain_decoders:
        obj = instantiate(dec_registry, decoder)
        return obj

    if dataset is None:
        print('Please specify dataset to instantiate encoder')
        return None

    if model is None:
        print('Please specify model to instantiate encoder')
        return None

    # Allow manual override of input feature dimension for decoders that
    # operate on backbone outputs (useful when model applies a final projection
    # to a different dim than model.d_model).
    manual_in = getattr(decoder, 'manual_input_dim', None)
    # Always pass the backbone width as in_features; let manual_input_dim control projection target
    in_features = model.d_model
    if dataset.num_classes is not None:
        out_features = dataset.num_classes
    else:
        out_features = dataset.d_data

    # Drop helper key to avoid passing it into the constructor
    try:
        from omegaconf import OmegaConf, open_dict
        dec_cfg = OmegaConf.create(decoder)
        # Preserve manual_input_dim for decoders that support it
        keep_manual_for = {
            'sequence-classifier',
            'double-conv-phase-pick',
            'double-conv-phase-pick-concat',
        }
        if getattr(dec_cfg, '_name_', None) not in keep_manual_for:
            with open_dict(dec_cfg):
                if 'manual_input_dim' in dec_cfg:
                    del dec_cfg['manual_input_dim']
    except Exception:
        dec_cfg = decoder

    if dec_cfg._name_ in phasepick_decoders:
        obj = instantiate(dec_registry, dec_cfg, d_model=in_features)
        return obj

    obj = instantiate(dec_registry, dec_cfg, in_features=in_features, out_features=out_features)

    return obj


def instantiate_decoder_simple(decoder, d_data, d_model):
    manual_in = getattr(decoder, 'manual_input_dim', None)
    # Always pass the backbone width as in_features; let manual_input_dim control projection target
    in_features = d_model
    try:
        from omegaconf import OmegaConf, open_dict
        dec_cfg = OmegaConf.create(decoder)
        # Preserve manual_input_dim for decoders that support it
        keep_manual_for = {
            'sequence-classifier',
            'double-conv-phase-pick',
            'double-conv-phase-pick-concat',
        }
        if getattr(dec_cfg, '_name_', None) not in keep_manual_for:
            with open_dict(dec_cfg):
                if 'manual_input_dim' in dec_cfg:
                    del dec_cfg['manual_input_dim']
    except Exception:
        dec_cfg = decoder

    if dec_cfg._name_ in phasepick_decoders:
        obj = instantiate(dec_registry, dec_cfg, d_model=in_features)
        return obj
    obj = instantiate(dec_registry, dec_cfg, in_features=in_features, out_features=d_data)
    return obj


def load_decoder_from_file(decoder_file, dataset: SequenceDataset = None, model=None):
    decoder_state_dict, hparams = torch.load(decoder_file, weights_only=False)
    dec_config = OmegaConf.create(hparams['decoder'])
    decoder = instantiate_decoder(dec_config, dataset=dataset, model=model)
    decoder.load_state_dict(decoder_state_dict)

    # freeze parameters
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False
    return decoder, hparams
