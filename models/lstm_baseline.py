import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


class LSTMBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            residual: bool = True,
            gating: bool = True,
            layer_norm: bool = True
    ):
        super(LSTMBlock, self).__init__()
        self.d_model = d_model
        self.residual = residual
        self.gating = gating
        self.layer_norm = layer_norm

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, state=None):
        """
        Forward pass for batched inputs
        :param x: [batch_size, seq_len, d_model]
        :param state: (hidden_state, cell state) e.g. (h_0, c_0). Defaults to zero if not provided
        :return: output, output_state
        """
        if self.layer_norm:
            res = self.norm(x)
        else:
            res = x
        out, out_state = self.lstm(res, state)
        if self.gating:
            out = out * F.silu(self.linear(res))
        if self.residual:
            out = out + res

        return out, out_state

    def step(self, x, state=None):
        """
        Step for sequential computation.
        :param x: [batch_size, 1, d_model] or [batch_size, d_model]
        :param state: (hidden_state, cell_state) e.g. (h_0, c_0). Defaults to zero
        :return: output, output_state
        """
        # make sure input is of shape [batch_size, 1, d_model]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        assert x.shape[1] == 1

        out, out_state = self.forward(x, state)
        return out.squeeze(1), out_state

    def default_state(self, batch_size: int = 0, device='cpu'):
        if batch_size > 0:
            return torch.zeros(1, batch_size, self.d_model), torch.zeros(1, batch_size, self.d_model).to(device)
        else:
            return torch.zeros(1, self.d_model), torch.zeros(1, self.d_model).to(device)


class LSTMSequenceModel(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layers: int,
            residual: bool = True,
            gating: bool = True,
            layer_norm: bool = True
    ):
        super(LSTMSequenceModel, self).__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.net = nn.Sequential(
            *[LSTMBlock(
                d_model=self.d_model,
                residual=residual,
                gating=gating,
                layer_norm=layer_norm
            ) for _ in range(self.n_layers)]
        )

    def forward(self, x, state=None):
        if state is None:
            #state = []
            for i in range(self.n_layers):
                x, s = self.net[i](x, None)
                #state.append(s)
        else:
            for i in range(self.n_layers):
                x, state[i] = self.net[i](x, state[i])
        return x, state

    def step(self, x, state=None):
        for i in range(self.n_layers):
            x, state[i] = self.net[i].step(x, state=state[i])
        return x, state

    def default_state(self, batch_size: int = 0, device='cpu'):
        return [layer.default_state(batch_size, device) for layer in self.net]


def lstm_test(norm, gating, residual):
    print('Testing LSTM...')
    print('Norm:', norm)
    print('Gating:', gating)
    print('Residual:', residual, '\n')
    d_model = 64
    batch_size = 16
    n_layers = 16
    seq_len = 1000

    model = LSTMSequenceModel(d_model=d_model, n_layers=n_layers, residual=residual, gating=gating, layer_norm=norm)

    x = torch.randn(batch_size, seq_len, d_model)
    with torch.no_grad():
        out, hidden = model(x, None)

    print('\nBatched forward pass')
    print('out:', out.shape)
    if hidden is not None:
        pass
        # print('hidden: ', hidden[0][0].shape, hidden[0][1].shape)
    else:
        print('hidden:', None)

    ys = []
    state = model.default_state(batch_size=batch_size)
    with torch.no_grad():
        for i in range(seq_len):
            o, state = model.step(x[:, i, :], state)
            ys.append(o)
    ys = torch.stack(ys, dim=1)

    print('\nSequential')
    print('ys:', ys.shape)
    print('state:', state[0][0].shape, state[0][1].shape)

    err = F.mse_loss(ys, out)
    print('\nERROR:', err, '\n\n')
    return err


def all_config_test():
    permutations = list(itertools.product([True, False], repeat=3))

    errors = []
    for p in permutations:
        errors.append(lstm_test(p[0], p[1], p[2]))

    print('All errors:', errors)


if __name__ == '__main__':
    #all_config_test()
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    model = LSTMSequenceModel(d_model=64, n_layers=32)
    print(count_parameters(model))
