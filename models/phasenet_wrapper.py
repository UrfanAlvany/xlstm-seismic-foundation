import seisbench
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm

import torch
import torch.nn as nn
import torch.nn.functional as F

#from dataloaders.seisbench_auto_reg import SeisBenchPhasePick
#from tasks.metrics import phase_pick_loss


class PhaseNetWrapper(nn.Module):
    def __init__(self, pretrained: str = None):
        super(PhaseNetWrapper, self).__init__()
        if pretrained is not None:
            self.model = sbm.PhaseNet.from_pretrained(pretrained)
        else:
            self.model = sbm.PhaseNet(phases="PSN", norm="peak")
        self.d_model = 3

    def forward(self, x, state=None, **kwargs):
        """
        Wrapped forward pass. State is ignored and just to comply with training setup.
        Input is transposed before being passed to model and output is transposed back.
        Returns logits, that have to be processed with softmax.
        :param x: [batch_size, 3001, 3]
        :param state: None
        :param kwargs:
        :return: [batch_size, 3001, 3]
        """
        # transpose to get x.shape = [batch_size, 3, 3001]
        x = x.transpose(1, 2)
        return self.model(x, logits=True).transpose(1, 2), None


def phasenet_test():
    data_config = {
        'sample_len': 3001,
        'bits': 0,
        'd_data': 3,
    }

    loader_config = {
        'batch_size': 4,
        'num_workers': 0,
        'shuffle': True,
    }

    model = PhaseNetWrapper()
    dataset = SeisBenchPhasePick(**data_config)
    loader = dataset.train_dataloader(**loader_config)

    batch = next(iter(loader))
    x, y = batch['X'], batch['y']
    print(f'x.shape: {list(x.shape)}', f'y.shape: {list(y.shape)}')
    with torch.no_grad():
        out, _ = model(x)
    print(f'out.shape: {list(out.shape)}')

    loss = phase_pick_loss(out, y)
    print(f'loss: {loss}')


if __name__ == '__main__':
    phasenet_test()
