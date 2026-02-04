import torch.nn as nn

try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None


def add_optimizer_hooks(
        model,
        bias_weight_decay=False,
        normalization_weight_decay=False,
):
    """Handle special optimizer logic by setting _optim attribute.

    Set weight_decay=0.0 for parameters in model.no_weight_decay, for parameters with
    attribute _no_weight_decay==True, for bias parameters if bias_weight_decay==False, for
    normalization parameters if normalization_weight_decay==False
    """

    # Separate out all parameters to those that will and won't experience regularizing weight decay
    blacklist_weight_modules = (nn.Embedding,)
    if not normalization_weight_decay:
        # Build list of normalization modules, filtering out None values
        norm_modules = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                       # Not compatible with Pytorch 1.8.1
                       # nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,
                       nn.GroupNorm, nn.SyncBatchNorm,
                       nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                       nn.LayerNorm, nn.LocalResponseNorm]
        
        # Add mamba modules only if they were successfully imported (not None)
        if RMSNormGated is not None:
            norm_modules.append(RMSNormGated)
        if LayerNorm is not None:
            norm_modules.append(LayerNorm)
            
        blacklist_weight_modules += tuple(norm_modules)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            if (not bias_weight_decay and pn.endswith('bias')) \
                    or getattr(p, '_no_weight_decay', False) \
                    or isinstance(m, blacklist_weight_modules):
                setattr(p, "_optim", {"weight_decay": 0.0})


def print_optim(optimizer, keys):
    """ Log values of particular keys from the optimizer's param groups """
    keys = sorted(keys)
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        n_params = sum(p.numel() for p in g['params'])
        print(' | '.join([
                             f"Optimizer group {i}",
                             f"{len(g['params'])} tensors",
                             f"{n_params} parameters",
                         ] + [f"{k} {v}" for k, v in group_hps.items()]))
