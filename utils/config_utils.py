from typing import Sequence, Mapping, Optional, Callable
import functools
import hydra
import omegaconf
from omegaconf import ListConfig, DictConfig


def is_list(x):
    return isinstance(x, Sequence) and not isinstance(x, str)


def to_list(x, recursive=False):
    """Convert an object to list.

    If Sequence (e.g. list, tuple, Listconfig): just return it

    Special case: If non-recursive and not a list, wrap in list
    """
    if is_list(x):
        if recursive:
            return [to_list(_x) for _x in x]
        else:
            return list(x)
    else:
        if recursive:
            return x
        else:
            return [x]


def instantiate(registry: dict, config, *args, partial=False, **kwargs):
    if config is None:
        return None
    if isinstance(config, str):
        _name_ = None
        _target_ = registry[config]
        config_copy = {}
    else:
        _name_ = config['_name_']
        config_copy = config.copy()
        with omegaconf.open_dict(config_copy):
            del config_copy['_name_']
        _target_ = registry[_name_]

    if isinstance(_target_, str):
        fn = hydra.utils.get_method(path=_target_)
    elif isinstance(_target_, Callable):
        fn = _target_
    else:
        raise NotImplementedError(f"Unsupported type {type(_target_)}")

    obj = functools.partial(fn, *args, **config_copy, **kwargs)



    if partial:
        return obj
    else:
        return obj()