"""
A helper function to get a default model for quick testing
"""
import os
from omegaconf import open_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

import torch
from cutie.model.cutie import CUTIE
from cutie.inference.utils.args_utils import get_dataset_cfg
from cutie.utils.download_models import download_models_if_needed

_CACHED_MODEL = None


def _get_default_tensor_type_name():
    return str(torch.tensor(0.0).type())


def _with_cpu_default_tensor_type(fn):
    prev_type = _get_default_tensor_type_name()
    switched = prev_type == "torch.cuda.FloatTensor"
    if switched:
        torch.set_default_tensor_type(torch.FloatTensor)
    try:
        return fn(), prev_type, switched
    finally:
        if switched:
            torch.set_default_tensor_type(torch.cuda.FloatTensor)


def get_default_model() -> CUTIE:
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL
    if not GlobalHydra.instance().is_initialized():
        initialize(version_base='1.3.2', config_path="../config", job_name="eval_config")
    cfg = compose(config_name="eval_config")

    weight_dir = download_models_if_needed()
    with open_dict(cfg):
        cfg['weights'] = os.path.join(weight_dir, 'cutie-base-mega.pth')
    get_dataset_cfg(cfg)

    # Load the network weights
    cutie = CUTIE(cfg).cuda().eval()
    model_weights, _, _ = _with_cpu_default_tensor_type(lambda: torch.load(cfg.weights))
    cutie.load_weights(model_weights)
    _CACHED_MODEL = cutie

    return cutie
