import attrs
from typing import Any, Dict, List, Optional

import wandb


@attrs.define
class Argument:
    name: str
    type: type
    default: Any


class Arguments:
    DATA_PATH = Argument('data.path', str, default=None)
    DATA_BATCH_SIZE = Argument('data.batch_size', int, default=256)
    REPRODUCIBILITY_SEED = Argument('reproducibility.seed', int, default=1)
    MODEL_NUM_HEROES = Argument('model.num_heroes', int, default=138)
    MODEL_SYMMETRIC = Argument('model.symmetric', bool, default=True)
    MODEL_LEARNING_RATE = Argument('model.learning_rate', float, default=1e-3)
    MODEL_LAYERS = Argument('model.layers', List[int], default=[32, 16])
    MODEL_WEIGHT_DECAY = Argument('model.weight_decay', float, default=1e-5)


def read_config(arg: Argument, config: Optional[Dict] = None):
    if config is None:
        config = wandb.run.config
    return extract_value(config, arg.name)


def extract_value(config: Dict, key: str):
    if isinstance(config[key], dict):
        return config[key]['value']
    return config[key]
