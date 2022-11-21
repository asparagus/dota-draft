"""Module for defining arguments to be used for the model training."""
import attrs
from typing import Any, Dict, List, Optional

import wandb


@attrs.define
class Argument:
    name: str
    type: type
    default: Any


class Arguments:
    """This class just holds the arguments that can be used and assigns a name to them."""
    DATA_ARTIFACT_ID = Argument('data.artifact_id', str, default=None)
    DATA_BATCH_SIZE = Argument('data.batch_size', int, default=256)
    REPRODUCIBILITY_SEED = Argument('reproducibility.seed', int, default=1)
    MODEL_NUM_HEROES = Argument('model.num_heroes', int, default=138)
    MODEL_EMBEDDING_SIZE = Argument('model.embedding_size', int, default=64)
    MODEL_SYMMETRIC = Argument('model.symmetric', bool, default=True)
    MODEL_LEARNING_RATE = Argument('model.learning_rate', float, default=1e-4)
    MODEL_LAYERS = Argument('model.layers', List[int], default=[32])
    MODEL_WEIGHT_DECAY = Argument('model.weight_decay', float, default=1e-4)


def read_config(arg: Argument, config: Optional[Dict] = None):
    """Read an argument value from config.

    If a config is not passed, the wandb run's config is used.

    Args:
        arg: Argument to retrieve from the config
        config: (Optional) Dict with config values
    """
    if config is None:
        config = wandb.run.config
    return extract_value(config, arg)


def extract_value(config: Dict, arg: Argument):
    """Extract a value from a config.

    The dictionary might either be formatted as either of these two:

    option1 = {
        key1: value1,
        key2: value2,
    }
    option2 = {
        key1: {'desc': desc1, 'value': value1},
        key2: {'desc': desc2, 'value': value2},
    }

    Args:
        config: Dict of config values
        arg: Argument to retrieve
    """
    if arg.name not in config and arg.default is None:
        raise IndexError(f'Missing required {arg.name} in config')
    content = config.get(arg.name, arg.default)
    if isinstance(content, dict):
        return content['value']
    return content
