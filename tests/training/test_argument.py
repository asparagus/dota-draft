import wandb

from draft.training.argument import Argument, read_config


def test_read_clean_config():
    arg = Argument(name='arg', type=int, default=None)
    clean_config = {'arg': 2}
    assert read_config(arg, clean_config) == 2


def test_read_wandb_format_config():
    arg = Argument(name='arg', type=int, default=None)
    clean_config = {'arg': {'desc': 'an argument', 'value': 2}}
    assert read_config(arg, clean_config) == 2


def test_read_wandb_config():
    arg = Argument(name='arg', type=int, default=None)
    config = wandb.Config()
    config['arg'] = 2
    assert read_config(arg, config) == 2