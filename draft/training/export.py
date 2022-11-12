"""Script for exporting a pretrained model."""
import argparse
import os

import torch
import wandb
import yaml

from draft.model.mlp import MLP, MLPConfig
from draft.model.wrapper import ModelWrapper, ModelWrapperConfig
from draft.training.argument import Arguments, read_config


CONFIG_FILE = 'config.yaml'
ROOT_DIR = '/tmp'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model to predict the winning team.')
    parser.add_argument('output', help='Output path')
    parser.add_argument('--checkpoint', required=True, help='Name of the saved checkpoint')
    parser.add_argument('--run_path', required=True, help='Path to the saved run')
    args = parser.parse_args()

    ## Define & Load
    root_dir = os.path.join(ROOT_DIR, args.run_path)
    ckpt_file = wandb.restore(args.checkpoint, run_path=args.run_path, root=root_dir)
    config_file = wandb.restore(CONFIG_FILE, run_path=args.run_path, root=root_dir)
    with open(config_file.name, 'r') as f:
        config = yaml.safe_load(f)
    mlp_config = MLPConfig(
        num_heroes=read_config(Arguments.MODEL_NUM_HEROES, config=config),
        layers=read_config(Arguments.MODEL_LAYERS, config=config),
    )
    wrapper_config = ModelWrapperConfig(
        symmetric=read_config(Arguments.MODEL_SYMMETRIC, config=config),
    )
    module = MLP(mlp_config)
    model = ModelWrapper(
        config=wrapper_config,
        module=module,
    )
    ckpt = torch.load(ckpt_file.name)
    model_ckpt = ckpt['state_dict']
    model.load_state_dict(model_ckpt)
    model_args = torch.ones((1, 10)).int()
    torch.onnx.export(model.to_torchscript(), model_args, args.output)
