"""Module for using a pretrained model to run eval on one example."""
from typing import Dict

import argparse
import os

import torch
import wandb
import yaml

from draft.data.api import Api
from draft.data.hero import Hero
from draft.model.mlp import MLP, MLPConfig
from draft.model.wrapper import ModelWrapper, ModelWrapperConfig
from draft.training.argument import Arguments, read_config


CONFIG_FILE = 'config.yaml'
ROOT_DIR = '/tmp'


def BuildHeroMap() -> Dict[str, Hero]:
    """Build a mapping from hero's name to their data."""
    result = Api().heroes()
    data = {}
    for hero in result:
        data[hero.localized_name] = hero
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model to predict the winning team.')
    parser.add_argument('--checkpoint', required=True, help='Name of the saved checkpoint')
    parser.add_argument('--run_path', required=True, help='Path to the saved run')
    args = parser.parse_args()
    hero_from_name = BuildHeroMap()

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

    ## Validate
    model.eval()

    matches = [
        # TI11 Grand Finals
        [
            # Radiant
            'Tusk', 'Mirana', 'Hoodwink', 'Naga Siren', 'Tidehunter',
            # Dire
            'Leshrac', 'Tiny', 'Enigma', 'Lich', 'Pudge',
        ],
        [
            # Radiant
            'Marci', 'Visage', 'Chaos Knight', 'Phoenix', 'Arc Warden',
            # Dire
            'Leshrac', 'Tusk', 'Chen', 'Bristleback', 'Morphling',
        ],
        [
            # Radiant team
            'Tiny', 'Mirana', 'Pangolier', 'Beastmaster', 'Medusa',
            # Dire team
            'Marci', 'Leshrac', 'Silencer', 'Naga Siren', 'Ember Spirit',
        ],
    ]

    # Radiant team is better than dire, they should output a high probability
    match_hero_ids = [
        [hero_from_name[hero_name]._id for hero_name in match]
        for match in matches
    ]
    draft = torch.tensor(match_hero_ids, dtype=torch.long)
    results = model(draft).detach().numpy()
    for match, result in zip(matches, results):
        print('Radiant:')
        for hero in match[:5]:
            print('- {hero}'.format(hero=hero))
        print('Dire:')
        for hero in match[5:]:
            print('- {hero}'.format(hero=hero))
        print('Odds:')
        print(result)
