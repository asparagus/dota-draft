from typing import Dict

import argparse

import torch
import wandb
import yaml

from draft.data.api import Api
from draft.data.hero import Hero
from draft.model.mlp import MLP, MLPConfig
from draft.model.wrapper import ModelWrapper, ModelWrapperConfig


CONFIG_FILE = "config.yaml"
ROOT_DIR = "/tmp"


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
    ckpt_file = wandb.restore(args.checkpoint, run_path=args.run_path, root=ROOT_DIR)
    config_file = wandb.restore(CONFIG_FILE, run_path=args.run_path, root=ROOT_DIR)
    with open(config_file.name, 'r') as f:
        config = yaml.safe_load(f)
    mlp_config = MLPConfig(num_heroes=138, layers=[32, 16])
    wrapper_config = ModelWrapperConfig(symmetric=True)
    module = MLP(mlp_config)
    model = ModelWrapper(
        config=ModelWrapperConfig(**config['model']['value']),
        module=module,
    )
    ckpt = torch.load(ckpt_file.name)
    model_ckpt = ckpt['state_dict']
    model.load_state_dict(model_ckpt)

    ## Validate
    model.eval()

    heroes = [
        # Radiant team
        'Tiny',
        'Mirana',
        'Pangolier',
        'Beastmaster',
        'Medusa',
        # Dire team
        'Marci',
        'Leshrac',
        'Silencer',
        'Naga Siren',
        'Ember Spirit',
    ]

    # Radiant team is better than dire, they should output a high probability
    hero_ids = [hero_from_name[hero_name]._id for hero_name in heroes]
    draft = torch.tensor([hero_ids], dtype=torch.long)
    print('Radiant:')
    for hero in heroes[:5]:
        print('- {hero}'.format(hero=hero))
    print('Dire:')
    for hero in heroes[5:]:
        print('- {hero}'.format(hero=hero))
    print('Odds:')
    print(model(draft).detach())
