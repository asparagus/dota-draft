from typing import Dict

import argparse

import torch
import wandb

from draft.data.api import Api
from draft.data.hero import Hero
from draft.model.simplemodel import SimpleModel


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
    torch.manual_seed(1)

    hero_from_name = BuildHeroMap()

    ## Define & Load
    wandb.restore(args.checkpoint, run_path=args.run_path)
    model = SimpleModel.load_from_checkpoint(
        args.checkpoint,
    )

    ## Validate
    model.eval()

    heroes = [
        # Radiant team
        'Huskar',
        'Dazzle',
        'Windranger',
        'Sven',
        'Drow Ranger',
        # Dire team
        'Rubick',
        'Chen',
        'Skywrath Mage',
        'Pangolier',
        'Shadow Demon',
    ]

    # Radiant team is better than dire, they should output a high probability
    hero_ids = [hero_from_name.get(hero_name)._id for hero_name in heroes]
    draft = torch.tensor(hero_ids, dtype=torch.long)
    print('Radiant:')
    for hero in heroes[:5]:
        print('- {hero}'.format(hero=hero))
    print('Dire:')
    for hero in heroes[5:]:
        print('- {hero}'.format(hero=hero))
    print('Odds:')
    print(model(draft).detach())
