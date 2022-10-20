import argparse

import numpy as np
import torch
import wandb

from draft.data import api
from draft.model.simplemodel import SimpleModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model to predict the winning team.')
    parser.add_argument('--checkpoint', required=True, help='Name of the saved checkpoint')
    parser.add_argument('--run_path', required=True, help='Path to the saved run')
    args = parser.parse_args()
    torch.manual_seed(1)

    def RetrieveHeroMap():
        result = api.Api().heroes()
        data = {}
        for hero_data in result:
            data[hero_data['localized_name']] = hero_data
        return data

    hero_from_name = RetrieveHeroMap()

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
    draft = torch.tensor([
        [hero_from_name.get(name)['id']
         for name in heroes]
    ], dtype=torch.long)
    print('Radiant:')
    for hero in heroes[:5]:
        print('- {hero}'.format(hero=hero))
    print('Dire:')
    for hero in heroes[5:]:
        print('- {hero}'.format(hero=hero))
    print('Odds:')
    print(model(draft).detach())
