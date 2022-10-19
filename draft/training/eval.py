import argparse

import numpy as np
import torch

from draft.data import api
from draft.model.net import Net


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model to predict the winning team.')
    parser.add_argument('--checkpoint', required=True, help='Path to saved checkpoint')
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
    params = torch.load(args.checkpoint)
    m = Net.load_from_checkpoint(
        args.checkpoint,
        # This should come from the config
        num_heroes=138,
        dimensions=[256, 256, 256],
    )

    ## Validate
    m.eval()

    draft = [
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
         for name in draft]
    ], dtype=torch.long)
    print(sigmoid(m(draft).detach()))
