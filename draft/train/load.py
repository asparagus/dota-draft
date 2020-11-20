import argparse

import numpy as np
import torch

from draft import api
from draft.train import model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model to predict the winning team.')
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
    embedding_dim = params['embeddings.embeddings.weight'].shape[1]
    m = model.Model(130, embedding_dim)
    m.load_state_dict(params)

    ## Validate
    m.eval()

    draft = [
        'Lifestealer',
        'Enchantress',
        'Skywrath Mage',
        'Pangolier',
        'Slardar',
        'Rubick',
        'Crystal Maiden',
        'Magnus',
        'Axe',
        'Troll Warlord',
    ]
    draft = torch.tensor([
        [hero_from_name.get(name)['id']
         for name in draft]
    ], dtype=torch.long)
    print(np.exp(m(draft).detach()))
