import json
import requests

import numpy as np
import torch

def RetrieveHeroInfo():
    result = json.loads(requests.get('https://api.opendota.com/api/heroes').text)
    data = {}
    for hero_data in result:
        data[hero_data['id']] = hero_data
    return data


hero_data = RetrieveHeroInfo()
params = torch.load('pytorch-embeddings.pt')
weights = params['embeddings.weight'].numpy()

with open('pytorch-embeddings.tsv', 'w') as f:
    for row in weights:
        f.write('\t'.join([str(w) for w in row]) + '\n')

default_data = {'localized_name': 'None', 'roles': ['None']}
with open('pytorch-embeddings.metadata', 'w') as f:
    f.write('Hero\tRole\n')
    for i, row in enumerate(weights):
        f.write(hero_data.get(i, default_data)['localized_name'] + '\t' +
                hero_data.get(i, default_data)['roles'][0] + '\n')
