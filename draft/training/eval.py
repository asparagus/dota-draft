"""Module for using a pretrained model to run eval on one example."""
from typing import Dict

import argparse

import numpy as np
import onnxruntime as ort
import wandb

from draft.data.api import Api
from draft.data.hero import Hero


CONFIG_FILE = 'config.yaml'
ROOT_DIR = '/tmp'

INPUT_NAME = 'processed.1'


def BuildHeroMap() -> Dict[str, Hero]:
    """Build a mapping from hero's name to their data."""
    result = Api().heroes()
    data = {}
    for hero in result:
        data[hero.localized_name] = hero
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the model to predict the winning team.')
    parser.add_argument('model_artifact_id', help='Artifact id for the saved model')
    args = parser.parse_args()
    hero_from_name = BuildHeroMap()

    ## Retrieve the onnx model
    wandb_api = wandb.Api()
    artifact = wandb_api.artifact(args.model_artifact_id)
    onnx_path = artifact.get_path('model.onnx').download()
    ort_session = ort.InferenceSession(onnx_path)

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
        np.array([hero_from_name[hero_name]._id for hero_name in match]).astype(np.int32)
        for match in matches
    ]
    results = [
        ort_session.run(None, {INPUT_NAME: np.expand_dims(hero_ids, axis=0)})
        for hero_ids in match_hero_ids
    ]
    for match, result in zip(matches, results):
        print('Radiant:')
        for hero in match[:5]:
            print('- {hero}'.format(hero=hero))
        print('Dire:')
        for hero in match[5:]:
            print('- {hero}'.format(hero=hero))
        print('Odds:')
        print(result)
