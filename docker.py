"""Script for building and pushing docker images."""
import os

from draft.providers import GAR


ENVIRONMENTS = [
    'collect',
    'compact',
    'compact-worker',
    'dota-draft',
    'eval',
    'train',
]


if __name__ == '__main__':
    for env in ENVIRONMENTS:
        os.system(f'docker build -t {env}:latest . -f environments/{env}/Dockerfile')
        os.system(f'docker tag {env} {GAR.location}-docker.pkg.dev/{GAR.project}/{GAR.repository}/{env}:latest')
        os.system(f'docker push {GAR.location}-docker.pkg.dev/{GAR.project}/{GAR.repository}/{env}:latest')
