import os
import setuptools


DOTA_DRAFT = 'dota-draft'

# Install a single environment by setting the ENVIRONMENT variable.
# See environments/README.md
environment = os.getenv('ENVIRONMENT') or DOTA_DRAFT

# Read from requirements.txt for consistency
requirements_path = os.path.join('environments', environment, 'requirements.txt')
try:
    with open(requirements_path) as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    raise ValueError(f'Invalid environment {environment}, could not install dependencies.')


PACKAGE_DICT = {
    'collect': ['draft.data'],
    'compact': ['draft.data'],
    'training': ['draft.data', 'draft.training', 'draft.model'],
    DOTA_DRAFT: ['draft.data', 'draft.training', 'draft.model'],
}


setuptools.setup(
    name=f'{DOTA_DRAFT}-{environment}' if environment != DOTA_DRAFT else DOTA_DRAFT,
    version='1.0',
    description='Dota draft package',
    author='Ariel Perez',
    author_email='arielperezch@gmail.com',
    url='https://github.com/asparagus/dota-draft',
    install_requires=requirements,
    packages=PACKAGE_DICT.get(environment),
)
