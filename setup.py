import os
import setuptools


DOTA_DRAFT = 'dota-draft'

# Install a single environment by setting the ENVIRONMENT variable.
# See environments/README.md
environment = os.getenv('ENVIRONMENT') or DOTA_DRAFT

# WORKDIR is set by the Docker images, otherwise use current setup.
workdir = os.getenv('WORKDIR') or os.getcwd()

# Read from requirements.txt for consistency
dirs = ','.join(os.listdir('.'))
environments_dir = os.path.join(workdir, 'environments')
requirements_dir = os.path.join(environments_dir, environment)
requirements_path = os.path.join(requirements_dir, 'requirements.txt')
if not os.path.exists(environments_dir):
    raise RuntimeError(f'No environments folder within {workdir} with dirs ({dirs})')
if not os.path.exists(requirements_dir):
    raise RuntimeError(f'Could not find the environment {environment} within {workdir} with dirs ({dirs})')
if not os.path.exists(requirements_path):
    raise RuntimeError(f'Could not find the requirements file at {requirements_path} within {workdir} with dirs ({dirs})')

try:
    with open(requirements_path) as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    raise RuntimeError(f'Invalid environment {environment}, could not install dependencies.')


setuptools.setup(
    name=f'{DOTA_DRAFT}-{environment}' if environment != DOTA_DRAFT else DOTA_DRAFT,
    version='1.0',
    description='Dota draft package',
    author='Ariel Perez',
    author_email='arielperezch@gmail.com',
    url='https://github.com/asparagus/dota-draft',
    install_requires=requirements,
    packages=['draft'],
    package_data={'draft': ['configs/*.yaml']},
)
