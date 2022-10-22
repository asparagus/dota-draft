import setuptools


# Read from requirements.txt for consistency
requirements = []
try:
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
except:
    pass


setuptools.setup(
    name='dota-draft',
    version='1.0',
    description='Dota draft package',
    author='Ariel Perez',
    author_email='arielperezch@gmail.com',
    url='https://github.com/asparagus/dota-draft',
    install_requires=requirements,
    packages=['draft', 'draft.data', 'draft.model', 'draft.training'],
)