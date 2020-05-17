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
    url='https://github.com/Ariel-Perez/dota-draft',
    install_requires=requirements,
    packages=['draft'],  # Omit test package
    # packages=setuptools.find_packages("src"),  # include all packages under src
    # package_dir={"": "src"}                    # tell distutils packages are under src
)