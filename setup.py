import setuptools


setuptools.setup(
    name='dota-draft',
    version='1.0',
    install_requires=[
        'google-cloud-storage==1.28.0',
        'requests==2.23.0',
    ],
    packages=setuptools.find_packages(),
)