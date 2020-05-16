import setuptools


setuptools.setup(
    name='dota-draft',
    version='1.0',
    description='Dota draft package',
    author='Ariel Perez',
    author_email='arielperezch@gmail.com',
    url='https://github.com/Ariel-Perez/dota-draft',
    install_requires=[
        'google-cloud-storage==1.28.0',
        'requests==2.23.0',
    ],
    packages=setuptools.find_packages("src"),  # include all packages under src
    package_dir={"": "src"}                    # tell distutils packages are under src
)