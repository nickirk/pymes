from setuptools import setup, find_packages

setup(
    name='pymes',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'spglib',
        # add any other necessary libraries here
    ],
    package_data={
        '': ['*.txt', '*.md'],
        'pymes': ['lib/ctf/*'],
    },
    include_package_data=True,
)