from setuptools import setup, find_packages

setup(
    name='pymes',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'spglib',
        'cython==0.29.37',
        'h5py'
        # add any other necessary libraries here
    ],
    package_data={
        '': ['*.txt', '*.md'],
        'pymes': ['lib/ctf/*'],
    },
    include_package_data=True,
)