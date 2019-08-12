from os import getenv
from os.path import dirname, realpath
from setuptools import find_packages, setup


setup(
    name='mae_envs',
    version='0.0.0',
    packages=find_packages(),
    package_data={
        '': ['*.pyx', '*.pxd', '*.pxi', '*.h'],
    })
