from setuptools import find_packages, setup


setup(
    name='rusp',
    version='0.0.0',
    packages=find_packages(),
    package_data={
        '': ['*.pyx', '*.pxd', '*.pxi', '*.h'],
    })
