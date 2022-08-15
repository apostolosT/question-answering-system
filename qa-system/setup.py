from glob import glob
from os.path import splitext, basename
from setuptools import find_packages, setup

setup(
    name='qa-system',
    packages=find_packages('.'),
    version='0.1.0',
    description='Question Answering system for AISERA interview project',
    author='Apostolos Tamvakis',
    license='MIT',
)
