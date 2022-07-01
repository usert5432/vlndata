#!/usr/bin/env python

import setuptools

def readme():
    with open('README.md') as f:
        return f.read()

setuptools.setup(
    name             = 'vlndata',
    version          = '0.0.1-alpha',
    author           = 'Dmitrii Torbunov',
    author_email     = 'dtorbunov@bnl.gov',
    classifiers      = [
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
    ],
    description      = \
        'Data primitives to work with columnar data and VL data',
    install_requires = [
        'numpy',
        'pandas',
        'h5py',
    ],
    license          = 'MIT',
    long_description = readme(),
    packages         = setuptools.find_packages(
        include = [ 'vlndata', 'vlndata.*' ]
    ),
)

