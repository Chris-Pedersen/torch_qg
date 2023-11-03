#!/usr/bin/env python

from setuptools import setup, find_packages

description = "Modelling of a 2-layer quasi-geostrophic system in pytorch 2.0"
version="0.0.1"

setup(name="torch_qg",
    version=version,
    description=description,
    url="https://github.com/Chris-Pedersen/torch_qg",
    author="Chris Pedersen",
    author_email="c.pedersen@nyu.edu",
    packages=find_packages(),
    )
