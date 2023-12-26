"""
Setup installation file for floyd.
"""

from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="Floyd",
    version="0.2.3",
    description="Floyd - Declarative Neuronal Network Simulator",
    long_description=long_description,
    url="https://github.com/jdmonaco/floyd",
    author="Joseph Monaco",
    author_email="joe@selfmotion.net",
    license="MIT",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    packages=["floyd"],
)
