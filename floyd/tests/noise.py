#!/usr/bin/env ipython --profile=sharpwaves -i

"""
Test noise module for changes in multiple inheritance.
"""

# import something
from floyd.noise import OUNoiseProcess
from inspect import getmro

# do something...
oup = OUNoiseProcess(N=36, tau=17.3)

print(oup)
print(getmro(OUNoiseProcess))

