#!/usr/bin/env ipython --profile=sharpwaves -i

"""
Script or testing comment goes here.
"""

# import something
from floyd.layout import HexagonalDiscLayout
from inspect import getmro

# do something...
layout = HexagonalDiscLayout(scale=0.05)

print(getmro(HexagonalDiscLayout))

