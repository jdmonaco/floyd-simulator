#!/usr/bin/env ipython --profile=sharpwaves -i

"""
Testing multiple inheritance and Spec init in neuron group classes.
"""

import inspect
from inspect import getmro, signature

from roto.null import Null

from floyd.groups import BaseUnitGroup
from floyd.neurons.coba import COBANeuronGroup
from floyd.neurons.adex import AEIFNeuronGroup
from floyd.state import State


class DummyContext(object):
    g_ASD_SDF = 1.0

State.context = DummyContext()
State.network = Null

grp = COBANeuronGroup(name='test', N=17, C_m=273.0)

print(grp)
print(grp.spec)

