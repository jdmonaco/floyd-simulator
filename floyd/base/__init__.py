"""
Base classes for neuron groups and synapses to connect them together.
"""

from .groups import BaseUnitGroup
from .input import InputGroup
from .neuron import BaseNeuronGroup
from .noisy import NoisyNeuronGroup
from .projection import BaseProjection