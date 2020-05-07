"""
Groups of synaptic connections for pairs of rate-based neuron groups.
"""

from toolbox.numpy import *
from specify import Slider

from ..base import BaseProjection


class RateSynapses(BaseProjection):

    I_max = Slider(1.0, start=0.0, end=1e1, step=1.0, doc='max. synaptic input')
    signal_dtype = 'f'

    extra_variables = ('I', 'I_peak', 'g_peak')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.I_peak = self.I_max
        self.g_peak = self.I_peak  # NOTE: hack to mirror g_peak for NetworkGraph
        self.I_total = zeros(self.post.size, self.signal_dtype)

    def update(self):
        """
        Update synaptic timing based on the current presynaptic rates.
        """
        super().update()
        self.I[self.ij] = self.I_peak[self.ij] * self.S[self.ij] * \
                                self.terminal[self.ij]
        self.I_total[:] = self.I.sum(axis=1)
