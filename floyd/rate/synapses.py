"""
Current-based synaptic projection for rate-based neuron groups.
"""

from toolbox.numpy import *
from specify import Slider

from ..base import BaseProjection


class RateBasedSynapses(BaseProjection):

    I_max = Slider(0.1, start=-1, end=1, step=0.05, doc='max. synaptic input')

    extra_variables = ('I', 'I_peak')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.I_peak = self.I_max
        self.I_total = zeros(self.post.size, self.signal_dtype)

    def update(self):
        """
        Update synaptic timing based on the current presynaptic rates.
        """
        super().update()

        self.I[self.ij] = self.I_peak[self.ij] * self.S[self.ij] * self.terminal[self.ij]
        self.I_total[:] = self.I.sum(axis=1)
