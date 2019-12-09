"""
Groups of synaptic connections between a pair of rate-based neuron groups.
"""

from toolbox.numpy import *
from specify import Slider

from ..matrix import pairwise_distances as distances
from ..state import State

from ..synapses import Synapses


class RateSynapses(Synapses):

    transmitter = 'glutamate'
    I_max       = Slider(default=1.0, start=0.0, end=1e1, step=1.0, doc='maximum synaptic current input')

    base_variables = ('C', 'S', 'I', 'I_peak', 'g_peak')

    def __init__(self, pre, post, **kwargs):
        self._initialized = False
        super().__init__(pre, post, **kwargs)

        self.pre = pre
        self.post = post
        self.recurrent = pre is post

        self.delay = None
        self.I_peak = self.I_max
        self.g_peak = self.I_peak  # NOTE: hack to mirror g_peak for NetworkGraph
        self.I_total = zeros(post.N, 'f')
        self.distances = distances(c_[post.x, post.y], c_[pre.x, pre.y])

        self._initialized = True

    def set_delays(self, cond_velocity=0.5):
        """
        Construct spike-transmission conduction delay lines.
        """
        if not hasattr(self, 'active_post'):
            raise RuntimeError('synapses must be connected first!')
        timing = self.distances / cond_velocity
        self.delay = DelayLines.from_delay_times(self.N, timing[self.ij],
                State.dt, dtype='f')
        self.out('max delay = {} timesteps', self.delay.delays.max())

    # TODO: Update this for the new current-routing strategy of net vs. total

    def update(self):
        """
        Update synaptic timing based on the current presynaptic rates.
        """
        syn_rates = pre_rates = self.pre.r[self.j]
        if self.delay is not None:
            self.delay.set(pre_rates)
            syn_rates = self.delay.get()

        self.I[self.ij] = self.I_peak[self.ij]*self.S[self.ij]*syn_rates
        self.I_total[:] = self.I.sum(axis=1)
