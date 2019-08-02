"""
Groups of synaptic connections between a pair of neuron groups.
"""

from toolbox.numpy import *
from tenko.mixins import RandomMixin

from ..matrix import pairwise_distances as distances
from ..groups import BaseUnitGroup
from ..spec import paramspec
from ..state import State

from ..synapses import Synapses

class RateSynapses(Synapses):

    @classmethod
    def get_spec(cls, return_factory=False, **keyvalues):
        """
        Return a Spec default factory function or instance with updated values.
        """
        if not hasattr(cls, '_Spec'):
            cls._Spec = paramspec(f'{cls.__name__}Spec',
                transmitter = 'glutamate',
                I_max       = 1.0,
            )
        if return_factory:
            return cls._Spec
        return cls._Spec(**keyvalues)

    base_variables = ('C', 'S', 'I', 'I_peak')

    def __init__(self, pre, post, spec, seed=None):
        self.get_spec().is_valid(spec)

        name = f'{pre.name}->{post.name}'
        BaseUnitGroup.__init__(self, (post.N, pre.N), name, spec=spec)

        self.pre = pre
        self.post = post
        self.recurrent = pre is post
        self.set_random_seed(seed)

        self.delay = None
        self.I_peak = self.p.I_max
        self.I_total = zeros(post.N, 'f')
        self.distances = distances(c_[post.x, post.y], c_[pre.x, pre.y])

        State.network.add_synapses(self)

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
