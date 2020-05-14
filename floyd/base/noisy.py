"""
Base class for unit groups of model neurons with stochastic noise sources.AGAAAAAAAAG
"""

__all__ = ('NoisyNeuronGroup',)


from toolbox.numpy import *
from specify import Param

from ..state import State, RunMode
from ..noise import OUNoiseProcess

from .neuron import BaseNeuronGroup


class NoisyNeuronGroup(BaseNeuronGroup):

    tau_noise = Param(10.0, units='ms', doc='time-constant of stochastic noise')
    stochastic = Param(False, doc='generate stochastic (O-U process) noise')

    def __init__(self, *args, **kwargs):
        """
        Initialize stochastic unit-noise sources for this neuron group.
        """
        self._initialized = False 
        super().__init__(*args, **kwargs)

        if self.stochastic:
            # Set up the intrinsic noise inputs (current-based only for
            # rate-based neurons). In interactive run mode, generators are used
            # to provide continuous noise.
            self.oup = OUNoiseProcess(N=self.size, tau=self.tau_noise,
                    seed=self.name + '_OU_noise_source')
            if State.run_mode == RunMode.INTERACT:
                self.eta_gen = self.oup.generator()
                self.eta = next(self.eta_gen)
            else:
                self.oup.compute()
                self.eta = self.oup.eta[...,0]
        else:
            self.oup = None
            self.eta = zeros(self.size)

    def update(self):
        """
        Update the model neurons.
        """
        self.update_noise()
        super().update()

    def update_noise(self):
        """
        Update the intrinsic noise sources (for those with nonzero gains).
        """
        if not self.stochastic: return

        if State.run_mode == RunMode.INTERACT:
            self.eta = next(self.eta_gen)
        else:
            self.eta = self.oup.eta[...,State.n]