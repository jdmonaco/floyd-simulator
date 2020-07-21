"""
Base class for current-based (CUBA) rate-coding neuron groups.
"""

from toolbox.numpy import *
from specify import Slider

from ..base import BaseNeuronGroup
from ..state import State


class RateBasedNeuronGroup(BaseNeuronGroup):

    tau_m     = Slider(10.0, start=1.0, end=100.0, step=0.1, units='ms', doc='')
    r_rest    = Slider(50.0, start=1.0, end=100.0, step=0.1, units='sp/s', doc='')
    r_max     = Slider(100.0, start=1.0, end=500.0, step=1.0, units='sp/s', doc='')
    I_DC_mean = Slider(0, start=-1e3, end=1e3, step=1e1, units='pA', doc='')

    extra_variables = ('excitability', 'I_syn', 'I_app', 'I_net', 'I_leak', 
                       'I_app_unit', 'I_total')

    def __init__(self, *args, **kwargs):
        """
        Initialize a rate-based neuron group with current-based inputs.
        """
        super().__init__(*args, **kwargs)

        # Initialize metrics and nonlinear activation function
        self.output = self.r_rest
        self.excitability = 1.0
        self.F = lambda r: self.r_max * (1 + tanh(r / self.r_max)) / 2

    def update(self):
        """
        Update the model neurons.
        """
        super().update()
        self.update_currents()
        self.update_rates()

    def update_rates(self):
        """
        Evolve the neuronal firing rate variable according to input currents.
        """
        self.output = self.F(self.output + (State.dt / self.tau_m) * self.I_total)
        self.output[self.output < 0] = 0.0

    def update_currents(self):
        """
        Update total input conductances for afferent synapses.
        """
        self.I_syn = 0.0
        for g in self.afferents.keys():
            self.I_syn += 10**self[g] * self.afferents[g].I_total

        self.I_leak  = self.r_rest - self.output
        self.I_app   = self.I_DC_mean + self.I_app_unit
        self.I_net   = self.I_syn + self.I_app
        self.I_total = (self.I_leak + self.I_net) * self.excitability