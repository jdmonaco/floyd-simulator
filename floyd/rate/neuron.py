"""
Base class for current-based (CUBA) rate-coding neuron groups.
"""

from toolbox.numpy import *
from specify import Slider

from ..base import BaseNeuronGroup
from ..state import State


class LinearUnitGroup(BaseNeuronGroup):

    """
    A group of threshold-linear units with no temporal dynamics.
    """

    I_thr = Slider(0.0, start=-500.0, end=500.0, step=5.0, doc='threshold')
    I_const = Slider(0, start=-100, end=100, step=0.1, doc='fixed input')

    extra_variables = ('excitability', 'I_syn', 'I_net', 'I_app', 'I_app_unit', 
                       'I_total')

    def __init__(self, *args, **kwargs):
        """
        Initialize rectified threshold-linear units with external inputs.
        """
        super().__init__(*args, **kwargs)
        self.excitability = 1.0

    def update(self):
        self.update_synaptic_input()
        self.update_total_inputs()
        self.update_output()
        super().update()

    def update_synaptic_input(self):
        """
        Update total input from afferent synapses.
        """
        self.I_syn = 0.0
        for g in self.afferents.keys():
            self.I_syn += 10**self[g] * self.afferents[g].I_total

    def update_total_inputs(self):
        self.I_app   = self.I_const + self.I_app_unit
        self.I_net   = self.I_syn + self.I_app
        self.I_total = self.I_net

    def update_output(self):
        self.output = self.excitability * (self.I_total - self.I_thr)
        self.output[self.output < 0] = 0.0


class RateBasedNeuronGroup(LinearUnitGroup):

    """
    A group of rate-based neurons with nonlinear, leaky integration.
    """

    tau = Slider(10.0, start=1.0, end=100.0, step=0.1, units='ms')
    r_rest = Slider(50.0, start=1.0, end=500.0, step=1.0, units='sp/s')
    r_max = Slider(100.0, start=1.0, end=500.0, step=1.0, units='sp/s')

    extra_variables = ('excitability', 'I_syn', 'I_net', 'I_app', 'I_app_unit', 
                       'I_leak', 'I_total')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.F = lambda r: self.r_max * (1 + tanh(r / self.r_max)) / 2

    def update_total_inputs(self):
        self.I_leak  = self.r_rest - self.output
        self.I_app   = self.I_const + self.I_app_unit
        self.I_net   = self.I_syn + self.I_app
        self.I_total = (self.I_leak + self.I_net) * self.excitability

    def update_rates(self):
        self.output = self.F(self.output + (State.dt / self.tau) * self.I_total)
        self.output[self.output < 0] = 0.0