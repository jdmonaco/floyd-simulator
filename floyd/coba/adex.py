"""
A conductance-based adaptive-exponential (AdEx) threshold neuron group.
"""

from toolbox.numpy import *
from specify import Slider

from .leaky import LeakyNeuronGroup

from ..state import State


class AdaptExpNeuronGroup(LeakyNeuronGroup):

    """
    A general conductance-based adaptive exponential integrate-fire neuron.
    """

    g_L           = 25.0
    E_L           = -58.0
    g_tonic_exc   = 2.0
    g_tonic_inh   = 3.0
    delta         = Slider(default=1.0, units='mV', start=0.5, end=4.0, step=0.1, doc='slope factor')
    V_t           = Slider(default=-50.0, units='mV', start=-65, end=-20, step=0.1, doc='voltage threshold')
    V_thr         = 20.0
    a             = Slider(default=2.0, units='nS', start=0.2, end=6, step=0.1, doc='voltage coupling')
    b             = Slider(default=100.0, units='pA', start=0.0, end=200, step=1, doc='spike-adaptation strength')
    tau_w         = Slider(default=30.0, units='ms', start=1.0, end=400, step=5, doc='adaptation time-constant')

    extra_variables = ('w',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.V_r = self.E_L

    def update_voltage(self):
        """
        Evolve the membrane voltage for neurons according to input currents.
        """
        self.v += (State.dt / self.C_m) * (
                    self.g_L*self.delta*exp((self.v - self.V_t) / self.delta)
                  + self.I_total
        )

    def update_adaptation(self):
        """
        Update the adaptation variable after spikes are computed.
        """
        self.w[self.spikes] += self.b
        self.w += (State.dt / self.tau_w) * (
                    self.a*(self.v - self.E_L) - self.w
        )

    def update_currents(self):
        """
        Subtract the adaptation current from the net current.
        """
        COBANeuronGroup.update_currents(self)
        self.I_net -= self.w
        self.I_total -= self.w * self.excitability
