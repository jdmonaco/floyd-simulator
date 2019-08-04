"""
A conductance-based adaptive-exponential integrate-fire neuron group.
"""

from toolbox.numpy import *

from .coba import COBANeuronGroup

from ..state import State


class AEIFNeuronGroup(COBANeuronGroup):

    """
    A general conductance-based adaptive exponential integrate-fire neuron.
    """

    C_m           = 200.0
    g_L           = 25.0
    E_L           = -58.0
    E_exc         = 0.0
    E_inh         = -75.0
    I_DC_mean     = 0.0
    I_noise       = 0.0
    g_tonic_exc   = 0.0
    g_tonic_inh   = 0.0
    g_noise_exc   = 0.0
    g_noise_inh   = 0.0
    V_r           = -55.0
    delta         = 1.0
    V_t           = -50.0
    V_thr         = 20.0
    a             = 2.0
    b             = 100.0
    tau_w         = 30.0
    tau_ref       = 1.0
    tau_noise     = 10.0
    tau_noise_exc = 3.0
    tau_noise_inh = 10.0
    scale         = 0.1
    radius        = 0.5
    origin        = (0.5, 0.5)
    extent        = (0, 1, 0, 1)
    orientation   = 0.0

    extra_variables = ('w',)

    def update_voltage(self):
        """
        Evolve the membrane voltage for neurons according to input currents.
        """
        self.v += (State.dt/self.C_m) * (
                      self.g_L*self.delta*exp((self.v - self.V_t)/self.delta)
                      + self.I_net
        )

    def update_adaptation(self):
        """
        Update the adaptation variable after spikes are computed.
        """
        self.w[self.spikes] += self.b
        self.w += (State.dt/self.tau_w) * (self.a*(self.v - self.V_r) - self.w)
