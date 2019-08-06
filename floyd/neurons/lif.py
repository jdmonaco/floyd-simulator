"""
A conductance-based leaky integrate-fire neuron group.
"""

from .coba import COBANeuronGroup


class LIFNeuronGroup(COBANeuronGroup):

    """
    A general conductance-based leaky integrate-fire neuron.
    """

    C_m           = 200.0
    g_L           = 25.0
    E_L           = -58.0
    E_exc         = 0.0
    E_inh         = -70.0
    V_r           = -55.0
    V_thr         = -48.0
