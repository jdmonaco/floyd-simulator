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
    I_DC_mean     = 0.0
    I_noise       = 0.0
    g_tonic_exc   = 0.0
    g_tonic_inh   = 0.0
    g_noise_exc   = 0.0
    g_noise_inh   = 0.0
    V_r           = -55.0
    V_thr         = -48.0
    tau_ref       = 1.0
    tau_noise     = 10.0
    tau_noise_exc = 3.0
    tau_noise_inh = 10.0
    scale         = 0.1
    radius        = 0.5
    origin        = (0.5, 0.5)
    extent        = (0, 1, 0, 1)
    orientation   = 0.0
