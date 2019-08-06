"""
Helper functions to create model neurons from specific references.
"""

__all__ = ['DonosoLIFInterneurons', 'DonosoLIFPyramids',
           'MalerbaAEIFInterneurons', 'MalerbaAEIFPyramids',
           'BretteAEIFPyramids']


from . import LIFNeuronGroup, AEIFNeuronGroup


class DonosoLIFInterneurons(LIFNeuronGroup):

    """
    Donoso et al. (2018): Ripples in interneuron networks.
    """

    C_m     = 100.0
    g_L     = 10.0
    E_L     = -65.0
    E_inh   = -75.0
    E_exc   = 0.0
    V_r     = -67.0
    V_thr   = -52.0
    tau_ref = 1.0


class DonosoLIFPyramids(LIFNeuronGroup):

    """
    Donoso et al. (2018): Ripples in interneuron networks.
    """

    C_m     = 275.0
    g_L     = 25.0
    E_L     = -67.0
    E_inh   = -68.0
    E_exc   = 0.0
    V_r     = -60.0
    V_thr   = -50.0
    tau_ref = 2.0


class MalerbaAEIFInterneurons(AEIFNeuronGroup):

    """
    Malerba et al. (2016): Ripples as inhibitory transients.
    """

    C_m   = 200.0
    g_L   = 10.0
    E_L   = -70.0
    E_inh = -80.0
    E_exc = 0.0
    V_t   = -50.0
    V_r   = -58.0
    V_thr = 0.0
    delta = 2.0
    a     = 2.0
    b     = 10.0
    tau_w = 30.0


class MalerbaAEIFPyramids(AEIFNeuronGroup):

    """
    Malerba et al. (2016): Ripples as inhibitory transients.
    """

    C_m   = 200.0
    g_L   = 10.0
    E_L   = -58.0
    E_inh = -80.0
    E_exc = 0.0
    V_t   = -50.0
    V_r   = -46.0
    V_thr = 0.0
    delta = 2.0
    a     = 2.0
    b     = 100.0
    tau_w = 120.0


class BretteAEIFPyramids(AEIFNeuronGroup):

    """
    Brette & Gerstner (2005): The adaptive-exponential integrate-fire neuron.
    """

    C_m           = 281.0
    g_L           = 30.0
    E_L           = -70.6
    E_inh         = -75.0
    E_exc         = 0.0
    V_t           = -50.4
    V_r           = -46.0
    V_thr         = 20.0
    delta         = 2.0
    a             = 4.0
    b             = 80.5
    tau_w         = 144.0
    tau_noise     = 10.0
    tau_noise_exc = 2.728
    tau_noise_inh = 10.49
    g_noise_exc   = 10.0
    g_noise_inh   = 15.0
