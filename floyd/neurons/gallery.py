"""
Helper functions to create model neuron groups from reference points.
"""

from . import LIFGroup as _LIFGroup, AdExGroup as _AdExGroup

from ..spec import paramspec as _paramspec


def create_LIF_interneurons(return_spec=False, **params):
    """
    Donoso LIF interneuron.
    """
    LIFSpec = _LIFGroup.get_spec(return_factory=True)
    spec = _paramspec('LIFIntSpec', parent=LIFSpec, instance=True,
        C_m     = 100.0,
        g_L     = 10.0,
        E_L     = -65.0,
        E_inh   = -75.0,
        E_exc   = 0.0,
        V_r     = -67.0,
        V_thr   = -52.0,
        tau_ref = 1.0,
    )
    if params:
        spec.update(**params)
    if return_spec:
        return spec
    return _LIFGroup('LIF_int', spec)

def create_LIF_pyramids(return_spec=False, **params):
    """
    Donoso LIF pyramidal cell.
    """
    LIFSpec = _LIFGroup.get_spec(return_factory=True)
    spec = _paramspec('LIFPyrSpec', parent=LIFSpec, instance=True,
        C_m     = 275.0,
        g_L     = 25.0,
        E_L     = -67.0,
        E_inh   = -68.0,
        E_exc   = 0.0,
        V_r     = -60.0,
        V_thr   = -50.0,
        tau_ref = 2.0,
    )
    if params:
        spec.update(**params)
    if return_spec:
        return spec
    return _LIFGroup('LIF_pyr', spec)

def create_AdEx_interneurons(return_spec=False, **params):
    """
    Malerba AdEx interneuron.
    """
    AdExSpec = _AdExGroup.get_spec(return_factory=True)
    spec = _paramspec('AdExIntSpec', parent=AdExSpec, instance=True,
        C_m   = 200.0,
        g_L   = 10.0,
        E_L   = -70.0,
        E_inh = -80.0,
        E_exc = 0.0,
        V_t   = -50.0,
        V_r   = -58.0,
        V_thr = 0.0,
        delta = 2.0,
        a     = 2.0,
        b     = 10.0,
        tau_w = 30.0,
    )
    if params:
        spec.update(**params)
    if return_spec:
        return spec
    return _AdExGroup('AdEx_int', spec)

def create_Malerba_AdEx_pyramids(return_spec=False, **params):
    """
    Malerba AdEx pyramidal cell.
    """
    AdExSpec = _AdExGroup.get_spec(return_factory=True)
    spec = _paramspec('AdExPyrSpec', parent=AdExSpec, instance=True,
        C_m   = 200.0,
        g_L   = 10.0,
        E_L   = -58.0,
        E_inh = -80.0,
        E_exc = 0.0,
        V_t   = -50.0,
        V_r   = -46.0,
        V_thr = 0.0,
        delta = 2.0,
        a     = 2.0,
        b     = 100.0,
        tau_w = 120.0,
    )
    if params:
        spec.update(**params)
    if return_spec:
        return spec
    return _AdExGroup('AdEx_pyr', spec)

def create_Brette_AdEx_pyramids(return_spec=False, **params):
    """
    Brette & Gerstner AdEx pyramidal cell.
    """
    AdExSpec = _AdExGroup.get_spec(return_factory=True)
    spec = _paramspec('AdExPyrSpec', parent=AdExSpec, instance=True,
        C_m   = 281.0,
        g_L   = 30.0,
        E_L   = -70.6,
        E_inh = -75.0,
        E_exc = 0.0,
        V_t   = -50.4,
        V_r   = -46.0,
        V_thr = 0.0,
        delta = 2.0,
        a     = 4.0,
        b     = 80.5,
        tau_w = 144.0,
    )
    if params:
        spec.update(**params)
    if return_spec:
        return spec
    return _AdExGroup('AdEx_pyr', spec)
