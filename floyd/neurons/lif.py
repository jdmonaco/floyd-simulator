"""
Conductance-based leaky integrate-and-fire neuron group.
"""

from toolbox.numpy import *

from .coba import COBANeuronGroup

from ..layout import HexagonalDiscLayout as HexLayout
from ..spec import paramspec, Param


class LIFGroup(COBANeuronGroup):

    @classmethod
    def get_spec(cls, return_factory=False, **keyvalues):
        """
        Return a Spec default factory function or instance with updated values.
        """
        if not hasattr(cls, '_Spec'):
            cls._Spec = paramspec(f'{cls.__name__}Spec',
                C_m           = Param(200, 50, 300, 5, 'pF'),
                g_L           = 25.0,
                E_L           = -58.0,
                E_exc         = 0.0,
                E_inh         = -70.0,
                I_DC_mean     = Param(0, -1e3, 1e3, 1e1, 'pA'),
                I_noise       = Param(0, -1e3, 1e3, 1e1, 'pA'),
                g_tonic_exc   = Param(0, 0, 100, 1, 'nS'),
                g_tonic_inh   = Param(0, 0, 100, 1, 'nS'),
                g_noise_exc   = Param(0, 0, 100, 1, 'nS'),
                g_noise_inh   = Param(0, 0, 100, 1, 'nS'),
                V_r           = Param(-55, -85, -35, 1, 'mV'),
                V_thr         = Param(-48, -85, 20, 1, 'mV'),
                tau_ref       = Param(1, 0, 10, 0.1, 'ms'),
                tau_noise     = 10.0,
                tau_noise_exc = 3.0,
                tau_noise_inh = 10.0,
                layout        = HexLayout.get_spec(
                    scale       = 0.1,
                    radius      = 0.5,
                    origin      = (0.5, 0.5),
                    extent      = (0, 1, 0, 1),
                    orientation = 0.0,
                ),
            )
        if return_factory:
            return cls._Spec
        return cls._Spec(**keyvalues)
