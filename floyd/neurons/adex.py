"""
Conductance-based adaptive exponential integrate-and-fire neuron group.
"""

from toolbox.numpy import *

from .coba import COBANeuronGroup

from ..layout import HexagonalDiscLayout as HexLayout
from ..spec import paramspec, Param


class AdExGroup(COBANeuronGroup):

    @classmethod
    def get_spec(cls, return_factory=False, **keyvalues):
        """
        Return a Spec default factory function or instance with updated values.
        """
        if not hasattr(cls, '_Spec'):
            cls._Spec = paramspec(f'{cls.__name__}Spec',
                C_m         = Param(200, 50, 300, 5, 'pF'),
                g_L         = 25.0,
                E_L         = -58.0,
                E_exc       = 0.0,
                E_inh       = -70.0,
                I_DC_mean   = Param(0, 0, 1e3, 1e1, 'pA'),
                g_tonic_exc = Param(0, 0, 100, 1, 'nS'),
                g_tonic_inh = Param(0, 0, 100, 1, 'nS'),
                V_r         = Param(-55, -85, -40, 1, 'mV'),
                delta       = Param(1, 0.25, 6, 0.05, 'mV'),
                V_t         = Param(-50, -65, -20, 1, 'mV'),
                V_thr       = Param(20, -50, 20, 1, 'mV'),
                a           = Param(2, 0, 10, 0.1, 'nS'),
                b           = Param(100, 0, 200, 5, 'pA'),
                tau_w       = Param(30, 1, 300, 1, 'ms'),
                tau_ref     = Param(1, 0, 10, 1, 'ms'),
                tau_eta     = 10.0,
                sigma       = Param(0, 0, 1e3, 1e1, 'pA'),
                layout      = HexLayout.get_spec(
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

    extra_variables = ('w',)

    def update_voltage(self):
        """
        Evolve the membrane voltage for neurons according to input currents.
        """
        self.v += (State.dt/self.p.C_m) * (
              self.p.g_L*self.p.delta*exp((self.v - self.p.V_t)/self.p.delta)
              + self.I_net
        )

    def update_adaptation(self):
        """
        Update the adaptation variable after spikes are computed.
        """
        self.w[self.spikes] += self.p.b
        self.w += (State.dt/self.p.tau_w) * (
                                    self.p.a*(self.v - self.p.V_r) - self.w)
