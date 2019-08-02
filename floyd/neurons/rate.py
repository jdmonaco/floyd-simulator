"""
Base class for conductance-based model neuron groups.
"""

from functools import partial

from toolbox.numpy import *

from ..layout import get_layout_from_spec
from ..noise import OrnsteinUhlenbeckProcess as OUProcess
from ..spec import paramspec, Param
from ..groups import BaseUnitGroup
from ..state import State, RunMode


class RateNeuronGroup(BaseUnitGroup):

    @classmethod
    def get_spec(cls, return_factory=False, **keyvalues):
        """
        Return a Spec default factory function or instance with updated values.
        """
        if not hasattr(cls, '_Spec'):
            cls._Spec = paramspec(f'{cls.__name__}Spec',
                tau_m     = Param(10.0, 1.0, 100.0, 0.1, 'ms'),
                r_rest    = Param(0.0, 1.0, 100.0, 0.1, 'sp/s'),
                r_max     = Param(100.0, 1.0, 500.0, 1.0, 'sp/s'),
                I_DC_mean = Param(0, -1e3, 1e3, 1e1, 'pA'),
                I_noise   = Param(0, -1e3, 1e3, 1e1, 'pA'),
                tau_noise = 10.0,
                layout    = FixedLayout()
            )
        if return_factory:
            return cls._Spec
        return cls._Spec(**keyvalues)

    base_variables = ('x', 'y', 'r', 'g_total', 'g_total_inh', 'g_total_exc',
                      'excitability', 'I_app', 'I_net', 'I_leak',
                      'I_total_inh', 'I_total_exc', 'I_proxy')

    def __init__(self, name, spec, gain_param=(0, 10, 0.1, 'gain')):
        """
        Construct the neuron group by computing layouts and noise.
        """
        # Get the spatial layout to get the number of units
        if spec.layout is None:
            raise ValueError('layout required to determine group size')
        self.layout = get_layout_from_spec(spec.layout)
        self.N = self.layout.N

        BaseUnitGroup.__init__(self, self.N, name, spec=spec)

        # Set up the intrinsic noise inputs (current-based, excitatory
        # conductance-based, and inhibitory conductance-based). In interactive
        # run mode, generators are used to provide continuous noise.
        self.oup = OUProcess(N=self.N, tau=spec.tau_noise, seed=self.name)
        if State.run_mode == RunMode.INTERACT:
            self.eta_gen = self.oup.generator()
            self.eta = next(self.eta_gen)
        else:
            self.oup.compute(context=State.context)
            self.eta = self.oup.eta[...,0]

        # Initialize data structures
        self.S_inh = {}
        self.S_exc = {}
        self.synapses = {}
        self.g = {}
        self.g = paramspec(f'{name}GainSpec', instance=True,
                    **{k:Param(State.context.p[k], *gain_param)
                        for k in State.context.p.keys()
                            if k.startswith(f'g_{name}_')}
        )

        # Intialize some variables
        self.x = self.layout.x
        self.y = self.layout.y

        State.network.add_neuron_group(self)
        self.out(self)

    def add_synapses(self, synapses):
        """
        Add afferent synaptic pathway to this neuron group.
        """
        if synapses.post is not self:
            self.out('{} does not project to {}', synapses.name, self.name,
                    error=True)
            return

        # Add synapses to list of inhibitory or excitatory afferents
        gname = 'g_{}_{}'.format(synapses.post.name, synapses.pre.name)
        if synapses.p.transmitter == 'GABA':
            self.S_inh[gname] = synapses
        else:
            self.S_exc[gname] = synapses
        self.synapses[gname] = synapses

        # Conductance gains are initialized as a spec, so context parameters
        # must provide `g_<post>_<pre>` values for all synaptic pathways.
        # Thus, an error is raised if the value has not been found.
        if gname not in self.g:
            raise ValueError('missing gain parameter: {}'.format(repr(gname)))

    def update(self):
        """
        Update the model neurons.
        """
        self.update_rates()
        self.update_currents()
        self.update_noise()

    def update_rates(self):
        """
        Evolve the membrane voltage for neurons according to input currents.
        """
        self.r += (State.dt / self.p.tau_m) * self.I_net

    def update_currents(self):
        """
        Update total input conductances for afferent synapses.
        """
        self.I_total_exc = self.p.I_tonic_exc
        self.I_total_inh = -self.p.I_tonic_inh

        for gname in self.S_exc.keys():
            self.I_total_exc += self.g[gname] * self.S_exc[gname].I_total
        for gname in self.S_inh.keys():
            self.I_total_inh -= self.g[gname] * self.S_inh[gname].I_total

        self.I_leak      = self.p.r_rest - self.r
        self.I_proxy     = self.p.I_noise * self.eta
        self.I_app       = self.p.I_DC_mean * self.excitability
        self.I_net       = self.I_leak + self.I_app + self.I_proxy + \
                               self.I_total_exc + self.I_total_inh

    def update_noise(self):
        """
        Update the intrinsic noise sources (for those with nonzero gains).
        """
        if State.run_mode == RunMode.INTERACT:
            if self.p.I_noise: self.eta = next(self.eta_gen)
        else:
            if self.p.I_noise: self.eta = self.oup.eta[...,State.n]

    def reset(self):
        """
        Reset the neuron model and gain parameters to spec defaults.
        """
        self.p.reset()
        self.g.reset()

    def rates(self):
        """
        Return the firing rates in the calculation window.
        """
        return self.r[:]

    def mean_rate(self):
        """
        Return the mean firing rate in the calculation window.
        """
        return self.r.mean()

    def active_mean_rate(self):
        """
        Return the mean firing rate in the calculation window.
        """
        return self.r[self.r>0].mean()

    def active_fraction(self):
        """
        Return the active fraction in the calculation window.
        """
        return (self.r > 0).sum() / self.N

    def set_pulse_metrics(self, active=(10, 90, 10), rate=(1, 100, 20),
        only_active=True):
        """
        Set (min, max, smoothness) for active fraction and mean rate.
        """
        pulse = lambda l, u, k, x: \
            (1 + 1/(1 + exp(-k * (x - u))) - 1/(1 + exp(k * (x - l)))) / 2

        self.active_pulse = partial(pulse, *active)
        self.rate_pulse = partial(pulse, *rate)
        self.pulse_only_active = only_active

    def pulse(self):
        """
        Return a [0,1] "pulse" metric of the healthiness of activity.
        """
        apulse = self.active_pulse(self.active_fraction())
        if self.pulse_only_active:
            rpulse = self.rate_pulse(self.get_active_mean_rate())
        else:
            rpulse = self.rate_pulse(self.mean_rate())

        # As in kurtosis calculations, use the 4th power to emphasize
        # the extremities, then the mean tells you at least one or the other is
        # currently at extremes.

        return (apulse**4 + rpulse**4) / 2
