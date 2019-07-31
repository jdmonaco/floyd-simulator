"""
Base class for conductance-based model neuron groups.
"""

from functools import partial

from toolbox.numpy import *

from ..layout import get_layout_from_spec
from ..noise import OrnsteinUhlenbeckProcess as OUProcess
from ..activity import FiringRateWindow
from ..spec import paramspec, Param
from ..groups import BaseUnitGroup
from ..config import Config
from ..state import State, RunMode


class COBANeuronGroup(BaseUnitGroup):

    @classmethod
    def get_spec(cls, return_factory=False, **keyvalues):
        """
        Return a Spec default factory function or instance with updated values.
        """
        if not hasattr(cls, '_Spec'):
            cls._Spec = paramspec(f'{cls.__name__}Spec',
                C_m           = Param(200, 50, 500, 5, 'pF'),
                g_L           = 30.0,
                E_L           = -60.0,
                E_exc         = 0.0,
                E_inh         = -75.0,
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

    base_dtypes = {'spikes':'?'}
    base_variables = ('x', 'y', 'v', 'spikes', 't_spike', 'g_total',
                      'g_total_inh', 'g_total_exc', 'excitability',
                      'I_app', 'I_net', 'I_leak', 'I_total_inh',
                      'I_total_exc', 'I_proxy')

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
        self.oup_exc = OUProcess(N=self.N, tau=spec.tau_noise_exc,
                seed=self.name+'_excitatory')
        self.oup_inh = OUProcess(N=self.N, tau=spec.tau_noise_inh,
                seed=self.name+'_inhitatory')
        if State.run_mode == RunMode.INTERACT:
            self.eta_gen = self.oup.generator()
            self.eta_gen_exc = self.oup_exc.generator()
            self.eta_gen_inh = self.oup_inh.generator()
            self.eta = next(self.eta_gen)
            self.eta_exc = next(self.eta_gen_exc)
            self.eta_inh = next(self.eta_gen_inh)
        else:
            self.oup.compute(context=State.context)
            self.oup_exc.compute(context=State.context)
            self.oup_inh.compute(context=State.context)
            self.eta = self.oup.eta[...,0]
            self.eta_exc = self.oup_exc.eta[...,0]
            self.eta_inh = self.oup_inh.eta[...,0]

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

        # Initialize metrics
        self.activity = FiringRateWindow(self)

        # Intialize some variables
        self.x = self.layout.x
        self.y = self.layout.y
        self.t_spike = -inf
        self.LFP_uV = 0.0  # uV, LFP signal from summed net synaptic input

        # Map from transmitters to reversal potentials
        self.E_syn = dict(GABA=self.p.E_inh, AMPA=self.p.E_exc,
                NMDA=self.p.E_exc, glutamate=self.p.E_exc, L=self.p.E_exc)

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
        self.update_voltage()
        self.update_spiking()
        self.update_adaptation()
        self.update_conductances()
        self.update_currents()
        self.update_noise()
        self.update_metrics()

    def update_voltage(self):
        """
        Evolve the membrane voltage for neurons according to input currents.
        """
        self.v += (State.dt / self.p.C_m) * self.I_net

    def update_spiking(self):
        """
        Update spikes (and refractory periods, etc.) for the current timstep.
        """
        # Enforce absolute refractory period
        dt = State.t - self.t_spike
        self.v[dt < self.p.tau_ref] = self.p.V_r

        # Perform voltage resets for threshold crossings
        self.spikes = self.v > self.p.V_thr
        self.v[self.spikes] = self.p.V_r

        # Update most-recent spike time for units that spiked
        self.t_spike[self.spikes] = State.t

    def update_adaptation(self):
        """
        Update any adaptation variables after spikes are computed.
        """
        pass

    def update_conductances(self):
        """
        Update total input conductances for afferent synapses.
        """
        self.g_total_exc = self.p.g_tonic_exc
        if self.p.g_noise_exc:
            self.g_total_exc += self.p.g_noise_exc * self.eta_exc

        self.g_total_inh = self.p.g_tonic_inh
        if self.p.g_noise_inh:
            self.g_total_inh += self.p.g_noise_inh * self.eta_inh

        for gname in self.S_exc.keys():
            self.g_total_exc += self.g[gname] * self.S_exc[gname].g_total
        for gname in self.S_inh.keys():
            self.g_total_inh += self.g[gname] * self.S_inh[gname].g_total

        self.g_total = self.g_total_exc + self.g_total_inh

    def update_currents(self):
        """
        Calculate total currents based on total conductances.
        """
        self.I_leak      = self.p.g_L * (self.p.E_L - self.v)
        self.I_total_exc = self.g_total_exc * (self.p.E_exc - self.v)
        self.I_total_inh = self.g_total_inh * (self.p.E_inh - self.v)
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
            if self.p.g_noise_exc: self.eta_exc = next(self.eta_gen_exc)
            if self.p.g_noise_inh: self.eta_inh = next(self.eta_gen_inh)
        else:
            if self.p.I_noise: self.eta = self.oup.eta[...,State.n]
            if self.p.g_noise_exc: self.eta_exc = self.oup_exc.eta[...,State.n]
            if self.p.g_noise_inh: self.eta_inh = self.oup_inh.eta[...,State.n]

    def update_metrics(self):
        """
        Update metrics such as activity calculations.
        """
        self.LFP_uV = -(
                self.I_total_exc.sum() + self.I_total_inh.sum()) / Config.g_LFP
        self.activity.update()

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
        return self.activity.get_rates()

    def mean_rate(self):
        """
        Return the mean firing rate in the calculation window.
        """
        return self.activity.get_mean_rate()

    def active_fraction(self):
        """
        Return the active fraction in the calculation window.
        """
        return self.activity.get_activity()

    def set_pulse_metrics(self, active=(10, 90, 30), rate=(1, 70, 25),
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
        apulse = self.active_pulse(self.activity.get_activity())
        if self.pulse_only_active:
            rpulse = self.rate_pulse(self.activity.get_active_mean_rate())
        else:
            rpulse = self.rate_pulse(self.activity.get_mean_rate())

        # As in kurtosis calculations, use the 4th power to emphasize
        # the extremities, then the mean tells you at least one or the other is
        # currently at extremes.

        return (apulse**4 + rpulse**4) / 2
