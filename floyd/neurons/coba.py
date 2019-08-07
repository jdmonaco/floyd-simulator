"""
Base class for conductance-based model neuron groups.
"""

import copy
from functools import partial

from toolbox.numpy import *
from specify import Param, Slider, Specified, is_param

from ..layout import HexagonalDiscLayout as HexLayout
from ..noise import OUNoiseProcess
from ..activity import FiringRateWindow
from ..groups import BaseUnitGroup
from ..config import Config
from ..state import State, RunMode


class COBANeuronGroup(Specified, BaseUnitGroup):

    C_m           = Slider(default=200, start=50, end=500, step=5, units='pF', doc='membrane capacitance')
    g_L           = Param(default=30.0, units='nS', doc='leak conductance')
    E_L           = Param(default=-60.0, units='mV', doc='leak reversal potential')
    E_exc         = Param(default=0.0, units='mV', doc='excitatory reversal potential')
    E_inh         = Param(default=-75.0, units='mV', doc='inhibitory reversal potential')
    I_DC_mean     = Slider(default=0, start=-1e3, end=1e3, step=1e1, units='pA', doc='mean constant input current')
    I_noise       = Slider(default=0, start=-1e3, end=1e3, step=1e1, units='pA', doc='noisy input current')
    g_tonic_exc   = Slider(default=0, start=0, end=100, step=1, units='nS', doc='tonic excitatory input conductance')
    g_tonic_inh   = Slider(default=0, start=0, end=100, step=1, units='nS', doc='tonic inhibitory input conductance')
    g_noise_exc   = Slider(default=0, start=0, end=100, step=1, units='nS', doc='noisy excitatory input conductance')
    g_noise_inh   = Slider(default=0, start=0, end=100, step=1, units='nS', doc='noisy inhibitory input conductance')
    V_r           = Slider(default=-55, start=-85, end=-35, step=1, units='mV', doc='reset voltage')
    V_thr         = Slider(default=-48, start=-85, end=20, step=1, units='mV', doc='voltage threshold')
    tau_ref       = Slider(default=1, start=0, end=10, step=0.1, units='ms', doc='spike-refractory period')
    tau_noise     = Param(default=10.0, units='ms', doc='time-constant of noisy input current')
    tau_noise_exc = Param(default=3.0, units='ms', doc='time-constant of noisy excitatory conductance')
    tau_noise_inh = Param(default=10.0, units='ms', doc='time-constant of noisy inhibitory conductance')

    base_dtypes = {'spikes':'?'}
    base_variables = ('x', 'y', 'v', 'spikes', 't_spike', 'g_total',
                      'g_total_inh', 'g_total_exc', 'excitability',
                      'I_app', 'I_net', 'I_leak', 'I_total_inh',
                      'I_total_exc', 'I_proxy')

    def __init__(self, *, name, N=None, layout=None, g_end=10.0, g_step=0.1,
        **kwargs):
        """
        Construct model variables for a model neuron group.
        """
        self._initialized = False
        if N:
            self.N = N
            self.layout = None
        elif layout:
            if not hasattr(layout, 'compute'):
                layout = HexLayout(**layout)
            layout.compute(context=State.context)
            self.layout = layout
            self.N = self.layout.N
        else:
            raise ValueError('either N or layout is required')

        super().__init__(name=name, N=self.N, **kwargs)

        # Set up the intrinsic noise inputs (current-based, excitatory
        # conductance-based, and inhibitory conductance-based). In interactive
        # run mode, generators are used to provide continuous noise.
        self.oup = OUNoiseProcess(N=self.N, tau=self.tau_noise, seed=self.name)
        self.oup_exc = OUNoiseProcess(N=self.N, tau=self.tau_noise_exc,
                seed=self.name+'_excitatory', nonnegative=True)
        self.oup_inh = OUNoiseProcess(N=self.N, tau=self.tau_noise_inh,
                seed=self.name+'_inhibitory', nonnegative=True)
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

        # Add any conductance gain values in the shared context as Params
        self.gain_keys = []
        self.gain_param_base = Slider(default=1.0, start=0.0, end=g_end,
                step=g_step, units='nS')
        for k, v in vars(State.context.__class__).items():
            if k.startswith(f'g_{name}_'):
                self._add_gain_spec(k, v)

        # Initialize metrics and variables
        self.activity = FiringRateWindow(self)
        self.set_pulse_metrics()  # uses default pulse curves
        self.v = self.E_L
        if self.layout:
            self.x = self.layout.x
            self.y = self.layout.y
        else:
            self.x = self.y = 0.0
        self.t_spike = -inf
        self.LFP_uV = 0.0  # uV, LFP signal from summed net synaptic input

        # Map from transmitters to reversal potentials
        self.E_syn = dict(GABA=self.E_inh, AMPA=self.E_exc,
                NMDA=self.E_exc, glutamate=self.E_exc, L=self.E_exc)

        if 'network' in State:
            State.network.add_neuron_group(self)

    def _add_gain_spec(self, gname, value):
        """
        Add Param (slider) objects for any `g_{post.name}_{pre.name}` class
        attributes (Param object or just default values) of the shared context.
        """
        _, post, pre = gname.split('_')
        new_param = copy.copy(self.gain_param_base)
        new_param.doc = f'{pre}->{post} max conductance'

        if is_param(value):
            for k in value.__slots__:
                if hasattr(value, k) and getattr(value, k) is not None and \
                        k in new_param.__slots__:
                    slotval = copy.copy(getattr(value, k))
                    object.__setattr__(new_param, k, slotval)
            value = new_param.default
        else:
            new_param.default = copy.deepcopy(value)

        self.__class__._add_param(gname, new_param)
        self.__dict__[new_param.attrname] = copy.deepcopy(new_param.default)
        self.gain_keys.append(gname)
        self.debug(f'added gain {gname!r} with value {new_param.default!r}')

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
        if synapses.transmitter == 'GABA':
            self.S_inh[gname] = synapses
        else:
            self.S_exc[gname] = synapses
        self.synapses[gname] = synapses

        # Check whether the conductance gain spec has already been found in the
        # shared context. If not, then add a new Param to the spec with a
        # default value of 1.0.
        #
        # Gain spec names take the form `g_<post.name>_<pre.name>`.

        if gname in self.gain_keys:
            self.debug(f'gain spec {gname!r} exists for {synapses.name!r}')
        else:
            self._add_gain_spec(gname, 1.0)
            self.debug(f'added gain spec {gname!r} for {synapses.name!r}')

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
        self.v += (State.dt / self.C_m) * self.I_net

    def update_spiking(self):
        """
        Update spikes (and refractory periods, etc.) for the current timstep.
        """
        # Enforce absolute refractory period
        dt = State.t - self.t_spike
        self.v[dt < self.tau_ref] = self.V_r

        # Perform voltage resets for threshold crossings
        self.spikes = self.v > self.V_thr
        self.v[self.spikes] = self.V_r

        # Update most-recent spike time for units that spiked
        self.t_spike[self.spikes] = State.t

    def update_adaptation(self):
        """
        Update any adaptation variables after spikes are computed.
        """

    def update_conductances(self):
        """
        Update total input conductances for afferent synapses.
        """
        self.g_total_exc = self.g_tonic_exc
        if self.g_noise_exc:
            self.g_total_exc += self.g_noise_exc * self.eta_exc

        self.g_total_inh = self.g_tonic_inh
        if self.g_noise_inh:
            self.g_total_inh += self.g_noise_inh * self.eta_inh

        for gname in self.S_exc.keys():
            self.g_total_exc += self[gname] * self.S_exc[gname].g_total
        for gname in self.S_inh.keys():
            self.g_total_inh += self[gname] * self.S_inh[gname].g_total

        self.g_total = self.g_total_exc + self.g_total_inh

    def update_currents(self):
        """
        Calculate total currents based on total conductances.
        """
        self.I_leak      = self.g_L * (self.E_L - self.v)
        self.I_total_exc = self.g_total_exc * (self.E_exc - self.v)
        self.I_total_inh = self.g_total_inh * (self.E_inh - self.v)
        self.I_proxy     = self.I_noise * self.eta
        self.I_app       = self.I_DC_mean * self.excitability
        self.I_net       = self.I_leak + self.I_app + self.I_proxy + \
                               self.I_total_exc + self.I_total_inh

    def update_noise(self):
        """
        Update the intrinsic noise sources (for those with nonzero gains).
        """
        if State.run_mode == RunMode.INTERACT:
            if self.I_noise: self.eta = next(self.eta_gen)
            if self.g_noise_exc: self.eta_exc = next(self.eta_gen_exc)
            if self.g_noise_inh: self.eta_inh = next(self.eta_gen_inh)
        else:
            if self.I_noise: self.eta = self.oup.eta[...,State.n]
            if self.g_noise_exc: self.eta_exc = self.oup_exc.eta[...,State.n]
            if self.g_noise_inh: self.eta_inh = self.oup_inh.eta[...,State.n]

    def update_metrics(self):
        """
        Update metrics such as activity calculations.
        """
        self.LFP_uV = -(
                self.I_total_exc.sum() + self.I_total_inh.sum()) / Config.g_LFP
        self.activity.update()

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

    def active_mean_rate(self):
        """
        Return the mean firing rate of neurons active the calculation window.
        """
        return self.activity.get_active_mean_rate()

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

    def get_neuron_sliders(self):
        """
        Return a tuple of Panel FloatSlider objects for neuron Param values.
        """
        return self.get_widgets(exclude=self.gain_keys)

    def get_gain_sliders(self):
        """
        Return a tuple of Panel FloatSlider objects for gain Param values.
        """
        return self.get_widgets(include=self.gain_keys)
