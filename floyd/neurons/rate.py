"""
Base class for current-based rate-coding neuron groups.
"""

import copy
from functools import partial

from toolbox.numpy import *
from specify import Param, Slider, LogSlider, Specified, is_param
from specify.utils import get_all_slots

from ..noise import OUNoiseProcess
from ..groups import BaseUnitGroup
from ..state import State, RunMode


class RateNeuronGroup(Specified, BaseUnitGroup):

    tau_m     = Slider(default=10.0, start=1.0, end=100.0, step=0.1, units='ms', doc='')
    r_rest    = Slider(default=0.0, start=1.0, end=100.0, step=0.1, units='sp/s', doc='')
    r_max     = Slider(default=100.0, start=1.0, end=500.0, step=1.0, units='sp/s', doc='')
    I_DC_mean = Slider(default=0, start=-1e3, end=1e3, step=1e1, units='pA', doc='')
    I_noise   = Slider(default=0, start=-1e3, end=1e3, step=1e1, units='pA', doc='')
    tau_noise = Param(default=10.0, units='ms', doc='time-constant of input current noise')

    base_variables = ('x', 'y', 'r', 'excitability', 'I_app', 'I_net',
                      'I_leak', 'I_total_inh', 'I_total_exc', 'I_proxy')

    def __init__(self, *, name, N, g_log_range=(-5, 5), g_step=0.05, **kwargs):
        """
        Construct rate-based neuron groups.
        """
        self._initialized = False
        self.N = N
        super().__init__(name=name, N=N, **kwargs)

        # Set up the intrinsic noise inputs (current-based only for rate-based
        # neurons. In interactive run mode, generators are used to provide
        # continuous noise.
        self.oup = OUNoiseProcess(N=self.N, tau=self.tau_noise, seed=self.name)
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

        # Add any conductance gain values in the shared context as Params
        self.gain_keys = []
        self.gain_param_base = LogSlider(default=0.0, start=g_log_range[0],
                end=g_log_range[1], step=g_step, units='nS')
        for k, v in vars(State.context.__class__).items():
            if k.startswith(f'g_{name}_'):
                self._add_gain_spec(k, v)

        # Initialize metrics and variables
        self.r = self.r_rest
        self.set_pulse_metrics() # uses default pulse curves

        # Create the activation nonlinearity
        self.F = lambda r: self.r_max * (1 + tanh(r/self.r_max)) / 2

        # NOTE: Synapses expects this so add some dummy values for now
        self.E_syn = dict(GABA=-75.0, AMPA=0.0, NMDA=0.0, glutamate=0.0, L=0.0)
        self.E_L = -65.0
        self.C_m = 200.0

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
        new_all_slots = get_all_slots(type(new_param))

        if is_param(value):
            old_all_slots = get_all_slots(type(value))
            for k in old_all_slots:
                if hasattr(value, k) and getattr(value, k) is not None and \
                        k in new_all_slots:
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
        # default value of 0.0 (log10(1)).
        #
        # Gain spec names take the form `g_<post.name>_<pre.name>`.

        if gname in self.gain_keys:
            self.debug(f'gain spec {gname!r} exists for {synapses.name!r}')
        else:
            self._add_gain_spec(gname, 0.0)
            self.debug(f'added gain spec {gname!r} for {synapses.name!r}')

    def update(self):
        """
        Update the model neurons.
        """
        self.update_rates()
        self.update_currents()
        self.update_noise()

    def update_rates(self):
        """
        Evolve the neuronal firing rate variable according to input currents.
        """
        self.r = self.F(self.r + (State.dt / self.tau_m) * (self.I_leak +
                        self.I_net))
        self.r[self.r<0] = 0.0

    def update_currents(self):
        """
        Update total input conductances for afferent synapses.
        """
        self.I_total_exc = 0.0
        self.I_total_inh = 0.0

        for gname in self.S_exc.keys():
            self.I_total_exc += 10**self[gname] * self.S_exc[gname].I_total
        for gname in self.S_inh.keys():
            self.I_total_inh -= 10**self[gname] * self.S_inh[gname].I_total

        self.I_leak  = self.r_rest - self.r
        self.I_proxy = self.I_noise * self.eta
        self.I_app   = self.I_DC_mean * self.excitability
        self.I_net   = self.I_proxy + self.I_app + self.I_total_exc + \
                           self.I_total_inh

    def update_noise(self):
        """
        Update the intrinsic noise sources (for those with nonzero gains).
        """
        if State.run_mode == RunMode.INTERACT:
            if self.I_noise: self.eta = next(self.eta_gen)
        else:
            if self.I_noise: self.eta = self.oup.eta[...,State.n]

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
