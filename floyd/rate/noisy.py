"""
Base class for noisy current-based (CUBA) rate-coding neuron groups.
"""

import copy
from functools import partial

from toolbox.numpy import *
from specify import Param, Slider, LogSlider, Specified, is_param
from specify.utils import get_all_slots

from ..base import NoisyNeuronGroup
from ..noise import OUNoiseProcess
from ..state import State, RunMode


class NoisyRateBasedNeuronGroup(NoisyNeuronGroup):

    tau_m     = Slider(10.0, start=1.0, end=100.0, step=0.1, units='ms', doc='')
    r_rest    = Slider(0.0, start=1.0, end=100.0, step=0.1, units='sp/s', doc='')
    r_max     = Slider(100.0, start=1.0, end=500.0, step=1.0, units='sp/s', doc='')
    I_DC_mean = Slider(0, start=-1e3, end=1e3, step=1e1, units='pA', doc='')
    I_noise   = Slider(0, start=-1e3, end=1e3, step=1e1, units='pA', doc='')

    extra_variables = ('excitability', 'I_app', 'I_net', 'I_leak', 'I_inh', 'I_exc', 
                       'I_proxy', 'I_app_unit', 'I_total')

    def __init__(self, *, name, N, g_log_range=(-5, 5), g_step=0.05, **kwargs):
        """
        Construct rate-based neuron groups.
        """
        self._initialized = False
        self.N = N
        super().__init__(name=name, N=N, **kwargs)

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
        self.excitability = 1.0

        # Create the activation nonlinearity
        self.F = lambda r: self.r_max * (1 + tanh(r/self.r_max)) / 2

        # Dummy values for reversals (expected by Synapses)
        self.E_L = -60.0
        self.C_m = 200.0
        self.E_syn = dict(GABA=-75.0, AMPA=0.0, NMDA=0.0, glutamate=0.0)

        if 'network' in State and State.network:
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

        self.add_param(gname, new_param)
        self.gain_keys.append(gname)
        self.debug(f'added gain {gname!r} with value {new_param.default!r}')

    def add_projection(self, synapses):
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
        self.r = self.F(self.r + (State.dt / self.tau_m) * self.I_total)
        self.r[self.r<0] = 0.0

    def update_currents(self):
        """
        Update total input conductances for afferent synapses.
        """
        self.I_exc = 0.0
        self.I_inh = 0.0

        for gname in self.S_exc.keys():
            self.I_exc += 10**self[gname] * self.S_exc[gname].I_total
        for gname in self.S_inh.keys():
            self.I_inh -= 10**self[gname] * self.S_inh[gname].I_total

        self.I_leak  = self.r_rest - self.r
        self.I_proxy = self.I_noise * self.eta
        self.I_app   = self.I_DC_mean + self.I_app_unit
        self.I_net   = self.I_exc + self.I_inh + self.I_proxy + self.I_app
        self.I_total = (self.I_leak + self.I_net) * self.excitability

    def update_noise(self):
        """
        Update the intrinsic noise sources (for those with nonzero gains).
        """
        if self.oup is None: return
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
            rpulse = self.rate_pulse(self.active_mean_rate())
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

