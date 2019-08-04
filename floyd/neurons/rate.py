"""
Base class for conductance-based model neuron groups.
"""

from functools import partial

from toolbox.numpy import *
from specify import Param, Slider, Specified, is_param

from ..noise import OrnsteinUhlenbeckProcess as OUProcess
from ..groups import BaseUnitGroup
from ..state import State, RunMode


class RateNeuronGroup(BaseUnitGroup):

    N         = Param(default=1, doc='number of neurons in the group')
    tau_m     = Slider(default=10.0, start=1.0, end=100.0, step=0.1, units='ms', doc='')
    r_rest    = Slider(default=0.0, start=1.0, end=100.0, step=0.1, units='sp/s', doc='')
    r_max     = Slider(default=100.0, start=1.0, end=500.0, step=1.0, units='sp/s', doc='')
    I_DC_mean = Slider(default=0, start=-1e3, end=1e3, step=1e1, units='pA', doc='')
    I_noise   = Slider(default=0, start=-1e3, end=1e3, step=1e1, units='pA', doc='')
    tau_noise = Param(default=10.0, units='ms', doc='time-constant of input current noise')

    base_variables = ('x', 'y', 'r', 'g_total', 'g_total_inh', 'g_total_exc',
                      'excitability', 'I_app', 'I_net', 'I_leak',
                      'I_total_inh', 'I_total_exc', 'I_proxy')

    def __init__(self, name, gain_max=10.0, gain_step=0.1, **specs):
        """
        Construct the neuron group by computing layouts and noise.
        """
        super(Specified, self).__init__(**specs)
        super(BaseUnitGroup, self).__init__(self, self.N, name)

        # Set up the intrinsic noise inputs (current-based only for rate-based
        # neurons. In interactive run mode, generators are used to provide
        # continuous noise.
        self.oup = OUProcess(N=self.N, tau=self.tau_noise, seed=self.name)
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
        self.gain_param_base = Slider(start=0.0, end=gain_max, step=gain_step,
                owner=self, units='nS')
        for k, v in vars(State.context.__class__):
            if k.startswith(f'g_{name}_'):
                self._add_gain_spec(k, v)

        # Initialize metrics and variables
        self.r = self.r_rest

        # Map from transmitters to reversal potentials
        self.E_syn = dict(GABA=self.E_inh, AMPA=self.E_exc,
                NMDA=self.E_exc, glutamate=self.E_exc, L=self.E_exc)

        State.network.add_neuron_group(self)
        self.out(self)

    def _add_gain_spec(self, gname, value):
        """
        Add Param (slider) objects for any `g_{post.name}_{pre.name}` class
        attributes (Param object or just default values) of the shared context.
        """
        _, post, pre = gname.split('_')
        if is_param(value):
            new_param = self.gain_param_base.copy()
            new_param.default = float(value.default)
            value = new_param
        else:
            value = Slider(default=float(value),
                           doc=f'{pre}->{post} max conductance')
            value._set_names(gname)
            value.update(self.gain_param_base)
        self.__class__.__dict__[gname] = value
        self.__dict__[gname] = copy.deepcopy(value.default)
        setattr(self, gname, value)
        self.gain_keys.append(gname)
        self.debug('added gain key {gname!r} with value {value.default!r}')

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

        # Check whether the conductance gain spec has already been found in the
        # shared context. If not, then add a new Param to the spec with a
        # default value of 1.0.
        #
        # Gain spec names take the form `g_<post.name>_<pre.name>`.

        if gname in self.gain_keys:
            self.debug('gain spec {gname!r} exists for {synapses.name!r}')
        else:
            self._add_gain_spec(gname, 1.0)
            self.debug('added gain spec {gname!r} for {synapses.name!r}')

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
        self.r += (State.dt / self.tau_m) * self.I_net

    def update_currents(self):
        """
        Update total input conductances for afferent synapses.
        """
        self.I_total_exc = 0.0
        self.I_total_inh = 0.0

        for gname in self.S_exc.keys():
            self.I_total_exc += self[gname] * self.S_exc[gname].I_total
        for gname in self.S_inh.keys():
            self.I_total_inh -= self[gname] * self.S_inh[gname].I_total

        self.I_leak      = self.r_rest - self.r
        self.I_proxy     = self.I_noise * self.eta
        self.I_app       = self.I_DC_mean * self.excitability
        self.I_net       = self.I_leak + self.I_proxy + self.I_app + \
                               self.I_total_exc + self.I_total_inh

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
        neuron_keys = [k for k in self if k not in self.gain_keys]
        return self.widgets(*neuron_keys)

    def get_gain_sliders(self):
        """
        Return a tuple of Panel FloatSlider objects for gain Param values.
        """
        return self.widgets(*self.gain_keys)
