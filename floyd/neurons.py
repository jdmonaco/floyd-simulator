"""
Groups of neurons for network models.
"""

try:
    import panel as pn
except ImportError:
    print('Warning: install `panel` to use interactive dashboards.')

from toolbox.numpy import *

from .layout import HexagonalDiscLayout as HexLayout, HexLayoutSpec
from .noise import OrnsteinUhlenbeckProcess as OUProcess
from .activity import FiringRateWindow
from .spec import paramspec, Param
from .groups import BaseUnitGroup
from .config import Config
from .state import State

LIFNeuronSpec = paramspec('LIFNeuronSpec',
            C_m         = Param(200, 50, 300, 5, 'pF'),
            g_L         = 25.0,
            E_L         = -58.0,
            E_exc       = 0.0,
            E_inh       = -70.0,
            I_DC_mean   = Param(0, 0, 1e3, 1e1, 'pA'),
            g_tonic_exc = Param(0, 0, 100, 1, 'nS'),
            g_tonic_inh = Param(0, 0, 100, 1, 'nS'),
            V_r         = Param(-55, -85, -40, 1, 'mV'),
            V_thr       = Param(-48, -50, 20, 1, 'mV'),
            tau_ref     = Param(1, 0, 10, 1, 'ms'),
            tau_eta     = 10.0,
            sigma       = Param(0, 0, 1e3, 1e1, 'pA'),
            layoutspec  = HexLayoutSpec,
)
AdExNeuronSpec = paramspec('AdExNeuronSpec', parent=LIFNeuronSpec,
            delta = Param(1, 0.25, 6, 0.05, 'mV'),
            V_t   = Param(-50, -65, -20, 1, 'mV'),
            V_thr = Param(20, -50, 20, 1, 'mV'),
            a     = Param(2, 0, 10, 0.1, 'nS'),
            b     = Param(100, 0, 200, 5, 'pA'),
            tau_w = Param(30, 1, 300, 1, 'ms'),
)


def create_LIF_int_group():
    """
    The Donoso LIF interneuron.
    """
    LIFint_spec = LIFNeuronSpec(
        C_m     = 100.0,
        g_L     = 10.0,
        E_L     = -65.0,
        E_inh   = -75.0,
        E_exc   = 0.0,
        V_r     = -67.0,
        V_thr   = -52.0,
        tau_ref = 1.0,
    )
    group = LIFNeuronGroup('LIF_int', LIFint_spec)
    return group

def create_AdEx_int_group():
    """
    The Malerba AdEx interneuron.
    """
    AdExint_spec = AdExNeuronSpec(
        C_m = 200.0,
        g_L = 10.0,
        E_L = -70.0,
        E_inh = -80.0,
        E_exc = 0.0,
        V_t = -50.0,
        V_r = -58.0,
        V_thr = 0.0,
        delta = 2.0,
        a = 2.0,
        b = 10.0,
        tau_w = 30.0,
    )
    group = AdExNeuronGroup('AdEx_int', AdExint_spec)
    return group

def create_LIF_pyr_group():
    """
    The Donoso LIF pyramidal cell.
    """
    LIFpyr_spec = LIFNeuronSpec(
        C_m     = 275.0,
        g_L     = 25.0,
        E_L     = -67.0,
        E_inh   = -68.0,
        E_exc   = 0.0,
        V_r     = -60.0,
        V_thr   = -50.0,
        tau_ref = 2.0,
    )
    group = LIFNeuronGroup('LIF_pyr', LIFpyr_spec)
    return group

def create_AdEx_pyr_group():
    """
    The Malerba AdEx pyramidal cell.
    """
    AdExpyr_spec = AdExNeuronSpec(
        C_m = 200.0,
        g_L = 10.0,
        E_L = -58.0,
        E_inh = -80.0,
        E_exc = 0.0,
        V_t = -50.0,
        V_r = -46.0,
        V_thr = 0.0,
        delta = 2.0,
        a = 2.0,
        b = 100.0,
        tau_w = 120.0,
    )
    group = AdExNeuronGroup('AdEx_pyr', AdExpyr_spec)
    return group


class LIFNeuronGroup(BaseUnitGroup):

    base_dtypes = {'spikes':'?'}
    base_variables = ('x', 'y', 'v', 'spikes', 't_spike', 'g_total',
                      'g_total_inh', 'g_total_exc', 'excitability',
                      'I_app', 'I_net', 'I_leak', 'I_total_inh',
                      'I_total_exc', 'I_noise')

    def __init__(self, name, spec, gain_param=(0, 10, 0.1, 'gain')):
        """
        Construct the neuron group by computing layouts and noise.
        """
        # Compute (or retrieve) the spatial layout to get the number of units
        self.layout = HexLayout(spec.layoutspec)
        self.layout.compute(context=State.context)
        self.N = self.layout.N

        BaseUnitGroup.__init__(self, self.N, name, spec=spec)

        # Set up the intrinsic noise inputs depending on interaction mode
        self.oup = OUProcess(N=self.N, tau=spec.tau_eta, seed=self.name)
        if State.interact:
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

        # Initialize metrics
        self.activity = FiringRateWindow(self)

        # Intialize some variables
        self.x = self.layout.x
        self.y = self.layout.y
        self.t_spike = -inf
        self.LFP_uV = 0.0  # uV, LFP signal from summed net synaptic input

        # Pre-compute intermediate values
        self.dt_over_C_m = State.dt / self.p.C_m

        # Map from transmitters to reversal potentials
        self.E_syn = dict(GABA=self.p.E_inh, AMPA=self.p.E_exc,
                NMDA=self.p.E_exc, glutamate=self.p.E_exc, L=self.p.E_exc)

        State.network.add_neuron_group(self)
        State.context.out(self, prefix='NeuronInit')

    def add_synapses(self, synapses):
        """
        Add afferent synaptic pathway to this neuron group.
        """
        if synapses.post is not self:
            State.context.out('{} does not project to {}', synapses.name,
                    self.name, prefix='NeuronGroup', error=True)
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
        self.update_conductances()
        self.update_currents()
        self.update_noise()
        self.update_metrics()

    def update_voltage(self):
        """
        Evolve the membrane voltage for neurons according to input currents.
        """
        self.v += self.dt_over_C_m * self.I_net

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

    def update_conductances(self):
        """
        Update total input conductances for afferent synapses.
        """
        self.g_total_exc = self.p.g_tonic_exc
        self.g_total_inh = self.p.g_tonic_inh
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
        self.I_noise     = self.p.sigma * self.eta
        self.I_app       = self.p.I_DC_mean * self.excitability
        self.I_net       = self.I_leak + self.I_app + self.I_noise + \
                               self.I_total_exc + self.I_total_inh

    def update_noise(self):
        """
        Update the intrinsic noise source.
        """
        if State.interact:
            self.eta = next(self.eta_gen)
            return
        self.eta = self.oup.eta[...,State.n]

    def update_metrics(self):
        """
        Update metrics such as activity calculations.
        """
        self.LFP_uV = -(self.I_total_exc.sum() + self.I_total_inh.sum()) \
                            / Config.g_LFP
        self.activity.update()

    def reset(self):
        """
        Reset the neuron model parameters to spec defaults.
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


class AdExNeuronGroup(LIFNeuronGroup):

    extra_variables = ('w',)

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
        self.v += self.dt_over_C_m * (
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
