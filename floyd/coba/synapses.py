"""
Groups of synaptic connections between a pair of neuron groups.
"""

__all__ = ('Synapses',)


from toolbox.numpy import *
from specify import Specified, Param
from roto.dicts import midlite, pprint as dict_pprint

from .matrix import pairwise_distances as distances
from .groups import BaseUnitGroup
from .delay import DelayLines
from .state import State


class Synapses(Specified, BaseUnitGroup):

    transmitter = Param('glutamate', doc="'GABA' | 'glutamate'")
    g_max       = Param(1.0, units="nS")
    tau_r       = Param(0.5, units="ms")
    tau_d       = Param(2.0, units="ms")
    tau_l       = Param(1.0, units="ms")
    tau_max     = Param(25.0, units="ms")
    failures    = Param(False, doc="enable synaptic failures")

    base_variables = ('C', 'S', 't_spike', 'dt_spike', 'g', 'g_peak', 'p_r')

    def __init__(self, pre, post, *, seed=None, **kwargs):
        self._initialized = False
        super().__init__(name=f'{pre.name}->{post.name}', N=(post.N, pre.N),
                **kwargs)

        self.E_syn = post.E_syn[self.transmitter]
        self.s = self.scale_constant()
        self.pre = pre
        self.post = post
        self.recurrent = pre is post

        self.delay = None
        self.g_peak = self.g_max
        self.t_spike = -inf
        self.dt_spike = -inf
        self.g_total = zeros(post.N, 'f')
        self.distances = distances(c_[post.x, post.y], c_[pre.x, pre.y])

        self.out('{} = {:.4g} mV', self.name,
                self.postsynaptic_potential().mean(), prefix='PostPotential')

        if 'network' in State:
            State.network.add_projection(self)
        self._initialized = True

    def scale_constant(self):
        """
        Return the conductance normalizing constant (s) given rise and decay.
        """
        delta = arange(0, self.tau_max + State.dt, State.dt)
        g_t = exp(-delta/self.tau_d) - exp(-delta/self.tau_r)
        return 1 / g_t.max()

    def postsynaptic_potential(self):
        """
        Postsynaptic potential at rest for membrane and conductance parameters.
        """
        delta = arange(0, self.tau_max + State.dt, State.dt)
        g_t = exp(-delta/self.tau_d) - exp(-delta/self.tau_r)
        s = 1 / g_t.max()
        dV = self.g_peak * s * trapz(g_t, dx=State.dt) * (
                                self.E_syn - self.post.E_L) / self.post.C_m
        return dV

    def display_connectivity(self):
        """
        Display the parameters and connectivity statistics for these synapses.
        """
        if not hasattr(self, 'active_post'):
            raise RuntimeError('synapses must be connected first!')

        msd = lambda x: '{:.3g} +/- {:.3g}'.format(np.nanmean(x), np.nanstd(x))
        nz = self.fanin.nonzero()
        stats = dict(
                connectivity_fraction = '{:.3f}'.format(
                                    float(self.N/(self.pre.N*self.post.N))),
                fanout_cells = f'{msd(self.fanout)}',
                fanin_cells = f'{msd(self.fanin)}',
                fanout_contacts = f'{msd(self.S.sum(axis=0))}',
                fanin_contacts = f'{msd(self.S.sum(axis=1))}',
                contacts_per_connection = '{}'.format(
                                  msd(self.S.sum(axis=1)[nz]/self.fanin[nz])),
        )
        self.out.printf(dict_pprint(stats, name=f'Connectivity({self.name})'))

    def set_delays(self, cond_velocity=0.5):
        """
        Construct spike-transmission conduction delay lines.
        """
        if not hasattr(self, 'active_post'):
            raise RuntimeError('synapses must be connected first!')
        timing = self.distances / cond_velocity
        self.delay = DelayLines.from_delay_times(self.N, timing[self.ij],
                State.dt, dtype='?')
        self.out('max delay = {} timesteps', self.delay.delays.max())

    def update(self):
        """
        Update synaptic timing based on the current presynaptic spike vector.
        """
        syn_spikes = pre_spikes = self.pre.spikes[self.j]
        if self.delay is not None:
            self.delay.set(pre_spikes)
            syn_spikes = self.delay.get()

        # If release probability is defined, randomly cause spikes to fail
        if self.failures and any_(syn_spikes):
            spk_ix = syn_spikes.nonzero()[0]
            fail = self.rnd.random_sample(spk_ix.size) >= self.p_r_j[spk_ix]
            syn_spikes[spk_ix[fail]] = False

        # Set spike times and update dts from most recent spike
        self.t_spike[self.i[syn_spikes], self.j[syn_spikes]] = State.t
        self.dt_spike[self.ij] = dt = State.t - self.t_spike[self.ij]

        # Find actively evolving conductance time-courses
        active = (dt >= self.tau_l) & (dt <= self.tau_max)
        delta = dt[active] - self.tau_l

        # Conductance matrix indexes for setting values
        inactive = invert(active)
        ij_inactive = self.i[inactive], self.j[inactive]
        ij_active = self.i[active], self.j[active]

        # Bi-exponential time-course of postsynaptic conductances, where pulses
        # during active conductances begin from the current conductance level
        self.g[ij_inactive] = 0.0
        g_t = self.g_peak[ij_active]*self.S[ij_active]*self.s*(
                    exp(-delta/self.tau_d) - exp(-delta/self.tau_r))
        self.g[ij_active] += g_t  # allow paired-pulse facilitation

        # Total conductances for connected postsynaptic neurons
        self.g_total[:] = self.g.sum(axis=1)

    def connect(self, p=True, kernel_fanout=None, allow_multapses=True):
        """
        Connect two groups of neurons with a synaptic projection.

        Note: p can be True for all-to-all connectivity; float for fixed
        probability; int for fixed fan-out; a KernelFunc on the distance
        between neurons that determines the probability of connection; or a
        full, pre-computed weight matrix (ndarray object) of the correct shape
        for all synapses, i.e., (post.N, pre.N).
        """
        if hasattr(self, 'active_post'):
            self.out('Previous connection will be overwritten', warning=True)
        self.out('Connecting {}', self.name)

        self.set(C=0, S=0)
        C = self.S  # accumulate no. of contacts if multapses allowed
        if type(p) is bool and p == True:
            C += 1
        elif np.isscalar(p) and p == float(p):
            C[:] = self.rnd.random_sample(C.shape) < p
        elif np.isscalar(p) and p == int(p):
            C[:int(p)] = 1
            for j in range(self.pre.N):
                C[:,j] = self.rnd.random_shuffle(C[:,j])
        elif hasattr(p, 'apply'):
            self._kernel_connect(C, p, kernel_fanout, allow_multapses)
        elif isinstance(p, ndarray) and p.shape == self.shape:
            self._direct_connect(C, p)
        else:
            raise ValueError(f'invalid connection specification: {p!r}')

        # Process the connectivity matrix into some useful data structures
        self.ij = C.nonzero()
        self.i, self.j = self.ij
        self.C[self.ij] = 1  # binary connectivity indicator
        self.active_post = unique(self.i)
        self.active_pre = unique(self.j)
        self.N = int(self.C.sum())
        self.fanin = self.C.sum(axis=1)
        self.fanout = self.C.sum(axis=0)
        self.convergence = {n:self.j[self.i == n] for n in self.active_post}
        self.divergence = {n:self.i[self.j == n] for n in self.active_pre}

        if self.failures:
            self.p_r_j = self.p_r[self.ij]

    def _kernel_connect(self, C, kernel, fanout, multapses):
        """
        Implement kernel-based connectivity using a parameter expression on
        pre and post variables and a fanout specification (which may be an int,
        a float on [0,1], or a SamplerType object). Connectivity matrix C is
        modified in-place.
        """
        N_post, N_pre = self.shape
        p_k = kernel(self.distances)
        if self.recurrent:
            p_k *= 1 - eye(N_pre)  # eliminate autapses
        p_k /= p_k.sum(axis=0)[AX]

        n = zeros(N_pre, 'i')
        if np.isscalar(fanout):
            if fanout == int(fanout):
                n += fanout
            elif 0.0 <= fanout <= 1.0:
                n += round(fanout*N_post)
            else:
                raise ValueError(f'invalid fanout value: {fanout!r}')
        elif hasattr(fanout, 'sample'):
            n += fanout(N_pre, state=self.rnd).astype('i')
            n[n<0] = 0
        else:
            raise ValueError(f'invalid fanout value: {fanout!r}')

        for j in range(N_pre):
            conn = self.rnd.choice(N_post, n[j], replace=multapses, p=p_k[:,j])
            for i in conn:
                C[i,j] += 1  # multapses increment connection count

    def _direct_connect(self, C, W_syn):
        """
        Implement an explicit specification of synaptic connectivity based on a
        full, pre-computed weight matrix.
        """
        assert W_syn.shape == self.shape, f'shape mismatch: {W_syn.shape}'
        C[W_syn.nonzero()] = 1
        self.g_peak = W_syn
