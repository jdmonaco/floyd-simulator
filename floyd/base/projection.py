"""
Groups of synaptic connections between a pair of neuron groups.
"""

__all__ = ('BaseProjection',)


import copy
from numbers import Number

from toolbox.numpy import *
from specify import Specified, Param
from roto.dicts import pprint as dict_pprint

from ..matrix import pairwise_distances as distances
from ..delay import DelayLines
from ..state import State

from .groups import BaseUnitGroup


class BaseProjection(Specified, BaseUnitGroup):

    """
    A generic, one-way connection from units of a source group (pre) to the 
    units of a destination group (post).
    """

    # Note: The main connection parameter, p, can take various values:
    #
    # True -- one-to-all connectivity (same as p == 1.0)
    # float (0.0, 1.0) -- fixed fan-out probability
    # int (== 1) -- one-to-one connectivity, up to min(pre.size, post.size)
    # int (> 1) -- fixed fan-out number
    # KernelFunc instance -- distance kernel for fan-out probability
    # array with shape (post.size, pre.size) -- explicit connectivity matrix
    #
    # If p is a KernelFunc object, then kernel_fanout should also be specified:
    #
    # kernel_fanout -- float, int, SamplerFunc object

    p = Param(True, doc='connectivity specification')
    kernel_fanout = Param(0.1, doc='fanout specification if p is a KernelFunc')
    allow_multapses = Param(False, doc='a connection may comprise >1 contacts')
    signal_dtype = Param('f', doc='output signal dtype')
    
    # Projection variables:
    #
    # C -- binary {0, 1} connectivity (adjacency) matrix
    # S -- integer connectivity matrix of total contacts per connection
    # delays -- transmission delay matrix (in simulation timesteps)
    # cursor -- cursor matrix indicating delay loop phase of each connection

    base_variables = ('C', 'S')
    base_dtypes = {'C': 'u1', 'S': 'u2'}

    def __init__(self, pre, post, **kwargs):
        self._initialized = False
        super().__init__(name=f'{pre.name}->{post.name}', 
            shape=(post.size, pre.size), **kwargs)

        # Set the transmitting (pre) and receiving (post) unit groups
        self.pre = pre
        self.post = post
        self.recurrent = pre is post
        self.terminal = zeros(self.shape, self.signal_dtype)  # signal readout

        if State.is_defined('network'):
            State.network.add_projection(self)
            
        self._initialized = True
        self._connected = False
        self._usedelays = False

    def print_stats(self):
        """
        Print the parameters and connectivity statistics for these synapses.
        """
        if not self._connected:
            raise RuntimeError('this projection is currently not connected')

        msd = lambda x: '{:.3g} +/- {:.3g}'.format(np.nanmean(x), np.nanstd(x))
        nz = self.fanin.nonzero()
        stats = dict(
            connectivity_fraction = '{:.3f}'.format(
                float(self.N_conns / (self.pre.size * self.post.size))),
            fanout_units = f'{msd(self.fanout)}',
            fanin_units = f'{msd(self.fanin)}',
            fanout_contacts = f'{msd(self.S.sum(axis=0))}',
            fanin_contacts = f'{msd(self.S.sum(axis=1))}',
            contacts_per_connection = '{}'.format(
                msd(self.S.sum(axis=1)[nz] / self.fanin[nz])),
        )
        self.out.printf(dict_pprint(stats, name=f'Projection({self.name})'))

    def set_delays(self, delays, timebased=False, dense=True, delay_dtype='u4'):
        """
        Add dense or sparse signal delay lines to the connections.
        """
        if dense == False:
            # TODO Sparse delays like previous code
            raise NotImplementedError('sparse delays')
        
        if timebased:
            delays = np.round(delays / State.dt)

        self.delays = 1 + expand_dims(delays, -1).astype(delay_dtype)
        self.cursor = zeros(self.delays.shape, 'i4')
        self.delay_lines = zeros(self.shape + (self.delays.max() + 1,), 
                                    self.signal_dtype)

        self._usedelays = True

    def update(self):
        """
        Update the projection with current transmitter (pre) output signals.
        """
        if self._usedelays:
            np.put_along_axis(self.delay_lines, self.cursor, 
                              self.pre.output[AX], axis=2)
            self.cursor[:] = (self.cursor + 1) % self.delays
            self.terminal[:] = np.take_along_axis(self.delay_lines, 
                                                  self.cursor, axis=2)
            return
        
        self.terminal[:] = self.pre.output[AX]
        
    def connect(self, **specs):
        """
        Build specified connectivity matrixes for this projection.
        """
        if self._connected:
            self.out('Overwriting previous connection', warning=True)

        # Update projection specs if provided in keyword arguments
        oldp = copy.deepcopy(self.p)
        self.update(specs)

        # Set connection (C) and contact (S) matrices to zero in order to
        # build new projections based on the contact matrix
        self.set(C=0, S=0)

        # Set or sample the connections according to connection spec (p)
        if hasattr(self.p, 'apply'):
            self.kernel_connect()
        elif isinstance(self.p, ndarray) and self.p.shape == self.shape:
            self.S[self.p.nonzero()] = 1
        elif type(self.p) is bool:
            self.S += int(self.p)
        elif isinstance(self.p, numbers.Number):
            if isinstance(self.p, int):
                if self.p == 0:
                    pass
                elif self.p == 1:
                    self.S = eye(*self.shape)
                elif 1 < self.p <= self.post.size:
                    if self.allow_multapses:
                        for j in range(self.pre.size)):
                            conn = self.rnd.randint(self.post.size, size=self.p)
                            for i in conn:
                                self.S[i,j] += 1  # increment contact counts
                    else:
                        self.S[:self.p] = 1
                        for j in range(self.pre.size):
                             self.rnd.shuffle(self.S[:,j])
                else:
                    raise ValueError(f'invalid fanout: {self.p!r}')
            elif 0 <= self.p <= 1:
                self.S = self.rnd.random_sample(self.S.shape) < self.p
            else:
                raise ValueError(f'invalid fanout: {self.p!r}')
        else:
            self.p = oldp
            raise ValueError(f'invalid fanout: {self.p!r}')

         # Eliminate autapses (cycles) if this is a recurrent projection
        if self.recurrent:
            self.S *= 1 - eye(self.pre.size)

        # Compute the connectivity stats and maps
        self.compute_connectivity()

    def compute_connectivity(self):
        """
        Process the connectivity matrices into stats and useful data structures.
        """
        self.ij = self.S.nonzero()
        self.i, self.j = self.ij  # unzipped matrix indices of all connections
        self.C[self.ij] = 1  # binary {0,1} connectivity (adjacency) matrix
        self.active_post = unique(self.i)
        self.active_pre = unique(self.j)
        self.N_connections = int(self.C.sum())
        self.N_contacts = int(self.S.sum())
        self.fanout = self.C.sum(axis=0)
        self.fanin = self.C.sum(axis=1)
        self.fanout_contacts = self.S.sum(axis=0)
        self.fanin_contacts = self.S.sum(axis=1)
        self.convergence = {n:self.j[self.i == n] for n in self.active_post}
        self.divergence = {n:self.i[self.j == n] for n in self.active_pre}

        self._connected = True
        self.out('Projection connected: {}', self.name)

    def kernel_connect(self):
        """
        Implement kernel-based connectivity using a RelativeDistanceKernel 
        object (or similar) and a fanout specification (which may be an int,
        a float on [0,1], or a SamplerType object). 
        
        The contact-count matrix S is modified in-place to reflect the 
        resulting projection connectivity.
        """
        # Compute the kernel-valued densities, eliminate autapses, and 
        # normalize probability distributions
        p_k = self.p.apply(self)  
        if self.recurrent:
            p_k *= 1 - eye(self.pre.size)  # eliminate autapses
        p_k /= p_k.sum(axis=0)[AX]

        # Set the fanouts for each pre unit according to sampling parameters
        f_k = self.kernel_fanout
        fanout = zeros(self.pre.size, 'i')
        if hasattr(f_k, 'sample'):
            fanout += f_k(self.pre.size, state=self.rnd).astype('i')
            fanout[fanout<0] = 0
        elif isinstance(f_k, numbers.Number):
            if isinstance(f_k, int):
                if f_k == 0:
                    pass
                elif f_k > 0:
                    fanout += f_k
                else:
                    raise ValueError(f'invalid fanout: {f_k!r}')
            elif 0 <= f_k <= 1:
                fanout += round(f_k * self.post.size)
            else:
                raise ValueError(f'invalid fanout: {f_k!r}')
        else:
            raise ValueError(f'invalid fanout: {f_k!r}')

        # Sample the calculated number of fanout connections
        for j in range(self.pre.size):
            conn = self.rnd.choice(self.post.size, fanout[j], p=p_k[:,j],
                        replace=self.allow_multapses) 
            for i in conn:
                self.S[i,j] += 1  # increment contact counts