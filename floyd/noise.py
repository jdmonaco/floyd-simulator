"""
Create frozen noise sigals for background synaptic stochasticity.
"""

__all__ = ['OUNoiseProcess']


import matplotlib.pyplot as plt

from pouty import printf, box

from toolbox.numpy import *
from tenko.persistence import AutomaticCache
from tenko.base import TenkoObject
from specify import Specified, Param

from .state import State


class OUNoiseProcess(Specified, AutomaticCache, TenkoObject):

    N = Param(1, doc='number of noise signals')
    tau = Param(10.0, doc='time-constant of noise filter')
    rand_init = Param(True, doc='random initialization')
    seed = Param(__name__, doc='string, seed key for RNG state')
    nonnegative = Param(False, doc='force non-negative outputs from process')

    _data_root = 'noise'
    _key_params = ('N', 'tau', 'rand_init', 'seed', 'nonnegative', 'duration',
                   'dt')
    _cache_attrs = ('t', 'eta')
    _save_attrs = ('Nt', 'std')

    def __init__(self, **kwargs):
        super().__init__(spec_produce=self._key_params,
                         duration=State.duration, dt=State.dt, **kwargs)

    def _compute(self):
        """
        Integrate the O-U processes to construct standardized noise signals.
        """
        self.t = t = arange(0, State.duration + State.dt, State.dt)

        self.Nt = t.size
        self.Nprogress = 0
        printf('OUNoise: ')

        tau = self.tau
        dt = State.dt
        sqrt_dt = sqrt(dt)

        eta = empty((self.N, self.Nt), 'd')
        randn = self.rnd.standard_normal(size=eta.shape)
        eta[:,0] = randn[:,0] if self.rand_init else 0
        for i in range(1, self.Nt):
            y = eta[:,i-1]
            eta[:,i] = y - y*dt/tau + randn[:,i]*sqrt_dt/tau
            if self.nonnegative:
                eta[eta[:,i]<0,i] = 0.0
            self.progressbar(i)
        printf('\n')

        self.mean = eta.mean(axis=1).mean()
        self.std = eta.std(axis=1).mean()
        self.eta = (eta - self.mean) / self.std

    def generator(self):
        """
        Get a generator of standardized noise vectors for interactive mode.
        """
        self._compute()  # needed for standardization (cf. yield line)

        tau = self.tau
        dt = State.dt
        sqrt_dt = sqrt(dt)

        eta = empty(self.N, 'd')
        randn = self.rnd.standard_normal(size=self.N)
        eta[:] = randn if self.rand_init else 0
        while True:
            y = eta[:]
            randn = self.rnd.standard_normal(size=self.N)
            eta[:] = y - y*dt/tau + randn*sqrt_dt/tau
            if self.nonnegative:
                eta[eta<0] = 0.0
            yield eta / self.std

    def progressbar(self, n, filled=False, color='ochre', width=80):
        """
        Once-per-update console output for a progress bar.
        """
        pct = (n + 1) / self.Nt
        barpct = self.Nprogress / width
        while barpct < pct:
            box(filled=filled, c=color)
            self.Nprogress += 1
            barpct = self.Nprogress / width

    def plot(self, max_lines=10):
        """
        Bring up a figure window and plot example traces.
        """
        plt.ioff()
        f = plt.figure(num=self.hash, clear=True, figsize=(7,4))
        f.suptitle(f'Ornstein-Uhlenbeck Process\ntau={self.tau:.4f}, '
                   f'dt={self.dt:.4f}, {min(max_lines,self.N)}/{self.N} shown')
        ax = plt.axes()

        if self.N <= max_lines:
            data = self.eta.T
        else:
            data = self.eta[random.choice(self.N, max_lines, replace=False)].T

        ax.plot(self.t, data, lw=0.8, alpha=0.7)
        ax.set_xlim(self.t[0]-self.dt, self.t[-1]+self.dt)
        ax.axhline(c='k', zorder=-10)
        ymax = absolute(ax.get_ylim()).max()
        ax.set_ylim(-ymax, ymax)

        plt.ion()
        plt.show()
