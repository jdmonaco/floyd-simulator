"""
Spatial layouts for neuron groups including hexagonal grids.
"""

__all__ = ['HexagonalDiscLayout', ]


import queue

import matplotlib.pyplot as plt

from toolbox.numpy import *
from pouty import printf, box
from toolbox.constants import twopi
from specify import Specified, Param
from tenko.persistence import AutomaticCache
from tenko.base import TenkoObject


class HexagonalDiscLayout(Specified, AutomaticCache, TenkoObject):

    """
    Spatial layout of a hexagonal grid circumscribed by a circle.
    """

    scale       = Param(default=0.1, units='mm', doc='scale of grid spacing')
    radius      = Param(default=0.5, units='mm', doc='disc radius')
    origin      = Param(default=(0.5, 0.5), units='mm', doc='(x,y) origin coordinates')
    extent      = Param(default=(0, 1, 0, 1), units='mm', doc='(left, right, bottom, top) extent')
    orientation = Param(default=0.0, units='radians', doc='angle of grid orientation')

    _data_root   = 'layout'
    _key_params  = ('scale', 'radius', 'origin', 'extent', 'orientation')
    _cache_attrs = ('x', 'y')
    _save_attrs  = ('N',)

    def __init__(self, **specs):
        super().__init__(spec_produce=self._key_params, **specs)

    def _compute(self):
        """
        Find the specified hexagonal grid circumscribed by the CA disc.
        """
        H = []
        scale = self.scale
        extent = self.extent
        angles = linspace(self.orientation, self.orientation + twopi, 7)[:-1]
        Q = queue.deque()
        Q.append(self.origin)

        printf('HexDisc: ', c='ochre')
        while Q:
            v = Q.pop()
            existing = False
            for u in H:
                if isclose(v[0], u[0]) and isclose(v[1], u[1]):
                    existing = True
                    break
            if existing:
                continue
            if not (extent[0] <= v[0] < extent[1]):
                continue
            if not (extent[2] <= v[1] < extent[3]):
                continue
            Q.extend([(v[0] + scale*cos(a), v[1] + scale*sin(a))
                      for a in angles])
            H.append(v)
            box(filled=False, c='purple')
        print()

        # Sort grid points from top-left to bottom-right
        Harr = array(H)
        H = Harr[lexsort(tuple(reversed(tuple(Harr.T))))]

        # Remove points outside the circle of the CA disc
        d = hypot(H[:,0] - self.origin[0], H[:,1] - self.origin[1])
        indisc = d < self.radius
        self.x, self.y = H[indisc].T
        self.N = self.x.size

    def plot(self):
        """
        Bring up a figure window and plot example traces.
        """
        plt.ioff()
        f = plt.figure(num=self.hash, clear=True, figsize=(5,5.5))
        f.suptitle(f'Hexagonal Disc Layer\nscale={self.scale:.4f}, '
               f'orientation={self.orientation:.4f}, extent={self.extent}')
        ax = plt.axes()

        ax.scatter(self.x, self.y, s=12, linewidths=0, edgecolors='none')
        ax.set_xlim(self.extent[0], self.extent[1])
        ax.set_ylim(self.extent[2], self.extent[3])
        ax.axis('equal')

        plt.ion()
        plt.show()
