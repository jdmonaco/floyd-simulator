"""
Create and handle caching storage for hexagonal cell layouts.
"""

import queue

import matplotlib.pyplot as plt

from pouty import printf, box
from toolbox.numpy import *
from toolbox.constants import twopi
from tenko.persistence import AutomaticCache

from .spec import paramspec


HexLayoutSpec = paramspec('HexLayoutSpec',
            scale=0.2,
            radius=1.0,
            origin=(0,0),
            extent=(-1,1,-1,1),
            orientation=0.0,
)


class HexagonalDiscLayout(AutomaticCache):

    data_root = 'layout'
    key_params = ('scale', 'radius', 'origin', 'extent', 'orientation')
    cache_attrs = ('x', 'y')
    save_attrs = ('N',)

    def __init__(self, spec):
        """
        Create a new layout from the given HexagonalDiscLayoutSpec.
        """
        spec.validate()
        self.p = spec
        AutomaticCache.__init__(self, **spec)

    def _compute(self):
        """
        Find the specified hexagonal grid circumscribed by the CA disc.
        """
        H = []
        scale = self.p.scale
        extent = self.p.extent
        angles = linspace(self.p.orientation, self.p.orientation + twopi, 7)[:-1]
        Q = queue.deque()
        Q.append(self.p.origin)

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
        d = hypot(H[:,0] - self.p.origin[0], H[:,1] - self.p.origin[1])
        indisc = d < self.p.radius
        self.x, self.y = H[indisc].T
        self.N = self.x.size

    def plot(self):
        """
        Bring up a figure window and plot example traces.
        """
        plt.ioff()
        f = plt.figure(num=self.hash, clear=True, figsize=(5,5.5))
        f.suptitle(f'Hexagonal Disc Layer\nscale={self.p.scale:.4f}, '
               f'orientation={self.p.orientation:.4f}, extent={self.p.extent}')
        ax = plt.axes()

        ax.scatter(self.x, self.y, s=12, linewidths=0, edgecolors='none')
        ax.set_xlim(self.p.extent[0], self.p.extent[1])
        ax.set_ylim(self.p.extent[2], self.p.extent[3])
        ax.axis('equal')

        plt.ion()
        plt.show()
