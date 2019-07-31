"""
Network graph figure.
"""

import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from toolbox.numpy import *

from .base import FloydObject
from .config import Config
from .state import State


class NetworkGraph(FloydObject):

    """
    Create and update a graph plot of the network structure
    """

    def __init__(self):
        FloydObject.__init__(self)
        if 'network' in State:
            self.G = State.network.G
        else:
            raise RuntimeError('network has not been initialized')

        self.ax = None
        self.fig = None
        self.cmap = None
        self.lw = None
        self.N = len(self.G)
        self.N_groups = len(State.network.neuron_groups)
        self.pos = None
        self.nodes = None
        self.labels = []
        self.arrows = None
        self.artists = []
        self.colors = ones((self.N, 4))

        State.netgraph = self

    def __str__(self):
        return f'{self.klass}(N={self.N})'

    def closefig(self):
        """
        Close the figure window with the graph plot.
        """
        if self.fig is not None:
            plt.close(self.fig)

    def plot(self, ax=None, figsize=(4, 4), lw=2, cmap='plasma', alpha=0.7,
        label_offset=0.12, axes_zoom=0.18):
        """
        Draw the initial graph and store node, edge, and artist information.
        """
        if ax is None:
            self.fig = plt.figure(num='network-graph', clear=True,
                    figsize=figsize, dpi=Config.screendpi)
            self.ax = plt.axes([0,0,1,1])
            self.ax.set_axis_off()
        if type(ax) is str and ax in State.simplot.axes:
            self.ax = State.simplot.axes[ax]
        self.cmap = plt.get_cmap(cmap)
        self.lw = lw

        # Get the layout position of the graph
        self.pos = pos = nx.nx_agraph.graphviz_layout(self.G)

        # Draw the nodes.
        # Node shapes can be any MPL marker: 'so^>v<dph8'.
        # matplotlib.collections.PathCollection instance
        self.nodes = nx.draw_networkx_nodes(self.G, pos, ax=self.ax,
                node_shape='o', node_size=1e3, alpha=alpha, linewidths=1,
                vmin=0, vmax=1, edgecolors='k', node_color='w')
        self.colors[:,-1] = alpha
        self.nodes.set_zorder(-10)

        # Draw the labels.
        # Dict of labels keyed on the nodes.
        labels = nx.draw_networkx_labels(self.G, pos, ax=self.ax,
                font_size=12, font_color='#2c88f0', font_weight='bold',
                zorder=10)
        self.labels = list(labels.values())

        # Add vertical offset and path effects outline to labels to avoid
        # visual conflict with node and arrows
        dx = label_offset * float(diff(self.ax.get_ylim()))
        dy = label_offset * float(diff(self.ax.get_ylim()))
        ymid = np.mean(self.ax.get_ylim())
        offset = max(dx, dy)
        for label in self.labels:
            x, y = label.get_position()
            if y >= ymid:
                label.set_position((x, y + offset))
            else:
                label.set_position((x, y - offset))
            label.set_path_effects([
                path_effects.Stroke(linewidth=3, foreground='white'),
                path_effects.Normal(),
            ])

        # Zoom to accommodate large labels and label offsets, etc.
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        dx = axes_zoom * (xmax - xmin)
        dy = axes_zoom * (ymax - ymin)
        self.ax.set_xlim(xmin - dx, xmax + dx)
        self.ax.set_ylim(ymin - dy, ymax + dy)

        # Draw the edges.
        # List of matplotlib.patches.FancyArrowPatch
        self.arrows = nx.draw_networkx_edges(self.G, pos, ax=self.ax,
                alpha=alpha, arrowstyle='-|>', arrowsize=36, arrows=True,
                connectionstyle='angle3')
        for arrow in self.arrows:
            arrow.set_zorder(0)

        # Give stimulators fancy arrows and inhibitory synapses bracket arrows.
        # Set linewidths to Z-score of total connection strength.
        if State.network.synapses:
            W, mu, sigma = self.zscore_weights()
        for i, edge in enumerate(self.G.edges.items()):
            (A, B), attrs = edge
            arrow = self.arrows[i]
            if A.startswith('Stim'):
                arrow.set_arrowstyle('wedge')
                arrow.set_color('#25f077')
                arrow.set_alpha(alpha)
                continue
            S = attrs['object']
            if S.transmitter == 'GABA':
                arrow.set_arrowstyle('-[')
            if sigma == 0.0:
                arrow.set_linewidth(lw)
                continue
            arrow.set_linewidth(lw * (W[i] - mu) / sigma)

        self.artists = [self.nodes]
        self.artists.extend(self.labels)
        self.artists.extend(self.arrows)

        if 'simplot' in State:
            State.simplot.register_network_graph(self)

    def zscore_weights(self):
        """
        Return (w, w_mu, w_sigma) tuples for Z-scoring connection strengths.
        """
        # Get all the fanins and conductances to Z-score them
        syn_strength = []
        for S in State.network.synapses:
            post = S.post.name
            pre = S.pre.name
            syn_strength.append(
                S.post.g[f'g_{post}_{pre}'] * S.fanin.mean() * S.p.g_max)
        syn_strength = array(syn_strength)
        syn_mu = syn_strength.mean()
        syn_sigma = syn_strength.std()
        return syn_strength, syn_mu, syn_sigma

    def update(self):
        """
        Update the network graph with connection strength and neuron metrics.
        """
        # Set the node face colors according to neuron group "pulse"
        self.colors[:] = self.nodes.get_facecolors()
        for i, node in enumerate(self.G.nodes.items()):
            name, attrs = node
            if 'object' not in attrs or name.startswith('Stim'):
                continue
            group = attrs['object']
            self.colors[i] = self.cmap(group.pulse())
        self.nodes.set_facecolors(self.colors)

        # Update linewidths to Z-score of total connection strength
        if State.network.synapses:
            W, mu, sigma = self.zscore_weights()
        for i, edge in enumerate(self.G.edges.items()):
            (A, B), attrs = edge
            if 'object' not in attrs or A.startswith('Stim'):
                continue
            S = attrs['object']
            arrow = self.arrows[i]
            if sigma == 0.0:
                arrow.set_linewidth(lw)
                continue
            arrow.set_linewidth(self.lw * (W[i] - mu) / sigma)
