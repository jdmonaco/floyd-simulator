"""
Network graph figure.
"""

__all__ = ('NetworkGraph',)


import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from toolbox.numpy import *

from tenko.base import TenkoObject
from tenko.state import Tenko

from .config import Config
from .state import State


class NetworkGraph(TenkoObject):

    """
    Create and update a graph plot of the network structure
    """

    def __init__(self):
        super().__init__()
        if 'network' in State:
            self.G = State.network.G
        else:
            raise RuntimeError('network has not been initialized')

        self.ax = None
        self.fig = None
        self.cmap = None
        self.lw = None
        self.N = len(self.G)
        self.N_groups = len(State.network.neuron_groups) + \
                        len(State.network.input_groups)
        self.pos = None
        self.nodes = []
        self.labels = []
        self.arrows = None
        self.artists = []
        self.metric_func = None
        self.metric_range = None
        self.colors = ones((self.N, 4))

        State.netgraph = self

    def closefig(self):
        """
        Close the figure window with the graph plot.
        """
        if self.fig is not None:
            plt.close(self.fig)

    def plot(self, ax=None, metric='active_fraction', metric_range=(0, 1),
        figsize=(4, 4), lw=2, cmap='plasma', alpha=0.8, label_offset=0,
        node_size=1e3, axes_zoom=0.16):
        """
        Draw the initial graph and store node, edge, and artist information.
        """
        if ax is None:
            self.fig = plt.figure(num='network-graph', clear=True,
                    figsize=figsize, dpi=Tenko.screendpi)
            self.ax = plt.axes([0,0,1,1])
            self.ax.set_axis_off()
        else:
            self.ax = State.simplot.get_axes(ax)
        self.cmap = plt.get_cmap(cmap)
        self.lw = lw
        self.metric_func = metric
        self.metric_min = metric_range[0]
        self.metric_ptp = diff(metric_range)

        # Get the layout position of the graph
        self.pos = pos = nx.nx_agraph.graphviz_layout(self.G)

        # Draw the nodes.
        # Node shapes can be any MPL marker: 'so^>v<dph8'.
        # matplotlib.collections.PathCollection instance
        self.nodes = nx.draw_networkx_nodes(self.G, pos, ax=self.ax,
                node_shape='o', node_size=node_size, alpha=alpha, linewidths=1,
                vmin=0, vmax=1, edgecolors='k', node_color='w')
        self.colors[:,-1] = alpha
        self.nodes.set_zorder(-10)

        # Draw the labels.
        # Dict of labels keyed on the nodes.
        labels = nx.draw_networkx_labels(self.G, pos, ax=self.ax,
                font_size=8, font_color='#2c88f0', font_weight='bold')
        self.labels = list(labels.values())
        [l.set_zorder(10) for l in self.labels]

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
                path_effects.Stroke(linewidth=2.5, foreground='white'),
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
        if self.G.edges:
            self.arrows = nx.draw_networkx_edges(self.G, pos, ax=self.ax,
                    alpha=alpha, arrowstyle='-|>', arrowsize=12, arrows=True,
                    node_size=node_size, connectionstyle='arc3,rad=-0.13')
            for arrow in self.arrows:
                arrow.set_zorder(0)
                arrow.set_alpha(alpha)

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
            if hasattr(S, 'transmitter') and S.transmitter == 'GABA':
                arrow.set_arrowstyle('->')
            if sigma == 0.0:
                arrow.set_linewidth(lw)
                continue
            arrow.set_linewidth(lw * (W[i] - mu) / sigma)

        self.artists = [self.nodes]
        self.artists.extend(self.labels)
        if self.arrows is not None:
            self.artists.extend(self.arrows)

        if 'simplot' in State:
            State.simplot.register_network_graph(self)

    def zscore_weights(self):
        """
        Return (w, w_mu, w_sigma) tuples for Z-scoring connection strengths.
        """
        # Get all the fanins and conductances to Z-score them
        syn_strength = []
        for S in State.network.synapses.values():
            post = S.post.name
            pre = S.pre.name
            if 'g_peak' in S:
                S_peak = S.g_peak.mean()
            elif 'I_peak' in S:
                S_peak = S.I_peak.mean()
            else:
                S_peak = 1.0
            syn_strength.append(
                S.post[f'g_{post}_{pre}'] * S.fanin.mean() * S_peak)
        syn_strength = array(syn_strength)
        syn_mu = syn_strength.mean()
        syn_sigma = syn_strength.std()
        return syn_strength, syn_mu, syn_sigma

    def update(self):
        """
        Update the network graph with connection strength and neuron metrics.
        """
        # Set the node face colors according to neuron group metric
        self.colors[:] = self.nodes.get_facecolors()
        for i, node in enumerate(self.G.nodes.items()):
            name, attrs = node
            if 'object' not in attrs or name.startswith('Stim'):
                continue
            group = attrs['object']
            metric = getattr(group, self.metric_func)()
            metric_norm = (metric - self.metric_min) / self.metric_ptp
            self.colors[i] = self.cmap(metric_norm)
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
                arrow.set_linewidth(self.lw)
                continue
            arrow.set_linewidth(self.lw * (W[i] - mu) / sigma)
