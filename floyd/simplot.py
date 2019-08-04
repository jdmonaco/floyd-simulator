"""
Simulation plotting for interactive dashboards or batched movie creation.
"""

__all__ = ['SimulationPlotter', ]


import functools

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tenko.base import TenkoObject
from roto.dicts import merge_two_dicts
from roto.null import Null

from .traces import RealtimeTracesPlot
from .state import State, RunMode


def block_record_mode(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if State.run_mode == RunMode.RECORD:
            return Null
        return func(*args, **kwargs)
    return wrapper


class SimulationPlotter(TenkoObject):

    """
    Manage axes, plots, data traces, and labels for interactive visualization.
    """

    def __init__(self, nrows=1, ncols=1):
        """
        After construction, use `gs` to create subplot specs to pass to the
        `set_axes()` method.
        """
        super().__init__()
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = (State.figw, State.figh)
        if State.run_mode == RunMode.RECORD:
            self.fig = Null
        else:
            self.fig = plt.figure(num='simplot', clear=True,
                                  figsize=self.figsize, dpi=State.figdpi)
            plt.ioff()
        self.gs = GridSpec(nrows, ncols, self.fig, 0, 0, 1, 1, 0, 0)
        self.axes = {}
        self.axes_objects = []
        self.plots = []
        self.artists = []
        self.traceplots = []
        self.initializer = None
        self.network_graph = None
        self.network_graph_in_fig = False

        State.simplot = self

    def __contains__(self, axname):
        """
        Test whether an axes with the given name is registered for plotting.
        """
        return axname in self.axes

    @block_record_mode
    def set_axes(self, **named_axes):
        """
        Create named axes plots from subplot specs based on the `gs` gridspec.
        """
        for name, spec in named_axes.items():
            if isinstance(spec, mpl.axes.Axes):
                ax = spec
            else:
                ax = self.fig.add_subplot(spec)
            self.axes[name] = ax
            self.axes_objects.append(ax)

        for ax in self.axes_objects:
            ax.set_axis_off()

    @block_record_mode
    def get_axes(self, *names):
        """
        Return a list of axes objects keyed by name (or a single axes).
        """
        if len(names) == 1:
            return self.axes[names[0]]
        return [self.axes[name] for name in names]

    @block_record_mode
    def update_plots(self):
        """
        Update the timestamp and all registered plots with new data.
        """
        [updater() for _, updater in self.plots]
        [rtp.update_plots() for rtp in self.traceplots]
        if State.run_mode == RunMode.INTERACT or self.network_graph_in_fig:
            self.network_graph.update()
        return self.fig

    @block_record_mode
    def update_traces_data(self):
        """
        Update only the data trace plots (for continuous tracing).
        """
        [rtp.update_data() for rtp in self.traceplots]

    @block_record_mode
    def register_plot(self, handle, updater):
        """
        Register a plot to be updated using the given update function.

        Note: `updater` a callback that should update the plot data.
        """
        self.plots.append((handle, updater))
        self.out('Artist: {}', handle, prefix='SimPlotRegistrar')

    @block_record_mode
    def register_network_graph(self, netgraph):
        """
        Set a NetworkGraph instance to be updated during the simulation.
        """
        self.network_graph = netgraph
        self.network_graph_in_fig = netgraph.ax in self.axes_objects
        self.out('Graph: {} (main figure = {})', netgraph,
                self.network_graph_in_fig, prefix='SimPlotRegistrar')

    @block_record_mode
    def add_realtime_traces_plot(self, *traceplot, **tracekw):
        """
        Create and register a RealtimeTracesPlot with the simulation.

        Note: All keyword arguments are passed to the RealtimeTracesPlot
        constructor. Callback updaters should be used for all traces.
        """
        rtp = RealtimeTracesPlot(*traceplot, **tracekw)
        self.out(f'Trace: {rtp!r}', prefix='SimPlotRegistrar')
        self.traceplots.append(rtp)

    @block_record_mode
    def draw_borders(self, color='k', linewidth=1, bottom=None, right=None):
        """
        Draw borders around each of the plots.

        Arguments
        ---------
        color : mpl color spec
            Line color to use for the borders

        linewidth : float
            Line width to use for the borders

        bottom : tuple of axes names, optional
            Axes with bottom != 0 but which require a bottom border line

        right : tuple of axes names, optional
            Axes with right != 1 but which require a right border line
        """
        for name, ax in self.axes.items():
            _, b, r, _ = ax.get_position().extents
            pkw = dict(c=color, lw=linewidth, transform=ax.transAxes)
            ax.plot([0,0], [0,1], **pkw)  # left
            ax.plot([0,1], [1,1], **pkw)  # top
            if b == 0.0 or (bottom is not None and name in bottom):
                ax.plot([0,1], [0,0], **pkw)  # bottom
            if r == 1.0 or (right is not None and name in right):
                ax.plot([1,1], [0,1], **pkw)  # right

    @block_record_mode
    def init(self, initializer):
        """
        Run the figure initialization function (only in interactive mode).
        """
        self.initializer = initializer
        if State.run_mode == RunMode.INTERACT:
            self.init_timestamp()
            initializer()
            self.closefig()

    @block_record_mode
    def closefig(self):
        """
        Close the figure.
        """
        plt.close(self.fig)
        if self.network_graph is not None:
            self.network_graph.closefig()

    def get_all_artists(self):
        """
        Save the list of artists in batch mode to be returned from initializer.
        """
        if State.run_mode == RunMode.INTERACT:
            return Null

        if self.artists:
            return self.artists

        self.init_timestamp()

        for plot, _ in self.plots:
            if type(plot) is list:
                self.artists.extend(plot)
            elif isinstance(plot, mpl.artist.Artist):
                self.artists.append(plot)
            else:
                self.out(plot, prefix='UnknownPlotType', warning=True)

        for rtp in self.traceplots:
            self.artists.extend(rtp.plot())

        if self.network_graph is not None:
            if self.network_graph.ax in self.axes_objects:
                self.artists.extend(self.network_graph.artists)

        return self.artists

    def init_timestamp(self, xy=(0.02, 0.02), axname=None, **txt):
        """
        Create the text object for the timestamp.
        """
        if hasattr(self, 'timestamp'):
            return

        ax = self.axes_objects[0] if axname is None else self.axes[axname]
        fmt = dict(color='gray', transform=ax.transAxes, va='bottom',
                    ha='left', fontweight='normal')
        fmt.update(txt)
        self.timestamp = ax.text(xy[0], xy[1], '', **fmt)

        def update_timestamp():
            self.timestamp.set_text(f't = {State.t:.2f} ms')
        self.register_plot(self.timestamp, update_timestamp)
