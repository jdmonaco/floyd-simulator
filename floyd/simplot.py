"""
Simulation plotting for interactive dashboards or batched movie creation.
"""

__all__ = ('SimulationPlotter',)


import functools

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects

from tenko.base import TenkoObject
from roto.dicts import merge_two_dicts
from roto.null import Null

from .traces import RealtimeTracesPlot
from .state import State, RunMode


def block_record_mode(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if State.run_mode in (RunMode.RECORD, RunMode.SAMPLE):
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
        if State.run_mode in (RunMode.RECORD, RunMode.SAMPLE):
            self.fig = Null
        else:
            self.fig = plt.figure(num='simplot', clear=True,
                                  figsize=self.figsize, dpi=State.figdpi)
            if 'context' in State:
                State.context.figure('simplot', handle=self.fig)
            plt.ioff()
        self.gs = GridSpec(nrows, ncols, self.fig, 0, 0, 1, 1, 0, 0)
        self.axes = {}  # map names to axes objects
        self.axes_names = {}  # reverse mapping of axes objects to their names
        self.plots = []
        self.artists = []
        self.axes_labels = {}
        self.traceplots = []
        self.timestamp = None
        self.timestamp_loc = None
        self.timestamp_ax = None
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
    def set_axes(self, timestamp_ax=None, timestamp_loc=None, **named_axes):
        """
        Create named axes plots from subplot specs based on the `gs` gridspec.
        """
        # Set timestamp axes (name) and position if specified
        if timestamp_ax is not None:
            self.timestamp_ax = timestamp_ax
        if timestamp_loc is not None:
            self.timestamp_loc = timestamp_loc

        # Create all the specified subplot axes objects
        for name, spec in named_axes.items():
            if isinstance(spec, mpl.axes.Axes):
                ax = spec
            else:
                ax = self.fig.add_subplot(spec)
            if name in self.axes:
                raise ValueError('an axis named {name!r} was set previously')
            if ax in self.axes_names:
                raise ValueError('axis {self.axes_names[ax]!r} already set')
            self.axes[name] = ax
            self.axes_names[ax] = name
            ax.set_axis_off()

    def _get_axes_object(self, ax_or_name):
        """
        Get an mpl.axes.Axes object for a given axes argument.
        """
        if ax_or_name in self.axes_names:
            ax = ax_or_name
        elif ax_or_name in self.axes:
            ax = self.axes[ax_or_name]
        else:
            raise KeyError(f'unknown axes value: {ax_or_name!r}')
        return ax

    def _get_axes_name(self, ax_or_name):
        """
        Get the name of a mpl.axes.Axes object for a given axes argument.
        """
        if ax_or_name in self.axes_names:
            name = self.axes_names[ax_or_name]
        elif ax_or_name in self.axes:
            name = ax_or_name
        else:
            raise KeyError(f'unknown axes value: {ax_or_name!r}')
        return name

    @block_record_mode
    def get_axes(self, *axargs):
        """
        Return a list of axes objects keyed by name (or a single axes).
        """
        if not axargs:
            return tuple(self.axes.values())
        if len(axargs) == 1:
            return self._get_axes_object(axargs[0])
        return [self._get_axes_object(ax) for ax in axargs]

    @block_record_mode
    def update_plots(self):
        """
        Update the timestamp and all registered plots with new data.
        """
        [updater() for _, updater in self.plots]
        [rtp.update_plots() for rtp in self.traceplots]
        if State.run_mode == RunMode.INTERACT or self.network_graph_in_fig:
            if self.network_graph is not None:
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

        Note: `updater` is a callback that should update the plot data.
        """
        self.plots.append((handle, updater))
        self.out('Artist: {}', handle, prefix='SimPlotRegistrar')

    @block_record_mode
    def register_network_graph(self, netgraph):
        """
        Set a NetworkGraph instance to be updated during the simulation.
        """
        if self.network_graph is not None:
            self.out('Network graph already registered', warning=True)
        self.network_graph = netgraph
        self.network_graph_in_fig = netgraph.ax in self.axes_names
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
    def draw_borders(self, *axes, color='k', linewidth=1, bottom=None, right=None):
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
        for ax in self.get_axes(*axes):
            name = self._get_axes_name(ax)
            _, b, r, _ = ax.get_position().extents
            pkw = dict(c=color, lw=linewidth, transform=ax.transAxes)
            ax.plot([0,0], [0,1], **pkw)  # left
            ax.plot([0,1], [1,1], **pkw)  # top
            if b == 0.0 or (bottom is not None and name in bottom):
                ax.plot([0,1], [0,0], **pkw)  # bottom
            if r == 1.0 or (right is not None and name in right):
                ax.plot([1,1], [0,1], **pkw)  # right

    @block_record_mode
    def draw_axes_labels(self, *axes, pad=None, loc='left', **fontkw):
        """
        Add axes names as plot labels in the given axes location.
        """
        if not axes:
            axes = tuple(self.axes.keys())
        loc = 'left' if loc is None else loc
        fontdict = dict(color='navy', fontweight='medium')
        fontdict.update(fontkw)
        for axarg in axes:
            name = self._get_axes_name(axarg)
            self.draw_label(name[0].title() + name[1:], name, pad=pad, loc=loc,
                            **fontdict)

    @block_record_mode
    def draw_label(self, s, ax, *, pad=None, loc=None, **fontkw):
        """
        Plot a label to the given axes and return its handle.
        """
        ax = self._get_axes_object(ax)

        # Get the position and alignment for the label location
        loc = 'upper center' if loc is None else loc
        x, y, ha, va = self._loc_to_position(pad=pad, loc=loc)

        # Plot the text label with a nice white-stroke outline for clarity
        fontdict = dict(ha=ha, va=va, transform=ax.transAxes, fontsize='large',
                        zorder=100)
        fontdict.update(fontkw)
        label = ax.text(x, y, str(s), **fontdict)
        label.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground='white'),
            path_effects.Normal(),
        ])

        # Add the label to an axes-specific list of labels
        if ax not in self.axes_labels:
            self.axes_labels[ax] = []
        self.axes_labels[ax].append(label)

        return label

    @staticmethod
    def _loc_to_position(pad=None, loc=None):
        """
        Get an (x, y, ha, va) tuple for `Axes.text` position and alignment.

        Location `loc` defaults to 'upper left' and vertical alignment
        defaults to 'lower' if omitted. Location should otherwise be specified
        as a two-word string from ('upper', 'center', 'lower') and ('left',
        'center', 'right'), respectively.
        """
        loc = 'lower left' if loc is None else loc
        loc = loc.split()
        if len(loc) == 1:
            vloc, hloc = 'lower', loc[0]
        elif len(loc) == 2:
            vloc, hloc = loc
        else:
            raise ValueError(f'bad location value {loc!r}')

        pad = 0.03 if pad is None else pad
        if vloc == 'upper':
            y, va = 1 - pad, 'top'
        elif vloc == 'center':
            y, va = 0.5, 'center'
        elif vloc == 'lower':
            y, va = pad, 'bottom'
        else:
            raise ValueError(f'bad vertical alignment {vloc!r}')

        if hloc == 'left':
            x, ha = pad, hloc
        elif hloc == 'center':
            x, ha = 0.5, hloc
        elif hloc == 'right':
            x, ha = 1 - pad, hloc
        else:
            raise ValueError(f'bad vertical alignment {hloc!r}')

        return x, y, ha, va

    @block_record_mode
    def init(self, initializer):
        """
        Run the figure initialization function (only in interactive mode).
        """
        self.initializer = initializer
        if State.run_mode == RunMode.INTERACT:
            initializer()
            self.init_timestamp()
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
            if self.network_graph_in_fig:
                self.artists.extend(self.network_graph.artists)

        return self.artists

    def init_timestamp(self, **txt):
        """
        Create the text object for the timestamp.
        """
        if self.timestamp is not None: return

        firstax = next(iter(self.axes_names.keys()))
        ax = firstax if self.timestamp_ax is None else \
                self._get_axes_object(self.timestamp_ax)
        loc = 'right' if self.timestamp_loc is None else self.timestamp_loc

        # Plot the timestamp text object
        x, y, ha, va = self._loc_to_position(loc=loc)
        fmt = dict(color='gray', ha=ha, va=va, transform=ax.transAxes)
        fmt.update(txt)
        self.timestamp = ax.text(x, y, '', **fmt)

        def update_timestamp():
            self.timestamp.set_text(f't = {State.t:.2f} ms')
        self.register_plot(self.timestamp, update_timestamp)
