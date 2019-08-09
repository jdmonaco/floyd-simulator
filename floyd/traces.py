"""
Streaming data-trace plots for simulations.
"""

__all__ = ['RealtimeTracesPlot']


import operator
from collections import deque

import numpy as np

from tenko.base import TenkoObject

from .state import State, RunMode


class RealtimeTracesPlot(TenkoObject):

    """
    A collection of real-time windowed trace plots with adaptive axes limits.
    """

    def __init__(self, *traces, units=None, fmt=None, datapad=0.05,
        datalim='auto', legend=True, legend_format={}):
        """
        Initialize with data traces specified as 2-, 3-, or 4-tuples:

        (ax, label) -- only axes and label; `ax` may specifiy a simplot axes
        (ax, label, updater) -- add a callback function for the next value
        (ax, label, [updater,] fmt) -- add a line plot format dict

        The plots will be automatically created in interactive mode. In non-
        interactive mode, the `plot()` method should be called manually. The
        `update()` method should be called at every time-step to update the
        data traces. Traces without a callback updater function need to be
        passed in as keyword arguments to `update()`.

        Keyword arguments:

        units : str, optional
            The dimensional units for the data for display in the legend

        fmt : dict, optional
            A dict of shared formatting parameters for the line plots

        datapad : float
            Fractional amount of symmetric padding around the data range

        datalim : 'auto' | float | 'expand' | 'none'
            Automatic limit setting for the data axis can be fully adaptive to
            the current data range ('auto'), relax smoothly when the data range
            shrinks (float, relaxation time-constant), partially adaptive by
            only expanding as necessary ('expand'), or turned off ('none').

        legend : True | False | 'last' | numpy function name | callable
            Simple plot legends using the data trace labels can be turned on or
            off with boolean values. Dynamic legends can be enabled that update
            with every update of the data traces: the last (current) value can
            be displayed ('last'), a numpy function result on the current data
            trace, or any python callable that receives the data trace and
            returns a scalar value.

        legend_format : dict, optional
            Legend formatting arguments can be provided
        """
        super().__init__()
        self.ax = {}
        self.axtraces = []
        self.fmt = {}
        self.names = []
        self.rolls = []
        self.updaters = {}
        for values in traces:
            trace_fmt = {}
            updater = None
            if len(values) == 2:
                ax, name = values
            elif len(values) == 3:
                ax, name, updater_or_fmt = values
                if callable(updater_or_fmt):
                    updater = updater_or_fmt
                else:
                    trace_fmt.update(updater_or_fmt)
            elif len(values) == 4:
                ax, name, updater, _fmt = values
                trace_fmt.update(_fmt)
            else:
                self.out(f'Invalid trace: {values!r}', error=True)
                continue

            # Check if a named axes is in the shared simulation plotter
            if ax in State.simplot:
                ax = State.simplot.get_axes(ax)
            self.names.append(name)
            self.ax[name] = ax

            # Set the updater callbacks
            if updater is not None:
                self.updaters[name] = updater

            # Combined shared and axes-specific format dicts
            trace_fmt['label'] = name
            self.fmt[name] = {}
            if fmt is not None:
                self.fmt[name].update(fmt)
            self.fmt[name].update(trace_fmt)

            # Invert the trace-axes mapping for adaptive data limits
            for axtrace in self.axtraces:
                if ax is axtrace[0]:
                    axtrace[1].append(name)
                    break
            else:
                self.axtraces.append([ax, [name]])

            # Track the traces with rolling windows
            rolled = self.fmt[name].pop('rolled', False)
            if rolled:
                self.rolls.append(name)

        self.q_max = q_max = int(State.tracewin / State.dt)
        self.q_t = deque([], q_max)
        self.q = {name:deque([], q_max) for name in self.names}
        self.q_rolls = {name:deque([], q_max) for name in self.rolls}
        self.axobjs = set(self.ax.values())

        if type(datalim) in (int, float):
            if (State.run_mode == RunMode.INTERACT and \
                    datalim <= State.dt_block) or \
                        (datalim <= State.dt) or (datalim <= 0):
                self.datalim = 'auto'
            else:
                self.datalim = 'relax'
                self._datalim_relax = datalim
                if State.run_mode == RunMode.INTERACT:
                    dt = State.dt_block
                else:
                    dt = State.dt
                self.shrink = dt / self._datalim_relax
        else:
            self.datalim = datalim
        self.datapad = datapad

        # Enable dynamic legends - last value, numpy function
        self.legfmt = dict(loc='upper left', frameon=False)
        self.legfmt.update(legend_format)
        self.legend = legend
        self.legend_fn = None
        self.legend_is_dynamic = True
        self._legends = []
        if legend == 'last':
            self.legend_fn = operator.itemgetter(-1)
        elif type(legend) is str and hasattr(np, legend):
            self.legend_fn = getattr(np, legend)
        elif callable(legend):
            self.legend_fn = legend
        else:
            self.legend_is_dynamic = False
        self.legend_label_templ = '{} = {:.3g}'
        if units is not None:
            self.legend_label_templ += f' {units}'

        if State.run_mode == RunMode.INTERACT:
            self.plot()

    def plot(self):
        """
        Initialize empty line plots for each data trace.
        """
        if hasattr(self, 'artists'):
            return self.artists
        self.lines = {}
        self.artists = []
        for name in self.names:
            self.lines[name] = self.ax[name].plot([], [], **self.fmt[name])[0]
            self.artists.append(self.lines[name])
        if not self.legend_is_dynamic and self.legend == True:
            self._legends = [ax.legend(**self.legfmt) for ax in self.axobjs]
        return self.artists

    def update_data(self, **traces):
        """
        Update data traces with new values.

        Note: a keyword argument should be provided for each named trace that
        does not have an associated updater callback.
        """
        self.q_t.append(State.t)
        N_q_t = len(self.q_t)
        for name in self.names:

            # Retrieve value from callback or keyword arguments
            if name in self.updaters:
                value = self.updaters[name]()
            elif name in traces:
                value = traces[name]
            else:
                self.out(f'Missing trace update: {name!r}', warning=True)
                continue

            # Skip adding value if queue somehow advanced already
            # (This happens irregularly when creating movies for some reason.)
            q_name = self.q[name]
            if N_q_t < self.q_max and len(q_name) >= N_q_t:
                continue

            # Add value to rolling window trace or simple data trace
            if name in self.rolls:
                roll = self.q_rolls[name]
                roll.append(float(value))
                q_sum = np.cumsum(roll)
                q_name.append((q_sum[-1] - q_sum[0])/len(roll))
            else:
                q_name.append(float(value))

    def update_plots(self):
        """
        Update trace line plots and axes limits with new data.
        """
        for name in self.names:
            self.lines[name].set_data(self.q_t, self.q[name])

        # For dynamic legends, recreate the plot legend with new data
        if self.legend_is_dynamic:
            [leg.remove() for leg in self._legends]
            for name in self.names:
                self.lines[name].set_label(self.legend_label_templ.format(name,
                    self.legend_fn(self.q[name])))
            self._legends = [ax.legend(**self.legfmt) for ax in self.axobjs]

        # Set trace window limits on time axis
        tlim = (self.q_t[0], max(self.q_t[-1], self.q_t[0] + State.tracewin))
        [ax.set_xlim(tlim) for ax in self.axobjs]

        # Set data limits on axis (e.g., y-axis if time axis is x-axis)
        if self.datalim in ('auto', 'relax'):
            for ax, names in self.axtraces:
                axdmin, axdmax = np.inf, -np.inf
                for name in names:
                    dmin = min(self.q[name])
                    dmax = max(self.q[name])
                    pad = self.datapad * (dmax - dmin)
                    axdmin = min(dmin - pad, axdmin)
                    axdmax = max(dmax + pad, axdmax, axdmin + self.datapad)
                if self.datalim == 'relax':
                    ymin, ymax = ax.get_ylim()
                    if axdmin > ymin:
                        axdmin -= self.shrink*(axdmin - ymin)
                    if axdmax < ymax:
                        axdmax += self.shrink*(ymax - axdmax)
                if axdmin == np.inf or axdmax == -np.inf:
                    __import__('pdb').set_trace()
                ax.set_ylim(axdmin, axdmax)

        elif self.datalim == 'expand':
            for name in self.names:
                dmin = min(self.q[name])
                dmax = max(self.q[name])
                pad = self.datapad * (dmax - dmin)
                dlim = self.ax[name].get_ylim()
                self.ax[name].set_ylim(min(dmin-pad, dlim[0]),
                                       max(dmax+pad, dlim[1]))
