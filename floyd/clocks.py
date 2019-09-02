"""
Classes for timing events against the simulation time.
"""

__all__ = ('SimulationClock', 'ProgressBar', 'timed', 'Clock', 'ArrayClock')


import functools

from numpy import r_, vstack
from scipy.interpolate import interp1d

from toolbox.numpy import inf, arange
from pouty import box, hline, debug
from tenko.base import TenkoObject

from .state import State, RunMode
from .config import Config


class SimulationClock(TenkoObject):

    """
    Special shared-state clock that governs the simulation timesteps.
    """

    def __init__(self, *, t_start=0.0):
        if 'simclock' in State and State.simclock is not None:
            raise RuntimeError(f'simulation clock exists: {State.simclock!r}')
        super().__init__()

        t_end = t_start + State.duration
        ts = arange(t_start, t_end + State.dt, State.dt)
        State.ts = ts[ts <= t_end]
        State.N_t = len(State.ts)
        State.t = t_start - State.dt
        State.n = -1  # simulation frame index, flag for unstarted simulation

        State.simclock = self
        self.debug(f'initialized at t = {t_start!r} with dt = {State.dt!r}')

    def __bool__(self):
        if State.run_mode == RunMode.INTERACT:
            return True
        return State.n < State.N_t

    def update(self):
        """
        Advance the simulation clock.
        """
        State.n += 1
        if State.run_mode == RunMode.INTERACT:
            State.t += State.dt
            return
        if State.n == 0:
            hline()
        elif State.n >= State.N_t:
            return
        State.t = State.ts[State.n]


class ProgressBar(TenkoObject):

    """
    Progress bar for tracking the advancement of the simulation.
    """

    def __init__(self, nchars=Config.progress_width, filled=False,
        color='purple'):
        if 'progressbar' in State and State.progressbar is not None:
            raise RuntimeError(f'progress bar exists: {State.progressbar!r}')
        super().__init__(color=color)

        self.nchars = nchars
        self.filled = filled
        self.Nprogress = 0
        self.progfrac = 0.0

        State.progressbar = self

    def update(self):
        """
        Update the display of the progress bar.
        """
        State.progress = (State.n + 1) / State.N_t
        while self.progfrac < State.progress:
            self.out.box(filled=self.filled)
            self.Nprogress += 1
            self.progfrac = self.Nprogress / self.nchars


def timed(func=None, *, dt=None, start=None):
    """
    Decorator for functions to be clocked callbacks with regular intervals.

    Function must accept an argument, which will be the Clock instance.
    """
    if func is None:
        dt = State.dt if dt is None else dt
        start = 0.0 if start is None else start
        return functools.partial(timed, dt=dt, start=start)
    Clock(dt=dt, t_start=start).add_callback(func)


class Clock(TenkoObject):

    """
    Minimal simulation-tracking clock with tick-based callbacks.
    """

    def __init__(self, *, dt=Config.dt, t_start=0.0):
        if dt < State.dt:
            raise RuntimeError(f'clock dt too small: {dt} < {State.dt}')

        self.dt = dt
        self.t_start = t_start
        self.t = self.t_start - self.dt
        self.t_next = self.t + self.dt
        self.n = -1
        self.callbacks = []

        State.network.add_clock(self)

    def add_callback(self, func):
        """
        Add a callback function to be called on every tick (with instance).
        """
        assert callable(func), f'callback must be callable: {func!r}'
        name = State.network._get_object_name(func)
        self.callbacks.append(func)
        debug(f'added tick-count callback {name!r}: {func!r}', prefix='clock')

    def update(self):
        """
        Advance the clock as close to the current simulation time as possible.
        """
        while State.t >= self.t_next:
            self.n += 1
            self.t += self.dt
            self.t_next = self.t + self.dt
            self._call_callbacks()

    def _call_callbacks(self):
        for fn in self.callbacks:
            fn(self)


class ArrayClock(Clock):

    """
    Clock for defined durations with timestep arrays.
    """

    def __init__(self, duration=None, timestamps=None, **kwargs):
        """
        Initialize the array-based clock with optional direct timestamps.
        """
        super().__init__(**kwargs)
        duration = Config.duration if duration is None else duration

        if timestamps is None:
            self.duration = duration
            self.t_end = self.t_start + self.duration
            ts = arange(self.t_start, self.t_end + self.dt, self.dt)
            self.ts = ts[ts <= self.t_end]
        else:
            self.t_start, self.t_end = timestamps[0], timestamps[-1]
            self.duration = self.t_end - self.t_start
            self.t = -1  # flag for unstarted clock
            self.ts = timestamps
        self.N_t = len(self.ts)
        self.t_next = self.ts[0]

    def update(self):
        """
        Update clock time (ticks) based on the current simulation time.
        """
        while State.t >= self.t_next:
            self.n += 1
            self.t = self.ts[self.n]
            if self.n >= self.N_t - 1:
                self.t_next = inf
            else:
                self.t_next = self.ts[self.n + 1]
            self._call_callbacks()


class TimedArray(TenkoObject):

    """
    Time-dependent data arrays that update throughout the simulation.
    """

    def __init__(self, T, X, state_key=None, interpolate=False):
        """
        Initialize the timed array. Multiple arrays in `X` should be
        column-stacked, with time on the first axis. Interpolation will
        preprocess the (T, X) data with linear interpolation at every timestep
        of the simulation. Setting `state_key` will update the given key in the
        shared state.
        """
        assert len(T) == len(X), f'dimension 0 mismatch: {len(T)} vs. {len(X)}'
        super().__init__()

        # Handle a late start by inserting an extra t=0 data point
        if T[0] > State.ts[0]:
            T = r_[State.ts[0], T]
            X = vstack((X[0], X))

        # Perform linear interpolation
        self.interpolate = interpolate
        if interpolate:
            T = arange(T[0], T[-1] + State.dt, State.dt)
            T = T[T <= T[-1]]
            lerp = interp1d(T, X, axis=0, bounds_error=False, fill_value=X[-1])
            X = lerp(T)

        self.T = T
        self.X = X
        self.state_key = state_key
        self.clock = ArrayClock(timestamps=self.T)

        if state_key is not None:
            self.clock.add_callback(self._state_update)

    def _state_update(self, clock=None):
        State[self.state_key] = self.value

    @property
    def value(self):
        if self.n == -1:
            raise RuntimeError('no timed array value before simulation start')
        return self.X[self.n]
