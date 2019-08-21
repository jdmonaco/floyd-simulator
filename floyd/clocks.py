"""
Classes for timing events against the simulation time.
"""

__all__ = ('SimulationClock', 'ProgressBar', 'BaseClock', 'ArrayClock')

from toolbox.numpy import inf
from pouty import box, hline, debug

from .state import State, RunMode
from .config import Config


class SimulationClock(object):

    """
    Special shared-state clock that governs the simulation timesteps.
    """

    def __init__(self, *, t_start=0.0):
        if State.simclock is not None:
            raise RuntimeError(f'simulation clock exists: {State.simclock!r}')

        t_end = t_start + self.duration
        ts = np.arange(t_start, t_end + State.dt, State.dt)
        State.ts = ts[ts <= t_end]
        State.N_t = len(State.ts)
        State.t = t_start - State.dt
        State.n = -1  # simulation frame index, flag for unstarted simulation

        State.simclock = self

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
        State.t = State.ts[State.n]


class ProgressBar(object):

    """
    Progress bar for tracking the advancement of the simulation.
    """

    def __init__(self, nchars=Config.progress_width, filled=False, c='purple'):
        self.nchars = nchars
        self.filled = filled
        self.color = color
        self.Nprogress = 0
        self.progfrac = 0.0

    def update(self):
        """
        Update the display of the progress bar.
        """
        State.progress = frac = (State.n + 1) / State.N_t
        while self.progfrac < frac:
            box(filled=self.filled, c=self.color)
            self.Nprogress += 1
            self.progfrac = self.Nprogress / self.nchars


class Clock(object):

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


class ArrayClock(BaseClock):

    """
    Clock for defined durations with timestep arrays.
    """

    def __init__(self, duration=Config.duration, **kwargs):
        super().__init__(**kwargs)

        self.duration = duration
        self.t_end = self.t_start + self.duration
        ts = np.arange(self.t_start, self.t_end + self.dt, self.dt)
        self.ts = ts[ts <= self.t_end]
        self.N_t = len(self.ts)
        self.t_next = self.ts[0]

    def update(self):
        """
        Use timestep array values to set the time up to the full duration.
        """
        while State.t >= self.t_next:
            self.n += 1
            self.t = self.ts[self.n]
            if self.n >= self.N_t - 1:
                self.t_next = inf
            else:
                self.t_next = self.ts[self.n + 1]
            self._call_callbacks()