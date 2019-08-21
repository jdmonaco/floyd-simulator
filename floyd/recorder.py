"""
Record state, model variables, and spike/event data across a simulation.
"""

__all__ = ('MovieRecorder', 'ModelRecorder')


import numpy as np
import pandas as pd

from tenko.base import TenkoObject

from .state import State, RunMode
from .clocks import ArrayClock
from .config import Config


class MovieRecorder(TenkoObject):

    """
    A reduced version of the ModelRecorder for triggering video frames.
    """

    def __init__(self, fps=None, compress=None):
        """
        Initialize a modulo-based time-tracker with shared state outputs.
        """
        super().__init__()

        # Compute the implicit frame interval from fps and compression
        State.fps = Config.fps if fps is None else fps
        compress = Config.compress if compress is None else compress
        if compress == 'dt':
            dt_frame = State.dt
        else:
            interval = 1e3 / State.fps  # ms / video frame
            dt_frame = compress * interval  # ms of simulation per frame
            if dt_frame < State.dt:
                dt_frame = State.dt
                self.out(f'Setting movie compression to '
                         f'{dt_frame*State.fps/1e3}: implicit interval was '
                         f'<dt ({State.dt!r})', warning=True)

        self.frame = ArrayClock(dt=dt_frame, duration=State.duration)
        State.network.set_movie_recorder(self)


class ModelRecorder(TenkoObject):

    """
    Automatic recording of state, model variables, and spike/event timing.

    Notes
    -----
    (1) In the simulation loop, ModelRecorder.update() should be called first
    so that initial values at t=0 are stored, followed by variable updates.

    (2) Variable arrays are stored by reference and require that updated
    variables use the same data buffers throughout the simulation.

    (3) Initial values to the constructor may be non-boolean arrays
    (variables), boolean arrays (spike/event), or keys (simulation state).
    Further, the value may be a tuple with two elements: (1) a data array as
    described in the last sentence, and (2) a `record` value to be passed to
    one of the `add_*_monitor` methods as described in note #4.

    (4) Set `record` to a list of indexes (for axis 0 of data) or a scalar
    integer index for selective recordings.
    """

    def __init__(self, **initial_values):
        """
        Add recording monitors for keyword-specified variables and states.

        Boolean arrays (dtype, '?') are automatically considered to be spike/
        event output. Spike/event timing will be recording for every simulation
        timestep (regardless of `dt_rec`) and appended to a pandas DataFrame
        object with 'unit' and 't' columns. Spike/event dataframes are also
        saved to the context datafile in the `save_recordings` method.
        """
        super().__init__()

        # Nothing to do here if we're not in a data collection simulation
        if State.run_mode != RunMode.RECORD:
            return

        # Recording time tracking and update callback
        self.clock = ArrayClock(dt=State.dt_rec, duration=State.duration)
        self.clock.add_callback(self.update)

        # Data storage keyed by variable names
        self.unit_slices = dict()
        self.traces = dict()
        self.variables = dict()
        self.state_traces = dict()

        # Spike/event timing storage
        self.units = dict()
        self.unit_indexes = dict()
        self.timing = dict()
        self.events = dict()

        # Use the keywords and intial values to automatically set up monitors
        # for model variables, states, and spike/event signals
        for name, data in initial_values.items():
            record = True
            if type(data) is tuple:
                if len(data) != 2:
                    self.out('Tuple values must be length 2', warning=True)
                    continue
                data, record = data

            if isinstance(data, np.ndarray):
                if data.dtype == bool:
                    self.add_spike_monitor(name, data, record=record)
                else:
                    self.add_variable_monitor(name, data, record=record)
            elif np.isscalar(data):
                try:
                    state_value = float(data)
                except ValueError:
                    self.out('Not a scalar: {!r}', data, warning=True)
                    continue
                else:
                    self.add_state_monitor(name, state_value)
            else:
                self.out('Not an array or state: {!r}', data, warning=True)

        State.recorder = self

    def _new_monitor_check(self, name):
        assert State.n == -1, 'simulation has already started'
        exists = f'monitor exists ({name!r})'
        assert name not in self.variables, f'variable {exists}'
        assert name not in self.events, f'spike/event {exists}'
        assert name not in self.state_traces, f'state {exists}'
        assert type(name) is str, 'data name must be a string'

    def add_variable_monitor(self, name, data, record=True):
        """
        Add a new monitor for a data array variable.
        """
        self._new_monitor_check(name)
        assert type(data) is np.ndarray, 'variable data must be an array'

        if type(record) is bool and record == True:
            self.unit_slices[name] = sl = slice(None)
        elif np.iterable(record):
            self.unit_slices[name] = sl = list(record)
        elif np.isscalar(record) and isinstance(record, int):
            if 0 <= record < data.shape[0]:
                self.unit_slices[name] = sl = [record]
        else:
            raise ValueError(f'invalid record value ({record})')
        rdata = data[sl]

        dtype = data.dtype
        self.traces[name] = np.zeros((self.clock.N_t,) + rdata.shape, dtype)
        self.variables[name] = data

        self.debug(f'added variable monitor: {name!r}, dtype={dtype!r}, '
                   f'record={record!r}')

    def add_spike_monitor(self, name, data, record=True):
        """
        Add a new monitor for a boolean spike or event vector.

        Note: Set `record` to a list of indexes (for axis 0 of data)
        or a scalar integer index for selective recordings.
        """
        self._new_monitor_check(name)
        assert type(data) is np.ndarray, 'spike/event data must be an array'
        assert data.dtype == bool, 'spike/event data must be boolean'

        if type(record) is bool and record == True:
            self.unit_slices[name] = sl = slice(None)
        elif np.iterable(record):
            self.unit_slices[name] = sl = list(record)
        elif np.isscalar(record) and isinstance(record, int):
            if 0 <= record < data.shape[0]:
                self.unit_slices[name] = sl = [record]
        else:
            raise ValueError(f'invalid record value ({record})')
        rdata = data[sl]

        self.units[name] = np.array([], 'i')
        self.timing[name] = np.array([], 'f')
        self.unit_indexes[name] = np.arange(data.shape[0])[sl]
        self.events[name] = data

        self.debug(f'added spike/events monitor: {name!r}, '
                   f'record={self.unit_slices[name]!r}')

    def add_state_monitor(self, name, state_value):
        """
        Add a new monitor for a state value (scalars of any type).
        """
        self._new_monitor_check(name)
        assert np.ndim(state_value) == 0, 'state value must be scalar'

        dtype = np.array(state_value).dtype
        self.state_traces[name] = np.zeros(self.clock.N_t, dtype)
        State[name] = state_value

        self.debug(f'added state monitor: {name!r}, dtype={dtype!r}')

    def update(self, clock=None):
        """
        Update the time series, variable monitors, and state monitors.
        """
        # Record spike/event timing at every simulation timestep
        for name, data in self.events.items():
            recdata = data[self.unit_slices[name]]
            units = self.unit_indexes[name][recdata]
            if units.size == 0:
                continue
            timing = np.zeros(units.size, 'f') + State.t
            self.units[name] = np.concatenate((self.units[name], units))
            self.timing[name] = np.concatenate((self.timing[name], timing))

        # Update data trace values
        for name, data in self.variables.items():
            self.traces[name][self.clock.n] = data[self.unit_slices[name]]

        # Update state trace values
        for name in self.state_traces.keys():
            self.state_traces[name][self.clock.n] = State[name]

    def save(self, *path, **root):
        """
        Save monitored state, variable, and spike/event recordings.
        """
        if State.run_mode != RunMode.RECORD:
            self.out('Must be in RECORD mode to save data', warning=True)
            return

        for name in self.events.keys():
            df = pd.DataFrame(data=np.c_[self.units[name], self.timing[name]],
                    columns=('unit', 't'))
            State.context.save_dataframe(df, *path, name, **root)

        State.context.save_array(self.clock.ts, *path, 't', **root)

        for name, data in self.traces.items():
            State.context.save_array(data, *path, name, **root)

        for name, data in self.state_traces.items():
            State.context.save_array(data, *path, name, **root)
