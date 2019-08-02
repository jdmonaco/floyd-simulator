"""
Record state, model variables, and spike/event data across a simulation.
"""

import numpy as np
import pandas as pd

from tenko.base import TenkoObject

from .state import State, RunMode
from .config import Config


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

    def __init__(self, show_progress=True, **initial_values):
        """
        Add recording monitors for keyword-specified variables and states.

        Boolean arrays (dtype, '?') are automatically considered to be spike/
        event output. Spike/event timing will be recording for every simulation
        timestep (regardless of `dt_rec`) and appended to a pandas DataFrame
        object with 'unit' and 't' columns. Spike/event dataframes are also
        saved to the context datafile in the `save_recordings` method.
        """
        assert State.dt_rec >= State.dt, 'recording interval < simulation dt'
        super().__init__(self)

        # Simulation time & progress tracking
        State.ts = np.arange(0, State.duration + State.dt, State.dt)
        State.N_t = len(State.ts)
        State.n = -1  # simulation frame index, flag for unstarted simulation
        State.t = -State.dt
        self.Nprogress = 0
        self.show_progress = show_progress and not (
                         State.debug or State.run_mode == RunMode.INTERACT)

        # Recording time tracking
        self.ts_rec = np.arange(0, State.duration + State.dt_rec, State.dt_rec)
        self.N_t_rec = len(self.ts_rec)
        self.n_rec = -1  # recording frame index
        self.t_rec = -State.dt_rec
        self._rec_mod = np.inf  # recording trigger; inf triggers at t=0

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
        if State.run_mode == RunMode.RECORD:
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

    def __bool__(self):
        if State.run_mode == RunMode.INTERACT:
            return True
        return State.n < State.N_t

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
        self.traces[name] = np.zeros((self.N_t_rec,) + rdata.shape, dtype)
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
        self.state_traces[name] = np.zeros(self.N_t_rec, dtype)
        State[name] = state_value

        self.debug(f'added state monitor: {name!r}, dtype={dtype!r}')

    def update(self):
        """
        Update the time series, variable monitors, and state monitors.
        """
        State.n += 1
        if State.run_mode == RunMode.INTERACT:
            State.t += State.dt
            return
        if not self:
            State.context.hline()
            self.out(f'Simulation complete (n = {State.n - 1:g} frames)',
                     anybar='green')
            return
        if State.n == 0:
            State.context.hline()

        # Update simulation time and progress bar output
        State.t = State.ts[State.n]
        if self.show_progress:
            self.progressbar()

        # No recording in anything but data collection mode
        if State.run_mode != RunMode.RECORD:
            return

        # Record spike/event timing at every simulation timestep
        for name, data in self.events.items():
            recdata = data[self.unit_slices[name]]
            units = self.unit_indexes[name][recdata]
            if units.size == 0:
                continue
            timing = np.zeros(units.size, 'f') + State.t
            self.units[name] = np.concatenate((self.units[name], units))
            self.timing[name] = np.concatenate((self.timing[name], timing))

        # Variable and state data traces are sampled with the recording clock.
        # Each sample is triggered by a modulo calculation of the simulation
        # time with the parameterized recording interval. This calculation is
        # bypassed by the conditional below if the sample interval `dt_rec` is
        # equal to the simulation interval `dt`. (It can't be any lower, as an
        # exception would be raised in the constructor.)
        #
        # The modulo calculation is that a sample is triggered when t % dt_rec
        # is observed to *decrease* (meaning that the simulation wrapped around
        # one full cycle of the sampling interval).
        #
        if State.dt_rec > State.dt:
            _rec_mod = State.t % State.dt_rec
            between_samples = _rec_mod >= self._rec_mod
            self._rec_mod = _rec_mod
            if between_samples:
                return

        # Update recording index and time
        if self.n_rec >= self.N_t_rec:
            self.out('Recording complete (t = {:.3f} ms)', self.ts_rec[-1])
            return
        self.n_rec += 1
        self.t_rec = self.ts_rec[self.n_rec]

        # Update data trace values
        for name, data in self.variables.items():
            self.traces[name][self.n_rec] = data[self.unit_slices[name]]

        # Update state trace values
        for name in self.state_traces.keys():
            self.state_traces[name][self.n_rec] = State[name]

    def progressbar(self, filled=False, color='purple'):
        """
        Once-per-update console output for a simulation progress bar.
        """
        State.progress = pct = (State.n + 1) / State.N_t
        barpct = self.Nprogress / Config.progress_width
        while barpct < pct:
            State.context.box(filled=filled, color=color)
            self.Nprogress += 1
            barpct = self.Nprogress / Config.progress_width

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

        State.context.save_array(self.ts_rec, *path, 't', **root)

        for name, data in self.traces.items():
            State.context.save_array(data, *path, name, **root)

        for name, data in self.state_traces.items():
            State.context.save_array(data, *path, name, **root)
