"""
Functions to generate simulation inputs.
"""

__all__ = ('step_pulse_series', 'triangle_wave', 'InputStimulator',)


from collections import namedtuple

from toolbox.numpy import *
from tenko.base import TenkoObject
from specify import is_specified

from .state import State


Timepoint = namedtuple('Timepoint', ['t', 'value'])


def step_pulse_series(N, duration, max_input):
    """
    Construct a series of input series from zero to a maximum level.
    """
    dwell = duration / N
    t_steps = r_[linspace(0, duration - dwell, N), duration]
    v_steps = r_[linspace(0, max_input, N), 0]
    return tuple(Timepoint(t, v) for t, v in zip(t_steps, v_steps))

def triangle_wave(t0, p):
    """
    Triangle wave on [0,1] with period p for time points t0.
    """
    t = t0 - p/4
    a = p/2
    b = floor(t/a + 1/2)
    return 0.5 + (1/a) * (t - a*b) * (-1)**b


class InputStimulator(TenkoObject):

    """
    Generate one-off or repeating temporal input stimulus patterns.
    """

    def __init__(self, target, variable, *timepoints, state_key=None,
        stimulate=None, repeat=False, repeat_until=None):
        """
        Target object parameter is manipulated according to Timepoint tuples.

        Note: Target should be a BaseUnitGroup subclass. The variable should be
        the name of an array variable or a Param in `target.spec`.
        """
        super().__init__(name=f'Stim:{variable}')
        self.target = target
        self.variable = variable
        self.state_key = state_key
        self.is_array_variable = False

        if hasattr(target, 'variables') and variable in target.variables:
            assert isinstance(getattr(target, variable), ndarray), \
                    'object-level attributes must be arrays'
            self.is_array_variable = True
        elif is_specified(target) and variable in target:
            pass
        else:
            raise ValueError('not an attribute array or Param key: '
                             f'{variable!r}')

        # Process `stimulate` as an index for array target variables
        self.index = None
        if self.is_array_variable:
            if stimulate is None:
                self.index = slice(None)
            else:
                try:
                    getattr(target, variable)[stimulate]
                except IndexError as e:
                    State.context.out('bad target index for {}.{}'.format(
                        target, variable), prefix=self.klass, error=True)
                    raise e
                else:
                    self.index = stimulate

        # Sort steps into a list of Timepoint tuples
        self.timepoints = [Timepoint(t, v) for t, v in
                           sorted(timepoints, key=lambda x: x[0])]
        self.t = array([tp.t for tp in self.timepoints])
        self.N = len(self.timepoints)
        self.previous = -1

        # Set up the repeat variables
        self.cycles = 1
        if repeat_until is not None:
            self.cycles = max(1, int(repeat_until / self.t.max()))
        elif type(repeat) is bool and repeat:
            self.cycles = inf
        elif type(repeat) is int:
            self.cycles = repeat
        self.delivered = 0
        self.exhausted = False
        self.exhausted_printed = False
        self.last_t_mod = -1

        State.network.add_stimulator(self)

    def __str__(self):
        if hasattr(self.target, 'name'):
            tname = f'{self.target.name!r}'
        else:
            tname = f'{self.target.__class__.__name__}'
        varname = f'{self.variable!r}'
        return f'{self.klass}(target={tname}, variable={varname})'

    def update(self):
        """
        Update the target variable with the current stimulator value.
        """
        if self.exhausted:
            if self.exhausted_printed:
                return
            self.out('Stimulation protocol exhausted (t = '
                    f'{State.t-2*State.dt:.2f} ms)')
            self.exhausted_printed = True
            return

        # Wrap simulation time around the range of timepoints and check whether
        # any points have passed yet
        t_mod = State.t % self.t.max()
        nz = (self.t <= t_mod).nonzero()[0]
        if nz.size == 0:
            return

        # Detect the modulo wraparound as a cycle termination signal
        if t_mod < self.last_t_mod:
            self.delivered += 1
            if self.delivered == self.cycles:
                self.exhausted = True
        self.last_t_mod = t_mod

        # Get the most recent timepoint and elapsed time since it started
        i = nz[-1]
        t0 = self.t[i]
        dt = t_mod - t0

        # Perform the stimulation based on a value or function call
        tp = self.timepoints[i]
        value = tp.value
        if callable(value):
            value = value(dt)
        if self.is_array_variable:
            getattr(self.target, self.variable)[self.index] = value
        else:
            setattr(self.target, self.variable, value)
        if self.state_key is not None:
            State[self.state_key] = value

        # Print message for passing each timepoint
        if i != self.previous:
            self.debug(f'timepoint {i}/{self.N-1}: stim({t0:.3f}) = {value!r}')
            self.previous = i
