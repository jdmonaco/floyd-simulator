"""
Functions to generate simulation inputs.
"""

from collections import namedtuple

from toolbox.numpy import *

from .state import State


Timepoint = namedtuple('Timepoint', ['t', 'value'])


def triangle_wave(t0, p):
    """
    Triangle wave on [0,1] with period p for time points t0.
    """
    t = t0 - p/4
    a = p/2
    b = floor(t/a + 1/2)
    return 0.5 + (1/a) * (t - a*b) * (-1)**b


class InputStimulator(object):

    """
    Generate one-off or repeating temporal input stimulus patterns.
    """

    def __init__(self, target, variable, *timepoints, stimulate=None,
        repeat=False, repeat_until=None):
        """
        Target object variable is manipulated according to Timepoint tuples.

        Note: val can be 'off' to disable updating until the next time point.
        """
        self.target = target
        self.variable = variable
        assert hasattr(target, variable), f"no target attribute ('{variable}')"

        # Process `stimulate` as an index for array target variables
        self.index = None
        self.is_array = isinstance(getattr(target, variable), ndarray)
        if self.is_array:
            if stimulate is None:
                self.index = slice(None)
            else:
                try:
                    getattr(target, variable)[stimulate]
                except IndexError as e:
                    State.context.out('bad target index for {}.{}'.format(
                        target, variable), prefix=self.__class__.__name__,
                        error=True)
                    raise e
                else:
                    self.index = stimulate

        # Sort steps into a list of Timepoint tuples
        self.timepoints = [Timepoint(t, v) for t, v in
                           sorted(timepoints, key=lambda x: x[0])]
        self.t = array([tp.t for tp in self.timepoints])
        self.N = len(self.timepoints)

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

        State.network.add_stimulator(self)

    def __str__(self):
        if hasattr(self.target, 'name'):
            tname = f'\'{self.target.name}\''
        else:
            tname = self.__class__.__name__
        varname = f'\'{self.variable}\''
        return f'{self.__class__.__name__}(target={tname}, variable={varname})'

    def update(self):
        """
        Update the target variable with the current stimulator value.
        """
        if self.exhausted:
            return

        # Wrap simulation time around the range of timepoints and check whether
        # any points have passed yet
        t_mod = State.t % self.t.max()
        nz = (self.t <= t_mod).nonzero()[0]
        if nz.size == 0:
            return

        # Get the most recent timepoint and elapsed time since it started
        i = nz[-1]
        dt = t_mod - self.t[i]

        # Perform the stimulation based on a value or function call
        tp = self.timepoints[i]
        value = tp.value
        if callable(value):
            value = value(dt)
        if self.is_array:
            getattr(self.target, self.variable)[self.index] = value
        else:
            setattr(self.target, self.variable, value)

        # Determine whether the end of the stimulation protocol was reached
        if i == self.N - 1:
            self.delivered += 1
            if self.delivered == self.cycles:
                self.exhausted = True
