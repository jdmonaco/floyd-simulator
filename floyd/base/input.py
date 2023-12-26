"""
Base group class for read-only inputs, i.e., "sources".
"""

__all__ = ('InputGroup',)


from toolbox.numpy import *
from specify import Specified, Param
from floyd.state import State

from .groups import BaseUnitGroup


class InputGroup(Specified, BaseUnitGroup):

    """
    Base class for a group source that may provide arbitrary inputs to another
    group (e.g., a neuron group) over a projection (e.g., synapses). This class
    automatically computes basic metrics of output level and activity fraction
    that may be used for visualization, etc.
    """

    k_frac = Param(0.2, doc='pulse metric smoothness')
    do_metrics = Param(True, doc='whether to auto-compute basic metrics')

    # Input units are characterized only by an optional position coordinate
    # and an output value

    base_variables = ('x', 'y', 'output')

    def __init__(self, *, name, shape, **kwargs):
        super().__init__(name=name, shape=shape, **kwargs)

        # Initialize the calculations of output/activity pulse metrics
        self._min_output = inf
        self._mean_output = 0.0
        self._max_output = -inf
        self._min_active = inf
        self._mean_active = 0.0
        self._max_active = -inf
        self._pulse_active = 0.0
        self._pulse_output = 0.0
        self._pulse = 0.0
        self._k_frac = 0.2
        self._compute_pulse = lambda l, u, k, x: \
            (1 + 1/(1 + exp(-k * (x - u))) - 1/(1 + exp(k * (x - l)))) / 2

        # Auto-add to network only if this initialization is for an *InputGroup
        # object and not an instance of a subclass which might belong to a
        # different category (e.g., a neuron group)
        if 'InputGroup' in self.__class__.__name__:
            if State.is_defined('network'):
                State.network.add_input_group(self)

    def update(self):
        """
        Update the model neurons.

        Subclasses should overload this to update the values in the unit
        variable `output`. Overloaded methods can call super().update()
        to calculate metrics.
        """
        if self.do_metrics:
            self.update_metrics()

    def update_metrics(self):
        """
        Calculate model-agnostic metrics on the unit output.
        """
        # Compute mean/min/max of unit output values
        self._mean_output = self.output.mean()
        self._min_output = min(self._min_output, self._mean_output)
        self._max_output = max(self._max_output, self._mean_output)

        # Compute mean/min/max of unit activity (nonzero output) fraction
        self._mean_active = (self.output > 0).mean()
        self._min_active = min(self._min_active, self._mean_active)
        self._max_active = max(self._max_active, self._mean_active)

        # Calculate smoothness coefficients based on current range
        k_output = self.k_frac * (self._max_output - self._min_output)
        k_active = self.k_frac * (self._max_active - self._min_active)

        # Compute range-adaptive sigmoidal nonlinearities as pulse metrics
        self._pulse_output = self._compute_pulse(
            self._min_output, self._max_output, k_output, self._mean_output)
        self._pulse_active = self._compute_pulse(
            self._min_active, self._max_active, k_active, self._mean_active)

        # As in kurtosis calculations, use the 4th power to emphasize the
        # extremities, then the mean tells you at least one or the other is
        # currently at extreme values.
        self._pulse = (self._pulse_active**4 + self._pulse_output**4) / 2

    def get_output_metric(self):
        return self._pulse_output

    def get_active_metric(self):
        return self._pulse_active

    def get_pulse_metric(self):
        return self._pulse
