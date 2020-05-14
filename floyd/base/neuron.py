"""
Base class for unit groups of model neurons.
"""

__all__ = ('BaseNeuronGroup',)


import copy
from functools import partial

from toolbox.numpy import *
from specify import Specified, Param, LogSlider, is_param
from specify.utils import get_all_slots

from ..state import State

from .groups import BaseUnitGroup


class BaseNeuronGroup(Specified, BaseUnitGroup):

    k_frac = Param(0.2, doc='pulse metric smoothness')
    do_metrics = Param(False, doc='whether to auto-compute basic metrics')
    
    base_variables = ('x', 'y', 'output')

    def __init__(self, *, name, shape, g_log_range=(-5, 5), g_step=0.05, 
        **kwargs):
        """
        Initialize data structures and context-specified gain parameters.
        """
        self._initialized = False 
        super().__init__(name=name, shape=shape, **kwargs)

        # Add any conductance gain values in the shared context as Params
        self.afferents = {}
        self.gain_keys = []
        self.gain_param_base = LogSlider(default=0.0, start=g_log_range[0],
                end=g_log_range[1], step=g_step, units='nS')
        for k, v in vars(State.context.__class__).items():
            if k.startswith(f'g_{name}_'):
                self._add_gain_spec(k, v)

        # Initialize pulse metrics with default parameters
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

        if State.is_defined('network'):
            State.network.add_neuron_group(self)
        
        self._initialized = True

    def _add_gain_spec(self, gname, value):
        """
        Add Param (slider) objects for any `g_{post.name}_{pre.name}` class
        attributes (Param object or just default values) of the shared context.
        """
        _, post, pre = gname.split('_')
        new_param = copy.copy(self.gain_param_base)
        new_param.doc = f'{pre}->{post} max conductance'
        new_all_slots = get_all_slots(type(new_param))

        if is_param(value):
            old_all_slots = get_all_slots(type(value))
            for k in old_all_slots:
                if hasattr(value, k) and getattr(value, k) is not None and \
                        k in new_all_slots:
                    slotval = copy.copy(getattr(value, k))
                    object.__setattr__(new_param, k, slotval)
            value = new_param.default
        else:
            new_param.default = copy.deepcopy(value)

        self.add_param(gname, new_param)
        self.gain_keys.append(gname)
        self.debug(f'added gain {gname!r} with value {new_param.default!r}')

    def add_afferent_projection(self, projection):
        """
        Add afferent (input) projection to this neuron group.
        """
        if projection.post is not self:
            self.out('{} does not project to {}', projection.name, self.name,
                    error=True)
            return
        
        if projection in self.afferents.values():
            self.out('{} already added to {}', projection.name, self.name,
                    error=True)
            return

        # Add projection to this group's afferent inputs
        #
        # Note: gain spec names take the form `g_<post.name>_<pre.name>`.
        gname = 'g_{}_{}'.format(projection.post.name, projection.pre.name)
        self.afferents[gname] = projection

        # Check whether the gain spec has already been found in the
        # context. If not, then add a new Param to the spec with a
        # default value of 0.0 (log10(1)).
        if gname in self.gain_keys:
            self.debug(f'gain spec {gname!r} exists for {projection.name!r}')
        else:
            self._add_gain_spec(gname, 0.0)
            self.debug(f'added gain spec {gname!r} for {projection.name!r}')

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

    def get_neuron_sliders(self):
        """
        Return a tuple of Panel FloatSlider objects for neuron Param values.
        """
        return self.get_widgets(exclude=self.gain_keys)

    def get_gain_sliders(self):
        """
        Return a tuple of Panel FloatSlider objects for gain Param values.
        """
        return self.get_widgets(include=self.gain_keys)
