"""
Base class for unit groups of model neurons.
"""

__all__ = ('BaseNeuronGroup',)


import copy

from toolbox.numpy import *
from specify import LogSlider, is_param
from specify.utils import get_all_slots

from ..state import State

from .input import InputGroup


class BaseNeuronGroup(InputGroup):

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
        
        Returns exit status code {0,1,2} so that subclasses or downstream
        consumers can determine whether the projection was successfully added.
        """
        if projection.post is not self:
            self.out('{} does not project to {}', projection.name, self.name,
                    error=True)
            return 1
        
        if projection in self.afferents.values():
            self.out('{} already added to {}', projection.name, self.name,
                    error=True)
            return 2

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
        
        return 0

    def update(self):
        """
        Update the model neurons. 
        
        Subclasses should overload this to update the values in the unit 
        variable `output` and to call super().update() for pulse metrics.
        """
        super().update()

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
