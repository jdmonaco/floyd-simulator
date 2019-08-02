"""
Container for keyword specifications.
"""

try:
    import panel as pn
except ImportError:
    print('Warning: install `panel` to use interactive dashboards.')

import copy
import inspect
from collections import namedtuple

from pouty.console import ConsolePrinter, log
from roto.dicts import AttrDict, Tree
from tenko.base import TenkoObject

from .state import State


"""
MySpec = paramspec('MySpec', a=1, b=2, c=3)
> MySpec is a pseudo-type to construct spec objects

myspec1 = MySpec()
> Defaults: a=1, b=2, c=3

myspec2 = MySpec(c=4)
> Defaults: a=1, b=2, c=4

NewSpec = paramspec('NewSpec', spec=myspec2, d=5, e=6, f=7)
> NewSpec is another pseudo-type constructed from myspec2 and other values

newspec1 = NewSpec(c=3)
> Defaults: a=1, b=2, c=3, d=5, e=6, f=7
> The `c=3` value overrides the `c=4` value from myspec2.
"""

# Code from Param (PyViz project)

def classlist(klass):
    return inspect.getmro(klass)[::-1]

def get_all_slots(klass):
    # A subclass's __slots__ attribute does not contain slots defined
    # in its superclass (the superclass' __slots__ end up as
    # attributes of the subclass).
    all_slots = []
    parent_param_classes = [c for c in classlist(klass)[1::]]
    for c in parent_param_classes:
        if hasattr(c, '__slots__'):
            all_slots += c.__slots__
    return all_slots

def get_occupied_slots(instance):
    return [slot for slot in get_all_slots(type(instance))
            if hasattr(instance, slot)]

# End Param code


class Param(object):

    """
    A class attribute instance defining properties of inheritable values.
    """

    __slots__ = ['default', 'start', 'end', 'step', 'dtype', 'doc', 'units',
                 'widget', 'owner', 'attrname']

    def __init__(self, default=None, start=None, end=None, step=None,
        dtype=None, doc=None, units=None, widget=None):
        if default is None:
            raise ValueError('Param instances must have a default value')
        self.default = default
        self.start = start
        self.end = end
        self.step = step
        self.dtype = dtype
        self.doc = doc
        self.units = units
        self.widget = widget
        self.owner = None
        self.attrname = None

    def __repr__(self):
        attrs = [f'{a}={getattr(self, a)!r}' for a in self.__slots__
                     if getattr(self, a) is not None and \
                         a not in ('owner', 'attrname')]
        return self.__class__.__name__ + '(' + ', '.join(attrs) + ')'

    def update(self, p):
        """
        Update slots with values from another Param instance.
        """
        for slot in get_all_slots(type(self)):
            if hasattr(p, slot):
                setattr(self, slot, getattr(p, slot))

    # TODO: I think Params need to be data descriptors to get the instance
    # attribute accessor to work

    def __get__(self, name, type=None):
        pass

    def __set__(self, name, type=None):
        pass

    # Code from Param (PyViz project)

    def __getstate__(self):
        state = {}
        for slot in get_occupied_slots(self):
            state[slot] = getattr(self, slot)
        return state

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    # End Param code


_attrname = lambda n: f'_{n}_specified_value'


class Specified(TenkoObject):

    """
    Self-validating attribute stores of restricted key-value sets.
    """

    def __init__(self, **keyvalues):
        """
        Spec keyword arguments are stored in the attribute dict (__dict__).

        I.e., the initial constructor keywords (and keywords of the optional
        spec 'parent' object') are the only keyword values that will ever be
        expressed by iteration.
        """
        object.__setattr__(self, '_initialized', False)
        self._params = AttrDict()
        self._widgets = {}
        self._watchers = {}
        super().__init__()

        inheritance = classlist(self.__class__)
        for cls in inheritance:
            for name, value in vars(cls).items():
                if type(value) is Param:
                    value = copy.copy(value)
                    value.owner = self
                    value.attrname = _attrname(name)
                    self._params[name] = value
                    self.debug(f'copied Param {name!r} from {cls.__name__}')

        for key, value in keyvalues.items():
            if key == '_spec_class' and value != self.klass:
                self.out(f'Serialized type name {value!r} does not match',
                         f'class {self.klass!r}', warning=True)
                continue

            if type(value) is Param:
                value = copy.copy(value)
                if key in self._params:
                    value.update(self._params[key])
                    self.debug('updated Param {key!r} with inherited values')
            else:
                value = Param(default=value)
                self.debug('created new Param {key!r} with default {value!r}')
            value.owner = self
            value.attrname = _attrname(key)
            self._params[key] = value

        for name, p in self._params.items():
            print(name, p, p.attrname, p.default)
            object.__setattr__(self, p.attrname, p.default)

        self._initialized = True

    def __repr__(self):
        if not hasattr(self, '_params') or len(self._params) == 0:
            return self.klass + '()'
        indent = ' '*4
        r = self.klass + '(\n'
        for k, v in self.items():
            v = self._params[k]
            lines = f'{k} = {repr(v)},'.split('\n')
            for line in lines:
                r += indent + line + '\n'
        return r + ')'

    def __contains__(self, name):
        return name in self._params

    def __getattr__(self, name):
        print('__getattr__', name)
        if not self._initialized or name not in self._params:
            return object.__getattribute__(self, name)
        return self[name]

    def __setattr__(self, name, value):
        print('__setattr__', name, value)
        if not self._initialized or name not in self._params:
            return object.__setattr__(self, name, value)
        self[name] = value

    def __getitem__(self, name):
        print('__getitem__', name)
        if name not in self._params:
            self.out(f'Unknown parameter {name!r}', error=True)
            raise KeyError(f'{name!r}')
        return object.__getattribute__(self, self._params[name].attrname)

    def __setitem__(self, name, value):
        print('__setitem__', name, value)
        if name not in self:
            self.out(f'Unknown parameter {name!r}', warning=True)
            return
        if type(value) is Param:
            value = copy.copy(value)
            if name in self._params:
                value.update(self._params[name])
            else:
                value.owner = self
                value.attrname = _attrname(name)
            self._params[name] = value
            self.debug('merged Param {name!r} with current values')
            return
        attrname = self._params[name].attrname
        curvalue = object.__getattribute__(self, attrname)
        if curvalue != value:
            start = self._params[name].start
            end = self._params[name].end
            if value < start:
                self.out(f'New {name!r} value {value!r} is below start'
                         f'parameter {start!r}', warning=True)
            if value > end:
                self.out(f'New {name!r} value {value!r} is above end'
                         f'parameter {end!r}', warning=True)
            object.__setattr__(self, attrname, value)
            self.debug(f'set attr {name!r} to value {value!r}')
        if name in self._widgets:
            widget = self._widgets[name]
            if widget.value != value:
                widget.value = value
                widget.param.trigger('value')
                self.debug(f'updated widget {name!r} value to {value!r}')

    def __iter__(self):
        return iter(self._params.keys())

    def items(self):
        for name, p in self._params.items():
            yield (name, p)

    def defaults(self):
        """
        Iterate over (name, default) tuples for all parameters.
        """
        for name, p in self._params.items():
            yield (name, p.default)

    def update(self, **kw):
        """
        Update current parameter values.
        """
        for key, value in kw.items():
            self[key] = value

    def reset(self):
        """
        Reset keyword values to default values (from parameters).
        """
        self.update(**self.defaults())

    def get_widgets(self, *names, exclude=None):
        """
        Return a tuple of Panel FloatSlider objects for Param values.
        """
        if 'context' not in State:
            self.out('Cannot create sliders outside of simulation context',
                     error=True)
            return

        # Use arguments or gather full list of parameter names with widgets
        if names:
            name = list(names)
        else:
            names = list([name for name in self._params
                          if self._params[name].widget is not None])

        # If exclusions specified as list or singleton, remove from list
        if exclude is not None:
            if type(exclude) in (list, tuple):
                for exc in exclude:
                    if exc in names:
                        names.remove(exc)
            elif exclue in names:
                names.remove(exclude)

        for name in self._widgets.keys():
            if name in names:
                del self._widgets[name]
        if len(self._params) == 0:
            self.debug('found no params')
            return ()

        # Construct the widgets
        for name, p in self._params.items():
            if p.widget == 'FloatSlider':
                self._widgets[name] = pn.widgets.FloatSlider(
                        name            = name,
                        value           = self[name],
                        start           = p.start,
                        end             = p.end,
                        step            = p.step,
                        callback_policy = 'mouseup',
                )
            else:
                self.out('Widget type {p.widget!r} not currently supported',
                         warning=True)

        # Define an event-based callback function
        def callback(*events):
            State.context.toggle_anybar()
            for event in events:
                widget = event.obj
                name = widget.name
                self[name] = event.new

        # Register the callback with each of the sliders
        for name, widget in self._widgets.items():
            self._watchers[name] = widget.param.watch(callback, 'value')

        self.debug('created {} new widgets', len(self._widgets))
        return tuple(self._widgets.values())

    def unlink_widgets(self):
        """
        Remove callbacks from Panel FloatSlider objects.
        """
        for name, widget in self._widgets.items():
            widget.param.unwatch(self._watchers[name])

    def as_dict(self, subtree=None, T=None):
        """
        Return a copy of the spec as a nested dict object.
        """
        if T is None:
            T = Tree()
        if subtree is None:
            subtree = self
        for name, value in subtree.items():
            if isinstance(subtree, Specified):
                value = subtree[name]
            if hasattr(value, 'items'):
                self.as_dict(value, T[name])
                continue
            T[name] = value
        if isinstance(subtree, Specified):
            T['_spec_class'] = self.klass
        return T
