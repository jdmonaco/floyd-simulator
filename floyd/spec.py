"""
Container for keyword specifications.
"""

try:
    import panel as pn
except ImportError:
    print('Warning: install `panel` to use interactive dashboards.')

from collections import namedtuple

from pouty.console import ConsolePrinter, log
from roto.dicts import AttrDict, Tree

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

def paramspec(name, parent=None, instance=False, **defaults):
    """
    Create a factory closure for named Spec objects with optional parent spec.
    """
    class TempSpec(Spec): pass
    TempSpec.__name__ = name
    parent = TempSpec(spec=parent, **defaults)
    def getspec(spec=None, **keyvalues):
        newspec = parent.copy()
        newspec.update(spec=spec, **keyvalues)
        return newspec
    if instance:
        return getspec()
    return getspec

def is_param(p):
    for attr in _param_attrs:
        if not hasattr(p, attr):
            return False
    return True

def is_spec(s):
    return hasattr(s, '_speckeys')


_param_attrs = ['value', 'start', 'end', 'step', 'note']

Param = namedtuple('Param', _param_attrs)


class Spec(object):

    """
    Self-validating attribute stores of restricted key-value sets.
    """

    def __init__(self, spec=None, **keyvalues):
        """
        Spec keyword arguments are stored in the attribute dict (__dict__).

        I.e., the initial constructor keywords (and keywords of the optional
        spec 'parent' object') are the only keyword values that will ever be
        expressed by iteration.

        Note: If an eisting spec is supplied by keyword `spec`, then that
        spec's keyword values are consumed first, followed by the other keyword
        arguments. The union of those keys becomes the default key set.
        """
        self.params = AttrDict()
        self.defaults = AttrDict()
        self.sliders = {}
        self.watchers = {}
        self._klass = self.__class__.__name__
        self._out = ConsolePrinter(prefix=self._klass)
        self._debug = lambda *msg: self._out(*msg, debug=True)

        def set_item(key, value, source):
            if is_param(value):
                self.params[key] = value
                self.defaults[key] = value.value
                setattr(self, key, value.value)
                self._debug(f'add Param key {key!r} from {source}')
                return
            if callable(value) and value.__name__ == 'getspec':
                value = value()
                self._debug(f'instantiated spec {value!r} from {source}')
            setattr(self, key, value)
            self.defaults[key] = value
            self._debug(f'set key {key!r} to {value!r} from {source}')

        if spec is not None:
            if callable(spec):
                spec = spec()
            for key, value in spec:
                if key in spec.params:
                    set_item(key, spec.params[key], spec._klass)
                    continue
                set_item(key, value, spec._klass)

        for key, value in keyvalues.items():
            if type(value) is dict and '_spec_type' in value:
                specname = value.pop('_spec_type')
                value = paramspec(specname, instance=True, **value)
            set_item(key, value, 'keywords')

        self._speckeys = tuple(self.defaults.keys())

    def __repr__(self):
        indent = ' '*4
        r = self.__class__.__name__ + '(\n'
        for k, v in self.items():
            if k in self.params:
                v = self.params[k]
            lines = f'{k} = {repr(v)},'.split('\n')
            for line in lines:
                r += indent + line + '\n'
        return r + ')'

    def __contains__(self, key):
        return key in self._speckeys

    def __getitem__(self, key):
        if key not in self:
            self._out(f'Unknown key: {key!r}', error=True)
            raise KeyError(f'{key!r}')
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key not in self:
            self._out(f'Unknown key: {key!r}', warning=True)
            return
        if is_param(value):
            if key in self.params and self.params[key] == value:
                return
            self.params[key] = value
            setattr(self, key, value.value)
            self._debug(f'set key {key!r} to Param value {value!r}')
            return
        if callable(value) and getattr(value, '__name__', '_') == 'getspec':
            value = value()
        curvalue = getattr(self, key)
        if curvalue != value:
            setattr(self, key, value)
            self._debug(f'set key {key!r} to value {value!r}')
        if key in self.params:
            if self.params[key].value != value:
                self.params[key] = Param(*((value,) + self.params[key][1:]))
                self._debug(f'updated param {key!r} value to {value!r}')
            if key in self.sliders:
                slider = self.sliders[key]
                if slider.value != value:
                    slider.value = value
                    slider.param.trigger('value')
                    self._debug(f'updated slider {key!r} value to {value!r}')

    def __iter__(self):
        return self.items()

    def items(self):
        """
        Iterate the current spec key-value pairs.
        """
        for key, value in self.__dict__.items():
            if key in self:
                yield (key, value)

    def keys(self):
        """
        Iterate the current spec keys.
        """
        for key in self._speckeys:
            yield key

    def update(self, spec=None, **keyvalues):
        """
        Update this spec with another spec and/or keyword arguments.
        """
        if spec is not None:
            if callable(spec):
                spec = spec()
            for key, value in spec:
                if key in spec.params:
                    self[key] = spec.params[key]
                    continue
                self[key] = value
        for key, value in keyvalues.items():
            self[key] = value

    def reset(self):
        """
        Reset keyword values to default values (from init args or paramspec).
        """
        self.update(**self.defaults)

    def copy(self):
        """
        Return a copy of the spec.
        """
        newspec = self.__class__(**self)
        newspec.params = self.params.copy()
        newspec.defaults = self.defaults.copy()
        return newspec

    def panel_sliders(self):
        """
        Return a tuple of Panel FloatSlider objects for Param values.
        """
        from .state import State
        if 'context' not in State:
            self._out('Cannot create sliders outside of simulation context',
                     error=True)
            return

        if self.sliders:
            self._debug('found old sliders')
            return self.sliders
        if len(self.params) == 0:
            self._debug('found no params')
            return ()

        # Construct the slider widgets
        for name, p in self.params.items():
            self.sliders[name] = pn.widgets.FloatSlider(
                    name=name,
                    value=self[name],
                    start=p.start,
                    end=p.end,
                    step=p.step,
                    callback_policy='mouseup',
            )

        # Define an event-based callback function
        def callback(*events):
            State.context.toggle_anybar()
            for event in events:
                slider = event.obj
                key = slider.name
                self[key] = event.new

        # Register the callback with each of the sliders
        for name, slider in self.sliders.items():
            self.watchers[name] = slider.param.watch(callback, 'value')

        self._debug('created {} new sliders', len(self.sliders))
        return tuple(self.sliders.values())

    def unlink_sliders(self):
        """
        Remove callbacks from Panel FloatSlider objects.
        """
        for name, slider in self.slider.items():
            slider.param.unwatch(self.watchers[name])

    def as_dict(self, subtree=None, T=None):
        """
        Return a copy of the spec as a nested dict object.
        """
        if T is None:
            T = Tree()
        if subtree is None:
            subtree = self
        for key, value in subtree.items():
            if hasattr(value, 'items'):
                self.as_dict(value, T[key])
                continue
            T[key] = value
        if is_spec(subtree):
            T['_spec_type'] = self._klass
        return T

    @classmethod
    def is_valid(cls, spec, raise_on_fail=True):
        """
        Validate the given spec by checking for the right keys.
        """
        if not hasattr(spec, 'validate'):
            log('Not a spec: {}', spec, prefix=cls.__name__, warning=True)
            if raise_on_fail:
                raise ValueError('Missing validate method: {}'.format(spec))
            return False

        return spec.validate()

    def validate(self, raise_on_fail=True):
        """
        Validate by checking for the right keys.
        """
        valid = True
        try:
            for key in self.defaults:
                if key not in self:
                    self._out(f'Missing key: {key!r}', warning=True)
                    valid = False
            for key in self._speckeys:
                if key not in self.defaults:
                    self._out(f'Invalid key: {key!r}', warning=True)
                    valid = False
        except Exception as e:
            self._out('Not a spec', warning=True)
            valid = False

        if raise_on_fail and not valid:
            raise ValueError(f'Could not validate spec: {self!r}')
        return valid
