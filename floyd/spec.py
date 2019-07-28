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

def _isparam(p):
    for attr in _param_attrs:
        if not hasattr(p, attr):
            return False
    return True


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
        self.out = ConsolePrinter(prefix=self.__class__.__name__)

        def set_item(key, value):
            if _isparam(value):
                setattr(self, key, value.value)
                self.defaults[key] = value.value
                self.params[key] = value
                self.out(f'add Param key {key!r}', debug=True)
                return
            if callable(value) and value.__name__ == 'getspec':
                value = value()
                self.out(f'instantiated spec {value!r}', debug=True)
            setattr(self, key, value)
            self.out(f'set key {key!r} to {value!r}', debug=True)
            self.defaults[key] = value

        if spec is not None:
            if callable(spec):
                spec = spec()
            for key, value in spec:
                if key in spec.params:
                    set_item(key, spec.params[key])
                    continue
                set_item(key, value)

        for key, value in keyvalues.items():
            set_item(key, value)

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
            self.out(f'Unknown key: {key!r}', error=True)
            raise KeyError(f'{key!r}')
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key not in self:
            self.out(f'Unknown key: {key!r}', warning=True)
            return
        if _isparam(value):
            setattr(self, key, value.value)
            self.params[key] = value
            return
        if callable(value) and value.__name__ == 'getspec':
            value = value()
        if key in self.params:
            self.params[key] = Param(*((value,) + self.params[key][1:]))
        setattr(self, key, value)

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
            self.out('Cannot create sliders outside of simulation context',
                     error=True)
            return

        if self.sliders:
            self.out('found old sliders', debug=True)
            return self.sliders
        if len(self.params) == 0:
            self.out('found no params', debug=True)
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
                self.out(f'{key}: {event.old} -> {self[key]}', debug=True)

        # Register the callback with each of the sliders
        for name, slider in self.sliders.items():
            self.watchers[name] = slider.param.watch(callback, 'value')

        self.out('created {} new sliders', len(self.sliders), debug=True)
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
                    self.out(f'Missing key: {key!r}', warning=True)
                    valid = False
            for key in self._speckeys:
                if key not in self.defaults:
                    self.out(f'Invalid key: {key!r}', warning=True)
                    valid = False
        except Exception as e:
            self.out('Not a spec', warning=True)
            valid = False

        if raise_on_fail and not valid:
            raise ValueError(f'Could not validate spec: {self!r}')
        return valid
