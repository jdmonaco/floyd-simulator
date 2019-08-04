"""
Debugging the Param inheritance issue in SimulatorContext subclasses.
"""

import copy
import enum
import inspect
import functools

from tenko.context import AbstractBaseContext #, step
from pouty import debug, debug_mode
from pouty.console import snow as hilite
from roto.dicts import AttrDict
from toolbox.numpy import *
# from specify import *
from specify.utils import *
# from floyd import *


class RunMode(enum.Enum):
    ANIMATE  = 'create_movie'
    INTERACT = 'launch_dashboard'
    RECORD   = 'collect_data'


debug_mode(True)

def is_param(obj):
    for klass in classlist(type(obj)):
        if klass.__name__ == 'Param':
            return True
    return False

def is_specified(obj):
    for klass in classlist(type(obj))[::-1]:
        if klass.__name__ in ('Specified', 'SpecifiedMetaclass'):
            return True
    return False


class Param(object):

    __slots__ = ['name', 'default', 'owner', 'attrname']

    def __init__(self, default=None):
        self.name = None
        self.default = default
        self.owner = None
        self.attrname = None

    def __repr__(self):
        indent = ' '*4
        r = f'{self.__class__.__name__}('
        empty = True
        for a in get_valued_slots(self):
            if a in ('owner', 'name', 'constant', 'attrname'):
                continue
            r += f'{a}={getattr(self, a)!r}, '
            empty = False
        if empty:
            return r + ')'
        return r[:-2] + ')'

    def _set_names(self, name):
        if None not in (self.name, self.owner) and name != self.name:
            raise AttributeError
        self.name = name
        self.attrname = f'_{name}_specified_value'
        debug(f'set names: {self.name!r} and {self.attrname!r}')

    def __get__(self, obj, cls):
        if obj is None:
            if self.default is None:
                raise TypeError(f'No default value set for {self.name!r}')
            debug(f'{cls}.__get__: returning default {self.default!r}')
            return self.default
        debug(f'__get__: returning current value or default')
        return obj.__dict__.get(self.attrname, self.default)

    def __set__(self, obj, value):
        if obj is None:
            self.default = value
            debug(f'{self.owner!r}: set {self.name!r} default to {value!r}')
        else:
            oldval = obj.__dict__[self.attrname]
            obj.__dict__[self.attrname] = value
            debug(f'set {self.name!r} ({self.attrname!r}) from {oldval!r} to {value!r}')


class State(AttrDict):
    pass

class Specs(AttrDict):
    pass


class SpecifiedMetaclass(type):

    def __new__(metacls, name, bases, clsdict):
        debug(f'SpecifiedMetaclass.__new__(name={name!r}, bases={bases!r})')

        specs = clsdict['spec'] = Specs()
        for base in bases:
            for superclass in classlist(base)[::-1]:
                if not isinstance(superclass, metacls):
                    continue
                for key, value in vars(superclass).items():
                    if key == 'name': continue
                    if not is_param(value):
                        continue
                    if key in clsdict:
                        clsvalue = clsdict[key]
                        if is_param(clsvalue):
                            continue
                        p = type(value)(default=copy.deepcopy(clsvalue))
                        specs[key] = clsdict[key] = p
                        debug(f'copied ancestor for {key!r} in '
                              f'{superclass!r} as {p!r}')
                    else:
                        specs[key] = value
                        debug(f'found ancestor {key!r} in {superclass!r}')

        return super().__new__(metacls, name, bases, clsdict)

    def __init__(cls, name, bases, clsdict):
        debug(f'SpecifiedMetaclass.__init__(cls={cls!r}, name={name!r}, '
              f'bases={bases!r})')
        type.__init__(cls, name, bases, clsdict)
        cls.name = name

        # Initialize all Params by setting names and inheriting properties
        for pname, param in clsdict.items():
            if not is_param(param) or pname == 'name':
                continue
            param._set_names(pname)
            cls.__param_inheritance(pname, param)
            cls.spec[pname] = param
            debug(f'initialized Param {pname!r} to {param!r}')

    def __param_inheritance(cls, pname, param):
        slots = {}
        for p_class in classlist(type(param))[1::]:
            slots.update(dict.fromkeys(p_class.__slots__))
        setattr(param, 'owner', cls)
        del slots['owner']
        for slot in slots.keys():
            superclasses = iter(classlist(cls)[::-1])
            while getattr(param, slot) is None:
                try:
                    param_super_class = next(superclasses)
                except StopIteration:
                    break
                ancestor = param_super_class.__dict__.get(pname)
                if ancestor is not None and hasattr(ancestor, slot):
                    setattr(param, slot, getattr(ancestor, slot))

    def __setattr__(cls, name, value):
        debug(f'__setattr__({cls!r}, {name!r}, {value!r})')
        parameter, owner = cls.get_param_descriptor(name)
        if parameter and not is_param(value):
            if owner != cls:
                parameter = copy.copy(parameter)
                parameter.owner = cls
                type.__setattr__(cls, name, parameter)
                debug(f'set {name!r} to copied parameter {parameter!r}')
                cls.spec[name] = parameter
            cls.__dict__[name].__set__(None, value)
        else:
            type.__setattr__(cls, name, value)
            if is_param(value):
                cls.__param_inheritance(name, value)
                cls.spec[name] = value
                debug(f'set new key {name!r} to {value!r}')
            else:
                if not (name.startswith('_') or \
                        name in ('name', 'spec')):
                    debug('setting non-Param class attribute '
                         f'{cls.__name__}.{name} to value {value!r}')

    def get_param_descriptor(cls, pname):
        classes = classlist(cls)
        for c in classes[::-1]:
            attribute = c.__dict__.get(pname)
            if is_param(attribute):
                return attribute, c
        return None, None


class TenkoObject(object):
    __counts = {}
    def __init__(self, name=None):
        super().__init__()
        if hasattr(self.__class__, 'name'):
            self.klass = self.__class__.name
        else:
            self.klass = self.__class__.__name__
        if name is None:
            if self.klass in self.__counts:
                self.__counts[self.klass] += 1
            else:
                self.__counts[self.klass] = 0
            c = self.__counts[self.klass]
            self.name = f'{self.klass}_{c:03d}'
        else:
            self.name = name

    def __repr__(self):
        return f'{self.klass}(name={self.name!r})'



class AbstractContext(TenkoObject):

    _desc = '_desc'
    _tag = '_tag'

    def __init__(self, desc=None, tag=None, **kwargs):
        super().__init__(**kwargs)
        self._desc = self._desc if desc is None else desc
        self._tag = self._tag if tag is None else tag


class Specified(TenkoObject, metaclass=SpecifiedMetaclass):

    """
    Self-validating attribute stores of restricted key-value sets.
    """

    def __init__(self, **kwargs):
        """
        Class-scope parameter default values are instantiated in the object.
        """
        super().__init__()
        debug(f'{self.__class__!r}.__init__(kwargs={kwargs!r})')
        self._initialized = False

        # Build list from Specified hierarchy of Param names to instantiate
        to_instantiate = {}
        for cls in classlist(type(self)):
            if not is_specified(cls):
                continue
            for name, value in vars(cls).items():
                if is_param(value) and name != 'name':
                    to_instantiate[name] = value
                    self.spec[name] = value

        # Set the internal instance attribute values to copied Param defaults
        for param in to_instantiate.values():
            key = param.attrname
            new_value_from_default = copy.deepcopy(param.default)
            self.__dict__[key] = new_value_from_default
            debug(f'init: set {key!r} to {new_value_from_default!r}')

        # Set the value of keyword arguments
        for key, value in kwargs.items():
            descriptor, _ = type(self).get_param_descriptor(key)
            if not descriptor:
                debug(f'setting non-Param attribute {key!r} to {value!r}')
            setattr(self, key, value)
            debug(f'init: set {key!r} to {value!r} from kwargs')

        self._initialized = True

    def __str__(self):
        indent = ' '*4
        r = f'{self.name}('
        if len(self.spec):
            r += '\n'
        for k, param in self.items():
            dflt = param.default
            val = getattr(self, param.attrname)
            if val == dflt:
                line = f'{k} = {val!r}'
            else:
                line = hilite(f'{k} = {val!r} [default: {dflt!r}]')
            lines = line.split('\n')
            for line in lines:
                r += indent + line + '\n'
        return r + ')'

    def __contains__(self, name):
        return name in self.spec

    def __iter__(self):
        return iter(self.spec.keys())

    def params(self):
        """
        Iterate over (name, Param object) tuples for all parameters.
        """
        for name, p in self.spec.items():
            yield (name, p)

    items = params  # alias for to_dict() method

    def values(self):
        """
        Iterate over (name, value) tuples for all current parameter values.
        """
        for name in self:
            yield (name, getattr(self, name))

    def defaults(self):
        """
        Iterate over (name, default) tuples for all parameters.
        """
        for name, p in self.params():
            yield (name, p.default)

    def update(self, **kwargs):
        """
        Update parameter values from keyword arguments.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def reset(self):
        """
        Reset parameters to default values.
        """
        self.update(**dict(self.defaults()))


def simulate(func=None, *, mode=None):
    """
    Decorator for simulation methods with keyword-only mode argument, which is
    only necessary for any non-standard simulation methods (i.e., user-defined
    methods not named `create_movie`, `collect_data`, or `launch_dashboard`.
    """
    if func is None:
        return functools.partial(simulate, mode=RunMode[str(mode).upper()])
    if mode is None:
        mode = RunMode(func.__name__)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        self = args[0]
        status = { 'OK': False }

        State.run_mode = mode
        debug(f'running {func.__name__} in {mode!r} mode')

        self._prepare_simulation(args, kwargs, finish_setup=True)
        # self._step_enter(func, args, kwargs)
        # self._prepare_simulation(args, kwargs)
        # res = self._step_execute(func, args, kwargs, status)
        # self._step_exit(func, args, kwargs, status)
        res = func(*args, **kwargs)
        return res
    return wrapped


class SpecifiedContext(AbstractContext, Specified):

    title = Param('Testing Context')
    show_debug = Param(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        debug('context: returned from super().__init__')

        debug_mode(self.show_debug)
        self._specfile_init = kwargs.get('specfile')
        self._prepare_simulation(None, kwargs, finish_setup=False)

    def _prepare_simulation(self, args, kwargs, finish_setup=True):
        debug('_prepare_simulation', repr(list(kwargs.keys())))

        # Consume keywords for parameters in class attribute `spec`
        spec_keys = [k for k in kwargs.keys() if k in self.spec]
        for key in spec_keys:
            setattr(self, key, kwargs.pop(key))
            debug(f'consuming key {key!r} with value {getattr(self, key)!r}',
                    prefix='_prepare')

    @simulate
    def create_movie(self, dpi=120):
        debug(f'create_movie: running in {dpi} resolution')
        self.setup_model()


class MyModel(SpecifiedContext):

    # title = 'My New Model'
    a = Param(1.0)
    b = Param(pi)

    def setup_model(self, **kwargs):
        debug(f'self = {self!r}')
        debug(f'a = {self.a!r}')
        debug(f'b = {self.b!r}')
        debug(f'kwargs keys = {list(kwargs.keys())}')


model = MyModel(desc='specified debug testing', a=1.234)
# model.setup_model()
