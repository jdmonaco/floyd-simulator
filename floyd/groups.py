"""
Base class for groups of model units.
"""

from functools import reduce
import operator as op

from toolbox.numpy import *
from pouty import log

from .funcs import SamplerType


class BaseUnitGroup(object):

    base_variables = ()
    extra_variables = ()
    base_dtypes = {}
    default_dtype = 'f'

    def __init__(self, N_or_shape, name, spec=None, dtype=None):
        """
        Create a group of model units with named variables.

        Class-defined unit variables are made available as instance attributes.
        The optional spec object is made available as instance attribute `p`.

        Arguments
        ---------
        N_or_shape : int | tuple of ints
            The size of the group (or shape if multiple dimensions needed)

        name : str
            Unique name given to the group

        spec : Spec subclass instance, optional (default None)
            A Spec instance provides shared parameters to the unit group

        dtype : '?' | 'u' | 'i' | 'f' | 'd', optional (default 'f')
            Default numpy dtype to use for initializing array variables
        """
        self.name = name
        self.shape = N_or_shape
        if np.iterable(self.shape):
            self.N = reduce(op.mul, self.shape)
        else:
            self.N = self.shape

        self.vardtypes = dict(self.base_dtypes)
        if type(dtype) is dict:
            self.vardtypes = merge_two_dicts(self.vardtypes, dtypes)
        elif type(dtype) is str and dtype[0] in '?uifd':
            self.dtype = dtype
        else:
            self.dtype = self.default_dtype

        if spec is not None:
            spec.validate()
            self.p = spec

        self.variables = list(set(self.base_variables + self.extra_variables))
        for varname in self.variables:
            thisdtype = self.vardtypes.get(varname, self.dtype)
            object.__setattr__(self, varname, zeros(self.shape, thisdtype))

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'{self.__class__.__name__}(name={self.name!r}, ' \
               f'N={self.shape}, spec={self.p})'

    def __format__(self, fmtspec):
        return self.name.__format__(fmtspec)

    def _evaluate(self, value):
        if isinstance(value, SamplerType):
            return value.sample(self.shape)
        return value

    def __setattr__(self, name, value):
        if hasattr(self, 'variables') and name in self.variables:
            getattr(self, name)[:] = self._evaluate(value)
            return
        if name == 'p' and not hasattr(value, '_speckeys'):
            log(f'Prohibiting attempt to set \'p\' with non-spec: {p}',
                    prefix=f'{self.name}UnitGroup', warning=True)
            return
        object.__setattr__(self, name, value)

    def set(self, **values):
        """
        Set variables using keyword arguments.
        """
        for name, value in values.items():
            setattr(self, name, value)
