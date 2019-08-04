"""
Base class for groups of model units.
"""

from functools import reduce
import operator as op

from toolbox.numpy import *
from tenko.base import TenkoObject


class BaseUnitGroup(TenkoObject):

    base_variables = ()
    extra_variables = ()
    base_dtypes = {}
    default_dtype = 'f'

    def __init__(self, N_or_shape, name, dtype=None):
        """
        Create a group of model units with named variables.

        Class-defined unit variables are made available as instance attributes.

        Arguments
        ---------
        N_or_shape : int | tuple of ints
            The size of the group (or shape if multiple dimensions needed)

        name : str
            Unique name given to the group

        dtype : '?' | 'u' | 'i' | 'f' | 'd', optional (default 'f')
            Default numpy dtype to use for initializing array variables
        """
        super(TenkoObject, self).__init__(name=name)
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

        self.variables = list(set(self.base_variables + self.extra_variables))
        for varname in self.variables:
            thisdtype = self.vardtypes.get(varname, self.dtype)
            object.__setattr__(self, varname, zeros(self.shape, thisdtype))

    def set(self, **values):
        """
        Set variables using keyword arguments.
        """
        for name, value in values.items():
            setattr(self, name, value)

    def _evaluate(self, value):
        if hasattr(value, 'sample'):
            return value.sample(self.shape)
        return value

    def __setattr__(self, name, value):
        if name in self.variables:
            getattr(self, name)[:] = self._evaluate(value)
            return
        super().__setattr__(name, value)
