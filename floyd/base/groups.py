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
    default_dtype = 'd'

    def __init__(self, *, name, shape, dtype=None, **kwargs):
        """
        Create a group of model units with named variables.

        Class-defined unit variables are made available as instance attributes
        and initialized to zero-valued arrays.

        Arguments
        ---------
        shape : int | tuple of ints
            The shape or size (1D) of the group

        dtype : dict | '?' | 'u' | 'i' | 'f' | 'd', optional (default 'f')
            Default numpy dtype to use for initializing array variables
        """
        super().__init__(name=name, **kwargs)

        if np.iterable(shape):
            self.shape = tuple(shape)
            self.size = reduce(op.mul, self.shape)
        else:
            self.size = int(shape)
            self.shape = (self.size,)

        self._vardtypes = dict(self.base_dtypes)
        dflt_dtype = self.default_dtype
        if type(dtype) is dict:
            self._vardtypes = merge_two_dicts(self._vardtypes, dtype)
        elif type(dtype) is str and dtype[0] in '?uifd':
            dflt_dtype = dtype
        
        # TODO: Traverse MRO to aggregate extra_variables so subclasses 
        # don't need to repeat the variables of parent classes

        allvars = set(self.base_variables + self.extra_variables)
        self._variables = tuple(sorted(allvars))
        for varname in self._variables:
            thisdtype = self._vardtypes.get(varname, dflt_dtype)
            self.__dict__[varname] = zeros(self.shape, thisdtype)

    def set(self, **values):
        """
        Set variables using keyword arguments.
        """
        for name, value in values.items():
            setattr(self, name, value)

    def __setattr__(self, name, value):
        """
        For named group variable, use in-place setting of array values.
        """
        if hasattr(self, '_variables') and name in self._variables:
            getattr(self, name)[:] = self._evaluate(value)
            return
        super().__setattr__(name, value)

    def _evaluate(self, value):
        if hasattr(value, 'sample'):
            return value.sample(self.shape, state=self.rnd)
        return value
