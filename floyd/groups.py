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

    def __init__(self, *, N, dtype=None, seed=None, **kwargs):
        """
        Create a group of model units with named variables.

        Class-defined unit variables are made available as instance attributes.

        Arguments
        ---------
        N : int | tuple of ints
            The size of the group (or shape if multiple dimensions needed)

        name : str
            Unique name given to the group

        dtype : dict | '?' | 'u' | 'i' | 'f' | 'd', optional (default 'f')
            Default numpy dtype to use for initializing array variables

        seed : str
            A seed-key string to be converted into a random seed for an
            instance-specific RandomState object at the `rnd` attribute
        """
        super().__init__(**kwargs)

        self.shape = N
        if np.iterable(self.shape):
            self.N = reduce(op.mul, self.shape)
        else:
            self.N = self.shape

        self.rnd = None
        self.set_random_seed(seed)

        self._vardtypes = dict(self.base_dtypes)
        dflt_dtype = self.default_dtype
        if type(dtype) is dict:
            self._vardtypes = merge_two_dicts(self._vardtypes, dtype)
        elif type(dtype) is str and dtype[0] in '?uifd':
            dflt_dtype = dtype

        self._variables = list(set(self.base_variables + self.extra_variables))
        for varname in self._variables:
            thisdtype = self._vardtypes.get(varname, dflt_dtype)
            self.__dict__[varname] = zeros(self.shape, thisdtype)

    def set_random_seed(self, seed):
        """
        Set random seed on `rnd` RandomState instance attribute from string.
        """
        seed = self.name if seed is None else seed
        self._rand_seed = sum(list(map(ord, seed)))
        self.rnd = np.random.RandomState(seed=self._rand_seed)
        self.out(f'{self._rand_seed} [key: \'{seed}\']')

    def set(self, **values):
        """
        Set variables using keyword arguments.
        """
        for name, value in values.items():
            setattr(self, name, value)

    def _evaluate(self, value):
        if hasattr(value, 'sample'):
            __import__('pdb').set_trace()
            return value.sample(self.shape, state=self.rnd)
        return value

    def __setattr__(self, name, value):
        if hasattr(self, '_variables') and name in self._variables:
            getattr(self, name)[:] = self._evaluate(value)
            return
        super().__setattr__(name, value)
