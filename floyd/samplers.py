"""
Probabilistic sampling functions.
"""

__all__ = ['SamplerFunc', 'RandomSampler', 'GaussianSampler',
           'LognormalSampler', 'ClippedGaussianSampler',
           'ClippedLognormalSampler', 'PositiveGaussianSampler']


import scipy.stats as st

from toolbox.numpy import *


class SamplerFunc(object):

    """
    Base sampler class.
    """

    def __init__(self, loc=0.0, scale=1.0):
        """
        Initialize with (loc, scale) or just (scale,) with loc==0.0 by default.
        """
        self.loc = loc
        self.scale = scale

    def __repr__(self):
        s = '{.__class__.__name__}(loc={.loc:.3g}, scale={.scale:.3g})'
        return s.format(*(self,)*3)

    def __call__(self, N_or_shape):
        return self.sample(N_or_shape)

    def sample(self, N_or_shape, state=None):
        if isscalar(N_or_shape):
            shape = (N_or_shape,)
        else:
            shape = tuple(map(int, N_or_shape))
        state = random if state is None else state
        return self._sample(shape, state)

    def _sample(self, shape, rnd):
        raise NotImplementedError


class ClippedSamples(object):

    """
    Mixin to provide clipped (truncated) random samples.
    """

    def __init__(self, s_min, s_max, loc=0.0, scale=1.0):
        """
        A sampler with output clipped to the range `(s_min, s_max)`.
        """
        SamplerFunc.__init__(self, loc=loc, scale=scale)
        self.s_min = s_min
        self.s_max = s_max

    def sample(self, N_or_shape, state=None):
        x = SamplerFunc.sample(self, N_or_shape, state=state)
        x = clip(x, self.s_min, self.s_max, out=x)
        return x


class PositiveSamples(ClippedSamples):

    """
    Mixin to provide non-negative random samples (by clipping).
    """

    def __init__(self, loc=0.0, scale=1.0):
        ClippedSamples.__init__(self, 0, inf, loc=loc, scale=scale)


class RandomSampler(SamplerFunc):

    def _sample(self, shape, rnd):
        """
        Sample values from the specified uniform distribution.
        """
        return self.loc + self.scale * rnd.random_sample(shape)


class GaussianSampler(SamplerFunc):

    def _sample(self, shape, rnd):
        """
        Sample values from the specified Gaussian distribution.
        """
        return self.loc + self.scale * rnd.standard_normal(shape)

class ClippedGaussianSampler(ClippedSamples, GaussianSampler):
    pass

class PositiveGaussianSampler(PositiveSamples, GaussianSampler):
    pass


class LognormalSampler(SamplerFunc):

    def __init__(self, sigma, loc=0, scale=1):
        SamplerFunc.__init__(self, loc, scale)
        self.sigma = sigma

    def _sample(self, shape, rnd):
        """
        Sample values from the specified Gaussian distribution.
        """
        return st.lognorm.rvs(self.sigma, loc=self.loc, scale=self.scale,
                    size=shape, random_state=rnd)

class ClippedLognormalSampler(ClippedSamples, LognormalSampler):
    pass
