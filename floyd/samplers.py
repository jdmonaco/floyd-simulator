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

    def sample(self, N_or_shape):
        if isscalar(N_or_shape):
            shape = (N_or_shape,)
        else:
            shape = tuple(map(int, N_or_shape))
        return self._sample(shape)

    def _sample(self, shape):
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

    def sample(self, N_or_shape):
        x = SamplerFunc.sample(self, N_or_shape)
        x = clip(x, self.s_min, self.s_max, out=x)
        return x


class PositiveSamples(ClippedSamples):

    """
    Mixin to provide non-negative random samples (by clipping).
    """

    def __init__(self, loc=0.0, scale=1.0):
        ClippedSamples.__init__(self, 0, inf, loc=loc, scale=scale)


class RandomSampler(SamplerFunc):

    def _sample(self, shape):
        """
        Sample values from the specified uniform distribution.
        """
        return self.loc + self.scale * random_sample(shape)


class GaussianSampler(SamplerFunc):

    def _sample(self, shape):
        """
        Sample values from the specified Gaussian distribution.
        """
        return self.loc + self.scale * random.standard_normal(shape)

class ClippedGaussianSampler(ClippedSamples, GaussianSampler):
    pass

class PositiveGaussianSampler(PositiveSamples, GaussianSampler):
    pass


class LognormalSampler(SamplerFunc):

    def __init__(self, sigma, loc=0, scale=1):
        SamplerFunc.__init__(self, loc, scale)
        self.sigma = sigma

    def _sample(self, shape):
        """
        Sample values from the specified Gaussian distribution.
        """
        return st.lognorm.rvs(self.sigma, loc=self.loc, scale=self.scale,
                    size=shape)

class ClippedLognormalSampler(ClippedSamples, LognormalSampler):
    pass
