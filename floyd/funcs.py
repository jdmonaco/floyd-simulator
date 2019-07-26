"""
Placeholder types for different kernels and functions.
"""

import scipy.stats as st

from toolbox.numpy import *


class FunctionType(object):
    pass


class SamplerType(FunctionType):

    def __init__(self, loc_or_scale, *scale, clip=None):
        """
        Initialize with (loc, scale) or just (scale,) with loc==0.0 by default.
        """
        if scale:
            self.loc = loc_or_scale
            self.scale = scale[0]
        else:
            self.loc = 0.0
            self.scale = loc_or_scale
        self.clip = clip

    def __call__(self, N):
        return self.sample(N)

    def sample(self, N):
        x = self._sample(N)
        if self.clip is not None:
            clip(x, self.clip[0], self.clip[1], out=x)
        return x

    def _sample(self, N):
        raise NotImplementedError


class KernelType(FunctionType):

    def __init__(self, X0_or_sigma, *sigma):
        """
        Initialize with (X0, sigma) or just (sigma,) with X0 == 0.0 by default.
        """
        if sigma:
            self.X0 = X0_or_sigma
            self.sigma = sigma[0]
        else:
            self.X0 = 0.0
            self.sigma = X0_or_sigma

    def __call__(self, X):
        return self.apply(X)

    def apply(self, X):
        raise NotImplementedError


class RandomSampler(SamplerType):

    def _sample(self, N_or_shape):
        """
        Sample values from the specified uniform distribution.
        """
        return self.loc + self.scale * random_sample(N_or_shape)


class GaussianSampler(SamplerType):

    def _sample(self, N_or_shape):
        """
        Sample values from the specified Gaussian distribution.
        """
        return self.loc + self.scale * random.standard_normal(N_or_shape)


class LognormalSampler(SamplerType):

    def __init__(self, sigma, loc=0, scale=1):
        SamplerType.__init__(self, loc, scale)
        self.sigma = sigma

    def _sample(self, N_or_shape):
        """
        Sample values from the specified Gaussian distribution.
        """
        return st.lognorm.rvs(self.sigma, loc=self.loc, scale=self.scale,
                    size=N_or_shape)


class GaussianKernel(KernelType):

    def apply(self, X):
        """
        Compute Gaussian kernel values for the given values.
        """
        return 1/sqrt(2*self.sigma**2)*exp(-(X-self.X0)**2/(2*self.sigma**2))
