"""
Kernel functions.
"""

__all__ = ('KernelFunc', 'GaussianKernel', 'GaussianDistanceKernel')


from toolbox.numpy import *


class KernelFunc(object):

    def __init__(self, X0=0.0, sigma=1.0):
        """
        Initialize with (X0, sigma) or just (sigma,) with X0 == 0.0 by default.
        """
        self.X0 = X0
        self.sigma = sigma

    def __call__(self, X):
        return self.apply(X)

    def apply(self, X):
        raise NotImplementedError


class GaussianKernel(KernelFunc):

    """
    A 1-dimensional Gaussian density kernel.
    """

    def apply(self, X):
        """
        Compute Gaussian kernel values for the given values.
        """
        return 1/sqrt(2*self.sigma**2)*exp(-(X-self.X0)**2/(2*self.sigma**2))


class GaussianDistanceKernel(KernelFunc):

    """
    A 2-dimensional distance-indexed Gaussian density kernel.
    """

    def apply(self, X):
        """
        Compute Gaussian kernel values for the distances to given points.
        """
        D = hypot(X[:,0] - self.X0[0], X[:,1] - self.X0[1])
        return 1/sqrt(2*self.sigma**2)*exp(-D**2/(2*self.sigma**2))
