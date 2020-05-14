"""
Kernel functions.
"""

__all__ = ('KernelFunc', 'GaussianKernel', 'PointDistanceKernel', 
           'RelativeDistanceKernel')


from toolbox.numpy import *

from .matrix import pairwise_distances


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


class PointDistanceKernel(KernelFunc):

    """
    A Gaussian density kernel for 2D points indexed by distance from a single
    reference point.
    """

    def apply(self, X):
        """
        Compute Gaussian kernel values for the distances to given points.
        """
        D = hypot(X[:,0] - self.X0[0], X[:,1] - self.X0[1])
        return 1/sqrt(2*self.sigma**2)*exp(-D**2/(2*self.sigma**2))


class RelativeDistanceKernel(KernelFunc):

    """
    A Gaussian density kernel for 2D points indexed by relative pair-wise 
    distances between the units in two connected groups. 
    """

    def apply(self, pre, post=None):
        """
        Calculate a probability kernel for connecting units from two groups.
        
        Note: Unit positions should be stored as 'x' and 'y' group variables.
        Constructor argument 'X0' provides a distance offset if nonzero.
        """
        if post is None:
            assert hasattr(pre, 'pre') and hasattr(pre, 'post'), \
                'kernel requires two groups or a projection ({pre})'
            pre, post = pre.pre, pre.post

        D = pairwise_distances(c_[post.x, post.y], c_[pre.x, pre.y])
        return 1/sqrt(2*self.sigma**2)*exp(-(D-self.X0)**2/(2*self.sigma**2))