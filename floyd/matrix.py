"""
Matrix operations for sharp-wave attractor models.
"""

from toolbox.numpy import empty, hypot, AX


DEBUGGING = False


def _check_ndim(Mstr, M, ndim):
    assert M.ndim == ndim, f'{Mstr}.ndim != {ndim}'

def _check_shape(Mstr, M, shape, axis=None):
    if axis is None:
        assert M.shape == shape, f'{Mstr}.shape != {shape}'
    else:
        assert M.shape[axis] == shape, f'{Mstr}.shape[{axis}] != {shape}'

def pairwise_distances(A, B):
    """
    Compute distances between every pair of two sets of points.
    """
    N_A = len(A)
    N_B = len(B)
    if DEBUGGING:
        _check_ndim('A', A, 2)
        _check_ndim('B', B, 2)
        _check_shape('A', A, 2, axis=1)
        _check_shape('B', B, 2, axis=1)

    # Broadcast the first position matrix
    AA = empty((N_A,N_B,2), A.dtype)
    AA[:] = A[:,AX,:]

    DD = AA - B[AX,...]
    return hypot(DD[...,0], DD[...,1])
