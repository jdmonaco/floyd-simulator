"""
Delay lines for signal propagation.
"""

__all__ = ['DelayLines', ]


from numpy import broadcast_to, zeros, arange, array

from pouty import log


DELAY_TYPE = 'u4'
SIGNAL_TYPE = '?'


class DelayLines(object):

    def __init__(self, N, delays, dtype=SIGNAL_TYPE):
        """
        Create delayed signal lines based on unit number and delays.
        """
        self.N = N
        try:
            self.delays = broadcast_to(delays, N).astype(DELAY_TYPE)
        except ValueError as e:
            log('delays ({}) must broadcast to line number ({})',
                    np.array(delays).shape, self.N, prefix='DelayLines',
                    error=True)
            raise e

        self.delays[self.delays == 0] = 1  # zero-delays are 1-delay updates
        self.cursor = (zeros(N, 'i8'), arange(N))
        self.lines = zeros((self.delays.max(), N), dtype)

    @classmethod
    def from_delay_times(cls, N, delay_times, dt, dtype=SIGNAL_TYPE):
        """
        Return a DelayLines object based on timing delays across timesteps.
        """
        timing = array(delay_times)
        delays = array(timing / dt).astype(DELAY_TYPE)
        return cls(N, delays, dtype=dtype)

    def set(self, x):
        """
        Set the current value of the signal.
        """
        self.lines[self.cursor] = x
        self.cursor[0][:] += 1
        self.cursor[0][:] %= self.delays

    def get(self):
        """
        Retrieve the delayed value of the signal.
        """
        return self.lines[self.cursor]
