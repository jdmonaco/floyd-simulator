"""
Classes for computing activity measurements.
"""

from toolbox.numpy import *

from .state import State


class FiringRateWindow(object):

    """
    Rolling-window firing rates and activity fraction for neuron groups.
    """

    def __init__(self, neuron_group):
        self.group = neuron_group
        self.N = self.group.N
        self.window = State.calcwin / 1e3  # convert from ms to s
        self.q_max = int(State.calcwin / State.dt)
        self.spikes = np.zeros((self.N, self.q_max), 'u1')
        self.R = np.zeros(self.N)
        self.active = np.zeros(self.N, '?')
        self.j = -1

    def update(self):
        """
        Update firing-rate traces.
        """
        # Update the circular cursor index
        self.j += 1
        self.j %= self.q_max

        # Update the current spike vector and windowed rate & active vectors
        self.spikes[:,self.j] = self.group.spikes
        self.R[:] = self.spikes.sum(axis=1) / self.window
        self.active[:] = any_(self.spikes, axis=1)

    def get_rates(self):
        """
        Return the firing rates for each unit in the current window.
        """
        return self.R[:]

    def get_mean_rate(self):
        """
        Return the population average firing rate.
        """
        return self.R.mean()

    def get_active_mean_rate(self):
        """
        Return the average firing rate for cells active in the current window.
        """
        if any_(self.active):
            return self.R[self.active].mean()
        return 0.0

    def get_mean_spikes(self):
        """
        Return the average spike count in the current window.
        """
        return self.spikes.sum(axis=1).mean()

    def get_active_mean_spikes(self):
        """
        Return the average spike count for cells active in the current window.
        """
        if any_(self.active):
            return self.spikes[self.active].sum(axis=1).mean()
        return 0.0

    def get_activity(self):
        """
        Return the fraction of units active in the current window.
        """
        return self.active.sum() / self.N
