"""
Input/output relationship for neurons.
"""

from functools import partial

from toolbox.numpy import *
from roto.plots import shaded_error
from specify import Param

from . import FloydContext, step
from .. import *


class InputOutputCurves(FloydContext):

    """
    Evaluate the f-I curves for Grace and Kechen.
    """

    # Simulation parameters
    title      = "Input/Output Analysis of Neurons"
    duration   = 5000.0   # ms total simulation length
    dt         = 0.2      # ms single-update timestep
    dt_rec     = 0.2      # ms recording sample timestep
    dt_block   = 25.0     # ms dashboard update timestep
    figw       = 9.0      # inches main figure width
    figh       = 9.0      # inches main figure height
    tracewin   = 200.0    # ms trace plot window
    calcwin    = 50.0     # ms rolling calculation window

    # Model parameters
    N           = Param(default=100, doc='number of neurons')
    N_pulses    = Param(default=33, doc='number of test pulses')
    N_ex_cells  = Param(default=6, doc='number of examples cells for traces')
    max_current = Param(default=1e3, units='pA', doc='peak pulse stimulation')
    modeltype   = Param(default='Brette_AEIF', doc="'LIF'|'AEIF'")
    neurontype  = Param(default='pyr', doc="'int'|'pyr'")
    I_noise     = Param(default=0.0, doc='noisy input current gain')
    g_tonic_exc = Param(default=0.0, doc='tonic excitatory input conductance')
    g_tonic_inh = Param(default=0.0, doc='tonic inhibitory input conductance')

    def setup_model(self):

        # Create the neuron group based on model and neuron types
        params = dict(
                N           = N,
                name        = 'IOGroup',
                I_noise     = I_noise,
                g_tonic_exc = g_tonic_exc,
                g_tonic_inh = g_tonic_inh,
        )
        key = (modeltype, neurontype)
        if key == ('LIF', 'int'):
            group = DonosoLIFInterneurons(**params)
        elif key == ('LIF', 'pyr'):
            group = DonosoLIFPyramids(**params)
        elif key == ('AEIF', 'int'):
            group = MalerbaAEIFInterneurons(**params)
        elif key == ('Malerba_AEIF', 'pyr'):
            group = MalerbaAEIFPyramids(**params)
        elif key == ('Brette_AEIF', 'pyr'):
            group = BretteAEIFPyramids(**params)
        else:
            raise ValueError(f'unknown neuron model: {key!r}')

        # Initialize voltage and hetergeneous excitability
        group.excitability = PositiveGaussianSampler(1.0, 0.1)

        # Create the step-pulse DC-current stimulator
        pulses = step_pulse_series(N_pulses, duration, max_current)
        stim = InputStimulator(group, 'I_DC_mean', *pulses, state_key='I_app',
                               repeat=3)

        if State.run_mode == RunMode.RECORD:
            self.save_array(pulses, 'stimulus_series')

        # Create the model recorder, figure, and tables
        recorder = ModelRecorder(
                spikes = group.spikes,
                v      = group.v,
                net    = group.I_net,
                I_app  = 0.0,
        )

        # Create markdown tables for input and output values
        tablemaker = TableMaker()
        tablemaker.add_markdown_table('output',
                ('Rate (sp/s)', lambda g: g.mean_rate()),
                ('Activity (frac)', lambda g: g.active_fraction()),
                fmt='.4g',
        )

        # Create grid of axes in the figure
        simplot = SimulationPlotter(2, 2)
        simplot.set_axes(
                stim    = simplot.gs[0,0],
                voltage = simplot.gs[0,1],
                rate    = simplot.gs[1,0],
                net     = simplot.gs[1,1],
        )
        simplot.get_axes('voltage').set_ylim(group.E_inh - 10,
                                             group.V_thr + 20)

        # Initialize stimulation current trace plot
        simplot.add_realtime_traces_plot(
                ('stim', r'$I_{\rm app}$', lambda: State.I_app),
                units = 'pA',
                datalim = 50.0,
                fmt = dict(lw=1),
                legend = 'last',
        )

        # Initialize example units trace plots of voltage, rate, and I_net
        ex_units = randint(group.N, size=N_ex_cells)
        ex_fmt = dict(lw=0.5, alpha=0.6)
        volt_fn = lambda i: group.v[i]
        voltage_traces = [
                ('voltage', f'Vm_{i}', partial(volt_fn, i)) for i in ex_units
        ]
        simplot.add_realtime_traces_plot(*voltage_traces, fmt=ex_fmt,
                datalim='none')
        rate_fn = lambda i: group.rates()[i]
        rate_traces = [
                ('rate', f'rate_{i}', partial(rate_fn, i)) for i in ex_units
        ]
        simplot.add_realtime_traces_plot(*rate_traces, fmt=ex_fmt)
        net_fn = lambda i: group.I_net[i]
        net_traces = [
                ('net', f'I_net_{i}', partial(net_fn, i)) for i in ex_units
        ]
        simplot.add_realtime_traces_plot(*net_traces, fmt=ex_fmt)

        # Creat the network graph visualization
        netgraph = NetworkGraph()

        # Register figure initializer to add artists to the figure
        def init_figure():
            simplot.draw_borders()
            netgraph.plot()
            return simplot.get_all_artists()
        simplot.init(init_figure)

    @step
    def plot_firing(self, tag=None):
        """
        Plot a simple figure of spiking and voltage traces.
        """
        fig = self.figure('iocurves', clear=True, figsize=(11,3.7),
                title=f'Stepped Series Stimulation of IOGroup')

        t = self.read_array('t', step='collect_data', tag=tag)
        v = self.read_array('v', step='collect_data', tag=tag)
        iapp = self.read_array('I_app', step='collect_data', tag=tag)
        spikes = self.read_dataframe('spikes', step='collect_data', tag=tag)
        self.out(f'Found {len(spikes)} spikes')

        # Plot the applied current
        ax = fig.add_subplot()
        ax.plot(t, iapp, label=r'$I_{\rm app}$')
        ax.set_ylabel('Applied DC Current, pA')
        ax.set_xlabel('Time, ms')

        # Plot some overlaying voltage traces
        ax = ax.twinx()
        ax.plot(t, v[:,:8], label=r'$V_m$')
        ax.set_ylim(-85, 0)
        ax.set_ylabel('Membrane voltage, mV')

        # Plot a spike raster
        ax.plot(spikes.t, -85 * (
                1 - spikes.unit/spikes.unit.max()), 'k.', ms=2, alpha=0.5,
                label='spikes')
        ax.legend(loc='upper left')

    @step
    def compute_curves(self, tag=None, slopefrac=0.6, stepfrac=0.5):
        """
        Compute f-I curves based on simulation data.
        """
        root = dict(tag=tag, step='collect_data')
        t = self.read_array('t', **root)
        v = self.read_array('v', **root)
        iapp = self.read_array('I_app', **root)
        spikes = self.read_dataframe('spikes', **root)
        steps = self.read_array('stimulus_series', **root)
        self.debug(steps)

        # Get information about the shape of the data
        N_t, N_units = v.shape
        t_stim, I_stim = steps.T  # zero return is the last "step"
        N_steps = len(steps) - 1
        dt_step = median(diff(t_stim))
        dt_rate = stepfrac*dt_step*1e-3

        # Initialize the matrix for the curves
        fI = empty((N_steps, N_units))

        # Iterate across steps and units
        self.out('Computing firing rates...')
        for i in range(N_steps):
            I = I_stim[i]
            t0 = t_stim[i] + (1 - stepfrac)*dt_step
            t1 = t_stim[i+1]
            for j in range(N_units):
                spk = (spikes.t > t0) & (spikes.t <= t1) & (spikes.unit == j)
                N_spk = spk.sum()
                R = N_spk / dt_rate
                fI[i,j] = R
            self.box()
        self.newline()

        # Compute the statistical curves
        fI_mean = fI.mean(axis=1)
        fI_std = fI.std(axis=1)
        fI_lower = fI_mean - fI_std
        fI_upper = fI_mean + fI_std

        # Compute the f-I slope
        dI = median(diff(I_stim))
        I_slope = I_stim.max() * slopefrac
        k = argmin(abs_(I_stim - I_slope))
        fI_slope = (fI_mean[k+1] - fI_mean[k-1]) / (2 * dI)
        restext = f'Mean f-I slope = {1e3*fI_slope:.1f} Hz/nA @ ' \
                  f'I_app = {I_stim[k]/1e3:.2f} nA'
        self.out(restext)

        # Compute the rheobase
        k_min = fI_mean.nonzero()[0][0]
        rheo = I_stim[k_min]
        rheotext = f'Mean f-I rheobase = {rheo/1e3:.2f} nA'
        self.out(rheotext)

        # Plot the raw, mean, and error f-I curves with results annotation
        self.out('Plotting f-I curves...')
        fig = self.figure('iocurves', clear=True, figsize=(6, 5),
                title=r'Rate vs. $I_{\rm app}$ for '
                      f'{self.modeltype}/{self.neurontype}')
        ax = fig.add_subplot()
        ax.plot(I_stim[:-1], fI, 'k-', lw=0.6, alpha=0.15, zorder=-10)
        ax.plot(I_stim[:-1], fI_mean, 'b-', lw=1.2, alpha=0.8, zorder=10,
                label='Mean')
        shaded_error(I_stim[:-1], fI_mean, fI_std, ax=ax, facecolor='m', lw=0,
                alpha=0.6, zorder=0)
        ax.text(0.025, 0.975, restext, transform=ax.transAxes, va='top',
                ha='left', c='0.3', weight='medium')
        ax.text(0.025, 0.925, rheotext, transform=ax.transAxes, va='top',
                ha='left', c='0.3', weight='medium')
        ax.set(xlabel=r'$I_{\rm app}$, pA', ylabel='Mean firing rate, sp/s')

        # Save the figure
        self.savefig(tag=f'{self.modeltype}_{self.neurontype}',
                tight_padding=0.2)
