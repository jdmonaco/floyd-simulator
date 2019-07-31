"""
Input/output relationship for neurons.
"""

from functools import partial

from toolbox.numpy import *
from roto.plots import shaded_error

from . import FloydContext, step
from ..neurons.gallery import *
from .. import *


class InputOutputCurves(FloydContext):

    """
    Evaluate the f-I curves for Grace and Kechen.
    """

    def setup_model(self, pfile=None, **params):
        self.set_simulation_parameters(params,
            title    = f"Analysis of {params['modeltype']}/"
                       f"{params['neurontype']} Neurons",
            rnd_seed = None,   # str, RNG seed (None: class name)
            duration = 5000.0, # ms, total simulation length
            dt       = 0.1,    # ms, single-update timestep
            dt_rec   = 0.1,    # ms, recording sample timestep
            dt_block = 25.0,   # ms, dashboard update timestep
            figw     = 9.0,    # inches, main figure width
            figh     = 9.0,    # inches, main figure height
            tracewin = 100.0,  # ms, trace plot window
            calcwin  = 50.0,   # ms, rolling calculation window
            interact = False,  # run in interactive mode
            debug    = False,  # run in debug mode
        )
        self.set_model_parameters(params, pfile=pfile,
            N_pulses    = 33,     # number of test pulses
            N_ex_cells  = 5,      # number of examples cells for traces
            max_current = 1000.0, # pA, peak pulse stimulation
            modeltype   = 'LIF',  # 'LIF', 'AdEx'
            neurontype  = 'int',  # 'int', 'pyr'
            CA_spacing  = 0.031,  # mm, 0.1 cf. Taxidis
            CA_radius   = 0.220,  # mm, CA3/1 disc radius
            sigma       = 0.0,
            g_tonic_exc = 0.0,
            g_tonic_inh = 0.0,
        )

        # Create the hexagonal disc layout spec for the group
        layout = HexagonalDiscLayout.get_spec(
            scale       = CA_spacing,
            radius      = CA_radius,
            origin      = (CA_radius,)*2,
            extent      = (0, 2*CA_radius, 0, 2*CA_radius),
            orientation = pi/6,
        )

        # Create the neuron group based on model and neuron types
        params = dict(
                sigma       = sigma,
                g_tonic_exc = g_tonic_exc,
                g_tonic_inh = g_tonic_inh,
                layout      = layout
        )
        key = (modeltype, neurontype)
        if key == ('LIF', 'int'):
            group = create_LIF_interneurons(**params)
        elif key == ('LIF', 'pyr'):
            group = create_LIF_pyramids(**params)
        elif key == ('AdEx', 'int'):
            group = create_AdEx_interneurons(**params)
        elif key == ('Malerba_AdEx', 'pyr'):
            group = create_Malerba_AdEx_pyramids(**params)
        elif key == ('Brette_AdEx', 'pyr'):
            group = create_Brette_AdEx_pyramids(**params)
        else:
            raise ValueError(f'unknown neuron model: {key!r}')

        # Initialize voltage and hetergeneous excitability
        group.set_pulse_metrics()
        group.set(
                v = group.p.E_L,
                excitability = PositiveGaussianSampler(1.0, 0.1),
        )

        # Create the step-pulse DC-current stimulator
        pulses = step_pulse_series(N_pulses, duration, max_current)
        stim = InputStimulator(group, 'I_DC_mean', *pulses,
                               state_key='I_app', repeat=True)
        if self.current_step() == 'collect_data':
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
        simplot.get_axes('voltage').set_ylim(group.p.E_inh - 10,
                                             group.p.V_thr + 20)

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
            self.debug('init_figure called')
            simplot.draw_borders()
            netgraph.plot()
            return simplot.get_all_artists()
        simplot.init(init_figure)

    @step
    def plot_firing(self):
        """
        Plot a simple figure of spiking and voltage traces.
        """
        fig = self.figure('iocurves', clear=True, figsize=(11,3.7),
                title=f'Stepped Series Stimulation of {self.p.groupname}')

        t = self.read_array('t', step='collect_data')
        v = self.read_array('v', step='collect_data')
        iapp = self.read_array('I_app', step='collect_data')
        spikes = self.read_dataframe('spikes', step='collect_data')
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
        self.load_parameters(step='collect_data', tag=tag)
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
                      f'{self.p.modeltype}/{self.p.neurontype}')
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
        self.savefig(tag=f'{self.p.modeltype}_{self.p.neurontype}',
                tight_padding=0.2)
