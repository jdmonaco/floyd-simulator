"""
Input/output relationship for neurons.
"""

from toolbox.numpy import *
from floyd import *

from roto.plots import shaded_error

from .neurons import create_LIF_int_spec, create_LIF_pyr_spec
from .neurons import create_AdEx_int_spec, create_AdEx_pyr_spec

from . import SharpwavesContext, step


class NeuronEvaluation(SharpwavesContext):

    """
    Evaluate the f-I curves for Grace and Kechen.
    """

    def setup_model(self, pfile=None, **params):
        self.set_simulation_parameters(params,
            title    = f"Analysis of {params['groupname']} Neuron Group",
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
            debug    = True,   # run in debug mode
        )
        self.set_model_parameters(params, pfile=pfile,
            N_pulses    = 33,     # number of test pulses
            max_current = 1000.0, # pA, peak pulse stimulation
            groupspec   = None,   # NeuronSpec object
            groupname   = None,   # neuron group name
            neurontype  = None,   # neuron group type string
            CA_spacing  = 0.031,  # mm, 0.1 cf. Taxidis
            CA_radius   = 0.220,  # mm, CA3/1 disc radius
        )

        # Set up the hexagonal CA layout for the neurons
        groupspec.update(layoutspec=HexLayoutSpec(
            scale       = CA_spacing,
            radius      = CA_radius,
            origin      = (CA_radius,)*2,
            extent      = (0, 2*CA_radius, 0, 2*CA_radius),
            orientation = pi/6,
        ))

        # Create the neuron group based on the spec and name
        if neurontype == 'LIF':
            NeuronGroup = LIFNeuronGroup
        elif neurontype == 'AdEx':
            NeuronGroup = AdExNeuronGroup
        else:
            raise ValueError(f'unknown neuron type: {neurontype!r}')
        group = NeuronGroup(groupname, groupspec)
        group.set(
                v=groupspec.E_L,
                excitability=PositiveGaussianSampler(1.0, 0.1),
        )

        # Create the step-pulse DC-current stimulator
        step_series = step_pulse_series(N_pulses, duration, max_current)
        stim = InputStimulator(group.p, 'I_DC_mean', *step_series,
                               state_key='I_app', repeat=True)
        self.save_array(step_series, 'stimulus_series')

        # Create the model recorder
        recorder = ModelRecorder(show_progress=False, spikes=group.spikes,
                                 v=group.v, net=group.I_net, I_app=0.0)

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
        ax.plot(spikes.t, self.p.groupspec.E_L * (
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
        fig = self.figure('iocurves', clear=True, figsize=(6,5),
                title=r'Rate vs. $I_{\rm app}$'f' for {self.p.groupname}')
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
        self.savefig(tag=self.p.groupname, tight_padding=0.2)
