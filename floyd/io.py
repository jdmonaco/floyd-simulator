"""
Input/output relationship for neurons.
"""

from .neurons import LIFNeuronSpec, LIFNeuronGroup, AdExNeuronSpec, \
                     AdExNeuronGroup
from .context import SimulatorContext
from .input import InputStimulator, Timepoint
from .recorder import ModelRecorder


class NeuronEvaluation(SimulatorContext):

    """
    Evaluate the stupid f-I curves for Grace and Kechen.
    """

    def create_LIF_neuron_group(self, **params):
        LIFNeuronSpec = paramspec('LIFNeuronSpec',
                    C_m        = Param(200, 100, 300, 5, 'pF'),
                    g_L        = 25.0,
                    E_L        = -58.0,
                    E_exc      = 0.0,
                    E_inh      = -70.0,
                    I_tonic    = Param(0, 0, 1e3, 1e1, 'pA'),
                    V_r        = Param(-55, -85, -40, 1, 'mV'),
                    V_thr      = Param(-48, -50, 20, 1, 'mV'),
                    tau_ref    = Param(1, 0, 10, 1, 'ms'),
                    tau_eta    = 10.0,
                    sigma      = Param(0, 0, 1e3, 1e1, 'pA'),
                    layoutspec = HexLayoutSpec,
        )
        spec = LIFNeuronSpec(**params)
        group = LIFNeuronGroup('LIF', LIFNeuronSpec)
        return group

    def create_AdEx_neuron_group(self, **params):
        spec = paramspec('AdExNeuronSpec', parent=LIFNeuronSpec,
                    delta = Param(1, 0.25, 6, 0.05, 'mV'),
                    V_t   = Param(-50, -65, -20, 1, 'mV'),
                    V_thr = Param(20, -50, 20, 1, 'mV'),
                    a     = Param(2, 0, 10, 0.1, 'nS'),
                    b     = Param(100, 0, 200, 5, 'pA'),
                    tau_w = Param(30, 1, 300, 1, 'ms'),
        )
        group = AdExNeuronGroup('AdEx', AdExNeuronSpec, **params)
        return group

    def setup_model(self, group, neuron_type='LIF', **params):
        self.set_simulation_parameters(params,
            title    = 'f-I Curves for Neuron Group',
            rnd_seed = None, # str, RNG seed (None: class name)
            duration = 100.0, # ms, total simulation length
            dt       = 0.1,       # ms, single-update timestep
            dt_rec   = 0.1,   # ms, recording sample timestep
            dt_block = 25.0,  # ms, dashboard update timestep
            figw     = 9.0,     # inches, main figure width
            figh     = 9.0,     # inches, main figure height
            tracewin = 100.0, # ms, trace plot window
            calcwin  = 50.0,  # ms, rolling calculation window
            interact = False, # run in interactive mode
            debug    = True,    # run in debug mode
        )
        self.set_model_parameters(params, pfile=pfile,
            dt_dwell = 100.0, # ms, input dwell time
            C_pyr      = 275.0, # pF, membrane capacitance, *
            C_int      = 100.0, # pF, membrane capacitance, *
            g_L_pyr    = 25.0,  # nS, leak conductance, *
            g_L_int    = 10.0,  # nS, leak conductance
            E_L_pyr    = -67.0, # nS, leak reversal, *
            E_L_int    = -65.0, # nS, leak reversal, *
            E_inh_pyr  = -68.0, # mV, inhibitory reversal for pyr.
            E_inh_int  = -75.0, # mV, inhibitory reversal for int.
            E_exc      = 0.0,   # mV, excitatory reversal
            a_pyr      = 2.0,   # nS, subthreshold adaptation gain
            a_int      = 2.0,   # nS, subthreshold adaptation gain
            b_pyr      = 100.0, # pA, spike-reset adaptation; 100 cf. Malerba
            b_int      = 10.0,  # pA, spike-reset adaptation; 10 cf. Malerba
            delta_pyr  = 2.0,   # mV, slope factor; 2 cf. Malerba
            delta_int  = 2.0,   # mV, slope factor; 2 cf. Malerba
            tau_w_pyr  = 150.0, # ms, adaptation time-constant; 120 cf. Malerba
            tau_w_CA3i = 30.0,  # ms, adaptation time-constant; 30 cf. Malerba
            tau_w_CA1i = 5.0,   # ms, adaptation time-constant; JDM
            V_t_pyr    = -50.0, # mV, spike-initiation voltage
            V_t_int    = -52.0, # mV, spike-initiation voltage, *
            V_r_pyr    = -60.0, # mV, spike-reset voltage, *
            V_r_int    = -67.0, # mV, spike-reset voltage, *
            V_thr      = 20.0,  # mV, spike threshold
            tau_eta    = 10.0,  # ms, noise time-constant; 100 Hz cf. Malerba
        )

        if neuron_type == 'LIF':
            group = self.create_LIF_neuron_group()
        elif neuron_type == 'AdEx':
            group = self.create_AdEx_neuron_group()
        else:
            raise ValueError('unknown neuron type: {neuron_type!r}')

        # Construct the timing and magnitudes of input current pulses
        N_pulses = 32
        max_current = 750.0 # pA
        t_pulse = linspace(0, N_pulses*dt_dwell, N_pulses)
        current_pulses = linspace(0, max_current, N_pulses)

        # Create the stimulator
        timing = []
        for t, val in zip(t_pulse, current_pulses):
            timing.append(Timepoint(t, val))
        timing = tuple(timing)
        stim = InputStimulator(group, 'I_tonic', *timing)

        # Create the model recorder
        recorder = ModelRecorder(spikes=group.spikes, v=group.v,
                                 net=group.I_net)
