"""
Base simulator context.
"""

try:
    import panel as pn
except ImportError:
    print('Warning: install `panel` to use interactive dashboards.')

import time

from panel.widgets import Toggle
from panel.pane import Matplotlib, Markdown
import panel as pn

from tenko.context import AbstractBaseContext, step
from tenko.mixins import RandomMixin
from roto.dicts import merge_two_dicts
from maps.geometry import EnvironmentGeometry
from pouty import debug_mode

from .spec import paramspec
from .network import Network
from .state import State, reset_state
from .config import Config


class SimulatorContext(RandomMixin, AbstractBaseContext):

    """
    Context base class for simulations.
    """

    def set_simulation_parameters(self, modparams=None, **simparams):
        """
        Set simulation parameters in global scope and shared state.
        """
        # Set the configured defaults to a paramspec attribute
        self.psim = paramspec('SimSpec', instance=True,
                title     = Config.title,
                rnd_seed  = Config.rnd_seed,
                duration  = Config.duration,
                dt        = Config.dt,
                dt_rec    = Config.dt_rec,
                dt_block  = Config.dt_block,
                blocksize = int(Config.dt_block / Config.dt),
                figw      = Config.figw,
                figh      = Config.figh,
                tracewin  = Config.tracewin,
                calcwin   = Config.calcwin,
                interact  = Config.interact,
                debug     = Config.debug,
        )

        # Add derived values (e.g., blocksize) to the simulation parameters
        simparams.update(blocksize=int(simparams.get('dt_block',
                Config.dt_block) / simparams.get('dt', Config.dt)))

        # Update from model parameters and log differences with defaults
        self.out(f'Simulation parameters:')
        self.psim.update(**simparams)
        for name, value in self.psim:
            if modparams is not None and name in modparams:
                self.psim[name] = modparams.pop(name)
                logmsg = f'* {name} = {self.psim[name]!r} [default: {value!r}]'
            else:
                logmsg = f'- {name} = {value!r}'.format(name, value)
            self.out(logmsg, hideprefix=True)

        # Update global scope and shared state, then write out a JSON file
        debug_mode(self.psim.debug)
        self.get_global_scope().update(self.psim)
        reset_state()
        State.update(self.psim)
        State.context = self
        self.write_json(self.psim, 'simulation')

        # Initialize the 'magic' network object
        State.network = Network()

        # Display a green AnyBar dot to signal the simulation is running
        self.launch_anybar()
        self.set_anybar_color('green')
        self.debug(f'State = {State!r}')

    def set_model_parameters(self, params=None, pfile=None, **defaults):
        """
        Set model parameters according to file, keywords, or defaults.

        Note: It is required to specify default values for all parameters here.
        """
        # Set the defaults to a paramspec attribute and write to file
        self.p = paramspec('ModelSpec', instance=True, **defaults)
        self.write_json(self.p.as_dict(), 'defaults', base='context')

        # Import values from parameters file if specified
        if params is None:
            params = {}
        if pfile is not None:
            fpath, fparams = self.get_json(pfile)
            params = merge_two_dicts(fparams, params)
            self.out(fpath, prefix='ParameterFile')

        # Write file with effective parameters (i.e., kwargs > file > defaults)
        self.p.update(**params)
        self.write_json(self.p.as_dict(), self.filename('params'))

        # Set parameters as global variables in the object's module scope
        self.out('Model parameters:')
        self.get_global_scope().update(self.p)
        for name, value in self.p:
            dflt = self.p.defaults[name]
            if value != dflt:
                logstr = f'* {name} = {value!r} [default: {dflt!r}]'
            else:
                logstr = f'- {name} = {value!r}'
            self.out(logstr, hideprefix=True)

    def load_environment_parameters(self, env):
        """
        Import environment geometry into the module and object scope.
        """
        from numpy import ndarray
        modscope = self.get_global_scope()
        modscope['Env'] = self.e = E = EnvironmentGeometry(env)
        Ivars = list(sorted(E.info.keys()))
        Avars = list(sorted(filter(lambda k: isinstance(getattr(E, k),
            ndarray), E.__dict__.keys())))
        modscope.update(E.info)
        modscope.update({k:getattr(E, k) for k in Avars})
        self.out(repr(E), prefix='Geometry')
        for v in Ivars:
            self.out(f'- {v} = {getattr(E, v)}', prefix='Geometry',
                hideprefix=True)
        for k in Avars:
            self.out('- {} ({})', k, 'x'.join(list(map(str, getattr(E,
                k).shape))), prefix='Geometry', hideprefix=True)

    def setup_model(self, pfile=None, **params):
        """
        Model construction: This must be overridden by subclasses.
        """
        self.set_simulation_parameters(params,
                title    = Config.title,    # str, full simulation title
                rnd_seed = Config.rnd_seed, # str, RNG seed (None: class name)
                duration = Config.duration, # ms, total simulation length
                dt       = Config.dt,       # ms, single-update timestep
                dt_rec   = Config.dt_rec,   # ms, recording sample timestep
                dt_block = Config.dt_block, # ms, dashboard update timestep
                figw     = Config.figw,     # inches, main figure width
                figh     = Config.figh,     # inches, main figure height
                tracewin = Config.tracewin, # ms, trace plot window
                calcwin  = Config.calcwin,  # ms, rolling calculation window
                interact = Config.interact, # run in interactive mode
                debug    = Config.debug,    # run in debug mode
        )
        self.set_model_parameters(params, pfile=pfile,
                param1 = 1.0, # this is a model parameter default
                param2 = 1.0, # this is another model parameter default
        )
        self.set_default_random_seed(rnd_seed)
        raise NotImplementedError('models must implement setup_model()')

    @step
    def create_movie(self, tag=None, pfile=None, **params):
        """
        Simulate the model in batch mode for movie generation.
        """
        # Run the simulation in the non-interactive animation mode
        params.update(interact=False)
        self.model_setup(pfile=pfile, **params)

        anim = FuncAnimation(
                fig       = State.simplot.fig,
                func      = State.network.animation_update,
                init_func = State.simplot.initializer,
                frames    = range(State.N_t),
                interval  = int(1000/Config.fps),
                repeat    = False,
                blit      = True,
        )

        # Save the animation as a movie file
        self['movie_file'] = self.filename(tag=tag, ext='mp4')
        anim.save(self.path(self.c.movie_file), fps=Config.fps, dpi=Config.dpi)
        self.closefig()
        self.hline()

        # Save simulation data traces and play the movie!
        State.recorder.save()
        self.hline()
        self.play_movie()

    @step
    def collect_data(self, tag=None, pfile=None, **params):
        """
        Simulate the model in batch mode for data collection.
        """
        # Run the simulation in the non-interactive data collection mode
        params.update(interact=False)
        self.model_setup(pfile=pfile, **params)

        # Run the main loop until exhaustion
        while State.recorder:
            State.network.model_update()

        # Save simulation data traces
        State.recorder.save()

    def launch_dashboard(self, return_panel=False, threaded=True,
        dpi=Config.dpi, pfile=None, **params):
        """
        Construct an interactive Panel dashboard for running the model.
        """
        # Run the simulation in the interactive dashboard mode
        params.update(interact=True)
        self.setup_model(pfile=pfile, **params)
        State.is_playing = False
        State.block = 0

        # Play-toggle widget for simulation control with block count
        main_figure = Matplotlib(object=State.simplot.fig, dpi=dpi)
        # play_toggle = Toggle(name='Play', value=False, button_type='primary',
                # sizing_mode='stretch_width')
        tictoc = Markdown('Block -- [-- ms]')

        # TODO: Figure out how to improve this mechanism for continuous
        # simulation with play/pause control
        player = pn.widgets.DiscretePlayer(value='1', options=['1', '2', '3'],
                interval=5000, name='Simulation Control', loop_policy='loop')

        # Markdown displays for each registered table output
        table_txt = {}
        for name, table in State.tablemaker:
            table_txt[name] = Markdown(table)

        def simulation(*events):
            # for event in events:
                # self.debug(event)

            # if play_toggle.value == False:
                # play_toggle.set_param(name='Play', button_type='primary')
                # return
            # play_toggle.set_param(name='Pause', button_type='warning')

            # Run a blocked update of the network with performance timing
            t0 = time.perf_counter()
            State.network.dashboard_update()
            dt = time.perf_counter() - t0

            # Update the tic-toc string and block count
            tictoc_str = 'Block {} [{} ms]'.format(State.block, int(1e3*dt))
            tictoc.object = tictoc_str
            State.block += 1
            self.debug(tictoc_str.lower())

            # Update the Markdown table objects with new data
            for label in table_txt.keys():
                table_txt[label].object = State.tablemaker.get(label)

            # Manually trigger main figure and play-toggle button to advance
            main_figure.param.trigger('object')
            player.interval = int(1.05e3*dt)  # ms plus some padding

            # if play_toggle.value:

            # self.doc.add_next_tick_callback(
                    # lambda: self.debug('<next tick callback>'))
                    # lambda: play_toggle.param.trigger('value'))
            # play_toggle.param.trigger('value')

        gain_row = pn.Row(
            *[pn.Column(f'### {grp.name} conductances', *grp.g.panel_sliders())
                for grp in State.network.neuron_groups])

        neuron_row = pn.Row(
            *[pn.Column(f'### {grp.name} neurons', *grp.p.panel_sliders())
                for grp in State.network.neuron_groups])

        parameter_controls = State.network.get_panel_controls()

        # play_toggle.param.watch(simulation, 'value')
        player.param.watch(simulation, 'value')

        panel = \
            pn.Row(
                pn.WidgetBox(
                    f'## {self.psim.title}',
                    main_figure,
                    pn.Column(player, tictoc),
                    # pn.Column(play_toggle, tictoc),
                ),
                pn.Column(
                    gain_row,
                    neuron_row,
                    pn.Row(
                        pn.WidgetBox('### Model data',
                                     *tuple(table_txt.values())),
                        parameter_controls,
                    ),
                ),
            )

        if return_panel:
            return panel
        elif threaded:
            try:
                panel.show(threaded=True)
            except KeyboardInterrupt:
                self.out('Shutting down the server...')
                self.quit_anybar(killall=True)
        else:
            panel.show()  # blocking call
