"""
Base simulator context.
"""

try:
    import panel as pn
except ImportError:
    print('Warning: install `panel` to use interactive dashboards.')

import time

from numpy import ndarray

from tenko.context import AbstractBaseContext, step
from tenko.mixins import RandomMixin
from maps.geometry import EnvironmentGeometry

from .spec import paramspec
from .network import Network
from .state import State, reset_state
from .config import Config


class SimulatorContext(RandomMixin, AbstractBaseContext):

    """
    Main base class for simulations.
    """

    def set_simulation_parameters(self, params, **simparams):
        """
        Set simulation parameters in global scope and shared state.
        """
        # Set the defaults to a paramspec attribute and log differences
        simparams.update(title=simparams.get('title', 'Network Simulation'))
        SimSpec = paramspec('SimSpec', **simparams)
        self.psim = SimSpec()
        self.out(f'{self.psim.title} parameters:')
        for name, value in self.psim:
            if name in params:
                self.psim[name] = params.pop(name)
                logmsg = f'* {name} = {self.psim[name]!r} [default: {value!r}]'
            else:
                logmsg = f'- {name} = {value!r}'.format(name, value)
            self.out(logmsg, hideprefix=True)

        # Update global scope with simulation parameters
        self.get_global_scope().update(self.psim)

        # Clear and update the shared state, then write out a JSON file
        reset_state()
        State.update(self.psim)
        State.context = self
        self.write_json(self.psim, 'simulation')

        # Initialize the 'magic' network object
        self.out('Initializing network...')
        State.network = Network()

        # Display a green AnyBar dot to signal the simulation is running
        State.context.launch_anybar()
        State.context.set_anybar_color('green')

        self.debug(f'[set_sim_params] {State!r}')

    def set_model_parameters(self, params, pfile=None, **defaults):
        """
        Set model parameters according to file, keywords, or defaults.

        Note: It is required to specify default values for all parameters here.
        """
        # Set the defaults to a paramspec attribute and write to file
        ModelSpec = paramspec('ModelSpec', **defaults)
        self.p = ModelSpec()
        self.write_json(self.p.as_dict(), 'defaults.json', base='context')

        # Import values from parameters file if specified
        if pfile is not None:
            fpath, fparams = self.get_json(pfile)
            fparams.update(params)
            params = fparams
            self.out(fpath, prefix='ParameterFile')

        # Write file with effective parameters (i.e., kwargs > file > defaults)
        self.p.update(**params)  # defaults are auto-saved
        self.write_json(self.p.as_dict(), self.filename(stem='params'))

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

    @step
    def simulate_movie(self, tag=None, paramfile=None, **params):
        """
        Simulate the model in batch mode for movie generation.
        """
        # Run the simulation as an animation in non-interactive mode
        params.update(interact=False)
        self.run(paramfile=paramfile, **params)
        anim = FuncAnimation(
                fig       = State.simplot.fig,
                func      = State.network.single_update,
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

    def show_panel(self, dt=0.1, dt_block=25.0, tracewin=100.0, calcwin=25.0,
        figsize=(12.0, 9.0)):
        """
        Construct an interactive Panel dashboard for running the model.
        """
        self.run(
            dt       = dt,
            dt_block = dt_block,
            interact = True,
            tracewin = tracewin,
            calcwin  = calcwin,
            figw     = figsize[0],
            figh     = figsize[1],
        )

        # Player widget for simulation control with block count
        player = pn.widgets.DiscretePlayer(value='1', options=['1','2','3'],
                interval=5000, name='Simulation Control', loop_policy='loop')
        tictoc = pn.pane.Markdown('')
        State.block = 0

        # Markdown displays for each registered table output
        table_txt = {}
        for name, table in State.tablemaker:
            table_txt[name] = pn.pane.Markdown('')

        @pn.depends(player.param.value)
        def simulation(_):
            # Run a blocked update of the network with performance timing
            t0 = time.perf_counter()
            State.network.block_update()
            dt = time.perf_counter() - t0

            # Adapt the player interval to actual performance
            player.interval = int(1.1e3*dt)  # ms plus some padding

            # Update the tic-toc string
            tictoc_str = 'Block {} [{} ms]'.format(State.block, int(1e3*dt))
            tictoc.object = tictoc_str
            self.out(tictoc_str, prefix='BlockCounter')

            # Update the Markdown table objects with new data
            for label in table_txt.keys():
                table_txt[label].object = State.tablemaker.get(label)

            State.block += 1
            return State.simplot.fig

        gain_row = pn.Row(
            *[pn.Column(f'### {grp.name} conductances', *grp.g.panel_sliders())
                for grp in State.network.neuron_groups])

        neuron_row = pn.Row(
            *[pn.Column(f'### {grp.name} neurons', *grp.p.panel_sliders())
                for grp in State.network.neuron_groups])

        parameter_controls = State.network.get_panel_controls()

        panel = \
            pn.Row(
                pn.Column(
                    f'## {self.psim.title}',
                    simulation,
                    pn.Column(player, tictoc),
                ),
                pn.Column(
                    gain_row,
                    neuron_row,
                    pn.Row(
                        pn.Column(*tuple(table_txt.values())),
                        parameter_controls,
                    ),
                ),
            )
        panel.show()  # blocking call
        self.quit_anybar()
