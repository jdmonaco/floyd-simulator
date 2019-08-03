"""
Base simulator context.
"""

try:
    import panel as pn
except ImportError:
    print('Warning: install `panel` to use interactive dashboards.')

import os
import time

from matplotlib.animation import FuncAnimation
from panel.widgets import Toggle
from panel.pane import Matplotlib, Markdown
import panel as pn

from tenko.context import AbstractBaseContext, step
from tenko.mixins import RandomMixin
from roto.dicts import merge_two_dicts
from maps.geometry import EnvironmentGeometry
from pouty.console import snow as hilite
from pouty import debug_mode
from specify import Specified, Param

from .network import Network
from .state import State, RunMode, reset_state
from .config import Config


class SimulatorContext(RandomMixin, AbstractBaseContext, Specified):

    """
    Context base class for simulations.
    """

    def __init__(self, **kwargs):
        tenkw = AbstractBaseContext.pop_tenko_args(kwargs)
        super(AbstractBaseContext, self).__init__(**tenkw)
        super(Specified, self).__init__(**kwargs)
        self.extra_widgets = []
        State.context = self

    def load_parameters(self, step='collect_data', tag=None):
        """
        Load simulation and/or model parameter specs from a context path.
        """
        path = step
        if tag is not None:
            path += f'+{tag}'
        for p, specname, base in [('psim', 'SimSpec', 'simulation'),
                                  ('p', 'ModelSpec', 'model')]:
            fn = f'{base}.json'
            stem = os.path.join(self._ctxdir, path, fn)
            try:
                p_json = self.read_json(stem)
            except Exception as e:
                self.out(stem, prefix='MissingJSONFile', warning=True)
            else:
                # Manually de-serialize the run_mode key
                if 'run_mode' in p_json:
                    modename = p_json['run_mode'].upper()
                    p_json['run_mode'] = RunMode[modename]
                setattr(self, p, paramspec(specname, instance=True, **p_json))
                self.out(stem, prefix=f'{base.title()}Parameters')

    def set_simulation_parameters(self, modparams=None, **simparams):
        """
        Set simulation parameters in global scope and shared state.
        """
        # Start off with the correct (or default) debug mode
        _debug = simparams.get('debug', Config.debug)
        if modparams is not None:
            debug_mode(modparams.get('debug', _debug))

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
                figdpi    = Config.figdpi,
                tracewin  = Config.tracewin,
                calcwin   = Config.calcwin,
                run_mode  = Config.run_mode,
                debug     = Config.debug,
        )

        # Add derived values (e.g., blocksize) to the simulation parameters
        simparams.update(blocksize=int(simparams.get('dt_block',
                Config.dt_block) / simparams.get('dt', Config.dt)))
        for p in (simparams, modparams):
            if p is None:
                continue
            if 'run_mode' in p and type(p['run_mode']) is str:
                modename = p['run_mode'].upper()
                p['run_mode'] = RunMode[modename]

        # Update from model parameters and log differences with defaults
        self.out(f'Simulation parameters:')
        self.psim.update(**simparams)
        for name, value in self.psim:
            logmsg = f'- {name} = {value!r}'.format(name, value)
            if modparams is not None and name in modparams:
                pval = modparams.pop(name)
                if pval != self.psim[name]:
                    logmsg = hilite(
                        f'* {name} = {pval!r} [default: {self.psim[name]!r}]')
                    self.psim[name] = pval
            self.out(logmsg, hideprefix=True)

        # Update global scope and shared state
        self.get_global_scope().update(self.psim)
        reset_state()
        State.update(self.psim)
        State.context = self

        # Manually serialize the run_mode key and then write a JSON file
        psim_dict = self.psim.as_dict()
        psim_dict.update(run_mode=psim_dict['run_mode'].name.lower())
        self.write_json(psim_dict, 'simulation')

        # Initialize the 'magic' network object
        State.network = Network()

        # Display a green AnyBar dot to signal the simulation is running
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
        self.write_json(self.p.as_dict(), 'model')

        # Set parameters as global variables in the object's module scope
        self.out('Model parameters:')
        self.get_global_scope().update(self.p)
        for name, value in self.p:
            dflt = self.p.defaults[name]
            if value != dflt:
                logstr = hilite(f'* {name} = {value!r} [default: {dflt!r}]')
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
                figdpi   = Config.figdpi,   # dots/inch, figure resolution
                tracewin = Config.tracewin, # ms, trace plot window
                calcwin  = Config.calcwin,  # ms, rolling calculation window
                run_mode = Config.run_mode, # set the run mode
                debug    = Config.debug,    # run in debug mode
        )
        self.set_model_parameters(params, pfile=pfile,
                param1 = 1.0, # this is a model parameter default
                param2 = 1.0, # this is another model parameter default
        )
        self.set_default_random_seed(rnd_seed)
        raise NotImplementedError('models must implement setup_model()')

    @step
    def create_movie(self, tag=None, pfile=None, dpi=Config.moviedpi,
        fps=Config.fps, **params):
        """
        Simulate the model in batch mode for movie generation.
        """
        # Run the simulation in the animation run mode
        params.update(run_mode=RunMode.ANIMATE)
        self.setup_model(pfile=pfile, **params)

        anim = FuncAnimation(
                fig       = State.simplot.fig,
                func      = State.network.animation_update,
                init_func = State.simplot.initializer,
                frames    = range(State.N_t),
                interval  = int(1000/Config.fps),
                repeat    = False,
                blit      = True,
        )

        # Save the animation as a movie file and play the movie!
        self['movie_file'] = self.filename(use_modname=True, use_runtag=True,
                ext='mp4')
        anim.save(self.path(self.c.movie_file), fps=fps, dpi=dpi)
        State.simplot.closefig()
        self.hline()
        self.play_movie()

    @step
    def collect_data(self, tag=None, pfile=None, **params):
        """
        Simulate the model in batch mode for data collection.
        """
        # Run the simulation in the data collection run mode
        params.update(run_mode=RunMode.RECORD)
        self.setup_model(pfile=pfile, **params)

        # Run the main loop until exhaustion
        while State.recorder:
            State.network.model_update()

        # Save simulation data traces
        self.set_anybar_color('blue')
        State.recorder.save()

    def add_dashboard_widgets(self, *widgets):
        """
        Add extra widgets to be displayed between figure and neurons.
        """
        self.extra_widgets.extend(widgets)

    def launch_dashboard(self, return_panel=False, threaded=False,
        dpi=Config.screendpi, pfile=None, **params):
        """
        Construct an interactive Panel dashboard for running the model.
        """
        # Run the simulation in the interactive run mode
        params.update(run_mode=RunMode.INTERACT)
        self.setup_model(pfile=pfile, **params)

        # Main figure Matplotlib pane object will be manually updated
        main_figure = Matplotlib(object=State.simplot.fig, dpi=dpi)

        # Create another figure for the network graph if it exists
        graph_figure = None
        if 'netgraph' in State and State.netgraph is not None:
            if not State.simplot.network_graph_in_fig:
                graph_figure = Matplotlib(object=State.netgraph.fig, dpi=dpi)
                self.debug('created separate network graph figure')

        # Set up discrete player widget for play/pause control
        State.block = 0
        dt_perf = int(1e3)
        tictoc = Markdown('Block -- [-- ms]')
        player = pn.widgets.DiscretePlayer(value='1', options=list(map(str,
            range(1, max(4, 1+int(self.psim.tracewin/self.psim.calcwin))))),
            interval=dt_perf, name='Simulation Control', loop_policy='loop')

        # Markdown displays for each registered table output
        table_txt = {}
        for name, table in State.tablemaker:
            table_txt[name] = Markdown(table)

        def simulation(*events):
            nonlocal dt_perf
            t0 = time.perf_counter()

            # Disable the player to stop interval during block simulation
            player.interval = int(1e6)  # ms plus some padding
            player.param.trigger('interval')

            # Run a blocked update of the network
            State.network.dashboard_update()

            # Update the tic-toc string and block count
            tictoc_str = f'Block {State.block} [{dt_perf} ms]'
            tictoc.object = tictoc_str
            State.block += 1
            self.debug(tictoc_str.lower())

            # Update the Markdown table objects with new data
            for label in table_txt.keys():
                table_txt[label].object = State.tablemaker.get(label)

            # Manually update main figure and graph figure if it exists
            main_figure.param.trigger('object')
            if graph_figure is not None:
                graph_figure.param.trigger('object')

            # Calculate the performance timing to update the player interval
            dt_perf = int(1e3*(time.perf_counter() - t0))
            player.interval = dt_perf
            player.param.trigger('interval')

        player.param.watch(simulation, 'value')

        gain_row = pn.Row(
            *[pn.Column(f'### {grp.name} conductances', *grp.g.get_widgets())
                for grp in State.network.neuron_groups])

        neuron_row = pn.Row(
            *[pn.Column(f'### {grp.name} neurons', *grp.p.get_widgets())
                for grp in State.network.neuron_groups])

        controls = State.network.get_panel_controls(single_column=True)

        last_column = (controls,)
        if graph_figure is not None:
            last_column = last_column + (graph_figure,)

        control_columns = [pn.Column(gain_row, neuron_row),
                           pn.Column(*last_column)]

        if self.extra_widgets:
            extra = pn.Column(*self.extra_widgets)
            control_columns.insert(0, extra)

        panel = pn.Row(
                    pn.WidgetBox(
                        f'## {self.psim.title}',
                        main_figure,
                        pn.Row(tictoc, player),
                        pn.Row('### Model data', *tuple(table_txt.values())),
                    ),
                    *control_columns,
                )

        if return_panel:
            return panel

        # Blocking call if threaded == True
        self.server = panel.show(threaded=threaded)
