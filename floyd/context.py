"""
Base simulator context.
"""

__all__ = ['SimulatorContext', 'simulate', 'step']


try:
    import panel as pn
except ImportError:
    print('Warning: install `panel` to use interactive dashboards.')

import os
import time
import functools

from matplotlib.animation import FuncAnimation
from panel.pane import Matplotlib, Markdown
import panel as pn

from specify import Specified, Param
from tenko.context import AbstractBaseContext, step
from maps.geometry import EnvironmentGeometry
from roto.strings import sluggify
from pouty.console import snow as hilite
from pouty import debug_mode, debug

from .network import Network
from .state import State, RunMode
from .config import Config


SPECFILE = 'specs.json'
DFLTFILE = 'defaults.json'


def simulate(func=None, *, mode=None):
    """
    Decorator for simulation methods with keyword-only mode argument, which is
    only necessary for any non-standard simulation methods (i.e., user-defined
    methods not named `create_movie`, `collect_data`, or `launch_dashboard`).
    """
    if func is None:
        return functools.partial(simulate, mode=RunMode[str(mode).upper()])
    if mode is None:
        mode = RunMode(func.__name__)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        self = args[0]
        status = { 'OK': False }

        # Initiate the context and prepare simulation parameters
        debug(f'running {func.__name__} in {mode!r}', prefix='launcher')
        self._step_enter(func, args, kwargs)
        self._prepare_simulation(mode, args, kwargs)

        # Call the user-defined model setup method and display the network
        self.setup_model()
        State.context.hline()
        State.network.deisplay_neuron_groups()
        State.network.display_projections()
        State.network.display_object_counts()

        # Execute the requested simulation loop
        res = self._step_execute(func, args, kwargs, status)
        self._step_exit(func, args, kwargs, status)
        return res
    return wrapped


class SimulatorContext(Specified, AbstractBaseContext):

    """
    Context base class for simulations.
    """

    title      = Param(default=Config.title, doc="string, simulation title")
    tag        = Param(default=Config.tag, doc="string or None, simulation label")
    rnd_seed   = Param(default=Config.rnd_seed, doc="string, random numbers seed key")
    duration   = Param(default=Config.duration, doc="ms, simulation duration")
    dt         = Param(default=Config.dt, doc="ms, simulation timestep")
    dt_rec     = Param(default=Config.dt_rec, doc="ms, recording interval")
    dt_block   = Param(default=Config.dt_block, doc="ms, block interval")
    blocksize  = Param(default=int(Config.dt_block / Config.dt), doc="int, timesteps per block")
    figw       = Param(default=Config.figw, doc="inches, main figure width")
    figh       = Param(default=Config.figh, doc="inches, main figure height")
    figdpi     = Param(default=Config.figdpi, doc="dots/inch, figure resolution")
    tracewin   = Param(default=Config.tracewin, doc="ms, length of trace plot window")
    calcwin    = Param(default=Config.calcwin, doc="ms, length of rolling calculation window")
    show_debug = Param(default=Config.show_debug, doc="boolean, whether to show debug statements")

    def __init__(self, **kwargs):
        """
        The constructor performs most of the preparatory steps for a simulation
        except for instantiating the network. Thus, the constructor processes
        kwargs as parameters for subsequent calls to the simulation methods
        (those decorated with @simulate). A benefit of this is that JSON files
        with default and specified parameter values are written out to the
        directory tree (which is created if necessary). These files can be
        examined and modified before making a simulation call.

        Subclasses must override the `setup_model()` method to define the
        'kernel' of the model that will be called by other simulation methods
        (i.e., `create_movie()`, `collect_data()`, and `launch_dashboard()`).
        Model components from floyd packages and modules should be instantiated
        and configured within the `setup_model()` implementation.
        """
        debug_mode(Config.show_debug)  # default to config before init
        super().__init__(**kwargs)
        self.hline()
        debug_mode(self.show_debug)  # use instance attribute after init
        self._specfile_init = kwargs.get('specfile')
        self._prepare_simulation(RunMode.INTERACT, None, kwargs,
                finish_setup=False)

    def __str__(self):
        return AbstractBaseContext.__str__(self)

    def printspec(self):
        self.printf(Specified.__str__(self))

    def _prepare_simulation(self, run_mode, args, kwargs, finish_setup=True):
        """
        Implicit method that prepares for an imminent simulation.
        """
        step = self.current_step()
        tag = self.current_tag()
        if step:
            rundir = os.path.join(self._ctxdir, step)
            if tag: prev_rundir += '+{}'.format(sluggify(tag))

            # Load a previous specfile from the anticipated rundir path
            spath = os.path.join(rundir, SPECFILE)
            try:
                sfdata = self.read_json(spath)
            except Exception as e:
                self.debug(spath, prefix=f'SpecFileNotFound')
            else:
                self.update(sfdata)
                self.out(spath, prefix=f'LoadedSpecFile')
            finally:
                debug_mode(self.show_debug)

        # Update from an alternative specfile if one was provided
        specfile = kwargs.pop('specfile', self._specfile_init)
        found_specfile = None
        found_specpath = None
        if specfile:
            json = self.get_json(specfile)
            if json is not None:
                found_specpath, fspecs = json
                self.update(fspecs)
                found_specfile = specfile
                self.out(found_specpath, prefix='LoadedSpecFile')

        # Consume keywords for parameters in class attribute `spec`
        spec_keys = list(filter(lambda k: k in self.spec, kwargs.keys()))
        for key in spec_keys:
            setattr(self, key, kwargs.pop(key))
            self.debug(f'consumed kwarg {key!r} with {getattr(self, key)!r}')
        if spec_keys:
            self.out(f'Updated {len(spec_keys)} parameter values',
                     prefix='KeywordSpecs')

        # Derived values to be updated (e.g., blocksize)
        self.blocksize = int(self.dt_block / self.dt)

        # Print out the resulting spec parameters if this is the actual run
        if finish_setup:
            self.printspec()

        # Update global scope and shared state
        specdict = dict(self.items())
        specdefaults = dict(self.defaults())
        extra_state = dict(run_mode=run_mode, context=self,
                           specfile=found_specfile, specpath=found_specpath)
        State.reset()
        State.update(specdict, **extra_state)
        self.get_global_scope().update(specdict, **extra_state)

        # Final point at which `show_debug` could possibly change
        debug_mode(self.show_debug)

        # Set the RNG seed if a seed key was provided
        if self.rnd_seed:
            self.set_default_random_seed(rnd_seed)

        # Write JSON files of current parameter values and defaults
        sfpath = self.write_json(specdict, SPECFILE)
        dfpath = self.write_json(specdefaults, DFLTFILE, base='context')
        self.out(sfpath, prefix='WroteSpecFile')
        self.out(dfpath, prefix='WroteDefaultsFile')

        # We want to process parameter updates and write out defaults and
        # specs files during both construction (__init__) and method calls to
        # begin simulations (@simulate decorated methods). However, we do not
        # want the constructor to create the network, etc.

        if not finish_setup:
            return

        # Initialize the simulation network object and assign to an instance
        # attribute (n.b., it goes into shared state anyway)
        self.network = Network()
        self.set_anybar_color('green')
        if debug_mode(): self.printf(State)

    def load_environment_parameters(self, env):
        """
        Import environment geometry into the global scope.
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

    def setup_model(self, *args, **kwargs):
        """
        Model construction: This method must be overridden by subclasses to
        define the model that will be simulated.
        """
        raise NotImplementedError('models must implement setup_model()')

    @simulate
    def create_movie(self, specfile=None, dpi=Config.moviedpi, fps=Config.fps):
        """
        Simulate the model in batch mode for movie generation.
        """
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
        self.c['movie_file'] = self.filename(use_modname=True, use_runtag=True,
                ext='mp4')
        anim.save(self.path(self.c.movie_file), fps=fps, dpi=dpi)
        State.simplot.closefig()
        self.play_movie()

    @simulate
    def collect_data(self, specfile=None):
        """
        Simulate the model in batch mode for data collection.
        """
        while State.recorder:
            State.network.model_update()

        # Save simulation recording data
        self.set_anybar_color('blue')
        State.recorder.save()

    @simulate
    def launch_dashboard(self, specfile=None, return_panel=False,
        threaded=False, dpi=Config.screendpi):
        """
        Construct an interactive Panel dashboard for running the model.
        """
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
            range(1, max(4, 1+int(self.tracewin/self.calcwin))))),
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
            *[pn.Column(f'### {grp.name} conductances',
                *grp.get_gain_sliders())
                    for grp in State.network.neuron_groups.values()])

        neuron_row = pn.Row(
            *[pn.Column(f'### {grp.name} neurons',
                *grp.get_neuron_sliders())
                    for grp in State.network.neuron_groups.values()])

        controls = State.network.get_panel_controls(single_column=True)

        last_column = (controls,)
        if graph_figure is not None:
            last_column = last_column + (graph_figure,)

        control_columns = [pn.Column(gain_row, neuron_row),
                           pn.Column(*last_column)]

        if self._widgets:
            context_column = pn.WidgetBox('### {self.name}', *self._widgets)
            control_columns.insert(0, context_column)

        panel = pn.Row(
                    pn.WidgetBox(
                        f'## {self.title}',
                        main_figure,
                        pn.Row(tictoc, player),
                        '### Model data',
                        pn.Row(*tuple(table_txt.values())),
                    ),
                    *control_columns,
                )

        if return_panel:
            return panel

        try:

            # Blocking call if threaded == True
            self.server = panel.show(threaded=threaded)

        except KeyboardInterrupt:
            pass
