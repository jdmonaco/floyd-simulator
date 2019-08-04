"""
Base simulator context.
"""

try:
    import panel as pn
except ImportError:
    print('Warning: install `panel` to use interactive dashboards.')

import os
import time
from decorator import decorator

from matplotlib.animation import FuncAnimation
from panel.pane import Matplotlib, Markdown
import panel as pn

from specify import Specified, Param
from tenko.context import AbstractBaseContext, step
from maps.geometry import EnvironmentGeometry
from roto.dicts import merge_two_dicts
from roto.strings import sluggify
from pouty.console import snow as hilite
from pouty import debug_mode

from .network import Network
from .state import State, RunMode
from .config import Config


SPECFILE = 'specs.json'
DFLTFILE = 'defaults.json'


@decorator
def simulate(_f_, *args, **kwargs):
    """
    Declare a method as a simulation in this context.
    """
    self = args[0]
    status = { 'OK': False }
    self._step_enter(_f_, args, kwargs)
    self._prepare_simulation(args, kwargs)
    res = self._step_execute(_f_, args, kwargs, status)
    self._step_exit(_f_, args, kwargs, status)

    return res


class SimulatorContext(AbstractBaseContext, Specified):

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
    run_mode   = Param(default=Config.run_mode, doc="'animate'|'interact'|'record', simulation type")
    show_debug = Param(default=Config.show_debug, doc="boolean, whether to show debug statements")

    def __init__(self, **kwargs):
        """
        Separate keyword arguments into tenko.AbstractBaseContext arguments and
        specify.Specified parameter values. The constructor performs most of
        the preparatory steps for a simulation except for instantiating the
        main network object; this mean that the constructor processes keyword
        arguments as parameter just subsequent calls to one of the simulation
        methods (decorated with @simulate). A benefit of this is that JSON
        files with default and specified parameter values are written out to
        the directory tree (which is created if necessary. These files can be
        examined and modified before making a call to a simulation method.

        Subclasses must override the `setup_model()` method, which should be
        considered to be the 'kernel' of the model called by the simulation
        methods (i.e., `create_movie()`, `collect_data()`, and
        `launch_dashboard()`). Model components from floyd packages and modules
        should be instantiated and configured within the `setup_model()`
        implementation.
        """
        tenkw = AbstractBaseContext.pop_tenko_args(kwargs)
        super(AbstractBaseContext, self).__init__(**tenkw)
        super(Specified, self).__init__(**kwargs)
        self._extra_widgets = []
        self._specfile_init = kwargs.get('specfile')
        self._prepare_simulation(None, kwargs, finish_setup=False)

    def __str__(self):
        return super(Specified, self).__str__()

    def printspec(self):
        self.out(str(self), hideprefix=True)

    def _prepare_simulation(self, args, kwargs, finish_setup=True):
        """
        Implicit method that prepares for an imminent simulation.
        """
        debug_mode(self.show_debug)
        step = self._lastcall['step']
        tag = self._lastcall['tag']
        rundir = os.path.join(self._ctxdir, step)
        if tag: prev_rundir += '+{}'.format(sluggify(tag))

        # Load a previous specfile from the anticipated rundir path
        spath = os.path.join(rundir, SPECFILE)
        try:
            sfdata = self.read_json(spath)
        except Exception as e:
            self.debug(spath, prefix='rundir specs ({SPECFILE!r}) not found')
        else:
            # De-serialize the run_mode key
            if 'run_mode' in sfdata:
                modename = sfdata['run_mode'].upper()
                sfdata['run_mode'] = RunMode[modename]
            self.update(**sfdata)
            self.out(spath, prefix=f'LoadedSpecFile')
        finally:
            debug_mode(self.show_debug)

        # Update from an alternative specfile if one was ... provided
        specfile = kwargs.pop('specfile', self._specfile_init)
        if specfile:
            fpath, fspecs = self.get_json(specfile)
            self.update(**fspec)
            self.out(fpath, prefix='LoadedSpecFile')

        # Update Param values with keyword arguments from the call
        self.update(**kwargs)
        if kwargs:
            self.out(repr(list(kwargs.keys())), prefix='LoadedKeywords')
        debug_mode(self.show_debug)

        # Derived (e.g., blocksize) and de-serialized (e.g., run_mode) values
        self.blocksize = int(self.dt_block / self.dt)
        if type(self.run_mode) is str:
            modename = self.run_mode.upper()
            self.run_mode = RunMode[modename]

        # Print out the resulting spec parameters
        self.out(f'Simulation parameters:')
        self.printspec()

        # Update global scope and shared state
        specdict = dict(self.values())
        State.reset()
        State.update(specdict)
        self.get_global_scope().update(**specdict)

        # Set the RNG seed if a seed key was provided
        if 'rnd_seed' in self and self.rnd_seed:
            self.set_default_random_seed(rnd_seed)

        # Manually serialize the run_mode key and then write a JSON file
        if isinstance(specdict['run_mode'], RunMode):
            specdict.update(run_mode=specdict['run_mode'].name.lower())
        self.write_json(specdict, SPECFILE)

        # Similarly, write the defaults to a JSON file
        dfltdict = dict(self.defaults())
        if isinstance(dfltdict['run_mode'], RunMode):
            dfltdict.update(run_mode=dfltdict['run_mode'].name.lower())
        self.write_json(dfltdict, DFLTFILE, base='context')

        # We want to process parameter updates and write out defaults and
        # specs files during both construction (__init__) and method calls to
        # begin simulations (@simulate decorated methods). However, we do not
        # want the constructor to create the network, etc.

        if not finish_setup:
            return

        # Display green AnyBar to signal the start of the simulation
        self.set_anybar_color('green')
        self.debug(f'State = {State}')

        # Initialize the simulation network object and store an instance
        # attribute reference (n.b., it goes into shared state anyway)
        State.context = self
        self.network = Network()

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

    def setup_model(self):
        """
        Model construction: This method must be overridden by subclasses to
        define the model that will be simulated.
        """
        raise NotImplementedError('models must implement setup_model()')

    @simulate
    def create_movie(self, specfile=None, dpi=Config.moviedpi, fps=Config.fps,
        **specs):
        """
        Simulate the model in batch mode for movie generation.
        """
        # Run the simulation in the animation run mode
        spec.update(run_mode=RunMode.ANIMATE)
        self.setup_model()

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

    @simulate
    def collect_data(self, specfile=None, **specs):
        """
        Simulate the model in batch mode for data collection.
        """
        # Run the simulation in the data collection run mode
        spec.update(run_mode=RunMode.RECORD)
        self.setup_model()

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
        self._extra_widgets.extend(widgets)

    @simulate
    def launch_dashboard(self, specfile=None, return_panel=False,
        threaded=False, dpi=Config.screendpi, **specs):
        """
        Construct an interactive Panel dashboard for running the model.
        """
        # Run the simulation in the interactive run mode
        spec.update(run_mode=RunMode.INTERACT)
        self.setup_model()

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
            *[pn.Column(f'### {grp.name} conductances',
                *grp.get_gain_sliders())
                    for grp in State.network.neuron_groups])

        neuron_row = pn.Row(
            *[pn.Column(f'### {grp.name} neurons',
                *grp.get_neuron_sliders())
                    for grp in State.network.neuron_groups])

        controls = State.network.get_panel_controls(single_column=True)

        last_column = (controls,)
        if graph_figure is not None:
            last_column = last_column + (graph_figure,)

        control_columns = [pn.Column(gain_row, neuron_row),
                           pn.Column(*last_column)]

        if self._extra_widgets:
            extra = pn.Column(*self._extra_widgets)
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
