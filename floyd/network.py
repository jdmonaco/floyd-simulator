"""
Network representation of groups and pathways as nodes and edges.
"""

__all__ = ['Network']


import time
import functools

import matplotlib.pyplot as plt
import networkx as nx

from pouty import debug_mode, printf
from roto.dicts import pprint as dict_pprint
from tenko.base import TenkoObject

from .state import State, RunMode


class Network(TenkoObject):

    """
    Container for neuron groups, synapses, and other simulation elements.
    """

    def __init__(self):
        super().__init__()

        self.neuron_groups = {}
        self.synapses = {}
        self.stimulators = {}
        self.state_updaters = {}
        self.buttons = {}
        self.watchers = {}
        self.counts = dict(neuron_groups=0, synapses=0, stimulators=0,
                           state_updaters=0)
        self.G = nx.DiGraph()
        State.network = self
        self.debug('network initialized')

    @staticmethod
    def _get_object_name(obj):
        """
        Given an unknown object, find a reasonable name for it.
        """
        if hasattr(updater, 'name'):
            name = updater.name
        elif hasattr(updater, '__name__'):
            name = updater.__name__
        elif hasattr(updater, '__qualname__'):
            name = updater.__qualname__
        else:
            s = str(name)
            r = repr(name)
            if len(s) < len(r):
                name = s
            else:
                name = r
        return name

    def get_panel_controls(self, single_column=False):
        """
        Get a Panel Row of controls for network parameters.
        """
        from panel.widgets import Checkbox, Button, TextInput
        from panel.pane import Markdown
        from panel import Row, Column, WidgetBox

        paramfile_input = TextInput(name='Filename',
            value='spec' if State.specfile is None else State.specfile)
        notes_input = TextInput(name='Notes',
            placeholder='Describe the results...')
        notes_txt = Markdown('', height=50)
        filename_txt = Markdown('' if State.specpath is None else
                                State.specpath)
        uniquify_box = Checkbox(name='Force unique', value=False)
        save_btn = Button(name='Save', button_type='primary')
        restore_btn = Button(name='Restore', button_type='success')
        defaults_btn = Button(name='Defaults', button_type='warning')
        zero_btn = Button(name='Disconnect', button_type='danger')

        self.buttons.update(
                save = save_btn,
                restore = restore_btn,
                defaults = defaults_btn,
                zero = zero_btn,
        )

        # A decorator to throttle callback calls from double-clicked buttons
        def throttle(func=None, dt_min=0.5):
            t_last = -dt_min
            if callable(func):
                @functools.wraps(func)
                def wrapper(value):
                    nonlocal t_last
                    t_now = time.perf_counter()
                    dt = t_now - t_last
                    t_last = t_now
                    if dt < dt_min:
                        self.debug('{}: skipping repeat call (dt = {:.2f} s)',
                                   func.__name__, dt)
                        return
                    func(value)
                return wrapper
            else:
                return functools.partial(throttle, dt_min=dt_min)

        @throttle(10.0)
        def save(value):
            State.context.toggle_anybar()
            self.debug('save button callback')
            psavefn = paramfile_input.value
            unique = uniquify_box.value
            params = dict(notes=notes_input.value)
            if State.context._widgets:
                params.update({name:slider.value for name, slider in
                    State.context._widgets.items()})
            for name, grp in self.neuron_groups.items():
                grp_params = {name:slider.value for name, slider in
                              grp._widgets.items()}
                params[name] = grp_params
            p = State.context.write_json(params, psavefn, base='context',
                    unique=unique)
            filename_txt.object = p

        @throttle
        def restore(value):
            State.context.toggle_anybar()
            self.debug('restore button callback')
            psavefn = paramfile_input.value
            json = State.context.get_json(psavefn)
            if json is None:
                filename_txt.object = f'**Could not find spec: {psavefn!r}**'
                return
            fullpath, params = json
            filename_txt.object = fullpath
            notes_input.value = params.pop('notes', '')
            for grp_name, grp in self.neuron_groups.items():
                grp_params = params[grp_name]
                for name, slider in grp._widgets.items():
                    if name in grp_params:
                        slider.value = grp_params.pop(name)
                del params[grp_name]
            for name, value in params.items():
                if name in State.context._widgets:
                    State.context._widgets[name].value = value

        @throttle
        def defaults(value):
            State.context.toggle_anybar()
            self.debug('defaults button callback')
            self.reset_neurons()

        @throttle
        def zero(value):
            State.context.toggle_anybar()
            self.debug('disconnet button callback')
            for grp in self.neuron_groups.values():
                for gname in grp.gain_keys:
                    if gname in grp._widgets:
                        grp._widgets[gname].value = 0.0

        # Add callback functions to the buttons and save the watcher objects
        for name, btn in self.buttons.items():
            self.watchers[name] = btn.param.watch(eval(name), 'clicks')

        # Add a client-side link between notes input and text display
        notes_input.link(notes_txt, value='object')

        # Create separete columns to construct as a final row or column
        file_column = Column('### Model spec files', paramfile_input,
                          uniquify_box, filename_txt, notes_input, notes_txt)
        control_column = Column('### Parameter controls',
                             *tuple(self.buttons.values()))

        if single_column:
            return WidgetBox(control_column, file_column)
        return WidgetBox(Row(file_column, control_column))

    def unlink_widgets(self):
        """
        Unlink all Panel widgets from their callback functions.
        """
        State.context.unlink_sliders()
        for grp in self.neuron_groups.values():
            grp.unlink_widgets()
        for name, button in self.buttons.items():
            button.param.unwatch(self.watchers[name])

    def dashboard_update(self):
        """
        For interactive mode, run a block of simulation then update outputs.
        """
        for i in range(State.blocksize):
            self.model_update()
            State.simplot.update_traces_data()
        self.display_update()

    def animation_update(self, n=None):
        """
        For animation mode, run one timestep and return updated artists.
        """
        self.model_update()
        State.simplot.update_traces_data()
        self.display_update()
        return State.simplot.artists

    def model_update(self):
        """
        Main once-per-loop update: the recorder, neuron groups, and synapses.
        """
        State.recorder.update()
        if not State.recorder:
            return

        for updater in self.state_updaters.values():
            updater.updater()
        for stimulator in self.stimulators.values():
            stimulator.update()
        for group in self.neuron_groups.values():
            group.update()
        for synapses in self.synapses.values():
            synapses.update()

    def display_update(self):
        """
        Update main simulation figure (and tables in interactive mode).
        """
        State.simplot.update_plots()
        if State.run_mode == RunMode.INTERACT:
            State.tablemaker.update()

    def reset_neurons(self):
        """
        Reset neuron group parameters to parameter defaults.
        """
        for grp in self.neuron_groups.values():
            grp.reset()

    def add_neuron_group(self, group):
        """
        Add an instance of NeuronGroup to the network.
        """
        if group.name in self.neuron_groups:
            self.out(group.name, prefix='AlreadyDeclared', warning=True)
            return
        self.neuron_groups[group.name] = group
        self.G.add_node(group.name, object=group)
        self.counts['neuron_groups'] += 1
        self.debug(f'added neuron group {group.name!r}')
        if debug_mode(): printf(f'{group!s}')

    def add_synapses(self, synapses):
        """
        Add an instance of Synapses to the network.
        """
        if synapses.name in self.synapses:
            self.out(synapses.name, prefix='AlreadyDeclared', warning=True)
            return
        self.synapses[synapses.name] = synapses
        synapses.post.add_synapses(synapses)
        self.G.add_edge(synapses.pre.name, synapses.post.name, object=synapses)
        self.counts['synapses'] += 1
        self.debug('added synapses {synapses!r}')
        if debug_mode(): printf(f'{synapses!s}')

    def add_stimulator(self, stim):
        """
        Add an instance of InputStimulator to the network.
        """
        if stim.name in self.stimulators:
            self.out(stimulators.name, prefix='AlreadyDeclared', warning=True)
            return
        self.stimulators[stim.name] = stim
        self.G.add_edge(stim.name, stim.target.name, object=stim)
        self.counts['stimulators'] += 1
        self.debug(f'added stimulator {stim.name!r} for {stim.target.name!r}')

    def add_state_updater(self, updater):
        """
        Add an object that updates the shared state.
        """
        if updater in list(self.state_updaters.values()):
            self.out(updater, prefix='AlreadyDeclared', warning=True)
            return
        name = self._get_object_name(updater)
        self.state_updaters[name] = updater
        self.counts['state_updaters'] += 1
        self.debug(f'added state updater {name!r}')

    def display_neural_connectivity(self):
        """
        Print out detailed fanin/fanout statistics for each projection.
        """
        self.out.hline()
        for synapses in self.synapses.values():
            synapses.connectivity_stats()
            self.out.hline()

    def display_object_counts(self):
        """
        Display a list of object counts for the current network.
        """
        self.out.printf(dict_pprint(self.counts, name=self.name))
