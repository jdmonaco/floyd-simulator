"""
Network representation of groups and pathways as nodes and edges.
"""

import time
import functools

import matplotlib.pyplot as plt
import networkx as nx

from tenko.base import TenkoObject

from .state import State, RunMode


class Network(TenkoObject):

    """
    Container for neuron groups, synapses, and other simulation elements.
    """

    def __init__(self):
        super().__init__()

        self.neuron_groups = []
        self._neuron_dict = {}
        self.synapses = []
        self._synapses_dict = {}
        self.stimulators = []
        self._stimulators_dict = {}
        self.state_updaters = []
        self.buttons = {}
        self.watchers = {}
        self.G = nx.DiGraph()
        State.network = self
        self.debug('network initialized')

    def get_panel_controls(self, single_column=False):
        """
        Get a Panel Row of controls for network parameters.
        """
        from panel.widgets import Checkbox, Button, TextInput
        from panel.pane import Markdown
        from panel import Row, Column, WidgetBox

        paramfile_input = TextInput(name='Filename', value='model-params')
        notes_input = TextInput(name='Notes',
                placeholder='Describe the results...')
        notes_txt = Markdown('', height=50)
        filename_txt = Markdown('')
        uniquify_box = Checkbox(name='Force unique', value=False)
        save_btn = Button(name='Save', button_type='primary')
        restore_btn = Button(name='Restore', button_type='success')
        defaults_btn = Button(name='Defaults', button_type='warning')
        zero_btn = Button(name='Zero', button_type='danger')

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
            for grp in self.neuron_groups:
                params.update({name:slider.value for name, slider in
                               grp.g.sliders.items()})
                params.update({f'{name}_{grp.name}':slider.value
                               for name, slider in grp.p.sliders.items()})
            p = State.context.write_json(params, psavefn, base='context',
                    unique=unique)
            filename_txt.object = p

        @throttle
        def restore(value):
            State.context.toggle_anybar()
            self.debug('restore button callback')
            psavefn = paramfile_input.value
            fullpath, params = State.context.get_json(psavefn)
            filename_txt.object = fullpath
            notes_input.value = params.pop('notes', '')
            for grp in self.neuron_groups:
                for gname, slider in grp.g.sliders.items():
                    if gname in params:
                        slider.value = params[gname]
                for name, slider in grp.p.sliders.items():
                    pname = f'{name}_{grp.name}'
                    if pname in params:
                        slider.value = params[pname]

        @throttle
        def defaults(value):
            State.context.toggle_anybar()
            self.debug('defaults button callback')
            for grp in self.neuron_groups:
                for which in 'gp':
                    spec = getattr(grp, which)
                    for name, slider in spec.sliders.items():
                        slider.value = spec.defaults[name]

        @throttle
        def zero(value):
            State.context.toggle_anybar()
            self.debug('zero button callback')
            for grp in self.neuron_groups:
                for slider in grp.g.sliders.values():
                    slider.value = 0.0
                for slider in grp.p.sliders.values():
                    slider.value = slider.start

        # Add callback functions to the buttons and save the watcher objects
        for name, btn in self.buttons.items():
            self.watchers[name] = btn.param.watch(eval(name), 'clicks')

        # Add a client-side link between notes input and text display
        notes_input.link(notes_txt, value='object')

        # Create separete columns to construct as a final row or column
        file_column = Column('### Parameter files', paramfile_input,
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
        for grp in self.neuron_groups:
            grp.g.unlink_sliders()
            grp.p.unlink_sliders()
        State.context.p.unlink_sliders()
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

        for updater in self.state_updaters:
            updater.updater()
        for stimulator in self.stimulators:
            stimulator.update()
        for group in self.neuron_groups:
            group.update()
        for synapses in self.synapses:
            synapses.update()

    def display_update(self):
        """
        Update main simulation figure (and tables in interactive mode).
        """
        State.simplot.update_plots()
        if State.run_mode == RunMode.INTERACT:
            State.tablemaker.update()

    def reset(self):
        """
        Reset neuron group parameters to spec defaults.
        """
        for grp in self.neuron_groups:
            grp.reset()

    def add_neuron_group(self, group):
        """
        Add an instance of NeuronGroup to the network.
        """
        if group in self.neuron_groups:
            self.out(group.name, prefix='AlreadyInNetwork', warning=True)
            return
        self.neuron_groups.append(group)
        self._neuron_dict[group.name] = group
        self.G.add_node(group.name, object=group)
        self.debug(f'added neuron group: {group!s}')

    def add_synapses(self, synapses):
        """
        Add an instance of Synapses to the network.
        """
        if synapses in self.synapses:
            self.out(synapses.name, prefix='AlreadyInNetwork', warning=True)
            return
        self.synapses.append(synapses)
        self._synapses_dict[synapses.name] = synapses
        synapses.post.add_synapses(synapses)
        self.G.add_edge(synapses.pre.name, synapses.post.name, object=synapses)
        self.debug('added synapses: {synapses!s}')

    def add_stimulator(self, stim):
        """
        Add an instance of InputStimulator to the network.
        """
        if stim in self.stimulators:
            self.out(stimulators.name, prefix='AlreadyInNetwork', warning=True)
            return
        self.stimulators.append(stim)
        self._stimulators_dict[stim.name] = stim
        self.G.add_edge(stim.name, stim.target.name, object=stim)
        self.debug(f'added stimulator: {stim!s}')

    def add_state_updater(self, updater):
        """
        Add an object that updates the shared state.
        """
        self.state_updaters.append(updater)

    def display_neural_connectivity(self):
        """
        Print out detailed fanin/fanout statistics for each projection.
        """
        self.out.hline()
        for S in self.synapses:
            S.connectivity_stats()
            self.out.hline()

    def get_neuron_group(self, key):
        """
        Return the named neuron group object.
        """
        return self._neuron_dict[key]

    def group_items(self):
        """
        Generator over neuron groups that provides (name, object) tuples.
        """
        return self._neuron_dict.items()

    def group_names(self):
        """
        Generator over neuron groups that provides group names.
        """
        return self._neuron_dict.keys()

    def group_values(self):
        """
        Generator over neuron groups that provides group objects.
        """
        return self._neuron_dict.values()

    def get_synapses(self, key):
        """
        Return the named neuron group object.
        """
        return self._synapses_dict[key]

    def synapse_items(self):
        """
        Generator over synapses that provides (name, object) tuples.
        """
        return self._synapses_dict.items()

    def synapse_names(self):
        """
        Generator over synapses that provides synapse names.
        """
        return self._synapses_dict.keys()

    def synapse_values(self):
        """
        Generator over synapses that provides synapse objects.
        """
        return self._synapses_dict.values()

    def get_stimulator(self, key):
        """
        Return the named neuron group object.
        """
        return self._stimulators_dict[key]

    def stimulator_items(self):
        """
        Generator over stimulators that provides (name, object) tuples.
        """
        return self._stimulators_dict.items()

    def stimulator_names(self):
        """
        Generator over stimulators that provides stimulator names.
        """
        return self._stimulators_dict.keys()

    def stimulator_values(self):
        """
        Generator over stimulators that provides stimulator objects.
        """
        return self._stimulators_dict.values()
