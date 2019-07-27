"""
Network representation of groups and pathways as nodes and edges.
"""

import time
import functools

import matplotlib.pyplot as plt
import networkx as nx

from pouty import debug

from .state import State


class Network(object):

    """
    Container for neuron groups, synapses, and other simulation elements.
    """

    def __init__(self):
        self.neuron_groups = []
        self.synapses = []
        self.stimulators = []
        self.buttons = {}
        self.watchers = {}
        self.G = nx.DiGraph()

        State.network = self

    def get_panel_controls(self):
        """
        Get a Panel Row of controls for network parameters.
        """
        from panel.widgets import Checkbox, Button, TextInput
        from panel.pane import Markdown
        from panel import Row, Column, WidgetBox

        paramfile_input = TextInput(name='Filename', value='model-params')
        notes_input = TextInput(name='Notes',
                placeholder='Describe the results...')
        notes_txt = Markdown('', height=200)
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
                        State.context.debug(
                                '{}: skipping repeat call (dt = {:.2f} s)',
                                func.__name__, dt)
                        return
                    func(value)
                return wrapper
            else:
                return functools.partial(throttle, dt_min=dt_min)

        @throttle(10.0)
        def save(value):
            State.context.toggle_anybar()
            State.context.debug('save button callback')
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
            State.context.debug('restore button callback')
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
            State.context.debug('defaults button callback')
            for grp in self.neuron_groups:
                for which in 'gp':
                    spec = getattr(grp, which)
                    for name, slider in spec.sliders.items():
                        slider.value = spec.defaults[name]

        @throttle
        def zero(value):
            State.context.toggle_anybar()
            State.context.debug('zero button callback')
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

        return WidgetBox(Row(
                    Column('### Parameter files', paramfile_input,
                           uniquify_box, filename_txt, notes_input, notes_txt),
                    Column('### Parameter controls',
                           *tuple(self.buttons.values()))))

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
        for stimulator in self.stimulators:
            stimulator.update()
        State.recorder.update()
        for group in self.neuron_groups:
            group.update()
        for synapses in self.synapses:
            synapses.update()

    def display_update(self):
        """
        Update main simulation figure (and tables in interactive mode).
        """
        State.simplot.update_plots()
        if State.interact:
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
        self.neuron_groups.append(group)
        self.G.add_node(group)

    def add_synapses(self, synapses):
        """
        Add an instance of Synapses to the network.
        """
        self.synapses.append(synapses)
        synapses.post.add_synapses(synapses)
        self.G.add_edge(synapses.pre, synapses.post, synapses=synapses)

    def add_stimulator(self, stimulator):
        """
        Add an instance of InputStimulator to the network.
        """
        self.stimulators.append(stimulator)
        self.G.add_edge(stimulator, stimulator.target)

    def display_neural_connectivity(self):
        """
        Print out detailed fanin/fanout statistics for each projection.
        """
        State.context.hline()
        for S in self.synapses:
            S.connectivity_stats()
            State.context.hline()

    def group_items(self):
        """
        Generator over neuron groups that provides (name, object) tuples.
        """
        for group in self.neuron_groups:
            yield group.name, group

    def group_names(self):
        """
        Generator over neuron groups that provides group names.
        """
        for group in self.neuron_groups:
            yield group.name

    def group_values(self):
        """
        Generator over neuron groups that provides group objects.
        """
        for group in self.neuron_groups:
            yield group

    def synapse_items(self):
        """
        Generator over synapses that provides (name, object) tuples.
        """
        for synapse in self.synapses:
            yield synapse.name, synapse

    def synapse_names(self):
        """
        Generator over synapses that provides synapse names.
        """
        for synapse in self.synapses:
            yield synapse.name

    def synapse_values(self):
        """
        Generator over synapses that provides synapse objects.
        """
        for synapse in self.synapses:
            yield synapse

    def draw(self, ax=None):
        """
        Plot the graph.
        """
        pos = nx.nx_agraph.graphviz_layout(self.G)
        nx.draw(self.G, pos=pos, ax=ax)
