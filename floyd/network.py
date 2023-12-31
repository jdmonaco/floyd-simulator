"""
Network representation of groups and pathways as nodes and edges.
"""

__all__ = ("Network",)


import functools
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pouty import debug_mode, printf
from roto.dicts import pprint as dict_pprint
from roto.strings import sluggify
from tenko.base import TenkoObject

from .clocks import ProgressBar, SimulationClock
from .state import RunMode, State


class Network(TenkoObject):

    """
    Container for neuron groups, synapses, and other simulation elements.
    """

    def __init__(self):
        super().__init__()

        self.clocks = []
        self.neuron_groups = {}
        self.input_groups = {}
        self.synapses = {}
        self.stimulators = {}
        self.state_updaters = {}
        self.buttons = {}
        self.watchers = {}
        self.counts = dict(
            neuron_groups=0,
            input_groups=0,
            synapses=0,
            stimulators=0,
            state_updaters=0,
            clocks=0,
        )
        self.G = nx.DiGraph()
        State.network = self
        self.simclock = SimulationClock()
        self.progressbar = ProgressBar()
        self.show_progress = not State.show_debug and State.run_mode not in (
            RunMode.INTERACT,
            RunMode.SAMPLE,
        )
        self.movie_recorder = None
        self.debug("network initialized")

    @staticmethod
    def _get_object_name(obj):
        """
        Given an unknown object, find a reasonable name for it.
        """
        if hasattr(obj, "name"):
            name = obj.name
        elif hasattr(obj, "__name__"):
            name = obj.__name__
        elif hasattr(obj, "__qualname__"):
            name = obj.__qualname__
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
        from panel import Column, Row, WidgetBox
        from panel.pane import Markdown
        from panel.widgets import Button, Checkbox, TextInput

        paramfile_input = TextInput(
            name="Filename", value="spec" if State.specfile is None else State.specfile
        )
        notes_input = TextInput(name="Notes", placeholder="Describe the results...")
        notes_txt = Markdown("", height=150)
        filename_txt = Markdown("" if State.specpath is None else State.specpath)
        uniquify_box = Checkbox(name="Force unique", value=False)
        save_btn = Button(name="Save", button_type="primary")
        restore_btn = Button(name="Restore", button_type="success")
        defaults_btn = Button(name="Defaults", button_type="warning")
        zero_btn = Button(name="Disconnect", button_type="danger")

        self.buttons.update(
            save=save_btn,
            restore=restore_btn,
            defaults=defaults_btn,
            zero=zero_btn,
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
                        self.debug(
                            "{}: skipping repeat call (dt = {:.2f} s)",
                            func.__name__,
                            dt,
                        )
                        return
                    func(value)

                return wrapper
            else:
                return functools.partial(throttle, dt_min=dt_min)

        @throttle(10.0)
        def save(value):
            State.context.toggle_anybar()
            self.debug("save button callback")
            psavefn = paramfile_input.value
            unique = uniquify_box.value
            params = dict(notes=notes_input.value)
            if State.context._widgets:
                params.update(
                    {
                        name: slider.value
                        for name, slider in State.context._widgets.items()
                    }
                )
            for name, grp in self.neuron_groups.items():
                grp_params = {
                    name: slider.value for name, slider in grp._widgets.items()
                }
                params[name] = grp_params
            p = State.context.write_json(params, psavefn, base="context", unique=unique)
            filename_txt.object = p

        @throttle
        def restore(value):
            State.context.toggle_anybar()
            self.debug("restore button callback")
            psavefn = paramfile_input.value
            json = State.context.get_json(psavefn)
            if json is None:
                filename_txt.object = f"**Could not find spec: {psavefn!r}**"
                return
            fullpath, params = json
            filename_txt.object = fullpath
            notes_input.value = params.pop("notes", "")
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
            self.debug("defaults button callback")
            self.reset_neurons()

        @throttle
        def zero(value):
            State.context.toggle_anybar()
            self.debug("disconnect button callback")
            for grp in self.neuron_groups.values():
                for gname in grp.gain_keys:
                    if gname in grp._widgets:
                        slider = grp._widgets[gname]
                        slider.value = slider.start  # log-slider -> min.

        # Add callback functions to the buttons and save the watcher objects
        for name, btn in self.buttons.items():
            self.watchers[name] = btn.param.watch(eval(name), "clicks")

        # Add a client-side link between notes input and text display
        notes_input.link(notes_txt, value="object")

        # Create separete columns to construct as a final row or column
        file_column = Column(
            "### Model spec files",
            paramfile_input,
            uniquify_box,
            filename_txt,
            notes_input,
            notes_txt,
        )
        control_column = Column("### Parameter controls", *tuple(self.buttons.values()))

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
        For animation mode, run enough timesteps for a video frame of the movie
        and return updated artists.
        """
        frame = self.movie_recorder.frame.n
        while self.movie_recorder.frame.n == frame:
            self.model_update()
            State.simplot.update_traces_data()
        self.display_update()
        self.debug("frame = {self.movie_recorder.n_frame}, t = {State.t}")
        return State.simplot.artists

    def model_update(self):
        """
        Main once-per-loop update: clocks, progress bar, state updaters,
        stimulators, input & neuron groups, and projections.
        """
        self.simclock.update()
        if self.show_progress:
            self.progressbar.update()
        for clock in self.clocks:
            clock.update()
        if not self.simclock:
            self.display_summary()
            return

        for updater in self.state_updaters.values():
            updater.update()
        for stimulator in self.stimulators.values():
            stimulator.update()
        for inpgrp in self.input_groups.values():
            inpgrp.update()
        for synapses in self.synapses.values():
            synapses.update()
        for nrngrp in self.neuron_groups.values():
            nrngrp.update()

    def display_update(self):
        """
        Update main simulation figure (and tables in interactive mode).
        """
        State.simplot.update_plots()
        if State.run_mode == RunMode.INTERACT and State.is_defined("tablemaker"):
            State.tablemaker.update()

    def display_summary(self):
        """
        Emit a simulation completion message with summary of time/frames.
        """
        if State.run_mode == RunMode.SAMPLE:
            return
        self.out.hline()
        msg = f"Simulation complete: n = {State.N_t - 1:g} timesteps"
        if State.run_mode == RunMode.RECORD:
            msg += f" / {State.recorder.clock.N_t:g} samples"
        elif State.run_mode == RunMode.ANIMATE:
            N_frames = self.movie_recorder.frame.N_t
            msg += f" / {N_frames:g} video frames"
        self.out(msg, anybar="green")
        self.out.hline()

    def reset_neurons(self):
        """
        Reset neuron group parameters to parameter defaults.
        """
        for grp in self.neuron_groups.values():
            grp.reset()

    def set_movie_recorder(self, recorder):
        """
        Set the MovieRecorder object to be used for this simulation.
        """
        if self.movie_recorder:
            raise RuntimeError(f"recorder exists: {self.movie_recorder!r}")

        self.movie_recorder = recorder

    def add_clock(self, clock):
        """
        Add a clock to be updated in the simulation loop.
        """
        self.clocks.append(clock)
        self.counts["clocks"] += 1
        self.debug(f"added clock {clock!r}")

    def add_neuron_group(self, group):
        """
        Add an instance of NeuronGroup to the network.
        """
        if group.name in self.neuron_groups:
            self.out(group.name, prefix="AlreadyDeclared", warning=True)
            return
        self.neuron_groups[group.name] = group
        self.G.add_node(group.name, object=group)
        self.counts["neuron_groups"] += 1
        self.debug(f"added neuron group {group.name!r}")
        if debug_mode():
            printf(f"{group!s}")

    def add_input_group(self, group):
        """
        Add an instance of InputGroup to the network.
        """
        if group.name in self.input_groups:
            self.out(group.name, prefix="AlreadyDeclared", warning=True)
            return
        self.input_groups[group.name] = group
        self.G.add_node(group.name, object=group)
        self.counts["input_groups"] += 1
        self.debug(f"added input group {group.name!r}")
        if debug_mode():
            printf(f"{group!s}")

    def add_projection(self, synapses):
        """
        Add an instance of Synapses to the network.
        """
        if synapses.name in self.synapses:
            self.out(synapses.name, prefix="AlreadyDeclared", warning=True)
            return
        self.synapses[synapses.name] = synapses
        synapses.post.add_afferent_projection(synapses)
        self.G.add_edge(synapses.pre.name, synapses.post.name, object=synapses)
        self.counts["synapses"] += 1
        self.debug(f"added synapses {synapses!r}")
        if debug_mode():
            printf(f"{synapses!s}")

    def add_stimulator(self, stim):
        """
        Add an instance of InputStimulator to the network.
        """
        if stim.name in self.stimulators:
            self.out(stimulators.name, prefix="AlreadyDeclared", warning=True)
            return
        self.stimulators[stim.name] = stim
        self.G.add_edge(stim.name, stim.target.name, object=stim)
        self.counts["stimulators"] += 1
        self.debug(f"added stimulator {stim.name!r} for {stim.target.name!r}")

    def add_state_updater(self, updater):
        """
        Add an object that updates the shared state.
        """
        if updater in list(self.state_updaters.values()):
            self.out(updater, prefix="AlreadyDeclared", warning=True)
            return
        name = self._get_object_name(updater)
        self.state_updaters[name] = updater
        self.counts["state_updaters"] += 1
        self.debug(f"added state updater {name!r}")

    def new_input_groups(self, *groupnames, inpclass, **specs):
        """
        Create a batch of neuron groups of a particular class using the given
        keyword specs and add them to the network.
        """
        if not groupnames:
            return
        self.out.hline()
        for name in groupnames:
            group = inpclass(name=name, **specs)
            if name not in self.input_groups:
                self.add_input_group(group)
                self.debug(f"group {name!r} did not add itself to network")
            self.out.printf(group)
            self.out.hline()

    def new_neuron_groups(self, *groupnames, nrnclass, **specs):
        """
        Create a batch of neuron groups of a particular class using the given
        keyword specs and add them to the network.
        """
        if not groupnames:
            return
        self.out.hline()
        for name in groupnames:
            group = nrnclass(name=name, **specs)
            if name not in self.neuron_groups:
                self.add_neuron_group(group)
                self.debug(f"group {name!r} did not add itself to network")
            self.out.printf(group)
            self.out.hline()

    def new_synaptic_projections(self, *prepost, synclass, **specs):
        """
        Create a batch of synaptic projections of a particular class using the
        given keyword specs and add them to the network.
        """
        if not prepost:
            return
        self.out.hline()
        for pre, post in prepost:
            if type(pre) is str:
                if pre in self.neuron_groups:
                    pre = self.neuron_groups[pre]
                elif pre in self.input_groups:
                    pre = self.input_groups[pre]
            if type(post) is str:
                post = self.neuron_groups[post]
            synapses = synclass(pre, post, **specs)
            if synapses.name not in self.synapses:
                self.add_projection(synapses)
                self.debug(f"synapses {name!r} did not add itself to network")
            self.out.printf(synapses)
            self.out.hline()

    def set_neuron_values(self, *groups, **specs):
        """
        Set specs on all or a subset of neuron groups (by name or object).
        """
        if not groups:
            groups = tuple(self.neuron_groups.values())
        for group in groups:
            if type(group) is str:
                group = self.neuron_groups[group]
            for key, value in specs.items():
                setattr(group, key, value)

    def set_synapse_values(self, *synapses, **specs):
        """
        Set specs on all or a subset of synapses (by name or object).
        """
        if not synapses:
            synapses = tuple(self.synapses.values())
        for syn in synapses:
            if type(syn) is str:
                syn = self.synapses[syn]
            for key, value in specs.items():
                setattr(syn, key, value)

    def neuron_groups_where(self, *name_substrings, condn=None):
        """
        Iterate over neuron groups whose names contain any of the substrings.
        """
        for name, group in self.neuron_groups.items():
            for sub in name_substrings:
                if sub in name:
                    yield group
                    continue
            if condn and condn(group):
                yield group

    def synapses_where(self, *pre_substrings, condn=None):
        """
        Iterate over synapse objects whose presynaptic groups' names contain
        any of the substrings.
        """
        for name, syn in self.synapses.items():
            for sub in pre_substrings:
                if sub in syn.pre.name:
                    yield syn
                    continue
            if condn and condn(syn):
                yield syn

    def display_input_groups(self):
        """
        Print out detailed parameters for input groups.
        """
        for group in self.input_groups.values():
            self.out.printf(group)
            self.out.hline()

    def display_neuron_groups(self):
        """
        Print out detailed parameters for neuron groups.
        """
        for group in self.neuron_groups.values():
            self.out.printf(group)
            self.out.hline()

    def display_projections(self):
        """
        Print out detailed parameters and connection statistics for synapses.
        """
        for synapses in self.synapses.values():
            self.out.printf(synapses)
            synapses.print_stats()
            self.out.hline()

    def display_object_counts(self):
        """
        Display a list of object counts for the current network.
        """
        self.out.printf(dict_pprint(self.counts, name=self.name))

    def export_graphs(self, tag=None):
        """
        Save weighted connectivity graph matrixes to external data files.
        """
        fn = "{}-network".format(State.context._projname)
        if tag:
            fn += "+{}".format(sluggify(tag))
        savepath = State.context.path(fn, base="context", unique=True)
        data = {}

        self.out("Calculating resultant synaptic weight matrixes...")
        for name, syn in self.synapses.items():
            post_pre_name = f"{syn.post.name}_{syn.pre.name}"
            g_post = syn.post[f"g_{post_pre_name}"]
            W = 10**g_post * syn.g_peak * syn.S
            data[post_pre_name] = W

        self.out("Constructing omnibus network matrix...")
        groups = list(self.neuron_groups.values())
        W_omni = None
        for post in groups:
            W = None
            for pre in groups:
                g_name = f"g_{post.name}_{pre.name}"
                if g_name not in post.synapses:
                    W_ = np.zeros((post.N, pre.N))
                else:
                    syn = post.synapses[g_name]
                    W_ = 10 ** post[g_name] * syn.g_peak * syn.S
                    if syn.transmitter == "GABA":
                        W_ *= -1
                if W is None:
                    W = W_
                else:
                    W = np.hstack((W, W_))
            if W_omni is None:
                W_omni = W
            else:
                W_omni = np.vstack((W_omni, W))
        data["W_all"] = W_omni

        self.out(list(data.keys()), prefix="WeightMatrixes")
        self.out("Saving compressed weight matrix data...")
        np.savez_compressed(savepath, **data)

        self.out(savepath, prefix="NetworkExported")
