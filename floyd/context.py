"""
Base simulator context.
"""

try:
    import simplejson as json
except ImportError:
    import json

try:
    import panel as pn
except ImportError:
    print('Warning: install `panel` to use interactive dashboards.')

from numpy import ndarray

from tenko.context import AbstractBaseContext
from maps.geometry import EnvironmentGeometry

from .spec import paramspec
from .network import Network
from .state import State, reset_state


class SimulatorContext(AbstractBaseContext):

    """
    Main base class for simulations.
    """

    def set_simulation_parameters(self, params, **simparams):
        """
        Set simulation parameters in global scope and shared state.
        """
        # Set the defaults to a paramspec attribute and log differences
        SimSpec = paramspec(f'{self.__class__.__name__}SimSpec', **simparams)
        self.psim = SimSpec()
        self.out('Simulation parameters:')
        for name, value in p.items():
            logmsg = '- {} = {}'.format(name, value)
            if name in params:
                p[name] = params.pop(name)
                logmsg = '* {} = {} [default: {}]'.format(name, p[name], value)
            self.out(logmsg, hideprefix=True)

        # Update global scope with simulation parameters
        self.get_global_scope().update(p)

        # Clear and update the shared state, then write out a JSON file
        reset_state()
        State.update(p)
        State.context = self
        self.write_json(p, 'simulation')

        # Initialize the 'magic' network object
        self.out('Initializing network...')
        State.network = Network()

        # Display a green AnyBar dot to signal the simulation is running
        State.context.launch_anybar()
        State.context.set_anybar_color('green')

    def set_model_parameters(self, params, pfile=None, **defaults):
        """
        Set model parameters according to file, keywords, or defaults.

        Note: It is required to specify default values for all parameters here.
        """
        # Set the defaults to a paramspec attribute and write to file
        ModelSpec = paramspec(f'{self.__class__.__name__}Spec', **defaults)
        self.p = ModelSpec()
        self.write_json(self.p.as_dict(), 'defaults.json', base='context')

        # Import values from parameters file if specified
        if pfile is not None:
            fpath, fparams = self.get_json(pfile)
            fparams.update(params)
            params = fparams
            self.out(fpath, prefix='ParameterFile')

        # Write file with effective parameters (i.e., kwargs > file > defaults)
        self.p.update(params)  # defaults are auto-saved
        self.write_json(self.p.as_dict(), self.filename(stem='params'))

        # Set parameters as global variables in the object's module scope
        self.out('Model parameters:')
        self.get_global_scope().update(self.p)
        for name, value in self.p.items():
            logstr = f'- {name} = {value}'
            dflt = self.p.defaults[name]
            if value != dflt:
                logstr = f'* {name} = {value} [default: {dflt}]'
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
