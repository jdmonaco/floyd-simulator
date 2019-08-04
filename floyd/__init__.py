"""
floyd ... a practical and wise neuronal networks simulator.

created ... july twenty-sixth, two thousand nineteen
author ... joseph d. monaco

in memoriam.
"""

__version__ = "0.1.0"


from .state    import State
from .config   import Config
from .network  import Network
from .recorder import ModelRecorder
from .context  import SimulatorContext
from .neurons  import *
from .synapses import *
from .input    import *
from .delay    import *
from .noise    import *
from .simplot  import *
from .graph    import *
from .traces   import *
from .mdtables import *
from .layout   import *
from .samplers import *
from .kernels  import *


# Miscellaneous setup

import warnings as _warnings
from matplotlib import MatplotlibDeprecationWarning as _MPLdeprecate
_warnings.filterwarnings('ignore', category=_MPLdeprecate)
