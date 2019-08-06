"""
floyd ... a practical and wise neuronal networks simulator.

created ... july twenty-sixth, two thousand nineteen
author ... joseph d. monaco

in memoriam.
"""

VERSION = "0.2.1"


from .state    import *
from .config   import *
from .network  import *
from .recorder import *
from .context  import *
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
