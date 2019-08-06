"""
Subpackage for simulations defined within floyd itself.
"""

VERSION = '0.2.1'


import os

import toolbox

from ..context import SimulatorContext, simulate, step
from ..config import Config


#
# Set up floyd as its own simulation context for test models, etc.
#

PROJNAME = Config.name
PROJDIR  = os.path.join(toolbox.PROJDIR, PROJNAME)
DATADIR  = os.path.join(PROJDIR, 'data')
REPODIR  = os.path.split(__file__)[0]
RESDIR   = os.path.join(PROJDIR, 'results')


class FloydContext(SimulatorContext):
    _projname = PROJNAME
    _version  = VERSION
    _rootdir  = PROJDIR
    _datadir  = DATADIR
    _repodir  = REPODIR
    _resdir   = RESDIR
    _logcolor = 'pink'
