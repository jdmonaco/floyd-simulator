"""
Subpackage for simulations defined within floyd itself.
"""

__version__ = '0.1.0'


import os

import toolbox
from tenko.context import step

from ..context import SimulatorContext
from ..config import Config


#
# Set up floyd as its own simulation context for test models, etc.
#

__repodir__ = os.path.split(__file__)[0]
__projname__ = Config.name
__projdir__ = os.path.join(toolbox.PROJDIR, __projname__)
__resdir__ = os.path.join(__projdir__, 'results')
__datadir__ = os.path.join(__projdir__, 'data')

FloydContext = SimulatorContext.factory(f'{__projname__.title()}Context',
    __projname__, __version__, __projdir__, __datadir__, __repodir__,
    __resdir__, logcolor='ochre')
