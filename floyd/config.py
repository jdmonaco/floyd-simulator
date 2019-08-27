"""
Shared configuration for the floyd simulator.
"""

__all__ = ('Config',)


from roto.dicts import AttrDict
from tenko.state import Tenko


class FloydConfig(AttrDict):
    pass


Config = FloydConfig()


#
# Configuration values
#

Config.name           = 'floyd'
Config.figdpi         = Tenko.screendpi
Config.moviedpi       = Tenko.screendpi
Config.fps            = 59.94  # 29.97 for SD  # 24.976 for 'cinematic'
Config.compress       = 1.0  # simulation frames per video frame
Config.progress_width = 80   # number of characters in the progress bar


#
# Simulation parameter defaults
#

Config.title      = 'Network Simulation'
Config.tag        = None
Config.seed       = None
Config.duration   = 200.0
Config.dt         = 0.1
Config.dt_rec     = 10.0
Config.dt_block   = 25.0
Config.figw       = 9.0
Config.figh       = 9.0
Config.tracewin   = 100.0
Config.calcwin    = 25.0
Config.show_debug = False
Config.run_mode   = None


#
# Domain-specific parameter defaults
#

Config.g_LFP = 1e3  # nS, ~1 uS
