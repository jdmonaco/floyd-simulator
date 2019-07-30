"""
Shared configuration for the floyd simulator.
"""

__all__ = ['Config']


from roto.dicts import AttrDict

from toolbox import IMACPRO_DPI, LG_ULTRAWIDE_DPI


Config = AttrDict()


#
# Configuration values
#

Config.name = 'floyd'
Config.figdpi = IMACPRO_DPI
Config.moviedpi = LG_ULTRAWIDE_DPI
Config.screendpi = LG_ULTRAWIDE_DPI
Config.fps = 100.0
Config.progress_width = 80


#
# Simulation parameter defaults
#

Config.title = 'Network Simulation'
Config.rnd_seed = None
Config.duration = 200.0
Config.dt = 0.1
Config.dt_rec = 10.0
Config.dt_block = 25.0
Config.figw = 12.0
Config.figh = 9.0
Config.tracewin = 100.0
Config.calcwin = 25.0
Config.interact = True
Config.debug = False

#
# Neurophysiology
#

Config.g_LFP = 1e3  # nS, ~1 uS
