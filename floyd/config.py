"""
Shared configuration for the floyd simulator.
"""

__all__ = ['Config']


from roto.dicts import AttrDict

from toolbox import IMACPRO_DPI


Config = AttrDict()


#
# Configuration values
#

Config.name = 'floyd'
Config.dpi = IMACPRO_DPI
Config.fps = 100.0
Config.progress_width = 80


#
# Neurophysiology
#

Config.g_LFP = 1e3  # nS, ~1 uS
