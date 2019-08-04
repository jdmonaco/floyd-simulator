"""
Global state for simulations.
"""

__all__ = ['State', 'RunMode']


from enum import Enum

from roto.dicts import AttrDict


class RunMode(Enum):
    ANIMATE  = 'create movies'
    INTERACT = 'launch dashboards'
    RECORD   = 'collect data'



class SharedState(AttrDict):
    def reset():
        for key in self.keys():
            self[key] = None


State = SharedState()


#
# Initial values
#

State.n         = 0
State.t         = 0.0
State.dt        = 0.25
State.duration  = 100.0
State.dt_rec    = 10.0
State.dt_block  = 50.0
State.blocksize = 200.0
State.tracewin  = 100.0
State.calcwin   = 25.0
State.figw      = 8
State.figh      = 6
State.figdpi    = 144
State.run_mode  = RunMode.INTERACT
