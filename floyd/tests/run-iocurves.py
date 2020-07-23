#!/usr/bin/env ipython --profile=sharpwaves -i

"""
Rebuilding the IOCurves test simulation to complete specify integration.
"""

from floyd.sim.iocurves import InputOutputCurves
from floyd import *

ioc = InputOutputCurves(desc='CA3 model', tag='adex update')

ioc.collect_data(
        tag         = 'exc6ns_inh6ns',
        duration    = 2e4,
        dt          = 0.2,
        dt_rec      = 0.2,
        calcwin     = 50.0,
        show_debug  = False,
        g_tonic_exc = 6.0,
        g_tonic_inh = 6.0,
        max_current = 3e3,
        N           = 220,
        N_pulses    = 47,
)
ioc.compute_curves(tag='no drive')
