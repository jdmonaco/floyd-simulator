"""
Base object class for model components.
"""

import pdb

from pouty.console import ConsolePrinter


class FloydObject(object):

    def __init__(self):
        self.klass = self.__class__.__name__
        self.out = ConsolePrinter(prefix=self.klass)
        self.debug = lambda *msg: self.out(*msg, debug=True)

    def __bp__(self, when=None):
        if when is not None and when == False:
            return
        pdb.set_trace()
