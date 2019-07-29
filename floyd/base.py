"""
Base object class for model components.
"""

import pdb

from pouty.console import ConsolePrinter


class FloydObject(object):

    def __init__(self, name=None):
        self.klass = prefix = self.__class__.__name__
        if name is not None:
            self.name = name
            prefix = f'{name}{prefix}'
        self.out = ConsolePrinter(prefix=prefix)
        self.debug = lambda *msg: self.out(*msg, debug=True)

    def __bp__(self, when=None):
        if when is not None and when == False:
            return
        pdb.set_trace()
