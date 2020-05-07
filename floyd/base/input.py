"""
Base group class for read-only inputs.
"""

from .groups import BaseUnitGroup


class BaseInputGroup(BaseUnitGroup):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def update(self):
        self.output = 0  # output variable indicates the signal 
                         # transmitted through the efferent projections