"""
Markdown tables of neuron group statistics.
"""

__all__ = ['MarkdownTable', 'TableMaker', ]


from tenko.base import TenkoObject
from .state import State


class MarkdownTable(object):

    """
    A Markdown table template for neuron group data.
    """

    def __init__(self, label, *columns, fmt='.5g', width=16):
        """
        Create an updating table based on (colname, func(group))-tuple columns.
        """
        self.label = label.lower()
        self.title = label.title()
        self.fmt = fmt
        self.columns = [name for name, _ in columns]
        self.funcs = [func for _, func in columns]
        self.ncols = len(columns)

        # Construct the header and header hline of the template
        spec = self.title.ljust(width)
        hline = '-'*(width + 1)
        for colname in self.columns:
            spec += '| ' + colname.ljust(width - 1)
            hline += '|' + '-'*width
        spec += '\n' + hline + '\n'

        # Add a row for each neuron group with format specifiers
        self.keys = []
        for group in State.network.neuron_groups:
            spec += '**{}**'.format(group.name).ljust(width)
            for i, colname in enumerate(columns):
                key = '{}_{}'.format(group.name, i)
                spec += ('| {' + '{}:{}'.format(key, fmt) + '}').ljust(width)
                self.keys.append(key)
            spec += '\n'
        self.spec = spec
        self.table = spec.format(**{k:0 for k in self.keys})

    def update(self):
        """
        Update the table based on the callback values for neuron groups.
        """
        data = {}
        for group in State.network.neuron_groups:
            for i, func in enumerate(self.funcs):
                key = '{}_{}'.format(group.name, i)
                data[key] = func(group)
        self.table = self.spec.format(**data)


class TableMaker(TenkoObject):

    """
    Auto-updating tables formatted by MarkdownTable objects.
    """

    def __init__(self):
        super().__init__()
        self.tables = {}
        State.tablemaker = self

    def __iter__(self):
        self.update()
        for name in self.tables.keys():
            yield (name, self.get(name))

    def register(self, table):
        """
        Register a MarkdownTable object for updating.
        """
        self.tables[table.label] = table
        self.out(f'{table.title}: {table.columns!r}', prefix='RegisteredTable')

    def update(self):
        """
        Update all registered tables.
        """
        [tbl.update() for tbl in self.tables.values()]

    def add_markdown_table(self, label, *columns, fmt='.5g', width=16):
        """
        Create a MarkdownTable and register it for updating.

        Note: All arguments are passed to the MarkdownTable constructor.
        """
        tbl = MarkdownTable(label, *columns, fmt=fmt, width=width)
        self.register(tbl)

    def get(self, label):
        """
        Get the current formatted table with the given label.
        """
        key = label.lower()
        if key in self.tables:
            return self.tables[key].table
        self.out(label, prefix='UnknownTable', warning=True)
