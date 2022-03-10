from abc import ABC, abstractmethod


class Domain:
    def __init__(self, values):
        self.values = tuple(values)

    def __hash__(self):
        return hash(self.values)

    def __eq__(self, other):
        return (
            self.__class__ == other.__class__ and
            self.values == other.values
        )


class RV:
    def __init__(self, domain, value=None):
        # e.g. binary vr, domain: (0, 1)
        self.domain = domain
        # example value: A = 0
        self.value = value
        self.nb = list()


class F:
    def __init__(self, table, nb):
        self.table = table
        self.nb = nb


class Graph:
    def __init__(self, rvs, fs):
        self.rvs = rvs
        self.fs = fs
        self.init_nb()

    def init_nb(self):
        for rv in self.rvs:
            rv.nb = list()
        for f in self.fs:
            for rv in f.nb:
                rv.nb.append(f)