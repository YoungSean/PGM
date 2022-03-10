import numpy as np
from Graph import F


class VE:
    def __init__(self, g, targets):
        self.g = g
        self.elimination_order = [rv for rv in g.rvs if rv not in targets]

    def eliminate_rv(self, target_rv):
        fs = target_rv.nb

        nb_rvs = set().union(*[set(f.nb) for f in target_rv.nb])
        nb_rvs.remove(target_rv)

        nb_rvs = list(nb_rvs)
        rvs = [target_rv] + nb_rvs
        rv_idx = {rv: i for i, rv in enumerate(rvs)}

        combined_table = np.ones([1] * len(rvs))

        for f in fs:
            self.g.fs.remove(f)

            table = f.table
            table = np.expand_dims(table, list(range(table.ndim, len(rvs))))
            old_idx = [i for i, rv in enumerate(f.nb)]
            new_idx = [rv_idx[rv] for rv in f.nb]
            table = np.moveaxis(table, old_idx, new_idx)

            combined_table = combined_table * table

        new_table = combined_table.sum(axis=0)
        new_f = F(new_table, nb_rvs)

        self.g.fs.add(new_f)
        self.g.rvs.remove(target_rv)
        self.g.init_nb()

    def run(self):
        while self.elimination_order:
            rv = self.elimination_order.pop()
            self.eliminate_rv(rv)

    def prob(self, rv):
        combined_table = np.ones(1)
        for f in rv.nb:
            combined_table = combined_table * f.table
        return combined_table

    def print_fs(self):
        print('#######################')
        for f in self.g.fs:
            print(f.nb)
            print(f.table)
