import random

import numpy as np
from Graph import F


class VE:
    def __init__(self, g, targets=None):
        self.g = g
        self.elimination_order = []#[rv for rv in g.rvs if rv not in targets]

    def eliminate_rv(self, target_rv):
        # assume we eliminate R.V. A, its neighbours are [B,C,D]
        # all neighbors of variable eliminated
        fs = target_rv.nb
        #
        nb_rvs = set().union(*[set(f.nb) for f in target_rv.nb])
        nb_rvs.remove(target_rv)
        # nb_rvs = [B,C,D]
        nb_rvs = list(nb_rvs)
        # let the variable A be the first element in rvs
        # rvs = [A, B, C, D]
        rvs = [target_rv] + nb_rvs
        # assign rv with idx
        rv_idx = {rv: i for i, rv in enumerate(rvs)}
        # initialize the factor with the scope of rvs
        combined_table = np.ones([1] * len(rvs))
        # factor products
        for f in fs:
            self.g.fs.remove(f)

            table = f.table
            table = np.expand_dims(table, list(range(table.ndim, len(rvs))))
            old_idx = [i for i, rv in enumerate(f.nb)]
            new_idx = [rv_idx[rv] for rv in f.nb]
            table = np.moveaxis(table, old_idx, new_idx)

            combined_table = combined_table * table
        # summing out variable A to get a new factor
        new_table = combined_table.sum(axis=0)
        new_f = F(new_table, nb_rvs)
        # keep new factor
        # remove the R.V. A from rvs of graphs
        self.g.fs.add(new_f)
        self.g.rvs.remove(target_rv)
        # update the neighbor lists of random variables
        self.g.init_nb()

    def run(self):
        # given an order, we eliminate R.V.s
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
            print([rv.name for rv in f.nb])
            print(f.table)

    def get_degree(self, rv):
        rvs = set().union(*[set(f.nb) for f in rv.nb])
        return len(rvs) - 1


    def min_degree(self):
        print("run min_degree")
        if len(self.g.rvs) == 0:
            print("No random variable exists.")
            return
        # find the minimum degree
        minDegree = min([self.get_degree(rv) for rv in self.g.rvs])
        # get a rv whose degree == minDegree
        rv_elimated = random.choice([rv for rv in self.g.rvs if self.get_degree(rv) == minDegree])
        print("Eliminating the random variable: ", rv_elimated.name)
        # put one rv into elimination_order
        self.elimination_order.append(rv_elimated)

    def ve_min_degree(self):
        print("run ve min degree")
        while self.g.rvs:
            self.min_degree()
            self.run()

    """Min Fill"""
    def get_edge_from_factor(self, factor):
        rvs = factor.nb
        edges = []
        N = len(rvs)
        if N < 2:
            return edges
        else:
            for i in range(N-1):
                for j in range(i+1, N):
                    if rvs[i].name < rvs[j].name:
                        edges.append((rvs[i], rvs[j]))
                        print((rvs[i].name, rvs[j].name))
                    else:
                        edges.append((rvs[j], rvs[i]))
                        print((rvs[j].name, rvs[i].name))

        return edges

    def get_fill(self, rv):
        factors = rv.nb
        total_edges = []
        for f in factors:
            edges = self.get_edge_from_factor(f)
            # print("edges from a factor: ", edges)
            total_edges = total_edges + edges
        #print("total_edges: ", total_edges)
        total_edges = set(total_edges)
        current_num_edges = len(total_edges)
        n = self.get_degree(rv) + 1
        num_clique_edges = n * (n-1) / 2
        num_fill = num_clique_edges - current_num_edges
        return num_fill

    def min_fill(self):
        print("run min_fill")
        if len(self.g.rvs) == 0:
            print("No random variable exists.")
            return
        # find the minimum degree
        minFill = min([self.get_fill(rv) for rv in self.g.rvs])
        print("Min Fill number: ", minFill)
        # get a rv whose degree == minDegree
        rv_elimated = random.choice([rv for rv in self.g.rvs if self.get_fill(rv) == minFill])
        print("Eliminating the random variable: ", rv_elimated.name)
        # put one rv into elimination_order
        self.elimination_order.append(rv_elimated)

    def ve_min_fill(self):
        print("run ve min fill")
        while self.g.rvs:
            self.min_fill()
            self.run()