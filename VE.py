import random
from collections import defaultdict

import numpy as np
from Graph import F


class VE:
    def __init__(self, g, targets=None):
        self.g = g
        self.elimination_order = []#[rv for rv in g.rvs if rv not in targets]
        self.width = 0
        self.order = []
        self.clusters = []
        self.given_order = []
        self.given_order_names = []

    def eliminate_rv(self, target_rv):
        # assume we eliminate R.V. A, its neighbours are [B,C,D]
        # all neighbors of variable eliminated
        fs = target_rv.nb
        #
        nb_rvs = set().union(*[set(f.nb) for f in target_rv.nb])
        ## save cluster here
        self.clusters.append(list([rv.name for rv in nb_rvs]))

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
            rv = self.elimination_order.pop(0)
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
        # print("run min_degree")
        if len(self.g.rvs) == 0:
            print("No random variable exists.")
            return
        # find the minimum degree
        minDegree = min([self.get_degree(rv) for rv in self.g.rvs])
        # get a rv whose degree == minDegree
        rv_elimated = random.choice([rv for rv in self.g.rvs if self.get_degree(rv) == minDegree])
        # print("Eliminating the random variable: ", rv_elimated.name)
        self.order.append(rv_elimated.name)
        self.width = max(self.width, minDegree)
        # put one rv into elimination_order
        self.elimination_order.append(rv_elimated)

    def ve_min_degree(self):
        #print("running variable elimination using min degree")
        while self.g.rvs:
            self.min_degree()
            self.run()

        self.print_result()

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
                        #print((rvs[i].name, rvs[j].name))
                    else:
                        edges.append((rvs[j], rvs[i]))
                        #print((rvs[j].name, rvs[i].name))

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
        # print("run min_fill")
        if len(self.g.rvs) == 0:
            print("No random variable exists.")
            return
        # find the minimum degree
        minFill = min([self.get_fill(rv) for rv in self.g.rvs])
        # print("Min Fill number: ", minFill)
        # get a rv whose degree == minDegree
        rv_elimated = random.choice([rv for rv in self.g.rvs if self.get_fill(rv) == minFill])
        # print("Eliminating the random variable: ", rv_elimated.name)
        self.order.append(rv_elimated.name)
        self.width = max(self.width, self.get_degree(rv_elimated))
        # put one rv into elimination_order
        self.elimination_order.append(rv_elimated)

    def ve_min_fill(self):
        # print("Running variable elimination using min fill")
        while self.g.rvs:
            self.min_fill()
            self.run()

        self.print_result()

    def print_result(self):
        for f in self.g.fs:
            Z = f.table
        print("Width along the given order: ", self.width)
        # print("Z: ", Z)
        print("Exact log10(Z): ", np.log10(Z))
        # print("Eliminating order: ", self.order)

    def set_order(self, order_file):
        with open(order_file, 'r') as f:
            orders = f.readline().split()
            orders = [int(name) for name in orders]
            self.given_order_names = orders
            for vr_name in orders:
                vr = self.g.rv_name_to_rv(vr_name)
                self.given_order.append(vr)


    def run_use_given_order(self):
        # given an order, we eliminate R.V.s
        while self.given_order:
            rv = self.given_order.pop(0)
            #print("we eliminate: ", rv.name)
            self.eliminate_rv(rv)

    def find_wCutset(self, w):
        C = set()
        max_size = self.find_max_size()
        while(max_size > w+1):
            X = self.find_X()
            C = C.union({X})
            self.remove_x(X)
            max_size = self.find_max_size()
        return C


    def find_X(self):
        X = -1
        freq = defaultdict(int)
        for cluster in self.clusters:
            for v in cluster:
                freq[v] += 1

        max_freq = 0
        possible_x = []
        for key, value in freq.items():
            if value > max_freq:
                possible_x = []
                max_freq = value
                possible_x.append(key)
            elif value == max_freq:
                possible_x.append(key)
        print(possible_x)
        X = random.choice(possible_x)
        return X


    def remove_x(self, X):
        for c in self.clusters:
            if X in c:
                c.remove(X)
        print("updated cluster {} using {}".format(self.clusters, X))


    def find_max_size(self):
        size_clusters = [len(c) for c in self.clusters]
        max_size = max(size_clusters)
        return max_size



