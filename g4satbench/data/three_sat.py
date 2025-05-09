import copy
import random

import numpy as np
import networkx as nx
from pysat.solvers import Cadical195 as Cadical
from torch.utils.data import IterableDataset
from g4satbench.utils.utils import VIG, clean_clauses, hash_clauses
from g4satbench.data.data import construct_lcg


class ThreeSatDataset(IterableDataset):
    def __init__(self, nr_gen_instances, min_n, max_n, min_m, max_m, min_l, max_l):
        self.nr_gen_instances = nr_gen_instances
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m
        self.min_l = min_l
        self.max_l = max_l
        self.hash_list = []

    def __len__(self):
        return 2 * self.nr_gen_instances

    def __iter__(self):
        for i in range(self.nr_gen_instances):
            n_vars, formula, sat = self._generate()
            data = construct_lcg(n_vars, formula)
            data.y = 1. if sat else 0.
            yield data

    def _generate(self):
        n = random.randint(self.min_n, self.max_n)
        m = random.randint(self.min_m, self.max_m)
        l = random.randint(self.min_l, self.max_l)
        solver = Cadical()
        clauses = []
        for i in range(m):
            clause = []
            for j in range(l):
                v = random.randint(1, n)
                if random.random() < 0.5:
                    clause.append(v)
                else:
                    clause.append(-v)
            solver.add_clause(clause)
            clauses.append(clause)
        sat = solver.solve()
        return n, clauses, sat


if __name__ == '__main__':
    dataset = ThreeSatDataset(1000, 4, 8, 15, 25, 2, 5)
    sat = 0
    unsat = 0
    for data in dataset:
        sat += data.y
        unsat += (1. - data.y)
    print(sat, unsat, sat / (sat + unsat))