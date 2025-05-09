import copy
import os
import random

import numpy as np
import networkx as nx
from pysat.solvers import Cadical195 as Cadical
from torch.utils.data import IterableDataset
from g4satbench.utils.utils import VIG, clean_clauses, hash_clauses
from g4satbench.data.data import construct_lcg


class SRDataset(IterableDataset):
    def __init__(self, nr_gen_instances, min_n, max_n, p_k_2=0.3, p_geo=0.4):
        self.nr_gen_instances = nr_gen_instances
        self.min_n = min_n
        self.max_n = max_n
        self.p_k_2 = p_k_2
        self.p_geo = p_geo
        self.hash_list = []

    def __len__(self):
        return 2 * self.nr_gen_instances

    def __iter__(self):
        for i in range(self.nr_gen_instances):
            n_vars, unsat, sat = self._generate()
            sat_data = construct_lcg(n_vars, sat)
            sat_data.y = 1.
            yield sat_data
            unsat_data = construct_lcg(n_vars, unsat)
            unsat_data.y = 0.
            yield unsat_data

    def _generate(self):
        """Copied from sr.py"""
        while True:
            n_vars = random.randint(self.min_n, self.max_n)
            solver = Cadical()
            clauses = []
            while True:
                # randomly choose k
                k_base = 1 if random.random() < self.p_k_2 else 2
                k = k_base + np.random.geometric(self.p_geo)

                # randomly choose k literals without replacement
                vs = np.random.choice(n_vars, size=min(n_vars, k), replace=False)
                clause = [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]

                solver.add_clause(clause)
                if solver.solve():
                    clauses.append(clause)
                else:
                    break

            unsat_clause = clause
            # flip the first literal in the last clause
            sat_clause = [-clause[0]] + clause[1:]

            clauses.append(unsat_clause)

            # ensure the graph in connected
            vig = VIG(n_vars, clauses)
            if not nx.is_connected(vig):
                continue

            # remove duplicate instances
            clauses = clean_clauses(clauses)
            h = hash_clauses(clauses)
            if h not in self.hash_list:
                self.hash_list.append(h)
                break

        sat_clauses = copy.deepcopy(clauses)
        clauses[-1] = sat_clause
        return n_vars, clauses, sat_clauses
