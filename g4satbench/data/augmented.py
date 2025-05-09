import copy
import random
from itertools import zip_longest

from torch.utils.data import Dataset

from g4satbench.data.data import construct_lcg
from g4satbench.data.sr import SRDataset


class SRAugmentedDataset(SRDataset, Dataset):
    def __init__(self, nr_gen_instances, nr_augm_per_instance, min_n, max_n, p_k_2=0.3, p_geo=0.4):
        super().__init__(nr_gen_instances, min_n, max_n, p_k_2, p_geo)
        self.nr_augm_per_instance = nr_augm_per_instance

    def __len__(self):
        return 2 * self.nr_gen_instances * (self.nr_augm_per_instance+1)

    def __iter__(self):
        for i in range(self.nr_gen_instances):
            n_vars, unsat, sat = self._generate()

            unsat_data = construct_lcg(n_vars, unsat)
            unsat_data.y = 0.
            yield unsat_data
            sat_data = construct_lcg(n_vars, sat)
            sat_data.y = 0.
            yield sat_data

            sat_gen = self._augment_instance(n_vars, sat, 1.)
            unsat_gen = self._augment_instance(n_vars, unsat, 0.)
            for (sat, unsat) in zip_longest(sat_gen, unsat_gen):
                if sat is not None:
                    yield sat
                if unsat is not None:
                    yield unsat

    def _augment_instance(self, n_vars, clauses, sat):
        for i in range(self.nr_augm_per_instance):
            new_n_vars, new_clauses = self._augment_formula(n_vars, clauses, sat)
            data = construct_lcg(new_n_vars, new_clauses)
            data.y = sat
            yield data

    def _augment_formula(self, n_vars, clauses, sat):
        clauses = copy.deepcopy(clauses)

        # simple permutations
        clauses = self._permutate(n_vars, clauses)

        # label preserving augmentations
        n_vars, clauses = self._label_preserving_augment(n_vars, clauses, sat)

        return n_vars, clauses

    def _permutate(self, n_vars, clauses):
        if random.random() < 0.5:
            # negate all variables
            for clause in clauses:
                for l in range(len(clause)):
                    clause[l] = -clause[l]
        else:
            # negate one variable
            var = random.randint(1, n_vars)
            for clause in clauses:
                for i in range(len(clause)):
                    if abs(clause[i]) == var:
                        clause[i] = -clause[i]

        return clauses

    def _label_preserving_augment(self, n_vars, clauses, sat):
        if sat:
            # remove a clause
            idx = random.randint(0, len(clauses) - 1)
            clauses.pop(idx)
        else:
            # add a clause
            clause = []
            for _ in range(3):
                var = random.randint(1, n_vars)
                clause.append(var)
            clause = list(map(lambda v: -v if random.random() < 0.5 else v, set(clause)))
            clauses.append(clause)

        return n_vars, clauses
