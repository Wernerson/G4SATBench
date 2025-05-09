from g4satbench.data.data import construct_lcg
from torch.utils.data import IterableDataset
from itertools import zip_longest
import os
import re

class MultiSATDataset(IterableDataset):
    BATCHES_REGEX = re.compile("(batch (\d+): sat: (\d+), unsat: (\d+))")

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def _get_batches(self):
        batch_file = os.path.join(self.data_dir, "batches.txt")
        with open(batch_file, "r") as f:
            for m in MultiSATDataset.BATCHES_REGEX.findall(f.read()):
                yield int(m[1]), int(m[2]), int(m[3])

    def __len__(self):
        total = 0
        for _, sat, unsat in self._get_batches():
            total += sat + unsat
        return total

    def _problems_in_file(self, file_path, sat):
        with open(file_path, 'r') as file:
            n = None
            m = None
            clauses = []
            for line in file:
                if line.startswith("c"):
                    continue
                elif line.startswith("p cnf"):
                    if len(clauses):
                        data = construct_lcg(n, clauses)
                        data.y = sat
                        yield data
                        clauses = []
                    n, m = map(int, line.split()[2:])
                else:
                    clauses.append(list(map(int, line.split()[:-1])))

    def __iter__(self):
        for batch, sat, unsat in self._get_batches():
            sat_file = os.path.join(self.data_dir, "sat", f"{batch}.cnf")
            unsat_file = os.path.join(self.data_dir, "unsat", f"{batch}.cnf")
            sat_gen = self._problems_in_file(sat_file, True)
            unsat_gen = self._problems_in_file(unsat_file, False)
            for (sat, unsat) in zip_longest(sat_gen, unsat_gen):
                if sat is not None:
                    yield sat
                if unsat is not None:
                    yield unsat