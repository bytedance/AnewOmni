# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import operator

import ray

import utils.register as R

from .base import BaseFilter, FilterResult, FilterInput
from .pyrosetta_utils.functions import pr_relax, score_interface, init_rosetta


# map string operators to functions
OPS = {
    ">" : operator.gt,
    "<" : operator.lt,
    "=" : operator.eq,
    "==": operator.eq,
    ">=": operator.ge,
    "<=": operator.le,
}

def parse_threshold(expr: str):
    """Parse an expression like '>=0.55' -> (op_func, value)."""
    m = re.match(r"(>=|<=|>|<|=)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expr.strip())
    if not m:
        raise ValueError(f"Invalid threshold expression: {expr}")
    op_str, val_str = m.groups()
    return OPS[op_str], float(val_str)


def check_threshold(value: float, expr: str) -> bool:
    """Check if a value satisfies a threshold expression."""
    op, target = parse_threshold(expr)
    return op(value, target)


@R.register('RosettaFilter')
class RosettaFilter(BaseFilter):
    def __init__(self, thresholds: dict=None, need_relax: bool=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        Args:
            thresholds: dict, define the thresholds for rosetta metrics. Supported metrics:
                interface_sc: interface shape complementarity, range in [0, 1], the larger, the better, usually should above 0.55
                interface_dG: interface delta G, unbounded range, the lower, the better (normally should below zero)
                interface_dSASA: delta solvent accessible surface area, larger value means larger buried interface
                interface_dG_SASA_ratio: dG/dSASA, usually below zero and the lower, the better
                interface_hydrophobicity: interface hydrophobicity, range in [0, 100](%)
                interface_nres: number of residues on the interface
                interface_interface_hbonds: number of interface hydrogen bonds
                interface_hbond_percentage: percentage of hydrogen bonds compared to the number of residues

                supported operators: >, <, =, >=, <=

                e.g. {
                    'interface_sc': '>0.55',
                    'interface_hydrophobicity': '<50'
                }

        '''
        self.thresholds = {} if thresholds is None else thresholds
        self._parsed_ops = { key: parse_threshold(self.thresholds[key]) for key in self.thresholds }
        self.need_relax = need_relax

        init_rosetta()  # initialize pyrosetta only once

    @property
    def name(self):
        return self.__class__.__name__ + f'(thresholds={self.thresholds}, need_relax={self.need_relax})'
    
    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):

        pdb_path = input.path_prefix + '.pdb'
        if self.need_relax: relax_path = input.path_prefix + '_rosetta.pdb'
        else: relax_path = pdb_path
        if not os.path.exists(relax_path):
            try: pr_relax(pdb_path, relax_path)
            except Exception as e: return FilterResult.ERROR, { 'error': f'relax failed: {e}' }

        try: interface_scores, _, _ = score_interface(relax_path, tgt_chains=input.tgt_chains, lig_chains=input.lig_chains)
        except Exception as e: return FilterResult.ERROR, { 'error': f'scoring failed: {e}' }

        flag = True
        for key in self._parsed_ops:
            if key not in interface_scores:
                return FilterResult.ERROR, { 'error': f'metric {key} not supported in rosetta filter' }
            op, th = self._parsed_ops[key]
            val = interface_scores[key]
            if not op(val, th):
                flag = False
                break

        return FilterResult.PASSED if flag else FilterResult.FAILED, interface_scores


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    f = RosettaFilter({ 'interface_sc': '>0.55' })
    print(f.run(FilterInput(path, ['A', 'B'], ['H'], None, None, None, None)))