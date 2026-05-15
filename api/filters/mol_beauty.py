# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import ray
import subprocess
import sys
import tempfile

import utils.register as R

from .base import BaseFilter, FilterResult, FilterInput


def _run_mol_beauty(smi, gpu):
    bin_file = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'beauty', 'deploy_gnn_standalone.py')
    bin_file = os.path.abspath(bin_file)
    with tempfile.NamedTemporaryFile(suffix='.csv') as fout:
        proc = subprocess.run(
            [sys.executable, bin_file, "--smiles", smi, "--output-csv", fout.name],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"mol_beauty exited with code {proc.returncode}")
        with open(fout.name, 'r') as fin:
            lines = [line.strip() for line in fin.readlines() if line.strip()]
        if len(lines) < 2:
            raise RuntimeError(f"mol_beauty produced invalid csv: {lines}")
        score = float(lines[1].split(',')[1])
    return score


@R.register('MolBeautyFilter')
class MolBeautyFilter(BaseFilter):
    def __init__(self, th: float=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        Args:
            th: minimal threshold for molecules (scores are aligned with standard Gaussian, with the larger the better)
        '''
        self.th = th
    
    @property
    def name(self):
        return self.__class__.__name__ + f'(th={self.th})'
    
    @ray.remote(num_cpus=1, num_gpus=1/8)
    def run(self, input: FilterInput):
        try:
            score = _run_mol_beauty(input.smiles, ray.get_gpu_ids()[0])
            # score = _run_mol_beauty(input.smiles, 0)
        except Exception as e:
            return FilterResult.ERROR, { 'error': str(e) }
        flag = FilterResult.PASSED
        if self.th is not None and score < self.th: flag = FilterResult.FAILED
        return flag, { 'mol_beauty': score }



if __name__ == '__main__':
    import sys
    smiles = sys.argv[1]
    f = MolBeautyFilter()
    print(f.run(FilterInput(None, [], [], smiles, '', 0, 0)))
