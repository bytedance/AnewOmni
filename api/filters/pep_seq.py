# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import ray

from Bio.SeqUtils.ProtParam import ProteinAnalysis

import utils.register as R

from .base import BaseFilter, FilterResult, FilterInput


@R.register('AvoidAAFilter')
class AvoidAAFilter(BaseFilter):
    def __init__(self, avoid_aas: list, skip_positions: list=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        Args:
            avoid_aa: list of amino acids to avoid (e.g. ['C', 'G'])
            skip_positions: list of positions to skip, started from 0 (e.g. [0, 4, 7])
        '''
        self.avoid_aas = avoid_aas
        self.skip_positions = {} if skip_positions is None else { pos: True for pos in skip_positions }

    @property
    def name(self):
        return self.__class__.__name__ + f'(avoid_aas={self.avoid_aas}, skip_positions={self.skip_positions})'
    
    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):

        seq_dict = {}
        for pos, aa in enumerate(input.seq):
            if pos in self.skip_positions: continue
            seq_dict[aa] = True
        
        for aa in self.avoid_aas:
            if aa in seq_dict: return FilterResult.FAILED, {}
        return FilterResult.PASSED, {}
    

@R.register('GRAVYFilter')
class GRAVYFilter(BaseFilter):
    def __init__(self, th: float=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # hydrophobicity (GRAVY) below certain threshold
        # about 75% natural peptides have a GRAVY below zero, otherwise it might be hard to synthesize
        self.th = th

    @property
    def name(self):
        return self.__class__.__name__ + f'(th={self.th})'
    
    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        x = ProteinAnalysis(input.seq)
        gravy = x.gravy()
        stats = { 'GRAVY': gravy }
        if gravy > self.th: return FilterResult.FAILED, stats
        else: return FilterResult.PASSED, stats