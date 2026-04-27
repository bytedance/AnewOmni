# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import random
from typing import Optional, List

import numpy as np

from utils import register as R

from .resample import ClusterResampler
from .base import BaseDataset, Summary
from .file_loader import MolType


@R.register('AntibodyDataset')
class AntibodyDataset(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            # cluster: Optional[str] = None,
            length_type: str = 'atom',
            cdr_type: List[str] = ['HCDR1', 'HCDR2', 'HCDR3', 'LCDR1', 'LCDR2', 'LCDR3'],
            cdr_type_sample_probs: dict = {},
            test_mode: bool = False, # extend all CDRs
            in_memory: bool = False
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index, in_memory)
        self.mmap_dir = mmap_dir
        # self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
        self.length_type = length_type
        self.cdr_type_sample_probs = cdr_type_sample_probs
        if len(cdr_type_sample_probs) != 0: assert len(cdr_type_sample_probs) == 6, f'Only {len(cdr_type_sample_probs)} CDRs have their probability defined'
        self.test_mode = test_mode

        self.idx_tup = []
        if test_mode:       
            for idx, prop in enumerate(self._properties):
                if 1 in prop['heavy_model_mark'] and 'HCDR1' in cdr_type: self.idx_tup.append((idx, 'HCDR1'))
                if 2 in prop['heavy_model_mark'] and 'HCDR2' in cdr_type: self.idx_tup.append((idx, 'HCDR2'))
                if 3 in prop['heavy_model_mark'] and 'HCDR3' in cdr_type: self.idx_tup.append((idx, 'HCDR3'))
                if 1 in prop['light_model_mark'] and 'LCDR1' in cdr_type: self.idx_tup.append((idx, 'LCDR1'))
                if 2 in prop['light_model_mark'] and 'LCDR2' in cdr_type: self.idx_tup.append((idx, 'LCDR2'))
                if 3 in prop['light_model_mark'] and 'LCDR3' in cdr_type: self.idx_tup.append((idx, 'LCDR3'))
        else:
            for idx, prop in enumerate(self._properties):
                flag = False
                if 1 in prop['heavy_model_mark'] and 'HCDR1' in cdr_type: flag = True 
                if 2 in prop['heavy_model_mark'] and 'HCDR2' in cdr_type: flag = True 
                if 3 in prop['heavy_model_mark'] and 'HCDR3' in cdr_type: flag = True 
                if 1 in prop['light_model_mark'] and 'LCDR1' in cdr_type: flag = True 
                if 2 in prop['light_model_mark'] and 'LCDR2' in cdr_type: flag = True 
                if 3 in prop['light_model_mark'] and 'LCDR3' in cdr_type: flag = True 
                if flag: self.idx_tup.append((idx, None))

        self.cdr_type = cdr_type

    ########## Start of Overloading ##########
    @property
    def mol_type(self):
        return MolType.ANTIBODY

    def __len__(self):
        return len(self.idx_tup)

    def get_len(self, idx):
        props = self._properties[self.idx_tup[idx][0]]
        if self.length_type == 'atom':
            return props['epitope_num_atoms'] + props['ligand_num_atoms']
        elif self.length_type == 'block':
            return props['epitope_num_blocks'] + props['ligand_num_blocks']
        else:
            raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_raw_data(self, idx):
        idx, _ = self.idx_tup[idx]
        return super().get_raw_data(idx)

    def get_summary(self, idx: int): # when called from __getitem__, the index is already transformed
        idx, cdr = self.idx_tup[idx]
        props = self._properties[idx]
        _id = self._indexes[idx][0]

        if cdr is None:
            assert not self.test_mode
            # randomly sample one available CDR
            choices = []
            for i in range(1, 4):
                if i in props['heavy_model_mark']: choices.append(f'HCDR{i}')
            for i in range(1, 4):
                if i in props['light_model_mark']: choices.append(f'LCDR{i}')
            if len(self.cdr_type_sample_probs) == 0: # unified distribution
                cdr = random.choice(list(set(choices).intersection(set(self.cdr_type))))
            else:
                choices = set(choices).intersection(set(self.cdr_type))
                probs = [self.cdr_type_sample_probs[cdr_type] for cdr_type in choices]
                probs = [p / sum(probs) for p in probs] # normalize
                cdr = np.random.choice(list(choices), size=1, p=probs)[0].item()

        # get indexes (pocket + peptide)
        epitope_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['epitope_block_id']]
        hchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['heavy_model_block_id']]
        lchain_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['light_model_block_id']]

        generate_mask = [0 for _ in epitope_block_ids]
        for m in props['heavy_model_mark']:
            if cdr.startswith('H') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)
        for m in props['light_model_mark']:
            if cdr.startswith('L') and m == int(cdr[-1]): generate_mask.append(1)
            else: generate_mask.append(0)

        # centering at the medium of two ends
        center_mask = [0 for _ in generate_mask]
        for i in range(len(center_mask)):
            if i + 1 < len(generate_mask) and generate_mask[i + 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1 # left end
            elif i - 1 > 0 and generate_mask[i - 1] == 1 and generate_mask[i] == 0:
                center_mask[i] = 1

        ref_seq = props['heavy_chain_sequence'] if cdr.startswith('H') else props['light_chain_sequence']
        mark = props['heavy_chain_mark'] if cdr.startswith('H') else props['light_chain_mark']

        start, end = mark.index(cdr[-1]), mark.rindex(cdr[-1])
        ref_seq = ref_seq[start:end + 1]

        return Summary(
            id=_id + '/' + cdr,
            ref_pdb=_id + '_ref.pdb',
            ref_seq=ref_seq, # the selected CDR
            target_chain_ids=props['target_chain_ids'],
            ligand_chain_ids=props['ligand_chain_ids'],
            select_indexes=epitope_block_ids + hchain_block_ids + lchain_block_ids,
            generate_mask=generate_mask,
            center_mask=center_mask
        )
    
    ########## End of Overloading ##########

    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        if len(item['bonds']) == 0: print(self.get_summary(idx))
        return item
    

if __name__ == '__main__':
    import sys
    dataset = AntibodyDataset(sys.argv[1], specify_index=sys.argv[2])
    print(dataset[0])
    print(len(dataset[0]['position_ids']))