# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional

from utils import register as R

from .resample import ClusterResampler
from .base import BaseDataset, Summary
from .file_loader import MolType


@R.register('GeneralDataset')
class GeneralDataset(BaseDataset):

    def __init__(
            self,
            mmap_dir: str,
            specify_data: Optional[str] = None,
            specify_index: Optional[str] = None,
            cluster: Optional[str] = None,
            length_type: str = 'atom',
            in_memory: bool = False
        ) -> None:
        super().__init__(mmap_dir, specify_data, specify_index, in_memory)
        self.mmap_dir = mmap_dir
        self.resampler = ClusterResampler(cluster) if cluster else None  # should only be used in training!
        self.length_type = length_type

        self.dynamic_idxs = [i for i in range(len(self))]
        self.update_epoch() # should be called every epoch

    def update_epoch(self):
        if self.resampler is not None:
            self.dynamic_idxs = self.resampler(len(self))

    ########## Start of Overloading ##########

    @property
    def mol_type(self):
        return MolType.NUCLEIC

    def get_id(self, idx):
        idx = self.dynamic_idxs[idx]
        return self._indexes[idx][0]

    def get_len(self, idx):
        idx = self.dynamic_idxs[idx]
        props = self._properties[idx]
        if self.length_type == 'atom':
            return props['pocket_num_atoms'] + props['ligand_num_atoms']
        elif self.length_type == 'block':
            return props['pocket_num_blocks'] + props['ligand_num_blocks']
        else:
            raise NotImplementedError(f'length type {self.length_type} not recognized')

    def get_summary(self, idx: int): # when called from __getitem__, the index is already transformed
        _id = self._indexes[idx][0]
        props = self._properties[idx]

        # get indexes (pocket + peptide)
        pocket_block_ids = [(chain, tuple(block_id)) for chain, block_id in props['pocket_block_id']]
        cplx = self.get_raw_data(idx)
        lig_block_ids = []
        for c in props['ligand_chain_ids']:
            lig_block_ids.extend([(c, block.id) for block in cplx[c]])
        generate_mask = [0 for _ in pocket_block_ids] + [1 for _ in lig_block_ids]
        center_mask = [1 for _ in pocket_block_ids] + [0 for _ in lig_block_ids]

        return Summary(
            id=_id,
            ref_pdb=_id + '_ref.pdb',
            ref_seq='|'.join(props['ligand_sequences']), # peptide has only one chain
            target_chain_ids=props['target_chain_ids'],
            ligand_chain_ids=props['ligand_chain_ids'],
            select_indexes=pocket_block_ids + lig_block_ids,
            generate_mask=generate_mask,
            center_mask=center_mask
        )
    
    ########## End of Overloading ##########

    def __getitem__(self, idx: int):
        idx = self.dynamic_idxs[idx]
        data = super().__getitem__(idx)
        return data
    

if __name__ == '__main__':
    import sys
    dataset = GeneralDataset(sys.argv[1])
    print(dataset[0])
    print(len(dataset[0]['A']))