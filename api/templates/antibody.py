# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from typing import List

import utils.register as R
from data.bioparse.utils import recur_index
from data.bioparse.hierarchy import Atom, Block
from data.file_loader import MolType, _extract_antibody_masks, AntibodyLoader

from .base import BaseTemplate, ComplexDesc


@R.register('Antibody')
class Antibody(BaseTemplate):

    def __init__(self, cdr_type='HCDR3', fr_len=3, size_min=None, size_max=None):
        super().__init__(size_min, size_max)
        self.cdr_type = cdr_type # H/LCDR1/2/3
        self.fr_len = fr_len
    
    @property
    def moltype(self) -> MolType:
        return MolType.ANTIBODY
    
    def default_filter_configs(self) -> List[dict]:
        return super().default_filter_configs() + [ { 'class': 'ChainBreakFilter' }, { 'class': 'LTypeAAFilter' } ]

    def sample_size(self, cplx_desc):
        raise ValueError(f'Do not sample size for CDR loop')

    def remove_ref_lig(self, cplx_desc):
        # do not remove the antibody, because we need the framework
        return cplx_desc.cplx
    
    def add_dummy_lig(self, cplx_desc):
        cplx = cplx_desc.cplx

        generate_mask, center_mask, lig_block_ids = _extract_antibody_masks(
            cplx, cplx_desc.lig_chains, cplx_desc.pocket_block_ids, self.cdr_type, self.fr_len
        )

        # remove ground truth cdr
        for m, block_id in zip(generate_mask, cplx_desc.pocket_block_ids + lig_block_ids):
            if m == 0: continue
            block: Block = recur_index(cplx, block_id)
            block.name = 'GLY'
            block.atoms = [Atom(    # we should retain the atom ids to ensure correct chemical bonding
                name='C',
                coordinate=[0, 0, 0],
                element='C',
                id=atom.id
            ) for atom in block.atoms]

        # set desc here
        cplx_desc.generate_mask = generate_mask
        cplx_desc.center_mask = center_mask
        cplx_desc.lig_block_ids = lig_block_ids
        
        return cplx
    
    def set_desc_attr(self, cplx_desc):
        # already set above. do not do it again
        return cplx_desc
    
    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        loader = AntibodyLoader(
            tgt_chains=cplx_desc.tgt_chains,
            lig_chains=cplx_desc.lig_chains,
            cdr_type=self.cdr_type,
            fr_len=self.fr_len
        )
        return loader.cplx_to_data(
            cplx_desc.cplx,
            cplx_desc.pocket_block_ids,
            cplx_desc.lig_block_ids
        )
    

@R.register('AntibodyMultipleCDR')
class AntibodyMultipleCDR(BaseTemplate):

    '''
        This is a dummy template, where the processing logics will be handled
        by later logics specifically by decomposing this template into several
        single-CDR templates
    '''
    
    def __init__(self, cdr_types, length_ranges={}, fr_len=3, specify_regions={}):
        super().__init__(size_min=None, size_max=None)
        self.cdr_types = cdr_types # H/LCDR1/2/3
        self.fr_len = fr_len
        self.length_ranges = length_ranges
        self.specify_regions = specify_regions

    @property
    def moltype(self) -> MolType:
        return MolType.ANTIBODY
    
    def default_filter_configs(self) -> List[dict]:
        return super().default_filter_configs() + [ { 'class': 'ChainBreakFilter' }, { 'class': 'LTypeAAFilter' } ]