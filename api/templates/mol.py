# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List
from copy import deepcopy

import torch
import numpy as np

from data.file_loader import MolType, MolLoader
from data.bioparse.utils import recur_index
from data.bioparse.parser.sdf_to_complex import sdf_to_complex
from data.resample import SizeSamplerByPocketSpace
from data.bioparse.hierarchy import Atom, Block, Bond
import utils.register as R
from models.modules.adapter.model import ConditionConfig

from .base import BaseTemplate, ComplexDesc


@R.register('Molecule')
class Molecule(BaseTemplate):

    def __init__(self, size_min=None, size_max=None):
        super().__init__(size_min, size_max)
        self.size_sampler = SizeSamplerByPocketSpace(size_min, size_max)
    
    @property
    def moltype(self) -> MolType:
        return MolType.MOLECULE
    
    def default_filter_configs(self) -> List[dict]:
        return super().default_filter_configs() + [ { 'class': 'MolSMARTSFilter' } ]

    def sample_size(self, cplx_desc):
        pocket_pos = []
        for _id in cplx_desc.pocket_block_ids:
            block = recur_index(cplx_desc.cplx, _id)
            for atom in block: pocket_pos.append(atom.coordinate)
        
        num_blocks = self.size_sampler(1, pocket_pos)[0]

        return num_blocks
    
    def dummy_lig_block_bonds(self, cplx_desc: ComplexDesc, size: int) -> List[Block]:
        blocks = [Block(
            name='UNK',
            atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
            id=(1, str(i))  # the same residue
        ) for i in range(size)]
        bonds = []
        return blocks, bonds

    def validate(self, cplx_desc):
        return True

    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        loader = MolLoader(
            tgt_chains=cplx_desc.tgt_chains,
            lig_chains=cplx_desc.lig_chains,
        )
        return loader.cplx_to_data(
            cplx_desc.cplx,
            cplx_desc.pocket_block_ids,
            cplx_desc.lig_block_ids
        )


@R.register('MoleculeResample')
class MoleculeResample(Molecule):
    def __init__(self, geom_tol_ratio = 0.001):
        super().__init__(size_min=None, size_max=None, geom_tol_ratio = geom_tol_ratio)

    ########## Override Basic Functions to Maintain the Original Ligand ##########
    @property
    def resample_mode(self):
        return True

    def remove_ref_lig(self, cplx_desc):
        return cplx_desc.cplx

    def add_dummy_lig(self, cplx_desc):
        return cplx_desc.cplx


@R.register('MoleculeGrow')
class MoleculeGrow(Molecule):
    def __init__(self, fragment_sdf, add_block_size_min, add_block_size_max, block_centers = None, block_center_stds = None, w = 1.0, geom_tol_ratio = 0.001):
        super().__init__(size_min=None, size_max=None, geom_tol_ratio = geom_tol_ratio)
        self.add_block_size_min = add_block_size_min
        self.add_block_size_max = add_block_size_max
        self.block_centers = [] if block_centers is None else block_centers
        self.block_center_stds = [1.0 for _ in self.block_centers] if block_center_stds is None else block_center_stds
        assert len(self.block_centers) < self.add_block_size_max, f'The number of specified block centers ({len(self.block_centers)}) should be less than (<) add_block_size_max ({self.add_block_size_max})'
        assert len(self.block_centers) == len(self.block_center_stds), f'The number of specified block centers ({len(self.block_centers)}) should be equal to the number of specified block center stds ({len(self.block_center_stds)})'
        self.w = w
        self.wt_fragments = sdf_to_complex(fragment_sdf)

    def sample_size(self, cplx_desc: ComplexDesc):
        return np.random.randint(self.add_block_size_min, self.add_block_size_max)

    def dummy_lig_block_bonds(self, cplx_desc: ComplexDesc, size: int) -> List[Block]:
        blocks = deepcopy(self.wt_fragments[0].blocks)
        for block in blocks: block.properties = {}  # delete original smiles
        bonds = []
        for bond in self.wt_fragments.bonds:
            bonds.append(Bond(
                index1=(bond.index1[0] + len(cplx_desc.cplx), bond.index1[1], bond.index1[2]),
                index2=(bond.index2[0] + len(cplx_desc.cplx), bond.index2[1], bond.index2[2]),
                bond_type=bond.bond_type
            ))
        for center, std in zip(self.block_centers, self.block_center_stds):
            center = (np.array(center) + std * np.random.randn(3)).tolist()
            blocks.append(Block(
                name='UNK',
                atoms=[Atom(name='C', coordinate=center, element='C', id=-1)],
                id=(1, str(len(blocks)))    # the same residue
            ))
        for _ in range(size - len(self.block_centers)):
            blocks.append(Block(
                name='UNK',
                atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
                id=(1, str(len(blocks)))    # the same residue
            ))
        return blocks, bonds

    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        data = super().to_data(cplx_desc)
        # add topo conditioning
        fix_block = [0 for _ in cplx_desc.lig_block_ids]
        for i in range(len(self.wt_fragments[0])): fix_block[i] = 1 # the given fragments
        # add coord conditioning
        fix_block_center = [m for m in fix_block]
        for i in range(len(self.block_centers)):    # additional center control
            fix_block_center[len(self.wt_fragments[0]) + i] = 1
        mask_2d = [0 for _ in cplx_desc.pocket_block_ids] + fix_block
        mask_3d = [0 for _ in cplx_desc.pocket_block_ids] + fix_block_center
        data['condition_config'] = ConditionConfig(
            mask_2d=torch.tensor(mask_2d, dtype=torch.bool),
            mask_3d=torch.tensor(mask_3d, dtype=torch.bool),
            mask_incomplete_2d=None,
            w=self.w
        )
        return data