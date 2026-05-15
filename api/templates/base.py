# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List, Optional
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from data.bioparse.hierarchy import remove_mols, Atom, Block, Molecule, Complex
from data.file_loader import MolType
from models.modules.adapter.model import ConditionConfig
import utils.register as R


@dataclass
class ComplexDesc:
    id: str
    cplx: Complex
    tgt_chains: List[str]
    lig_chains: List[str]
    pocket_block_ids: List[tuple]
    lig_block_ids: List[tuple] = None
    center_mask: List[int] = None
    generate_mask: List[int] = None
    # topo_generate_mask: Optional[List[int]] = None
    condition_config: Optional[ConditionConfig] = None
    additional_props: Optional[dict] = None


@R.register('BaseTemplate')
class BaseTemplate:
    def __init__(self, size_min=None, size_max=None):
        self.size_min = size_min
        self.size_max = size_max

    @property
    def name(self) -> str:
        return self.__class__.__name__
    
    @property
    def moltype(self) -> MolType:
        raise NotImplementedError('Molecule type not specified!')
    
    @property
    def resample_mode(self) -> bool:    # resample in the neighborhood of the given molecule
        return False

    @property
    def additional_sample_opt(self) -> dict:    # additional sample options for VAE/LDM
        return {}

    def default_filter_configs(self) -> List[dict]:
        return [
            {'class': 'AbnormalConfidenceFilter'},
            {'class': 'PhysicalValidityFilter'}
        ]

    def sample_size(self, cplx_desc: ComplexDesc):
        return np.random.randint(self.size_min, self.size_max)

    def remove_ref_lig(self, cplx_desc: ComplexDesc) -> Complex:
        cplx = remove_mols(cplx_desc.cplx, cplx_desc.lig_chains)
        return cplx

    def dummy_lig_block_bonds(self, cplx_desc: ComplexDesc, size: int) -> List[Block]:
        blocks = [Block(
            name='UNK',
            atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
            id=(i + 1, '')
        ) for i in range(size)]
        bonds = []
        return blocks, bonds

    def add_dummy_lig(self, cplx_desc: ComplexDesc) -> Complex:
        # sample size
        size = self.sample_size(cplx_desc)

        blocks, bonds = self.dummy_lig_block_bonds(cplx_desc, size)

        # set dummy molecule
        lig_chain = cplx_desc.lig_chains[0]
        dummy_mol = Molecule(
            name='dummy',
            blocks=blocks,
            id=lig_chain,
        )
        cplx = cplx_desc.cplx
        
        return Complex(
            name=cplx.name,
            molecules=cplx.molecules + [dummy_mol],
            bonds=cplx.bonds + bonds,
            properties=cplx.properties
        )
    
    def set_desc_attr(self, cplx_desc: ComplexDesc) -> ComplexDesc:
        '''
            Set necessary attributes for describing the generated complex
        '''
        lig_chain = cplx_desc.lig_chains[0]
        cplx_desc.lig_chains = [lig_chain]
        cplx_desc.lig_block_ids = [(lig_chain, block.id) for block in cplx_desc.cplx[lig_chain]]
        cplx_desc.generate_mask = [0 for _ in cplx_desc.pocket_block_ids] + [1 for _ in cplx_desc.lig_block_ids]
        cplx_desc.center_mask = [1 for _ in cplx_desc.pocket_block_ids] + [0 for _ in cplx_desc.lig_block_ids]
        return cplx_desc

    ########## Exposed API ##########

    def validate(self, cplx_desc: ComplexDesc):
        return True

    def __call__(self, cplx_desc: ComplexDesc) -> ComplexDesc:
        cplx_desc = deepcopy(cplx_desc)
        cplx_desc.cplx = self.remove_ref_lig(cplx_desc)
        cplx_desc.cplx = self.add_dummy_lig(cplx_desc)
        return self.set_desc_attr(cplx_desc)
    
    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        raise NotImplementedError(f'to_data not implemented!')