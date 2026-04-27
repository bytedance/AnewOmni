# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F

from data.bioparse import VOCAB, const
from utils.singleton import singleton
from utils.nn_utils import expand_like


@singleton
class Map:
    def __init__(self):
        self.max_n_atoms = max([len(atoms) for atoms in  VOCAB.atom_canonical_orders.values()])
        mapping = []
        for i in range(VOCAB.get_num_block_type()):
            elements = VOCAB.abrv_to_elements(VOCAB.idx_to_abrv(i))
            idxs = [VOCAB.atom_to_idx(e) for e in elements]
            idxs.extend([VOCAB.get_atom_dummy_idx() for _ in range(self.max_n_atoms - len(idxs))])
            mapping.append(idxs)
        self.mapping = torch.tensor(mapping, dtype=torch.long) # [num_block_type, M]
        self.mask = self.mapping != VOCAB.get_atom_dummy_idx() # [num_block_type, M], 1 for atom

        self.max_n_bonds = max([len(bonds) for bonds in VOCAB.chemical_bonds.values()])
        bond_mapping = []
        for i in range(VOCAB.get_num_block_type()):
            bonds = VOCAB.abrv_to_bonds(VOCAB.idx_to_abrv(i))
            bond_mapping.append(
                [(bond[0], bond[1], bond[2].value) for bond in bonds] + \
                [(0, 0, 0) for _ in range(self.max_n_bonds - len(bonds))]
            )
        self.bond_mapping = torch.tensor(bond_mapping, dtype=torch.long) # [num_block_type, E, 3]
        self.bond_mask = self.bond_mapping[..., -1] != 0 # [num_block_type, E]
        
    def __call__(self, block_types):
        atom_types = self.mapping[block_types] # [N, M]
        mask = self.mask[block_types] # [N, M]
        bonds = self.bond_mapping[block_types]
        bond_mask = self.bond_mask[block_types]
        return atom_types, mask, bonds, bond_mask
    
    def to(self, device):
        self.mapping = self.mapping.to(device)
        self.mask = self.mask.to(device)
        self.bond_mapping = self.bond_mapping.to(device)
        self.bond_mask = self.bond_mask.to(device)
        return self


def block_to_atom_map(block_types, assign_ids):
    '''
        Args:
            block_types: [N]
            assign_ids: [N], the indices of each block, used to generate new block_ids
        Returns:
            A: [Natom]
            block_ids: [Natom]
            local_bonds: [E, 3] (with local index)
    '''
    map_instance = Map().to(block_types.device)
    atom_types, mask, bonds, bond_mask = map_instance(block_types)  # [N, M], [N, max_E_per_block, 3], 
    A = atom_types[mask] # [Natom]
    block_ids = expand_like(assign_ids, atom_types)[mask] # [Natom]
    # local index to global index
    n_atoms =  mask.long().sum(-1) # [N]
    cum_n_atoms = F.pad(torch.cumsum(n_atoms, dim=0)[:-1], pad=(1, 0), value=0) # [N]
    bonds = torch.stack([
        bonds[..., 0] + cum_n_atoms.unsqueeze(-1),
        bonds[..., 1] + cum_n_atoms.unsqueeze(-1),
        bonds[..., 2]
    ], dim=-1) # [N, max_E_per_block, 3]
    bonds = bonds[bond_mask] # [E, 3]
    return A, block_ids, bonds