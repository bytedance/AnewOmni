# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass

import torch

from scripts.data_process.antibody.sabdab import _get_model_id_mask

from .base import transform_data
from .bioparse.parser.mmcif_to_complex import mmcif_to_complex
from .bioparse.parser.pdb_to_complex import pdb_to_complex
from .bioparse.parser.sdf_to_complex import sdf_to_complex
from .bioparse.hierarchy import merge_cplx
from .bioparse.interface import compute_pocket
from .bioparse.numbering import get_nsys


class MolType(Enum):
    MOLECULE = 1
    PEPTIDE = 2
    ANTIBODY = 3
    NUCLEIC = 4


@dataclass
class BaseLoader:

    # mode 1: one file for both target protein and ligand, where tgt_chains and lig_chains must be provided
    cplx_file: Optional[str] = None
    
    # mode 2: split file for target protein and ligand
    tgt_file: Optional[str] = None
    lig_file: Optional[str] = None

    tgt_chains: Optional[List[str]] = None
    lig_chains: Optional[List[str]] = None

    tgt_pocket_given: bool = False    # only pocket is given in the target file
    pocket_th: float = 10.0

    def load_cplx(self):
        if self.cplx_file is None: # split file mode
            assert self.tgt_file is not None and self.lig_file is not None
            assert self.tgt_file.endswith('.pdb')
            assert self.lig_chains is None, 'Cannot specify lig_chains when using a single file for the ligand'
            tgt = pdb_to_complex(self.tgt_file, selected_chains=self.tgt_chains)
            if self.lig_file.endswith('.pdb'): lig = pdb_to_complex(self.lig_file)
            elif self.lig_file.endswith('.sdf'): lig = sdf_to_complex(self.lig_file)
            else: raise NotImplementedError(f'file format not supported for {self.lig_file}')
            tgt_chains, lig_chains = [mol.id for mol in tgt], [mol.id for mol in lig]
            cplx = merge_cplx(tgt, lig)
            # overwrite tgt_chains and lig_chains
            self.tgt_chains, self.lig_chains = tgt_chains, lig_chains
        else: # single file mode
            assert self.tgt_chains is not None, f'target chains (tgt_chains) need to be specified'
            assert self.lig_chains is not None, f'ligand chains (lig_chains) need to be specified'
            if self.cplx_file.endswith('.pdb'):
                cplx = pdb_to_complex(self.cplx_file, selected_chains=self.tgt_chains + self.lig_chains)
            elif self.cplx_file.endswith('.cif'):
                cplx = mmcif_to_complex(self.cplx_file, selected_chains=self.tgt_chains + self.lig_chains)
            else:
                raise ValueError('only pdb or mmcif file are supported for complex input')
        
        # get pocket ids
        if self.tgt_pocket_given:
            pocket_block_ids = []
            for chain in self.tgt_chains:
                mol = cplx[chain]
                for block in mol: pocket_block_ids.append((mol.id, block.id))
        else:
            pocket_block_ids, _ = compute_pocket(cplx, self.tgt_chains, self.lig_chains, dist_th=self.pocket_th)
        
        # get ligand block ids
        ligand_block_ids = []
        for chain in self.lig_chains:
            mol = cplx[chain]
            for block in mol: ligand_block_ids.append((mol.id, block.id))
        return cplx, pocket_block_ids, ligand_block_ids

    def cplx_to_data(self, cplx, pocket_block_ids, ligand_block_ids, return_ids=False): # need to specify for each type of molecules
        raise NotImplementedError()
    

@dataclass
class MolLoader(BaseLoader):

    def cplx_to_data(self, cplx, pocket_block_ids, ligand_block_ids, return_ids=False):
        assert len(self.lig_chains) == 1
        data = transform_data(cplx, pocket_block_ids + ligand_block_ids)
        data['generate_mask'] = torch.tensor([0 for _ in pocket_block_ids] + [1 for _ in ligand_block_ids], dtype=torch.bool)
        data['center_mask'] = torch.tensor([1 for _ in pocket_block_ids] + [0 for _ in ligand_block_ids], dtype=torch.bool)
        data['position_ids'][data['generate_mask']] = 0
        data['is_aa'][data['generate_mask']] = False
        if return_ids: return data, pocket_block_ids, ligand_block_ids
        return data
    

@dataclass
class PeptideLoader(BaseLoader):
    
    def cplx_to_data(self, cplx, pocket_block_ids, ligand_block_ids, return_ids=False):
        assert len(self.lig_chains) == 1
        data = transform_data(cplx, pocket_block_ids + ligand_block_ids)
        data['generate_mask'] = torch.tensor([0 for _ in pocket_block_ids] + [1 for _ in ligand_block_ids], dtype=torch.bool)
        data['center_mask'] = torch.tensor([1 for _ in pocket_block_ids] + [0 for _ in ligand_block_ids], dtype=torch.bool)
        # position ids start from 1
        gen_mask = data['generate_mask']
        pep_position_ids = data['position_ids'][gen_mask]
        pep_position_ids = pep_position_ids - pep_position_ids.min() + 1
        data['position_ids'][gen_mask] = pep_position_ids
        if return_ids: return data, pocket_block_ids, ligand_block_ids
        return data
    

def _extract_antibody_masks(cplx, lig_chains, pocket_block_ids, cdr_type, fr_len):
    Nsys = get_nsys()
    # identify heavy chain and light chain
    if len(lig_chains) == 2:  # assume heavy chain goes first
        hc, lc = lig_chains
    elif len(lig_chains) == 1:
        if cdr_type.startswith('H'):
            hc, lc = lig_chains[0], None
        else:
            hc, lc = None, lig_chains[0]
    else:
        raise ValueError(f'Number of antibody chains not correct: got {len(lig_chains)}, but expect 1 or 2')
    
    # mark residues in the variable domain
    # heavy chain
    if hc is not None:
        blocks = [block for block in cplx[hc] \
                  if block.id[0] >= Nsys.HFR1[0] and block.id[0] <= Nsys.HFR4[-1]
                  ]
        ids = [block.id for block in blocks]
        heavy_cdr = Nsys.mark_heavy_seq([_id[0] for _id in ids])
        heavy_block_ids = [(hc, _id) for _id in ids]
        # modeling blocks (CDR +fr_len blocks and -fr_len blocks)
        heavy_model_block_id, heavy_model_mark = _get_model_id_mask(heavy_block_ids, heavy_cdr, fr_len)
    else:
        heavy_model_block_id, heavy_model_mark = [], []
    
    # light chain
    if lc is not None:
        blocks = [block for block in cplx[lc] \
                  if block.id[0] >= Nsys.LFR1[0] and block.id[0] <= Nsys.LFR4[-1]
                  ]
        ids = [block.id for block in blocks]
        light_cdr = Nsys.mark_light_seq([_id[0] for _id in ids])
        light_block_ids = [(lc, _id) for _id in ids]
        light_model_block_id, light_model_mark = _get_model_id_mask(light_block_ids, light_cdr, fr_len)
    else:
        light_model_block_id, light_model_mark = [], []

    # get generate mask
    generate_mask = [0 for _ in pocket_block_ids]
    cdr = cdr_type
    for m in heavy_model_mark:
        if cdr.startswith('H') and m == int(cdr[-1]): generate_mask.append(1)
        else: generate_mask.append(0)
    for m in light_model_mark:
        if cdr.startswith('L') and m == int(cdr[-1]): generate_mask.append(1)
        else: generate_mask.append(0)

    # centering at the medium of two ends
    center_mask = [0 for _ in generate_mask]
    for i in range(len(center_mask)):
        if i + 1 < len(generate_mask) and generate_mask[i + 1] == 1 and generate_mask[i] == 0:
            center_mask[i] = 1 # left end
        elif i - 1 > 0 and generate_mask[i - 1] == 1 and generate_mask[i] == 0:
            center_mask[i] = 1

    # only model the interface
    ligand_block_ids = heavy_model_block_id + light_model_block_id

    return generate_mask, center_mask, ligand_block_ids


@dataclass
class AntibodyLoader(BaseLoader):

    cdr_type: str = 'HCDR3'
    fr_len: int = 3

    def cplx_to_data(self, cplx, pocket_block_ids, ligand_block_ids, return_ids=False):

        generate_mask, center_mask, ligand_block_ids = _extract_antibody_masks(
            cplx, self.lig_chains, pocket_block_ids, self.cdr_type, self.fr_len
        )
        data = transform_data(cplx, pocket_block_ids + ligand_block_ids)
        data['generate_mask'] = torch.tensor(generate_mask, dtype=torch.bool)
        data['center_mask'] = torch.tensor(center_mask, dtype=torch.bool)
        if return_ids: return data, pocket_block_ids, ligand_block_ids
        return data