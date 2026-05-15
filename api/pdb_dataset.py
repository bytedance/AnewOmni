# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from typing import List
from copy import deepcopy

import torch

from data.bioparse.hierarchy import Molecule, Complex
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
from data.bioparse.interface import compute_pocket
from data.bioparse.utils import recur_index, assign_new_chain_id, expand_nc_block_ids
from data.base import transform_data
from models.modules.adapter.model import ConditionConfig
from utils.logger import print_log

from .templates import BaseTemplate, ComplexDesc


class PDBDataset(torch.utils.data.Dataset):
    def __init__(self, pdb_paths: List[str], tgt_chains: List[List[str]], template_config: BaseTemplate, lig_chains: List[List[str]]=None, hotspots: List[List[tuple]]=None, n_samples: int=1):
        super().__init__()
        self.pdb_paths = pdb_paths
        self.tgt_chains = tgt_chains
        assert not (lig_chains is None and hotspots is None), 'Either reference compound (lig_chains) or hotspots should be provided!'
        if (lig_chains is not None) and (hotspots is not None):
            print_log('Both lig_chains and hotspots are provided! Default using lig_chains to define the binding site!', level='WARN')
        self.lig_chains = [None for _ in pdb_paths] if lig_chains is None else lig_chains
        # hotspots for distinguishing pockets when reference compound is not available
        self.hotspots = [None for _ in pdb_paths] if hotspots is None else hotspots
        self.cplxs, self.pocket_block_ids = [], []
        for i, (path, tgt, lig, htspt) in enumerate(zip(self.pdb_paths, self.tgt_chains, self.lig_chains, self.hotspots)):
            cplx, pocket_block_ids, lig = self._process_pdb(path, tgt, lig, htspt)
            self.lig_chains[i] = lig
            self.cplxs.append(cplx)
            self.pocket_block_ids.append(pocket_block_ids)
        self.n_samples = n_samples
        self.config = template_config

    def __getitem__(self, idx):
        idx = idx // self.n_samples
        cplx, pocket_block_ids = self.cplxs[idx], self.pocket_block_ids[idx]

        # edit the cplx with template (create new cplx of ligand and merge)
        cplx_desc = self.config(ComplexDesc(
            id=os.path.basename(os.path.splitext(self.pdb_paths[idx])[0]),
            cplx=cplx,
            tgt_chains=list(self.tgt_chains[idx]),
            lig_chains=list(self.lig_chains[idx]),
            pocket_block_ids=pocket_block_ids
        ))

        data = self.config.to_data(cplx_desc)
        data['cplx_desc'] = cplx_desc

        return data

    def __len__(self):
        return len(self.cplxs) * self.n_samples
    
    def _process_pdb(self, pdb_path, tgt_chains, lig_chains, hotspots, pocket_dist_th=10.0):
        '''
        hotspots: in the format of tuples (chain id, seq number, insert code), e.g. [('A', 113, ''), ('A', 114, 'A')]
        '''
        if pdb_path.endswith('.pdb'):
            cplx = pdb_to_complex(pdb_path, tgt_chains + ('' if lig_chains is None else lig_chains))
        elif pdb_path.endswith('.cif'):
            cplx = mmcif_to_complex(pdb_path, tgt_chains + ('' if lig_chains is None else lig_chains))
        else: raise ValueError(f'File format {pdb_path} not recognized. Supported: .pdb, .cif')
        if lig_chains is not None:
            frag_lig_chains = []
            for mol in cplx:
                for c in lig_chains:
                    if c + '_' in mol.id: frag_lig_chains.append(mol.id) # small molecules fragmented
            chain_ids_for_pocket_detection = list(lig_chains) + frag_lig_chains
        else:  # hotspots should not be None
            residues = []
            for chain_id, seq_number, icode in hotspots:
                seq_number, icode = int(seq_number), icode.strip()
                try:
                    expanded_block_ids = expand_nc_block_ids(cplx[chain_id], (seq_number, icode))
                    for s, ic in expanded_block_ids:
                        residues.append(recur_index(cplx, (chain_id, (s, ic))))
                except KeyError: raise KeyError(f'Hotspot {(chain_id, seq_number, icode)} does not exist in the provided protein file')
            new_chain_id = assign_new_chain_id(tgt_chains)
            # add these residues as dummy ligand
            dummy_ligand = Molecule(new_chain_id, [deepcopy(res) for res in residues], id=new_chain_id)
            cplx.id2idx[new_chain_id] = len(cplx.molecules)
            cplx.molecules.append(dummy_ligand)
            chain_ids_for_pocket_detection = [new_chain_id]
            lig_chains = new_chain_id
        
        pocket_block_ids, _ = compute_pocket(cplx, tgt_chains, chain_ids_for_pocket_detection, dist_th=pocket_dist_th)
        
        return cplx, pocket_block_ids, lig_chains
    
    def collate_fn(self, batch):
        results = {}
        for key in batch[0]:
            values = [item[key] for item in batch]
            if key == 'lengths':
                results[key] = torch.tensor(values, dtype=torch.long)
            elif key == 'bonds': # need to add offsets
                offset = 0
                for i, bonds in enumerate(values):
                    bonds[:, :2] = bonds[:, :2] + offset # src/dst
                    offset += len(batch[i]['A'])
                results[key] = torch.cat(values, dim=0)
            elif key == 'cplx_desc':
                results[key] = values
            elif key == 'condition_config':
                results[key] = ConditionConfig.batchify(values)
            else:
                results[key] = torch.cat(values, dim=0)
        return results
    

if __name__ == '__main__':
    import sys
    dataset = PDBDataset([sys.argv[1]], sys.argv[2], sys.argv[3], BaseTemplate(10, 12)) # e.g. xxx.pdb AB CD
    print(dataset[0])