# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import shutil
import argparse
from typing import List, Tuple, Optional
from dataclasses import dataclass

from Bio import PDB

from data.mmap_dataset import create_mmap
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.interface import compute_pocket
from data.bioparse.vocab import VOCAB
from data.bioparse.utils import recur_index, is_standard_aa
from data.bioparse.hierarchy import Complex
from data.bioparse.numbering import Chothia
from utils.logger import print_log
from utils.parallel_func import parallel_func


def parse():
    parser = argparse.ArgumentParser(description='Process antigen-antibody complexes')
    parser.add_argument('--index', type=str, default=None, help='Index file of the dataset')
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    parser.add_argument('--pocket_th', type=float, default=10.0,
                        help='Threshold for determining binding site')
    parser.add_argument('--fr_len', type=int, default=3,
                        help='Framework extended from CDRs to model')
    return parser.parse_args()


TMPDIR = os.path.join(os.path.dirname(__file__), 'tmp')


@dataclass
class Item:

    id: str
    heavy_chain: str
    light_chain: str
    antigen_chains: List[str]
    date: str
    scfv: bool
    vhh: bool
    resolution: float
    affinity: Optional[float]
    pdb_path: str

    # later set
    heavy_seq: str = None
    light_seq: str = None
    antigen_seqs: List[str] = None
    linker: Tuple[str, str, str] = None # only for scfv, start-linker-end
    heavy_cdr: str = None
    heavy_block_ids: list = None
    light_cdr: str = None
    light_block_ids: list = None

# class Chothia:
#     # heavy chain
#     HFR1 = (1, 25)
#     HFR2 = (33, 51)
#     HFR3 = (57, 94)
#     HFR4 = (103, 113)
# 
#     H1 = (26, 32)
#     H2 = (52, 56)
#     H3 = (95, 102)
# 
#     # light chain
#     LFR1 = (1, 23)
#     LFR2 = (35, 49)
#     LFR3 = (57, 88)
#     LFR4 = (98, 107)
# 
#     L1 = (24, 34)
#     L2 = (50, 56)
#     L3 = (89, 97)
# 
#     @classmethod
#     def mark_heavy_seq(cls, pos: List[int]):
#         mark = ''
#         for p in pos:
#             if p < cls.HFR1[0] or p > cls.HFR4[1]: mark += 'X'
#             elif cls.H1[0] <= p and p <= cls.H1[1]: mark += '1'
#             elif cls.H2[0] <= p and p <= cls.H2[1]: mark += '2'
#             elif cls.H3[0] <= p and p <= cls.H3[1]: mark += '3'
#             else: mark += '0'
#         return mark
#     
#     @classmethod
#     def mark_light_seq(cls, pos: List[int]):
#         mark = ''
#         for p in pos:
#             if p < cls.LFR1[0] or p > cls.LFR4[1]: mark += 'X'
#             elif cls.L1[0] <= p and p <= cls.L1[1]: mark += '1'
#             elif cls.L2[0] <= p and p <= cls.L2[1]: mark += '2'
#             elif cls.L3[0] <= p and p <= cls.L3[1]: mark += '3'
#             else: mark += '0'
#         return mark


def str2float(s: str):
    if s == 'None' or s == 'NOT' or s == '': return None
    return float(s)


def format_chain(s: str):
    if s == 'NA': return None
    return s


def format_ag_chains(s: str, ag_type: str):
    ag_type = ag_type.replace(' ', '').split('|')
    s = s.replace(' ', '').split('|')
    chains = []
    for t, c in zip(ag_type, s):
        if t not in ['protein', 'peptide']: continue
        chains.append(c)
    return chains


def set_seq_data(item: Item, cplx: Complex):
    chains = item.antigen_chains + [item.heavy_chain]
    if item.light_chain is not None: chains.append(item.light_chain)

    # antigen sequences
    item.antigen_seqs = []
    for c in item.antigen_chains:
        item.antigen_seqs.append(''.join([
            VOCAB.abrv_to_symbol(block.name) for block in cplx[c]
        ]))
    
    # heavy chain
    if item.heavy_chain is not None:
        blocks = [block for block in cplx[item.heavy_chain] \
                  if block.id[0] >= Chothia.HFR1[0] and block.id[0] <= Chothia.HFR4[-1]
                  ]
        assert len(blocks) == len(cplx[item.heavy_chain])
        item.heavy_seq = ''.join([
            VOCAB.abrv_to_symbol(block.name) for block in blocks
        ])
        ids = [block.id for block in blocks]
        item.heavy_cdr = Chothia.mark_heavy_seq([_id[0] for _id in ids])
        item.heavy_block_ids = [[item.heavy_chain, _id] for _id in ids]
        assert 'X' not in item.heavy_cdr, 'X in heavy chain CDR marking'
    
    # light chain
    if item.light_chain is not None:
        blocks = [block for block in cplx[item.light_chain] \
                  if block.id[0] >= Chothia.LFR1[0] and block.id[0] <= Chothia.LFR4[-1]
                  ]
        assert len(blocks) == len(cplx[item.light_chain])
        item.light_seq = ''.join([
            VOCAB.abrv_to_symbol(block.name) for block in blocks
        ])
        ids = [block.id for block in blocks]
        item.light_cdr = Chothia.mark_light_seq([_id[0] for _id in ids])
        item.light_block_ids = [[item.light_chain, _id] for _id in ids]
        assert 'X' not in item.light_cdr, 'X in light chain CDR marking'
    
    return item


def _get_model_id_mask(ids, marks, fr_len): # fr_len is the number residues before and after the CDR to be considered
     model_mask = [-1 for _ in marks]
     for i, m in enumerate(marks):
         if m != '0': model_mask[i] = int(m) # CDR
         else:
             if (i + fr_len < len(marks)) and (marks[i + fr_len] != '0'):
                 model_mask[i] = int(m)
             elif (i - fr_len >= 0) and (marks[i - fr_len] != '0'):
                 model_mask[i] = int(m)
     model_block_id = [_id for i, _id in enumerate(ids) if model_mask[i] >= 0]
     model_mark = [m for m in model_mask if m >= 0]
     return model_block_id, model_mark


def extract_chains(input_pdb, output_pdb, antigen_chain_ids, heavy_chain_id, light_chain_id):
    """
    Extract specified chains from a PDB file and save them to a new PDB file.
    
    Args:
        input_pdb (str): Path to the input PDB file.
        output_pdb (str): Path to the output PDB file.
        antigen_chain_ids (list): List of chain IDs for antigen to extract.
        heavy_chain_id (str): Chain ID of the heavy chain
        light_chain_id (str): Chain ID of the light chain
    """
    # Create a PDB parser and structure object
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", input_pdb)

    # Create a PDB writer
    writer = PDB.PDBIO()
    
    # Select chains to keep
    class ChainSelect(PDB.Select):
        def accept_chain(self, chain):
            chain_id = chain.get_id()
            return (chain_id in antigen_chain_ids) or \
                   (chain_id == heavy_chain_id) or \
                   (chain_id == light_chain_id)

        def accept_residue(self, residue):
            chain_id = residue.get_parent().id
            if chain_id in antigen_chain_ids:
                return True
            elif chain_id == heavy_chain_id:
                return residue.id[1] >= Chothia.HFR1[0] and residue.id[1] <= Chothia.HFR4[1]
            elif chain_id == light_chain_id:
                return residue.id[1] >= Chothia.LFR1[0] and residue.id[1] <= Chothia.LFR4[1]
            return False

        def accept_atom(self, atom):
            return atom.element != 'X' # unknown

    # Save selected chains to a new file
    writer.set_structure(structure)
    writer.save(output_pdb, select=ChainSelect())


def missing_ratio(cplx: Complex):
    has_cnt, total_cnt = 0, 0
    for mol in cplx:
        for block in mol:
            if is_standard_aa(block.name):
                total_cnt += len(VOCAB.abrv_to_atoms(block.name))
                has_cnt += len(block)
    return 1 - (has_cnt / total_cnt)


def worker(item: Item, pocket_th: float, fr_len: int):
    ab_chains = []
    if item.heavy_chain is not None: ab_chains.append(item.heavy_chain)
    if item.light_chain is not None: ab_chains.append(item.light_chain)

    # prepare cleaned PDB. Some pdbs have more than 10000 atoms with multiple symmetric units
    tmp_pdb_path = os.path.join(TMPDIR, item.id + '_' + ''.join(ab_chains) + '.pdb')
    extract_chains(item.pdb_path, tmp_pdb_path, item.antigen_chains, item.heavy_chain, item.light_chain)
    cplx = pdb_to_complex(tmp_pdb_path, selected_chains=item.antigen_chains + ab_chains)
    os.remove(tmp_pdb_path)

    mr = missing_ratio(cplx)
    if mr > 0.2: # e.g. 7cw3, all CA without other atoms, or 3j3o, only has very few side-chain atoms
        raise ValueError(f'{cplx.name} missing too many atoms ({mr}%)')

    set_seq_data(item, cplx)

    # epitope blocks
    if len(item.antigen_chains):
        epitope_block_id, _ = compute_pocket(cplx, item.antigen_chains, ab_chains, dist_th=pocket_th)
    else: epitope_block_id = []
    epitope_blocks = [recur_index(cplx, _id) for _id in epitope_block_id]

    # framework blocks (CDR +fr_len blocks and -fr_len blocks)
    heavy_model_block_id, heavy_model_mark = [], []
    if item.heavy_chain is not None:
        heavy_model_block_id, heavy_model_mark = _get_model_id_mask(item.heavy_block_ids, item.heavy_cdr, fr_len)
    
    light_model_block_id, light_model_mark = [], []
    if item.light_chain is not None:
        light_model_block_id, light_model_mark = _get_model_id_mask(item.light_block_ids, item.light_cdr, fr_len)

    ligand_num_atoms = 0
    for _id in heavy_model_block_id:
        ligand_num_atoms += len(recur_index(cplx, _id))
    for _id in light_model_block_id:
        ligand_num_atoms += len(recur_index(cplx, _id))

    data = cplx.to_tuple()

    properties = {
        'epitope_num_blocks': len(epitope_block_id),
        'ligand_num_blocks': len(heavy_model_block_id) + len(light_model_block_id),
        'epitope_num_atoms': sum([len(block) for block in epitope_blocks]),
        'ligand_num_atoms': ligand_num_atoms,
        'target_chain_ids': item.antigen_chains,
        'ligand_chain_ids': ab_chains,
        'target_sequences': item.antigen_seqs,
        'heavy_chain_id': item.heavy_chain,
        'light_chain_id': item.light_chain,
        'heavy_chain_sequence': item.heavy_seq,
        'light_chain_sequence': item.light_seq,
        'heavy_chain_mark': item.heavy_cdr,
        'light_chain_mark': item.light_cdr,
        'epitope_block_id': epitope_block_id,
        'heavy_model_block_id': heavy_model_block_id,
        'heavy_model_mark': heavy_model_mark,
        'light_model_block_id': light_model_block_id,
        'light_model_mark': light_model_mark,
    }

    _id = f'{item.id}_{"".join(item.antigen_chains)}_{"" if item.heavy_chain is None else item.heavy_chain}_{"" if item.light_chain is None else item.light_chain}'

    return _id, data, properties



def process_iterator(items: List[Item], pocket_th: float, fr_len: int=3):

    # for cnt, item in enumerate(items):
    #     outputs = worker(item, pocket_th, fr_len)
    #     pdb_id, data, properties = outputs
    #     yield pdb_id, data, properties, cnt

        

    generator = parallel_func(worker, [(item, pocket_th, fr_len) for item in items], n_cpus=8)

    cnt = 0
    for outputs in generator:
        cnt += 1
        if outputs is None: continue
        pdb_id, data, properties = outputs
        yield pdb_id, data, properties, cnt

    # for cnt, item in enumerate(items):
    #     ab_chains = [item.heavy_chain]
    #     if item.light_chain is not None: ab_chains.append(item.light_chain)
    #     cplx = pdb_to_complex(item.pdb_path, selected_chains=item.antigen_chains + ab_chains)

    #     set_seq_data(item, cplx)

    #     # epitope blocks
    #     if len(item.antigen_chains):
    #         epitope_block_id, _ = compute_pocket(cplx, item.antigen_chains, ab_chains, dist_th=pocket_th)
    #     else: epitope_block_id = []
    #     epitope_blocks = [recur_index(cplx, _id) for _id in epitope_block_id]

    #     # framework blocks (CDR +fr_len blocks and -fr_len blocks)
    #     heavy_model_block_id, heavy_model_mark = [], []
    #     if item.heavy_chain is not None:
    #         heavy_model_block_id, heavy_model_mark = _get_model_id_mask(item.heavy_block_ids, item.heavy_cdr, fr_len)
    #     
    #     light_model_block_id, light_model_mark = [], []
    #     if item.light_chain is not None:
    #         light_model_block_id, light_model_mark = _get_model_id_mask(item.light_block_ids, item.light_cdr, fr_len)

    #     ligand_num_atoms = 0
    #     for _id in heavy_model_block_id:
    #         ligand_num_atoms += len(cplx[item.heavy_chain][_id])
    #     for _id in light_model_block_id:
    #         ligand_num_atoms += len(cplx[item.light_chain][_id])

    #     data = cplx.to_tuple()

    #     properties = {
    #         'epitope_num_blocks': len(epitope_block_id),
    #         'ligand_num_blocks': len(heavy_model_block_id) + len(light_model_block_id),
    #         'epitope_num_atoms': sum([len(block) for block in epitope_blocks]),
    #         'ligand_num_atoms': ligand_num_atoms,
    #         'target_chain_ids': item.antigen_chains,
    #         'ligand_chain_ids': ab_chains,
    #         'target_sequences': item.antigen_seqs,
    #         'heavy_chain_sequence': item.heavy_seq,
    #         'light_chain_sequence': item.light_seq,
    #         'heavy_chain_mark': item.heavy_cdr,
    #         'light_chain_mark': item.light_cdr,
    #         'epitope_block_id': epitope_block_id,
    #         'heavy_model_block_id': heavy_model_block_id,
    #         'heavy_model_mark': heavy_model_mark,
    #         'light_model_block_id': light_model_block_id,
    #         'light_model_mark': light_model_mark
    #     }
    #     yield item.id, data, properties, cnt


def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    pdb_dir = os.path.join(os.path.dirname(args.index), 'all_structures', 'chothia')
    
    # 1. Read summary file into Items
    with open(args.index, 'r') as fin: lines = fin.read().strip().split('\n')
    headers = lines[0].split('\t')
    head2idx = { h: i for i, h in enumerate(headers) }
    items, existing = [], {}
    for line in lines[1:]:
        line = line.split('\t')
        item = Item(
            id=line[head2idx['pdb']],
            heavy_chain=format_chain(line[head2idx['Hchain']]),
            light_chain=format_chain(line[head2idx['Lchain']]),
            antigen_chains=format_ag_chains(line[head2idx['antigen_chain']], line[head2idx['antigen_type']]),
            date=line[head2idx['date']],
            scfv=line[head2idx['scfv']] == 'True',
            vhh=format_chain(line[head2idx['Lchain']]) is None,
            resolution=str2float(line[head2idx['resolution']].split(',')[0]),
            affinity=str2float(line[head2idx['affinity']]),
            pdb_path=os.path.join(pdb_dir, line[head2idx['pdb']] + '.pdb')
        )
        uid = item.id + f'_{item.heavy_chain}_{item.light_chain}'
        if uid in existing: continue
        existing[uid] = True
        items.append(item)


    # 2. process pdb files into our format (mmap)
    os.makedirs(TMPDIR, exist_ok=True)
    create_mmap(
        process_iterator(items, args.pocket_th),
        args.out_dir, len(items), commit_batch=100)
    shutil.rmtree(TMPDIR)
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())