# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

import torch

from data.mmap_dataset import create_mmap
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.parser.sdf_to_complex import sdf_to_complex
from data.bioparse.interface import compute_pocket
from data.bioparse.hierarchy import merge_cplx
from data.bioparse.vocab import VOCAB
from data.bioparse.utils import recur_index
from utils.logger import print_log
from utils.parallel_func import parallel_func

def parse():
    parser = argparse.ArgumentParser(description='Process pocket-molecule complexes')
    parser.add_argument('--split', type=str, required=True, help='Split file of the dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of the dataset')
    parser.add_argument('--atom_level', action='store_true', help='Decompose in atom level')
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    return parser.parse_args()


def worker(data_dir, pocket_file, sdf_file, atom_level):
    _id = pocket_file + '|' + sdf_file
    
    pocket_file = os.path.join(data_dir, pocket_file)
    sdf_file = os.path.join(data_dir, sdf_file)

    pocket = pdb_to_complex(pocket_file)
    ligand = sdf_to_complex(sdf_file, atom_level=atom_level)
    cplx = merge_cplx(pocket, ligand)

    assert len(ligand) == 1

    rec_chains = [mol.id for mol in pocket]
    lig_chain = ligand.molecules[0].id

    pocket_num_atoms, target_seqs, pocket_block_id = 0, [], []
    for mol in pocket:
        seq = ''
        for block in mol:
            pocket_num_atoms += len(block)
            seq += VOCAB.abrv_to_symbol(block.name)
            pocket_block_id.append((mol.id, block.id))
        target_seqs.append(seq)

    lig_blocks = cplx[lig_chain].blocks

    data = cplx.to_tuple()

    properties = {
        'pocket_num_blocks': sum([len(mol) for mol in pocket]),
        'ligand_num_blocks': len(lig_blocks),
        'pocket_num_atoms': pocket_num_atoms,
        'ligand_num_atoms': sum([len(block) for block in lig_blocks]),
        'target_chain_ids': rec_chains,
        'ligand_chain_ids': [lig_chain],
        'target_sequences': target_seqs,
        'ligand_sequences': [ligand[lig_chain].get_property('smiles')],
        'pocket_block_id': pocket_block_id,
    }

    
    return _id, data, properties


def process_iterator(items, data_dir, atom_level):

    generator = parallel_func(worker, [(data_dir, pocket_file, sdf_file, atom_level) for (pocket_file, sdf_file) in items], n_cpus=8)

    cnt = 0
    for outputs in generator:
        cnt += 1
        if outputs is None: continue
        _id, data, properties = outputs
        yield _id, data, properties, cnt

    # for cnt, (pocket_file, sdf_file) in enumerate(items):

    #     _id = pocket_file + '|' + sdf_file
    #     
    #     pocket_file = os.path.join(data_dir, pocket_file)
    #     sdf_file = os.path.join(data_dir, sdf_file)

    #     pocket = pdb_to_complex(pocket_file)
    #     ligand = sdf_to_complex(sdf_file, atom_level=atom_level)
    #     cplx = merge_cplx(pocket, ligand)

    #     assert len(ligand) == 1

    #     rec_chains = [mol.id for mol in pocket]
    #     lig_chain = ligand.molecules[0].id

    #     pocket_num_atoms, target_seqs, pocket_block_id = 0, [], []
    #     for mol in pocket:
    #         seq = ''
    #         for block in mol:
    #             pocket_num_atoms += len(block)
    #             seq += VOCAB.abrv_to_symbol(block.name)
    #             pocket_block_id.append((mol.id, block.id))
    #         target_seqs.append(seq)

    #     lig_blocks = cplx[lig_chain].blocks

    #     data = cplx.to_tuple()

    #     properties = {
    #         'pocket_num_blocks': sum([len(mol) for mol in pocket]),
    #         'ligand_num_blocks': len(lig_blocks),
    #         'pocket_num_atoms': pocket_num_atoms,
    #         'ligand_num_atoms': sum([len(block) for block in lig_blocks]),
    #         'target_chain_ids': rec_chains,
    #         'ligand_chain_ids': [lig_chain],
    #         'target_sequences': target_seqs,
    #         'ligand_sequences': [ligand[lig_chain].get_property('smiles')],
    #         'pocket_block_id': pocket_block_id,
    #     }

    #     
    #     yield _id, data, properties, cnt


def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1. get index file
    split = torch.load(args.split)
    indexes = split['train'] + split['test'] # no validation in the provided split

    # 2. process pdb files into our format (mmap)
    create_mmap(
        process_iterator(indexes, args.data_dir, args.atom_level),
        args.out_dir, len(indexes), commit_batch=1000, abbr_desc_len=30)
    
    # 3. create train/test split
    with open(os.path.join(args.out_dir, 'index.txt'), 'r') as fin: lines = fin.readlines()
    id2lines = { line.split('\t')[0]: line for line in lines }

    # manually divide 100 entries for validation
    split['valid'] = split['train'][-100:]
    split['train'] = split['train'][:-100]

    for split_name in split:
        with open(os.path.join(args.out_dir, split_name + '_index.txt'), 'w') as fout:
            for pocket_file, sdf_file in split[split_name]:
                index = pocket_file + '|' + sdf_file
                if index in id2lines: fout.write(id2lines[index])

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())