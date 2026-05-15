# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

from data.mmap_dataset import create_mmap
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.interface import compute_pocket
from data.bioparse.vocab import VOCAB
from data.bioparse.utils import recur_index
from data.bioparse.fragment import brics_complex
from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Process protein-peptide complexes')
    parser.add_argument('--index', type=str, default=None, help='Index file of the dataset')
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    parser.add_argument('--pocket_th', type=float, default=10.0,
                        help='Threshold for determining binding site')
    parser.add_argument('--remove_het', action='store_true', help='Remove HETATM (for test set)')
    return parser.parse_args()


def process_iterator(items, pocket_th, remove_het=False):

    for cnt, pdb_id in enumerate(items):
        summary = items[pdb_id]
        rec_chain, lig_chain = summary['rec_chain'], summary['pep_chain']
        cplx = pdb_to_complex(summary['pdb_path'], selected_chains=[rec_chain, lig_chain], remove_het=remove_het)

        target_blocks = cplx[rec_chain].blocks
        lig_blocks = cplx[lig_chain].blocks
        target_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in target_blocks])
        lig_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in lig_blocks])
        
        pocket_block_id, _ = compute_pocket(cplx, [rec_chain], [lig_chain], dist_th=pocket_th)
        pocket_blocks = [recur_index(cplx, _id) for _id in pocket_block_id]

        data = cplx.to_tuple()

        properties = {
            'pocket_num_blocks': len(pocket_block_id),
            'ligand_num_blocks': len(lig_blocks),
            'pocket_num_atoms': sum([len(block) for block in pocket_blocks]),
            'ligand_num_atoms': sum([len(block) for block in lig_blocks]),
            'target_chain_ids': [rec_chain],
            'ligand_chain_ids': [lig_chain],
            'target_sequences': [target_seq],
            'ligand_sequences': [lig_seq],
            'pocket_block_id': pocket_block_id,
        }

        
        yield pdb_id, data, properties, cnt


def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1. get index file
    with open(args.index, 'r') as fin:
        lines = fin.readlines()
    indexes = {}
    root_dir = os.path.dirname(args.index)
    for line in lines:
        line = line.strip().split('\t')
        pdb_id = line[0]
        indexes[pdb_id] = {
            'rec_chain': line[1],
            'pep_chain': line[2],
            'pdb_path': os.path.join(root_dir, 'pdbs', pdb_id + '.pdb')
        }

    # 3. process pdb files into our format (mmap)
    create_mmap(
        process_iterator(indexes, args.pocket_th, args.remove_het),
        args.out_dir, len(indexes))
    
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())