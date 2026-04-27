# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
import shutil
import argparse

from data.mmap_dataset import create_mmap
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.parser.sdf_to_complex import sdf_to_complex
from data.bioparse.writer.complex_to_mmcif import complex_to_mmcif
from data.bioparse.interface import compute_pocket
from data.bioparse.hierarchy import merge_cplx
from data.bioparse.vocab import VOCAB
from data.bioparse.utils import recur_index
from utils.logger import print_log
from utils.parallel_func import parallel_func

from scripts.data_process.antibody.split import clustering

def parse():
    parser = argparse.ArgumentParser(description='Process PDBbind')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of the dataset')
    parser.add_argument('--pocket_th', type=float, default=10.0,
                        help='Threshold for determining binding site')
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    return parser.parse_args()


def parse_index(fpath):
    with open(fpath, 'r') as fin:
        lines = fin.readlines()
    
    data = []
    for line in lines:
        if line.startswith('#'):
            continue
        line = re.split(r'\s+', line)
        pdb_id, resolution, year, kd = line[:4]
        data.append({
            'pdb_id': pdb_id,
            'resolution': resolution,
            'year': year,
            'kd': kd
        })
    return data


def rectify_chain_id(id):
    if len(id) > 1: return id[0]
    return id

def find_new_chain_id(exist_ids):
    for i in range(26):
        i = chr(ord('A') + i)
        if i not in exist_ids: return i


def worker_pl(data_dir, item, pocket_th, cif_out_dir):

    pdb_id = item['pdb_id']

    rec_file = os.path.join(data_dir, pdb_id, pdb_id + '_protein.pdb')
    lig_file = os.path.join(data_dir, pdb_id, pdb_id + '_ligand.sdf')

    receptor = pdb_to_complex(rec_file)
    ligand = sdf_to_complex(lig_file)
    ligand.molecules[0].id = find_new_chain_id([mol.id for mol in receptor])
    cplx = merge_cplx(receptor, ligand)

    assert len(ligand) == 1
    
    rec_chains = [mol.id for mol in receptor]
    lig_chain = ligand.molecules[0].id

    target_seqs = []
    for mol in receptor:
        target_seqs.append(''.join([VOCAB.abrv_to_symbol(block.name) for block in mol]))
    lig_blocks = cplx[lig_chain].blocks
    lig_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in lig_blocks])
    
    pocket_block_id, _ = compute_pocket(cplx, rec_chains, [lig_chain], dist_th=pocket_th)
    pocket_blocks = [recur_index(cplx, _id) for _id in pocket_block_id]

    data = cplx.to_tuple()

    # save cif
    complex_to_mmcif(cplx, os.path.join(cif_out_dir, pdb_id + '.cif'))

    properties = {
        'pocket_num_blocks': len(pocket_block_id),
        'ligand_num_blocks': len(lig_blocks),
        'pocket_num_atoms': sum([len(block) for block in pocket_blocks]),
        'ligand_num_atoms': sum([len(block) for block in lig_blocks]),
        'target_chain_ids': rec_chains,
        'ligand_chain_ids': [lig_chain],
        'target_sequences': target_seqs,
        'ligand_sequences': [lig_seq],
        'pocket_block_id': pocket_block_id,
        'resolution': item['resolution'],
        'kd': item['kd'],
        'year': item['year']
    }

    return pdb_id, data, properties


def process_iterator_PL(items, data_dir, pocket_th, cif_out_dir):

    os.makedirs(cif_out_dir, exist_ok=True)

    generator = parallel_func(worker_pl, [(data_dir, item, pocket_th, cif_out_dir) for item in items], n_cpus=8)

    cnt = 0
    for outputs in generator:
        cnt += 1
        if outputs is None: continue
        _id, data, properties = outputs
        yield _id, data, properties, cnt


def _cluster(out_dir, index_file, seq_id=0.4):
    with open(os.path.join(out_dir, index_file), 'r') as fin: lines = fin.readlines()
    id2lines = { line.split('\t')[0]: line for line in lines }
    id2tgtseqs = {}
    for id in id2lines:
        seqs = json.loads(id2lines[id].rstrip().split('\t')[-1])['target_sequences']
        id2tgtseqs[id] = 'X'.join(seqs)
    
    # make temporary directory
    tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
    if os.path.exists(tmp_dir):
        print_log(f'Working directory {tmp_dir} exists! Deleting it.', level='WARN')
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    
    # split by 40% seq-id clustering
    fasta = os.path.join(tmp_dir, 'seq.fasta')
    with open(fasta, 'w') as fout:
        for _id in id2tgtseqs:
            fout.write(f'>{_id}\n{id2tgtseqs[_id]}\n') # item[_id][0] is the antigen sequence
    id2clu, clu2id = clustering(fasta, tmp_dir, seq_id)
    
    shutil.rmtree(tmp_dir)
    return id2clu, clu2id


def write_cluster(out_dir, index_file):
    id2clu, clu2id = _cluster(out_dir, index_file)

    # write clustering
    out_path = os.path.join(out_dir, 'index.cluster')
    with open(out_path, 'w') as fout:
        for id in id2clu:
            fout.write(f'{id}\t{id2clu[id]}\t{len(clu2id[id2clu[id]])}\n')


def split_val_test(dir):
    index_file = os.path.join(dir, 'index.txt')
    id2clu, clu2id = _cluster(dir, index_file)

    # find the smallest 200 cluster for testing, and the second smallest 200 clusters for validation
    sorted_clu = sorted(list(clu2id.keys()), key=lambda c: len(clu2id))
    test_clu, val_clu, train_clu = sorted_clu[:100], sorted_clu[100:200], sorted_clu[200:]

    # write results
    with open(index_file, 'r') as fin: lines = fin.readlines()
    id2lines = { line.split('\t')[0]: line for line in lines }

    def find_reso_best(ids):
        best_reso = 10000
        best_id = ids[0]
        for id in ids:
            prop = json.loads(id2lines[id].strip().split('\t')[-1])
            reso = prop['resolution']
            try: reso = float(reso)
            except ValueError: reso = 100
            if reso < best_reso:
                best_reso, best_id = reso, id
        return best_id

    for clu, name in zip([test_clu, val_clu, train_clu], ['test', 'valid', 'train']):
        out_file = open(os.path.join(dir, name + '_index.txt'), 'w')
        out_clu_file = open(os.path.join(dir, name + '.cluster'), 'w')
        for c in clu:
            if name == 'train': ids = clu2id[c]
            else:
                ids = [find_reso_best(clu2id[c])] # non-redundant
                print(f'{name}: cluster {c}, size {len(clu2id[c])}, selected {ids[0]}')
            for id in ids:
                out_file.write(id2lines[id])
                out_clu_file.write(f'{id}\t{c}\t{len(ids)}\n')
        out_file.close()
        out_clu_file.close()


def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1. refined set
    out_dir = os.path.join(args.out_dir, 'refined-set')
    indexes = parse_index(os.path.join(
        args.data_dir, 'refined-set', 'index', 'INDEX_refined_set.2020'
    ))
    create_mmap(
        process_iterator_PL(indexes, os.path.join(args.data_dir, 'refined-set'), args.pocket_th, os.path.join(out_dir, 'mmcif')),
        out_dir, len(indexes), commit_batch=1000, abbr_desc_len=30
    )
    print_log('Finished refined-set')

    # 2. all set minus refined set
    out_dir = os.path.join(args.out_dir, 'v2020-other-PL')
    indexes = parse_index(os.path.join(
        args.data_dir, 'v2020-other-PL', 'index', 'INDEX_general_PL.2020'
    ))
    indexes = [item for item in indexes if os.path.exists(os.path.join(args.data_dir, 'v2020-other-PL', item['pdb_id']))]
    create_mmap(
        process_iterator_PL(indexes, os.path.join(args.data_dir, 'v2020-other-PL'), args.pocket_th, os.path.join(out_dir, 'mmcif')),
        out_dir, len(indexes), commit_batch=1000, abbr_desc_len=30
    )
    print_log('Finished v2020-other-PL')


    # # 3. NL
    # out_dir = os.path.join(args.out_dir, 'NL')
    # indexes = parse_index(os.path.join(
    #     args.data_dir, 'NL', 'index', 'INDEX_general_NL.2020'
    # ))
    # create_mmap(
    #     process_iterator_PL(indexes, os.path.join(args.data_dir, 'NL'), args.pocket_th, os.path.join(out_dir, 'mmcif')),
    #     out_dir, len(indexes), commit_batch=1000, abbr_desc_len=30
    # )
    # print_log('Finished NL')

    # 4. clustering by target sequence
    splits = [
        os.path.join(args.out_dir, 'refined-set'),
        os.path.join(args.out_dir, 'v2020-other-PL'),
    ]
    for split_name in splits:
        print_log(f'Clustering for {split_name}')
        write_cluster(split_name, os.path.join(split_name, 'index.txt'))
        print_log('Clustering done')
    
    # 5. split validation and test set
    print_log(f'Split train/validation/test on refined-set')
    split_val_test(os.path.join(args.out_dir, 'refined-set'))

    print_log('Finished!')


if __name__ == '__main__':
    main(parse())