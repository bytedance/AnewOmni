# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
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
    parser = argparse.ArgumentParser(description='Process pocket-molecule complexes')
    parser.add_argument('--index_file', type=str, required=True, help='Path to BioLiP_nr.txt')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory of the dataset (BioLiP_updated_set)')
    parser.add_argument('--atom_level', action='store_true', help='Decompose in atom level')
    parser.add_argument('--pocket_th', type=float, default=10.0,
                        help='Threshold for determining binding site')
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    return parser.parse_args()


def rectify_chain_id(id):
    if len(id) > 1: return id[0]
    return id


def worker(data_dir, item, atom_level, pocket_th, cif_out_dir):

    rec_file = os.path.join(data_dir, 'receptor_nr', item['pdb_id'] + item['rec_chain'] + '.pdb')
    lig_file = os.path.join(data_dir, 'ligand_nr', f'{item["pdb_id"]}_{item["ccd_id"]}_{item["lig_chain"]}_{item["lig_serial_number"]}.pdb')

    receptor = pdb_to_complex(rec_file)
    ligand = pdb_to_complex(lig_file)
    receptor.molecules[0].id = rectify_chain_id(receptor.molecules[0].id)
    ligand.molecules[0].id = rectify_chain_id(ligand.molecules[0].id)
    if ligand.molecules[0].id == receptor.molecules[0].id:
        ligand.molecules[0].id = chr(ord(receptor.molecules[0].id) + 1)
    cplx = merge_cplx(receptor, ligand)

    assert len(receptor) == 1
    assert len(ligand) == 1
    
    rec_chain = receptor.molecules[0].id
    lig_chain = ligand.molecules[0].id

    _id = f'{item["pdb_id"]}_{item["ccd_id"]}_{rec_chain}_{lig_chain}_{item["lig_serial_number"]}'

    target_blocks = cplx[rec_chain].blocks
    lig_blocks = cplx[lig_chain].blocks
    target_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in target_blocks])
    lig_seq = ''.join([VOCAB.abrv_to_symbol(block.name) for block in lig_blocks])
    
    pocket_block_id, _ = compute_pocket(cplx, [rec_chain], [lig_chain], dist_th=pocket_th)
    pocket_blocks = [recur_index(cplx, _id) for _id in pocket_block_id]

    data = cplx.to_tuple()

    # save cif
    complex_to_mmcif(cplx, os.path.join(cif_out_dir, _id + '.cif'))

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
        'resolution': item['resolution'],
        'ccd_id': item['ccd_id']
    }

    return _id, data, properties


def process_iterator(items, data_dir, atom_level, pocket_th, cif_out_dir):

    # cnt = 0
    # for item in items:
    #     cnt += 1
    #     _id, data, properties = worker(data_dir, item, atom_level, pocket_th, cif_out_dir)
    #     yield _id, data, properties, cnt

    os.makedirs(cif_out_dir, exist_ok=True)

    generator = parallel_func(worker, [(data_dir, item, atom_level, pocket_th, cif_out_dir) for item in items], n_cpus=8)

    cnt = 0
    for outputs in generator:
        cnt += 1
        if outputs is None: continue
        _id, data, properties = outputs
        yield _id, data, properties, cnt


def load_index(path):
    with open(path, 'r') as fin:
        lines = fin.read().strip().split('\n')
    items, existed = [], {}
    repeat_cnt = 0
    for line in lines:
        line = line.split('\t')
        item = {
            'pdb_id': line[0],
            'rec_chain': line[1],
            'resolution': -1.0 if line[2] == '' else float(line[2]), # -1.0 for lack of resolution (e.g. NMR)
            'bs_number_code': line[3],
            'ccd_id': line[4],
            'lig_chain': line[5],
            'lig_serial_number': line[6],
        }
        non_repeat_check_id = f'{item["pdb_id"]}_{item["ccd_id"]}_{item["rec_chain"]}_{item["lig_chain"]}_{item["lig_serial_number"]}'
        if non_repeat_check_id in existed:
            repeat_cnt += 1
            print_log(f'{non_repeat_check_id} repeated', level='WARN')
        existed[non_repeat_check_id] = True
        items.append(item)
    print_log(f'{repeat_cnt} entries repeated')
    return items


def write_cluster(out_dir, index_file):
    with open(os.path.join(out_dir, index_file), 'r') as fin: lines = fin.readlines()
    id2lines = { line.split('\t')[0]: line for line in lines }
    id2tgtseqs = {}
    for id in id2lines:
        seqs = json.loads(id2lines[id].rstrip().split('\t')[-1])['target_sequences']
        assert len(seqs) == 1
        id2tgtseqs[id] = seqs[0]
    
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
    id2clu, clu2id = clustering(fasta, tmp_dir, 0.4)
    
    shutil.rmtree(tmp_dir)

    # write clustering
    out_path = os.path.join(out_dir, index_file.split('_')[0] + '_cluster.txt')
    with open(out_path, 'w') as fout:
        for id in id2clu:
            fout.write(f'{id}\t{id2clu[id]}\t{len(clu2id[id2clu[id]])}\n')


def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1. get index file
    indexes = load_index(args.index_file)

    # split by resolution
    resolution_th = 2.5 # below 2.5 are good, otherwise not good (but can be used for pretraining)
    good_cnt = 0
    for item in indexes:
        res = item['resolution']
        if res > 0 and res < resolution_th: good_cnt += 1
    print_log(f'Samples with resolution < {resolution_th}: {good_cnt}')

    # 2. process pdb files into our format (mmap)
    create_mmap(
        process_iterator(indexes, args.data_dir, args.atom_level, args.pocket_th, os.path.join(args.out_dir, 'mmcif')),
        args.out_dir, len(indexes), commit_batch=1000, abbr_desc_len=30)
    
    # 3. create pretrain/finetune split
    with open(os.path.join(args.out_dir, 'index.txt'), 'r') as fin: lines = fin.readlines()
    id2lines = { line.split('\t')[0]: line for line in lines }

    # divide by resolution
    split = { 'pretrain': [], 'finetune': [] }
    for id in id2lines:
        resolution = json.loads(id2lines[id].strip('\n').split('\t')[-1])['resolution']
        if resolution < 0 or resolution > resolution_th: split['pretrain'].append(id)
        else: split['finetune'].append(id)

    for split_name in split:
        with open(os.path.join(args.out_dir, split_name + '_index.txt'), 'w') as fout:
            for id in split[split_name]:
                if id in id2lines: fout.write(id2lines[id])
    
    # 4. clustering by target sequence
    for split_name in split:
        print_log(f'Clustering for {split_name}')
        write_cluster(args.out_dir, split_name + '_index.txt')
        print_log('Clustering done')
    
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())