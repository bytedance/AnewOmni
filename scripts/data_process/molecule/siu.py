# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
import shutil
import argparse

from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from data.mmap_dataset import create_mmap
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.parser.sdf_to_complex import sdf_to_complex
from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
from data.bioparse.writer.complex_to_mmcif import complex_to_mmcif
from data.bioparse.interface import compute_pocket
from data.bioparse.hierarchy import merge_cplx, Complex
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
    parser.add_argument('--pdb_dir', type=str, required=True, help='Path to PDB database')
    parser.add_argument('--n_cpus', type=int, default=8, help='Number of CPUs')
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')

    # split parts
    parser.add_argument('--n_parts', type=int, default=None, help='Number of split parts')
    parser.add_argument('--part_index', type=int, default=None, help='Local part index')
    return parser.parse_args()


def parse_index(data_dir, pdb_dir, out_dir, n_parts, part_index):
    out_path = os.path.join(out_dir, 'items.json')
    if os.path.exists(out_path):
        print_log(f'Loaded pre-scanned items from {out_path}')
        items = json.load(open(out_path, 'r'))
        return items
    print_log(f'Looping through {data_dir} to get indexes')
    items = []
    # cnt = 0
    uniprots = list(os.listdir(data_dir))
    if n_parts is not None: # split into several parts
        uniprots = sorted(uniprots)
        span = (len(uniprots) + n_parts - 1) // n_parts
        start = part_index * span
        end = start + span
        if part_index == n_parts - 1: end = len(uniprots)
        uniprots = uniprots[start:end]
        print_log(f'Processing part {part_index + 1} / {n_parts}, start: {start}, end: {end}')
    for uniprot in tqdm(uniprots):
        d1 = os.path.join(data_dir, uniprot)
        for prot_conf in os.listdir(d1):
            pdb_path, sm_dirs = None, []
            _, pdb_id, _, chain, _ = prot_conf.split('_')
            pdb_path = os.path.join(pdb_dir, pdb_id[1:3], pdb_id + '.cif')
            d2 = os.path.join(d1, prot_conf)
            if not os.path.exists(pdb_path):
                print_log(f'PDB file not found for {d2}, skip', level='WARN')
                continue
            for f in os.listdir(d2):
                d3 = os.path.join(d2, f)
                # if f.endswith('.pdb'): pdb_path = d3
                if os.path.isdir(d3): sm_dirs.append(d3)
            for d3 in sm_dirs:
                files = []
                for sdf_file in os.listdir(d3):
                    files.append(os.path.join(d3, sdf_file))
                for i, sdf_file in enumerate(sorted(files)):
                    items.append({
                        'uniprot': uniprot,
                        'prot_conf_id': prot_conf,
                        'pdb_id': pdb_id,
                        'chain_id': chain,
                        'sm_id': os.path.basename(d3),
                        'pdb': pdb_path,
                        'sdf': sdf_file,
                        'pose_id': i,
                        'num_poses': len(files)
                    })
            # break
        # cnt += 1
        # if cnt == 2: break
    json.dump(items, open(out_path, 'w'))
    return items


def find_new_chain_id(exist_ids):
    for i in range(26):
        i = chr(ord('A') + i)
        if i not in exist_ids: return i


def clean_rec(cplx: Complex):
    cleaned_bonds = []
    for bond in cplx.bonds:
        atom1, atom2 = recur_index(cplx, bond.index1), recur_index(cplx, bond.index2)
        dist = np.linalg.norm(np.array(atom1.get_coord()) - np.array(atom2.get_coord()))
        if dist < 2: cleaned_bonds.append(bond)
        else: print(f'drop {atom1} {atom2}')
    cplx.bonds = cleaned_bonds
    return cplx


def worker_pl(item, pocket_th, cif_out_dir):

    _id = item['uniprot'] + '_' + item['pdb_id'] + '_' + item['sm_id'] + '_' + f'p{item["pose_id"]}'

    # receptor = pdb_to_complex(item['pdb'])
    receptor = mmcif_to_complex(item['pdb'], selected_chains=[item['chain_id']], remove_het=True)

    # clean receptor, as SIU has renumbered residues, leading to chemical bonds between unconnected residues
    # receptor = clean_rec(receptor)

    ligand = sdf_to_complex(item['sdf'])
    ligand.molecules[0].id = find_new_chain_id([mol.id for mol in receptor])
    cplx = merge_cplx(receptor, ligand)
    smi = Chem.MolToSmiles(Chem.SDMolSupplier(item['sdf'])[0])

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
    complex_to_mmcif(cplx, os.path.join(cif_out_dir, _id + '.cif'))

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
        'num_poses': item['num_poses'],
        'smiles': smi
    }

    return _id, data, properties


def process_iterator_PL(items, pocket_th, cif_out_dir, n_cpus=8):

    os.makedirs(cif_out_dir, exist_ok=True)

    # res = [worker_pl(item, pocket_th, cif_out_dir) for item in items]

    generator = parallel_func(worker_pl, [(item, pocket_th, cif_out_dir) for item in items], n_cpus=n_cpus, unordered=True)

    cnt = 0
    for outputs in generator:
        cnt += 1
        if outputs is None: continue
        _id, data, properties = outputs
        yield _id, data, properties, cnt


def _generate_scaffold(smiles, include_chirality=False):
    """return scaffold string of target molecule"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return smiles
    scaffold = MurckoScaffold\
        .MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold

def _cluster_scaffold(id2seqs):
    id2clu, clu2id = {}, {}
    for _id in id2seqs:
        smi = id2seqs[_id]
        scaffold = _generate_scaffold(smi, include_chirality=True)
        id2clu[_id] = scaffold
        if scaffold not in clu2id:
            clu2id[scaffold] = []
        clu2id[scaffold].append(_id)
    return id2clu, clu2id


def _cluster(out_dir, index_file, seq_id=0.4):
    with open(os.path.join(out_dir, index_file), 'r') as fin: lines = fin.readlines()
    id2lines = { line.split('\t')[0]: line for line in lines }
    id2tgtseqs, id2scaffolds = {}, {}
    for id in id2lines:
        uid = '_'.join(id.split('_')[:-1])  # get rid of p0/p1/...
        if uid in id2tgtseqs: continue  # remove redundance
        props = json.loads(id2lines[id].rstrip().split('\t')[-1])
        seqs = props['target_sequences']
        id2tgtseqs[uid] = 'X'.join(seqs)
        id2scaffolds[uid] = _generate_scaffold(props['smiles'])
    
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

    # split by scaffold
    id2clu_scaffold, clu2id_scaffold = _cluster_scaffold(id2scaffolds)

    # product cluster (tgt * scaffold)
    product_id2clu, product_clu2id = {}, {}
    for id in id2clu:
        clu = id2clu[id] + '_X_' + id2clu_scaffold[id]
        product_id2clu[id] = clu
        if clu not in product_clu2id: product_clu2id[clu] = []
        product_clu2id[clu].append(id)

    return product_id2clu, product_clu2id


def write_cluster(out_dir, index_file):
    id2clu, clu2id = _cluster(out_dir, index_file)

    # write clustering
    out_path = os.path.join(out_dir, 'index.cluster')
    with open(os.path.join(out_dir, index_file), 'r') as fin: lines = fin.readlines()
    id2lines = [ (line.split('\t')[0], line) for line in lines ]
    with open(out_path, 'w') as fout:
        for id, line in id2lines:
            uid = '_'.join(id.split('_')[:-1])  # get rid of p0/p1/...
            num_poses = json.loads(line.rstrip().split('\t')[-1])['num_poses']
            # the "num_clusters" is tricky, because we want to first sample one data from target*ligand clusters,
            # then sample one conformation, so the 1/p should be proprotion to 1/(n_data_in_cluster * num_poses)
            fout.write(f'{id}\t{id2clu[uid]}\t{len(clu2id[id2clu[uid]]) * num_poses}\n')


def main(args):

    out_dir = args.out_dir
    if args.n_parts is not None:
        out_dir = os.path.join(out_dir, f'part{args.part_index}')
        assert args.part_index < args.n_parts
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # # 1. refined set
    # items = parse_index(args.data_dir, args.pdb_dir, out_dir, args.n_parts, args.part_index)
    # create_mmap(
    #     process_iterator_PL(items, args.pocket_th, os.path.join(out_dir, 'mmcif'), args.n_cpus),
    #     out_dir, len(items), commit_batch=1000, abbr_desc_len=30
    # )
    # print_log('Finished refined-set')

    print_log(f'Clustering')
    write_cluster(out_dir, 'index.txt')
    print_log('Clustering done')
    
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())