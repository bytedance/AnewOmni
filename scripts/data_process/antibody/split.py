# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import json
import shutil
import argparse
from collections import defaultdict

import numpy as np

from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Split antibody data')
    parser.add_argument('--index', type=str, required=True, help='Directory of the database')
    parser.add_argument('--rabd_summary', type=str, required=True, help='Summary of test set')
    parser.add_argument('--valid_ratio', type=float, default=0.05, help='Ratio of validation set')
    return parser.parse_args()


def read_index(index, test_summary):
    # load test set (RAbD)
    with open(test_summary, 'r') as fin:
        lines = fin.readlines()
    test_ids = {}
    for line in lines:
        item = json.loads(line)
        _id = item['pdb'] + '_' + ''.join(item['antigen_chains']) + '_' + item['heavy_chain'] + '_' + item['light_chain']
        test_ids[_id] = 0
    items, test_items = {}, {}
    with open(index, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            values = line.strip().split('\t')
            _id, props = values[0], json.loads(values[-1])
            if len(props['target_sequences']) == 0: continue
            ag_seq = 'X'.join(props['target_sequences'])
            if _id in test_ids:
                test_items[_id] = (ag_seq, line)
                test_ids[_id] = 1
            else:
                items[_id] = (ag_seq, line)
    for _id in test_ids:
        if test_ids[_id] == 0: print_log(f'Testing item {_id} not found!', level='WARN')
    return items, test_items


def exec_mmseq(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text


def clustering(fasta, tmp_dir, seq_id):

    # clustering
    db = os.path.join(tmp_dir, 'DB')
    cmd = f'mmseqs createdb {fasta} {db}'
    exec_mmseq(cmd)
    db_clustered = os.path.join(tmp_dir, 'DB_clu')
    cmd = f'mmseqs cluster {db} {db_clustered} {tmp_dir} --min-seq-id {seq_id} -c 0.95 --cov-mode 1'  # simlarity > 0.4 in the same cluster
    res = exec_mmseq(cmd)
    num_clusters = re.findall(r'Number of clusters: (\d+)', res)
    if not len(num_clusters):
        raise ValueError('cluster failed!')

    # write clustering results
    tsv = os.path.join(tmp_dir, 'DB_clu.tsv')
    cmd = f'mmseqs createtsv {db} {db} {db_clustered} {tsv}'
    exec_mmseq(cmd)
    
    # read tsv of class \t pdb
    with open(tsv, 'r') as fin:
        entries = fin.read().strip().split('\n')
    id2clu, clu2id = {}, defaultdict(list)
    for entry in entries:
        cluster, _id = entry.strip().split('\t')
        id2clu[_id] = cluster

    for _id in id2clu:
        cluster = id2clu[_id]
        clu2id[cluster].append(_id)
    
    clu_cnt = [len(clu2id[clu]) for clu in clu2id]
    print(f'cluster number: {len(clu2id)}, member number ' +
          f'mean: {np.mean(clu_cnt)}, min: {min(clu_cnt)}, ' +
          f'max: {max(clu_cnt)}')
    
    return id2clu, clu2id


def main(args):

    # load index file
    items, test_items = read_index(args.index, args.rabd_summary)

    # make temporary directory
    tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
    if os.path.exists(tmp_dir):
        print_log(f'Working directory {tmp_dir} exists! Deleting it.', level='WARN')
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    
    # split by 40% seq-id clustering
    fasta = os.path.join(tmp_dir, 'seq.fasta')
    with open(fasta, 'w') as fout:
        for _id in items:
            fout.write(f'>{_id}\n{items[_id][0]}\n') # item[_id][0] is the antigen sequence
        for _id in test_items:
            fout.write(f'>test_{_id}\n{test_items[_id][0]}\n')
    id2clu, clu2id = clustering(fasta, tmp_dir, 0.4)
    # get test clusters if given test set
    test_clusters = {}
    for _id in test_items:
        test_clusters[id2clu['test_' + _id]] = True
    # random split by clusters
    clusters = sorted([clu for clu in clu2id.keys() if clu not in test_clusters])
    np.random.shuffle(clusters)

    valid_size = int(args.valid_ratio * len(clusters))
    if len(test_items):
        test_size = 0 # do not split test set from the dataset
    sizes = [len(clusters) - valid_size - test_size, valid_size, test_size]
    fnames = ['train', 'valid', 'test']

    start = 0
    root_dir = os.path.dirname(args.index)
    for name, size in zip(fnames, sizes):
        assert 0 <= size and size <= len(clusters)
        if size == 0:
            continue
        cnt = 0
        end = start + size
        list_path = os.path.join(root_dir, name + '.txt')
        fpath = os.path.join(root_dir, name + '_index.txt')
        cls_fpath = os.path.join(root_dir, name + '.cluster')
        list_out, fout, cls_fout = open(list_path, 'w'), open(fpath, 'w'), open(cls_fpath, 'w')
        for c in clusters[start:end]:
            for _id in clu2id[c]:
                list_out.write(_id + '\n')
                fout.write(items[_id][-1])
                cls_fout.write(f'{_id}\t{c}\t{len(clu2id[c])}\n') # identity, cluster, cluster size
                cnt += 1
        list_out.close()
        fout.close()
        cls_fout.close()
        start = end
        print_log(f'Save {size} clusters, {cnt} entries to {fpath}. Clustering details in {cls_fpath}')

    if len(test_items):
        list_path = os.path.join(root_dir, 'test.txt')
        fpath = os.path.join(root_dir, 'test_index.txt')
        cls_fpath = os.path.join(root_dir, 'test.cluster')
        list_out, fout, cls_fout = open(list_path, 'w'), open(fpath, 'w'), open(cls_fpath, 'w')
        clu_cnt = {}
        for _id in test_items:
            list_out.write(_id.lstrip('test_') + '\n')
            fout.write(test_items[_id][-1])
            c = id2clu['test_' + _id]
            cls_fout.write(f'{_id}\t{c}\t{len(clu2id[c])}\n') # identity, cluster, cluster size
            clu_cnt[c] = True
        list_out.close()
        fout.close()
        cls_fout.close()
        print_log(f'Save {len(clu_cnt)} clusters, {len(test_items)} entries to {fpath}. Clustering details in {cls_fpath}')

    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    np.random.seed(12)
    main(parse())
