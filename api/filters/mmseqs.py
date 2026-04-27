# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import tempfile
from typing import Union, List
from collections import defaultdict

import ray

import utils.register as R

from .base import BaseFilter, FilterResult, FilterInput


def _exec_mmseq(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text


def _write_fasta(id2seq, fasta):
    with open(fasta, 'w') as fout:
        for _id in id2seq:
            fout.write(f'>{_id}\n{id2seq[_id]}\n')


def _mmseqs_clustering(fasta, tmp_dir, coverage=None, cov_mode=None, seq_id=0.3):
    # for cov_mode, please refer to: https://github.com/soedinglab/MMseqs2/issues/73#issuecomment-373644484
    # i.e., A minimum coverage (option -c [0,1], which is defined by the number of aligned residue pairs divided by
    # either the minimum of the length of query/centre and target/non-centre sequences (default mode, --cov-mode 0),
    # or by the length of the target/non-centre sequence (--cov-mode 1),
    # or by the length of the query/centre (--cov-mode 2)

    # clustering
    db = os.path.join(tmp_dir, 'DB')
    cmd = f'mmseqs createdb {fasta} {db}'
    _exec_mmseq(cmd)
    db_clustered = os.path.join(tmp_dir, 'DB_clu')
    cmd = f'mmseqs cluster {db} {db_clustered} {tmp_dir} --min-seq-id {seq_id}'  # simlarity > seq_id in the same cluster
    if coverage is not None: cmd += f' -c {coverage}'
    if cov_mode is not None: cmd += f' --cov-mode {cov_mode}'
    res = _exec_mmseq(cmd)
    num_clusters = re.findall(r'Number of clusters: (\d+)', res)
    if not len(num_clusters):
        raise ValueError('cluster failed!')

    # write clustering results
    tsv = os.path.join(tmp_dir, 'DB_clu.tsv')
    cmd = f'mmseqs createtsv {db} {db} {db_clustered} {tsv}'
    _exec_mmseq(cmd)
    
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
    
    return id2clu, clu2id


def _all2all_seq_id(query_fasta, target_fasta, tmp_dir):
    querydb = os.path.join(tmp_dir, 'queryDB')
    _exec_mmseq(f'mmseqs createdb {query_fasta} {querydb} --shuffle 0')
    targetdb = os.path.join(tmp_dir, 'targetDB')
    _exec_mmseq(f'mmseqs createdb {target_fasta} {targetdb} --shuffle 0')
    fake_prep_path = os.path.join(os.path.dirname(__file__), 'fake_pref.sh')
    allvsallpref = os.path.join(tmp_dir, 'allvsallpref')
    _exec_mmseq(f'bash {fake_prep_path} {querydb} {targetdb} {allvsallpref}')
    allvsallaln = os.path.join(tmp_dir, 'allvsallaln')
    _exec_mmseq(f'mmseqs align {querydb} {targetdb} {allvsallpref} {allvsallaln} -e inf --alignment-mode 3 --seq-id-mode 1')
    res_path = os.path.join(tmp_dir, 'results.tsv')
    _exec_mmseq(f'mmseqs convertalis {querydb} {targetdb} {allvsallaln} {res_path}')
    # load results
    with open(res_path, 'r') as fin: lines = fin.readlines()
    pairs = []
    for line in lines:
        line = line.strip().split('\t')
        pairs.append((line[0], line[1], float(line[2])))
    return pairs


@R.register('SeqIDFilter')
class SeqIDFilter(BaseFilter):
    def __init__(self, reference_seqs: Union[str, List[str]], th: float=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # sequence id by mmseqs2 below certain threshold (default 30%)
        # reference seqs can be either a list of strings, or path to a txt file in which each line is a sequence
        self.th = th
        if isinstance(reference_seqs, str): # a path to a txt file
            with open(reference_seqs, 'r') as fin:
                self.reference_seqs = fin.read().strip().split('\n')
        else: self.reference_seqs = reference_seqs

    @property
    def name(self):
        return self.__class__.__name__ + f'(reference_seqs={self.reference_seqs}, th={self.th})'
    
    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):

        # get tmp dir
        temp_dir = tempfile.TemporaryDirectory()

        id2seq_model = { 'model': input.seq }
        id2seq_ref = { f'ref{i}': seq for i, seq in enumerate(self.reference_seqs) }

        # write fasta
        query_fasta = os.path.join(temp_dir.name, 'query.fasta')
        _write_fasta(id2seq_model, query_fasta)
        ref_fasta = os.path.join(temp_dir.name, 'refs.fasta')
        _write_fasta(id2seq_ref, ref_fasta)

        # all to all seq id
        pair_seq_ids = _all2all_seq_id(query_fasta, ref_fasta, temp_dir.name)

        # cleanup
        temp_dir.cleanup()  

        stats = { 'pair_seq_ids': pair_seq_ids }

        for id1, id2, seqid in pair_seq_ids:
            assert id1 == 'model'
            if seqid > self.th: return FilterResult.FAILED, stats
        else: return FilterResult.PASSED, stats


if __name__ == '__main__':
    f = SeqIDFilter(['HKTDSFVGLM'])
    print(f.name)
    print(f.run(FilterInput(None, None, None, None, 'DMFYAFM', None, None)))
    