# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

import numpy as np

from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Split peptide data')
    parser.add_argument('--train_index', type=str, required=True, help='Path for training index')
    parser.add_argument('--valid_index', type=str, required=True, help='Path for validation index')
    parser.add_argument('--test_index', type=str, default=None, help='Path for test index')
    parser.add_argument('--all_index_for_non_standard', type=str, default=None, help='Path for all index to generate index for peptides with non-standard amino acids')
    parser.add_argument('--processed_dir', type=str, required=True, help='processed directory')
    return parser.parse_args()


def read_index(mmap_dir):
    items = {}
    index = os.path.join(mmap_dir, 'index.txt')
    with open(index, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            values = line.strip().split('\t')
            items[values[0]] = line
    return items


def transform(items, path, out):
    ids = {}
    with open(path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            ids[line.split('\t')[0]] = 1
    with open(out, 'w') as fout:
        for _id in ids: fout.write(items[_id])


def transform_non_standard(items, path, out):
    ids = {}
    with open(path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            _id, _, _, non_standard = line.strip().split('\t')
            if int(non_standard) == 1:
                ids[_id] = 1
    with open(out, 'w') as fout:
        for _id in ids: fout.write(items[_id])



def main(args):

    # load index file
    items = read_index(args.processed_dir)

    # load training/validation/(test)
    transform(items, args.train_index, os.path.join(args.processed_dir, 'train_index.txt'))
    transform(items, args.valid_index, os.path.join(args.processed_dir, 'valid_index.txt'))
    if args.test_index is not None:
        transform(items, args.test_index, os.path.join(args.processed_dir, 'test_index.txt'))
    if args.all_index_for_non_standard is not None:
        transform_non_standard(items, args.all_index_for_non_standard, os.path.join(args.processed_dir, 'non_standard_index.txt'))

    print_log('Done')


if __name__ == '__main__':
    np.random.seed(12)
    main(parse())
