# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import io
import gzip
import json
import mmap
from typing import Optional
from tqdm import tqdm
from multiprocessing import shared_memory

import torch
import numpy as np


def compress(x):
    serialized_x = json.dumps(x).encode()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as f:
        f.write(serialized_x)
    compressed = buf.getvalue()
    return compressed


def decompress(compressed_x):
    buf = io.BytesIO(compressed_x)
    with gzip.GzipFile(fileobj=buf, mode="rb") as f:
        serialized_x = f.read().decode()
    x = json.loads(serialized_x)
    return x


def _find_measure_unit(num_bytes):
    size, measure_unit = num_bytes, 'Bytes'
    for unit in ['KB', 'MB', 'GB']:
        if size > 1000:
            size /= 1024
            measure_unit = unit
        else:
            break
    return size, measure_unit


def create_mmap(iterator, out_dir, total_len=None, commit_batch=10000, abbr_desc_len=-1):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data_file_path = os.path.join(out_dir, 'data.bin')
    data_file = open(data_file_path, 'wb')
    index_file = open(os.path.join(out_dir, 'index.txt'), 'w')

    i, offset, n_finished = 0, 0, 0
    progress_bar = tqdm(iterator, total=total_len, ascii=True)
    for _id, x, properties, entry_idx in iterator:
        progress_bar.set_description(f'Processing {(_id[:abbr_desc_len] + "...") if abbr_desc_len > 0 else _id}')
        compressed_x = compress(x)
        bin_length = data_file.write(compressed_x)
        properties = json.dumps(properties)
        index_file.write(f'{_id}\t{offset}\t{offset + bin_length}\t{properties}\n') # tuple of (_id, start, end), data slice is [start, end)
        offset += bin_length
        i += 1

        if entry_idx > n_finished:
            progress_bar.update(entry_idx - n_finished)
            n_finished = entry_idx
            if total_len is not None:
                expected_size = os.fstat(data_file.fileno()).st_size / n_finished * total_len
                expected_size, measure_unit = _find_measure_unit(expected_size)
                progress_bar.set_postfix({f'{i} saved; Estimated total size ({measure_unit})': expected_size})

        if i % commit_batch == 0:
            data_file.flush()  # save from memory to disk
            index_file.flush()

        
    data_file.close()
    index_file.close()


def repack_properties(txt_file, convert_file=None):
    assert txt_file.endswith('.txt')
    if convert_file is not None:
        assert convert_file.endswith('.npy')
        prop_name = convert_file.replace('.npy', '.prop')
        root_name = convert_file.replace('.npy', '')
    else:
        prop_name = txt_file.replace('.txt', '.prop')
        root_name = txt_file.replace('.txt', '')
    prop_file = open(prop_name, 'wb')
    offset = [0]
    starts = []
    ends = []
    with open(txt_file, 'r') as f:
        for line in tqdm(f.readlines()):
            messages = line.strip().split('\t')
            _id, start, end = messages[:3]
            _property = json.loads(messages[3])
            # _indexes.append((_id, int(start), int(end)))
            _property.update({'id': _id, 'start': int(start), 'end': int(end)})
            prop_compressed = compress(_property)
            prop_len = prop_file.write(prop_compressed)
            offset.append(offset[-1] + prop_len)
            starts.append(int(start))
            ends.append(int(end))
    offset = np.array([offset[:-1], offset[1:], starts, ends], dtype=np.int64).transpose(1, 0)
    np.save(root_name, offset)
    prop_file.close()


class MMAPProperties:
    def __init__(self, npy_path: str, prop_path: str):
        self._npy_path = npy_path
        self._prop_path = prop_path
        self._indexes = np.load(self._npy_path, mmap_mode='r')
        self._mmap_file = open(self._prop_path, 'rb')
        self._mmap = mmap.mmap(self._mmap_file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return self._indexes.shape[0]
    
    def __del__(self):
        self._mmap.close()
        self._mmap_file.close()

    def __getitem__(self, index: int) -> Optional[dict]:
        if index < 0 or index >= len(self):
            return None
        prop_start = int(self._indexes[index][0])
        prop_end = int(self._indexes[index][1])
        data = decompress(self._mmap[prop_start:prop_end])
        return data


class MMAPDataset(torch.utils.data.Dataset):
    
    def __init__(self, mmap_dir: str, specify_data: Optional[str]=None, specify_index: Optional[str]=None, in_memory: bool=False) -> None:
        super().__init__()

        self._indexes = []
        self._properties = []
        _index_path = os.path.join(mmap_dir, 'index.txt') if specify_index is None else specify_index
        mmap_prop_idx = _index_path.replace('.txt', '.npy')
        mmap_prop_data = _index_path.replace('.txt', '.prop')
        if os.path.exists(mmap_prop_idx) and os.path.exists(mmap_prop_data): prop_load_mmap = True
        else: prop_load_mmap = False
        with open(_index_path, 'r') as f:
            for line in f.readlines():
                messages = line.strip().split('\t')
                _id, start, end = messages[:3]
                self._indexes.append((_id, int(start), int(end)))
                if not prop_load_mmap:
                    _property = json.loads(messages[3])
                    self._properties.append(_property)
        if prop_load_mmap: self._properties = MMAPProperties(mmap_prop_idx, mmap_prop_data)

        _data_path = os.path.join(mmap_dir, 'data.bin') if specify_data is None else specify_data
        self._data_file = open(_data_path, 'rb')
        self._mmap = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.in_memory = in_memory
        self.cache = [None for _ in range(len(self._indexes))]

    def __del__(self):
        self._mmap.close()
        self._data_file.close()

    def __len__(self):
        return len(self._indexes)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        
        data = self.cache[idx]
        if data is None:
            _, start, end = self._indexes[idx]
            data = decompress(self._mmap[start:end])
            if self.in_memory: self.cache[idx] = data

        return data