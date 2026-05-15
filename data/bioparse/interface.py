# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List

import numpy as np
from scipy.spatial import KDTree

from .hierarchy import Block, Complex
from .utils import is_aa, extract_atom_coords


def add_cb(input_array):
    #from protein mpnn
    #The virtual Cβ coordinates were calculated using ideal angle and bond length definitions: b = Cα - N, c = C - Cα, a = cross(b, c), Cβ = -0.58273431*a + 0.56802827*b - 0.54067466*c + Cα.
    N,CA,C,O = input_array
    b = CA - N
    c = C - CA
    a = np.cross(b,c)
    CB = np.around(-0.58273431*a + 0.56802827*b - 0.54067466*c + CA,3)
    return CB #np.array([N,CA,C,CB,O])


def _all_not_none(vals: List):
    if len(vals) == 0: return True
    return vals[0] is not None and _all_not_none(vals[1:])


def blocks_to_cb_coords(blocks):
    cb_coords = []
    for block in blocks:
        coords = extract_atom_coords(block, ['CB', 'N', 'CA', 'C', 'O'])
        if coords[0] is not None: cb_coords.append(coords[0])
        elif _all_not_none(coords[1:]): cb_coords.append(add_cb(np.array(coords[1:])))
        else:
            coords = [atom.get_coord() for atom in block]
            cb_coords.append(np.mean(coords, axis=0))
    return np.array(cb_coords)


def compute_pocket(cplx: Complex, id_set1: List[str], id_set2: List[str], dist_th: float=10.0):
    '''
        Compute the pocket block indexes between two parts defined by id_set1 and id_set2.
        For amino acids, the coordinate of Cb is used to calculate distances.
        For small molecules, the center of mass is used to calculate distances.
        dist_th defines the distance cutoff for extracting the pocket
    '''
    def _extract_block_index(id_set):
        blocks, indexes = [], []
        for _id in id_set:
            molecule = cplx[_id]
            blocks.extend(molecule)
            for block in molecule: indexes.append((_id, block.id)) # (chain id, block id)
        return blocks, indexes
    
    blocks1, indexes1 = _extract_block_index(id_set1)
    blocks2, indexes2 = _extract_block_index(id_set2)

    cb_coords1 = blocks_to_cb_coords(blocks1)
    cb_coords2 = blocks_to_cb_coords(blocks2)
    dist = np.linalg.norm(cb_coords1[:, None] - cb_coords2[None, :], axis=-1)  # [N1, N2]
    
    on_interface = dist < dist_th
    if_indexes1 = np.nonzero(on_interface.sum(axis=1) > 0)[0]
    if_indexes2 = np.nonzero(on_interface.sum(axis=0) > 0)[0]

    return (
        [indexes1[i] for i in if_indexes1], # pocket on chains in id_set1
        [indexes2[i] for i in if_indexes2]  # pocket on chains in id_set2
    )


def compute_interacting_pairs(cplx: Complex, id_set1: List[str], id_set2: List[str], dist_th: float=5.0, efficient=False):
    '''
        contacting residue pairs defined by atom-level distances
    '''
    def _extract_block_index(id_set):
        blocks, indexes = [], []
        for _id in id_set:
            molecule = cplx[_id]
            blocks.extend(molecule)
            for block in molecule: indexes.append((_id, block.id)) # (chain id, block id)
        return blocks, indexes
    
    blocks1, indexes1 = _extract_block_index(id_set1)
    blocks2, indexes2 = _extract_block_index(id_set2)

    if efficient:
        def get_coords(blocks):
            coords, indexes = [], []
            for i, block in enumerate(blocks):
                coords.extend([atom.get_coord() for atom in block])
                indexes.extend([i for _ in block])
            return np.array(coords), np.array(indexes)
        pos1, pos_idx1 = get_coords(blocks1)
        pos2, pos_idx2 = get_coords(blocks2)
        kdtree = KDTree(pos1)
        distances, nearest_idxs = kdtree.query(pos2, p=2, distance_upper_bound=dist_th, k=1)   # potential problem: when one residue has multiple interacting partners
        interact_flag = distances < dist_th  # [Nlig], get rid of inf
        idx1, idx2, distances = pos_idx1[nearest_idxs[interact_flag]], pos_idx2[interact_flag], distances[interact_flag] # redundant
        recorded = {}
        for i, j, d in zip(idx1, idx2, distances):
            if (i, j) not in recorded: recorded[(i, j)] = d
            else: recorded[(i, j)] = min(recorded[(i, j)], d)
        pairs = []
        for (i, j) in recorded:
            pairs.append((indexes1[i], indexes2[j], i, j, recorded[(i, j)]))
    else:
        # TODO: some bugs in the distance function, which is not computing atom-level pairwise distances between residues
        dist = dist_matrix_from_blocks(blocks1, blocks2)
        idx1, idx2 = np.where(dist < dist_th)
        pairs = []
        for i, j in zip(idx1, idx2):
            pairs.append((indexes1[i], indexes2[j], i, j, dist[i][j]))
    return pairs


def blocks_to_coords(blocks):
    max_n_unit = 0
    coords, masks = [], []
    for block in blocks:
        coords.append([atom.get_coord() for atom in block.atoms])
        max_n_unit = max(max_n_unit, len(coords[-1]))
        masks.append([1 for _ in coords[-1]])
    
    for i in range(len(coords)):
        num_pad =  max_n_unit - len(coords[i])
        coords[i] = coords[i] + [[0, 0, 0] for _ in range(num_pad)]
        masks[i] = masks[i] + [0 for _ in range(num_pad)]
    
    return np.array(coords), np.array(masks).astype('bool')  # [N, M, 3], [N, M], M == max_n_unit, in mask 0 is for padding


def dist_matrix_from_coords(coords1, masks1, coords2, masks2):
    dist = np.linalg.norm(coords1[:, None] - coords2[None, :], axis=-1)  # [N1, N2, M]
    dist = dist + np.logical_not(masks1[:, None] * masks2[None, :]) * 1e6  # [N1, N2, M]
    dist = np.min(dist, axis=-1)  # [N1, N2]
    return dist


def dist_matrix_from_blocks(blocks1, blocks2):
    blocks_coord, blocks_mask = blocks_to_coords(blocks1 + blocks2)
    blocks1_coord, blocks1_mask = blocks_coord[:len(blocks1)], blocks_mask[:len(blocks1)]
    blocks2_coord, blocks2_mask = blocks_coord[len(blocks1):], blocks_mask[len(blocks1):]
    dist = dist_matrix_from_coords(blocks1_coord, blocks1_mask, blocks2_coord, blocks2_mask)
    return dist