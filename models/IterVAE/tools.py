# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from torch_scatter import scatter_sum

from data.bioparse import VOCAB
from utils.nn_utils import expand_like

from ..modules.GET.tools import fully_connect_edges, knn_edges

default_vdw_radii = {
    'H': 1.2,  # Hydrogen
    'C': 1.7,  # Carbon
    'N': 1.55, # Nitrogen
    'O': 1.52, # Oxygen
    'F': 1.47, # Fluorine
    'P': 1.8, # Phosphorus
    'S': 1.8, # Sulfur
    'Cl': 2.27, # Chlorine
    'Br': 1.85 # Bromine
}


def vdw_radii_tensor(device='cpu'):
    vdw = torch.zeros(VOCAB.get_num_atom_type(), device=device)
    for atom_type in default_vdw_radii:
        idx = VOCAB.atom_to_idx(atom_type)
        vdw[idx] = default_vdw_radii[atom_type]
    return vdw


def _detect_clash(A, X_t, batch_ids, block_ids, chain_ids, generate_mask, tolerance=0.4):
    # get bonds
    # local neighborhood for clash detection
    row, col = fully_connect_edges(batch_ids[block_ids])

    # left end in generation, right end in pocket (different chains)
    select_mask = (generate_mask[block_ids[row]] & (~generate_mask[block_ids[col]])) & \
                  (chain_ids[block_ids[row]] != chain_ids[block_ids[col]])
    row, col = row[select_mask], col[select_mask]   # [E]

    # calculate distances
    distances = torch.norm(X_t[row] - X_t[col], dim=-1)
    
    # project element index to vdw
    vdw_radii = vdw_radii_tensor(A.device)
    vdw_sums_with_tolerance = vdw_radii[A[row]] + vdw_radii[A[col]] - tolerance

    # check whether distances are below clash threshold
    clash_detected = distances < vdw_sums_with_tolerance
    # print(clash_detected.sum() / (batch_ids.max() + 1))

    # get clashes
    distances, vdw_sums_with_tolerance = distances[clash_detected], vdw_sums_with_tolerance[clash_detected]
    row, col = row[clash_detected], col[clash_detected]

    return row, col, distances, vdw_sums_with_tolerance


@torch.no_grad()
def _inter_clash_guidance(t, A, X_t, batch_ids, block_ids, chain_ids, generate_mask, tolerance=0.4):

    row, col, distances, vdw_sums_with_tolerance = _detect_clash(
        A, X_t, batch_ids, block_ids, chain_ids, generate_mask, tolerance
    )

    # repulsion (only move atoms in the generation part)
    repulsion_strength = vdw_sums_with_tolerance - distances
    force = (X_t[row] - X_t[col]) / (distances[:, None] + 1e-10) * repulsion_strength[:, None]
    force = scatter_sum(force, row, dim=0, dim_size=A.shape[0])  # [Natom, 3]

    # weights
    w = min(t / (1 - t + 1e-10), 10)
    return w * force


@torch.no_grad()
def _avoid_clash(A, X_t, batch_ids, block_ids, chain_ids, generate_mask, is_aa=None, tolerance=0.3):

    # get clash
    row, col, distances, vdw_sums_with_tolerance = _detect_clash(
        A, X_t, batch_ids, block_ids, chain_ids, generate_mask, tolerance
    )

    # repulsion (only move atoms in the generation part)
    repulsion_strength = vdw_sums_with_tolerance - distances + 1e-5
    force = (X_t[row] - X_t[col]) / (distances[:, None] + 1e-10) * repulsion_strength[:, None]
    force = scatter_sum(force, row, dim=0, dim_size=A.shape[0])  # [Natom, 3]

    if is_aa is not None:   # do not add force on amino acids, which is already able to generate good geometry
        is_aa_atom = expand_like(is_aa[block_ids], force)
        force = torch.where(is_aa_atom, torch.zeros_like(force), force)

    X_t = X_t + force

    return X_t