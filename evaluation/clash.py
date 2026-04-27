# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
    Adapted from https://github.com/EDAPINENUT/CBGBench/blob/master/repo/tools/geometry/eval_steric_clash.py
'''
import numpy as np
from Bio import PDB
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist

from data.bioparse.const import covalent_radii


ca_dist = 3.6574 # from https://arxiv.org/pdf/2409.06744


def get_ca_coordinates(pdb_file, selected_chains):
    # Create a parser object to read the PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    ca_coordinates = []

    for model in structure:  # Loop through models (usually there's only one)
        for chain in model:
            if chain.id in selected_chains:
                for residue in chain:
                    # Check if the residue has an alpha carbon (CA)
                    if 'CA' in residue:
                        atom = residue['CA']
                        ca_coordinates.append(atom.coord)

    return np.array(ca_coordinates)


def inner_clash_ratio(ca_coords: np.array):
    '''
        Args:
            ca_coords: [N, 3], sequential coordinates of CA
    '''
    num_residues = len(ca_coords)

    # void self-loop and sequentially connecting residues
    pair_mask = np.eye(num_residues, num_residues, dtype=bool)
    pair_mask[np.arange(num_residues - 1), np.arange(1, num_residues)] = True
    pair_mask[np.arange(1, num_residues), np.arange(num_residues - 1)] = True

    dist = np.linalg.norm(ca_coords[:, None] - ca_coords[None, :], axis=-1) # [N, N]
    
    clash = (dist < ca_dist) & (~pair_mask)
    clash_indices = np.where(clash)
    clash_num_residues = len(np.unique(clash_indices[0]))

    return clash_num_residues / num_residues


def outer_clash_ratio(ca_coords1: np.array, ca_coords2: np.array):

    dist = np.linalg.norm(ca_coords1[:, None] - ca_coords2[None, :], axis=-1) # [N, M]

    clash = dist < ca_dist
    clash_indices = np.where(clash)
    clash_num_residues1 = len(np.unique(clash_indices[0]))
    clash_num_residues2 = len(np.unique(clash_indices[1]))

    clash_ratio1 = clash_num_residues1 / len(ca_coords1)
    clash_ratio2 = clash_num_residues2 / len(ca_coords2)

    return clash_ratio1, clash_ratio2



def eval_pdb_clash(pdb_path, target_chains, ligand_chains):
    target_ca_coords = get_ca_coordinates(pdb_path, target_chains)
    ligand_ca_coords = get_ca_coordinates(pdb_path, ligand_chains)

    clash_inner = inner_clash_ratio(ligand_ca_coords)
    clash_outer, _ = outer_clash_ratio(ligand_ca_coords, target_ca_coords)

    return clash_inner, clash_outer


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


def detect_atom_clash(pos_tgt, pos_lig, element_tgt, element_lig, vdw_radii=default_vdw_radii, tolerance=0.4, efficient=True):
    '''
    Adapted from https://github.com/EDAPINENUT/CBGBench/blob/master/repo/tools/geometry/eval_steric_clash.py#L93
    '''
    
    tgt_vdw_radii = np.array([vdw_radii.get(a, -1) for a in element_tgt])
    lig_vdw_radii = np.array([vdw_radii.get(a, -1) for a in element_lig])

    if efficient:   # use KDTree algorithm. suitable for large target
        kdtree = KDTree(np.array(pos_tgt))
        distances, nearest_idxs = kdtree.query(np.array(pos_lig))   # Nlig
        vdw_sums_with_tolerance = lig_vdw_radii + tgt_vdw_radii[nearest_idxs] - tolerance   # Nlig
        clashes = (distances < vdw_sums_with_tolerance)

        clash_atoms = np.where(clashes)[0]
    else:
        distances = cdist(pos_lig, pos_tgt)
        vdw_sums_with_tolerance = lig_vdw_radii[:, np.newaxis] + tgt_vdw_radii - tolerance
        clashes = (distances < vdw_sums_with_tolerance)

        # Get indices where clashes occur
        clash_indices = np.where(clashes)
        # clashed_distances = distances[clash_indices]
        clash_atoms = np.unique(clash_indices[0])

    return clash_atoms 


if __name__ == '__main__':
    import sys
    clash_inner, clash_outer = eval_pdb_clash(sys.argv[1], [sys.argv[2]], [sys.argv[3]])
    print(clash_inner, clash_outer)