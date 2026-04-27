# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os

from itertools import combinations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdchem import BondType
from rdkit.Geometry import Point3D


def check_bond_lengths(mol, confId=0, tolerance=0.30):
    """
    tolerance: allowed deviation in Å
    """
    conf = mol.GetConformer(confId)
    pt = Chem.GetPeriodicTable()
    problems = []

    total_cnt = 0
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        d = rdMolTransforms.GetBondLength(conf, i, j)

        ri = pt.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum())
        rj = pt.GetRcovalent(mol.GetAtomWithIdx(j).GetAtomicNum())
        expected = ri + rj

        if abs(d - expected) > tolerance:
            problems.append({
                "atoms": (i, j),
                "observed": d,
                "expected": expected
            })
        total_cnt += 1

    return problems, total_cnt



def ideal_angle_from_hybridization(atom):
    hyb = atom.GetHybridization()
    if hyb == Chem.rdchem.HybridizationType.SP3:
        return 109.5
    elif hyb == Chem.rdchem.HybridizationType.SP2:
        return 120.0
    elif hyb == Chem.rdchem.HybridizationType.SP:
        return 180.0
    else:
        return None


def check_bond_angles(mol, confId=0, tolerance=20.0):
    """
    tolerance: allowed deviation in degrees
    """
    conf = mol.GetConformer(confId)
    problems = []

    total_cnt = 0
    for atom in mol.GetAtoms():
        center = atom.GetIdx()
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()]

        if len(neighbors) < 2:
            continue

        ideal = ideal_angle_from_hybridization(atom)
        if ideal is None:
            continue

        for i in range(len(neighbors)):
            for j in range(i+1, len(neighbors)):
                a = neighbors[i]
                c = neighbors[j]

                angle = rdMolTransforms.GetAngleDeg(conf, a, center, c)

                if abs(angle - ideal) > tolerance:
                    problems.append({
                        "atoms": (a, center, c),
                        "observed": angle,
                        "expected": ideal
                    })
                total_cnt += 1

    return problems, total_cnt


def validate_geometry(mol, atom_coords=None):
    if atom_coords is not None:
        num_atoms = mol.GetNumAtoms()
        conformer = Chem.Conformer(num_atoms)
        for i, (x, y, z) in enumerate(atom_coords):
            conformer.SetAtomPosition(i, Point3D(x, y, z))

        # Attach the conformer to the molecule
        mol.AddConformer(conformer)
    if mol.GetNumConformers() == 0:
        return False, "No 3D coordinates"

    bond_issues, bond_cnt = check_bond_lengths(mol)
    angle_issues, angle_cnt = check_bond_angles(mol)

    return {
        'bond_length_problems': bond_issues,
        'bond_cnt': bond_cnt,
        'bond_angle_problems': angle_issues,
        'angle_cnt': angle_cnt
    }


if __name__ == '__main__':
    import sys
    from rdkit import Chem

    mol = Chem.SDMolSupplier(sys.argv[1])[0]