# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

def rdkit_mol_to_sdf(mol: Chem.Mol, coordinates: list, out_path: str):
    '''
        Transforming a RDKit Mol object into a SDF file
        Args:
            mol: Chem.Mol
            coordinates: list of xyz coordinates for these atoms (ordered as the RDKit Mol)
            out_path: str, outputing path of the SDF file
    '''
    num_atoms = mol.GetNumAtoms()
    conformer = Chem.Conformer(num_atoms)
    for i, (x, y, z) in enumerate(coordinates):
        conformer.SetAtomPosition(i, Point3D(x, y, z))

    # Attach the conformer to the molecule
    mol.AddConformer(conformer)

    w = Chem.SDWriter(out_path)
    w.write(mol)
    w.close()

def complete_rdkit_mol_to_sdf(mol: Chem.Mol, out_path: str):
    w = Chem.SDWriter(out_path)
    w.write(mol)
    w.close()


if __name__ == '__main__':
    import sys

    mol = Chem.MolFromSmiles("CCO")
    coordinates = [
        (0.0, 0.0, 0.0),  # C1
        (1.5, 0.0, 0.0),  # C2
        (2.0, 1.0, 0.0),  # O1
    ]

    rdkit_mol_to_sdf(mol, coordinates, sys.argv[1])