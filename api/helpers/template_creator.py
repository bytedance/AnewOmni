# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import random
import tempfile
from copy import deepcopy

from rdkit import Chem

from data.bioparse import VOCAB
from data.bioparse.hierarchy import Block, Atom, Bond
from data.bioparse.parser.sdf_to_complex import sdf_to_complex
from data.bioparse.writer.rdkit_mol_to_sdf import rdkit_mol_to_sdf
from utils.logger import print_log


class Creator:
    def __init__(self, mol_index):
        self.mol_index = mol_index  # will be used in bond definition
        self.blocks = []
        self.bonds = []

    def add_block(self, abrv, id, coords=None):
        # TODO: if not from vocabulary, do fragment decomposition
        atom_names = VOCAB.abrv_to_atoms(abrv)
        elements = VOCAB.abrv_to_elements(abrv)
        
        block = Block(
            name=abrv,
            atoms=[Atom(
                name=atom_name,
                coordinate=[0, 0, 0] if coords is None else coords[i],
                element=element,
                id=i
            ) for i, (atom_name, element) in enumerate(zip(atom_names, elements))],
            id=id
        )
        self.blocks.append(block)

        # add intra-block bonds
        for bond in VOCAB.abrv_to_bonds(block.name):
            self.bonds.append(Bond(
                (self.mol_index, len(self.blocks) - 1, bond[0]),
                (self.mol_index, len(self.blocks) - 1, bond[1]),
                bond[2]
            ))
        
    def add_bond(self, block_i, block_j, atom_i, atom_j, bond_type):
        self.bonds.append(Bond(
            (self.mol_index, block_i, atom_i),
            (self.mol_index, block_j, atom_j),
            bond_type
        ))

    def decompose_and_add_fragment(self, mol, seq_pos_number, coords=None):
        tmp = tempfile.NamedTemporaryFile(delete=True, suffix='.sdf')
        if coords is None: coords = [(0, 0, 0) for _ in mol.GetAtoms()]
        mol.SetProp('_Name', 'Fragment')
        writer = Chem.SDWriter(tmp.name)
        writer.write(mol)
        writer.close()
        mol_cplx = sdf_to_complex(tmp.name)
        tmp.close()
        # add blocks and bonds
        offset = len(self.blocks)
        added_blocks, added_bonds = [], []
        for i, block in enumerate(mol_cplx[0]):
            block.id = (seq_pos_number, str(i)) # chr(ord('A') + i))
            self.blocks.append(block)
            added_blocks.append(block)
        for bond in mol_cplx.bonds:
            new_bond = Bond(
                (self.mol_index, offset + bond.index1[1], bond.index1[2]),
                (self.mol_index, offset + bond.index2[1], bond.index2[2]),
                bond.bond_type
            )
            self.bonds.append(new_bond)
            added_bonds.append(new_bond)
        return added_blocks, added_bonds
    
    def add_bond_by_atom_name(self, block_i, block_j, atom_name_i, atom_name_j, bond_type):
        atom_i, atom_j = None, None
        for i, atom in enumerate(self.blocks[block_i]):
            if atom.name == atom_name_i:
                atom_i = i
                break
        for j, atom in enumerate(self.blocks[block_j]):
            if atom.name == atom_name_j:
                atom_j = j
                break
        assert atom_i != None, f'atom name {atom_name_i} not found in {self.blocks[block_i]}'
        assert atom_j != None, f'atom name {atom_name_j} not found in {self.blocks[block_j]}'
        self.add_bond(block_i, block_j, atom_i, atom_j, bond_type)

    def add_unk(self, id, is_aa=False):
        block = Block(
            name='GLY' if is_aa else VOCAB.idx_to_abrv(VOCAB.get_block_dummy_idx()),
            atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=0)],
            id=id
        )
        self.blocks.append(block)

    def visualize(self, out_path):
        pass

    def get_results(self):
        return self.blocks, self.bonds