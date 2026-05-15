# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os

from rdkit import Chem

from . import const
from .hierarchy import BondType
from .tokenizer.tokenize_3d import TOKENIZER
from .utils import bond_type_from_rdkit

class MoleculeVocab:

    def __init__(self):

        # load fragments (manually append single atoms)
        frags = []
        # add principal subgraphs
        for smi in TOKENIZER.get_frag_smiles():
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            if len(mol.GetAtoms()) == 1: continue
            frags.append((f'f{len(frags)}', smi))
        # add atoms
        for element in const.periodic_table:
            frags.append((f'f{len(frags)}', f'[{element}]'))

        # block level vocab
        self.block_dummy = ('X', 'UNK')
        self.idx2block = [self.block_dummy] + const.aas + const.bases + frags
        self.symbol2idx, self.abrv2idx = {}, {}
        self.aa_mask = []
        for i, (symbol, abrv) in enumerate(self.idx2block):
            self.symbol2idx[symbol] = i
            self.abrv2idx[abrv] = i
            self.aa_mask.append(True if abrv in const.AA_GEOMETRY else False)

        # atom level vocab
        self.atom_dummy = 'dummy'
        self.idx2atom = [self.atom_dummy] + const.periodic_table
        self.atom2idx = {}
        for i, atom in enumerate(self.idx2atom):
            self.atom2idx[atom] = i
        
        # atomic canonical orders & chemical bonds in each fragment
        self.atom_canonical_orders, self.element_canonical_orders = {}, {}
        self.chemical_bonds = {}
        for symbol, _ in const.aas:
            self.atom_canonical_orders[symbol] = const.backbone_atoms + const.sidechain_atoms[symbol]
            self.element_canonical_orders[symbol] = [name[0] for name in const.backbone_atoms + const.sidechain_atoms[symbol]] # only C, N, O, S
            atom2order = { a: i for i, a in enumerate(self.atom_canonical_orders[symbol]) }
            self.chemical_bonds[symbol] = [
                (atom2order[bond[0]], atom2order[bond[1]], BondType(bond[2])) for bond in const.aa_bonds[symbol]
            ]
        for symbol, abrv in const.bases:
            assert symbol not in self.atom_canonical_orders
            assert symbol not in self.element_canonical_orders
            assert symbol not in self.chemical_bonds
            self.atom_canonical_orders[symbol] = const.base_atoms[symbol]
            self.element_canonical_orders[symbol] = [name[0] for name in const.base_atoms[symbol]]
            atom2order = { a: i for i, a in enumerate(self.atom_canonical_orders[symbol]) }
            self.chemical_bonds[symbol] = [
                (atom2order[bond[0]], atom2order[bond[1]], BondType(bond[2])) for bond in const.base_bonds[symbol]
            ]
        for symbol, smi in frags:
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            self.atom_canonical_orders[symbol] = [atom.GetSymbol() for atom in mol.GetAtoms()] 
            self.element_canonical_orders[symbol] = [atom.GetSymbol() for atom in mol.GetAtoms()]
            self.chemical_bonds[symbol] = [
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond_type_from_rdkit(bond)) for bond in mol.GetBonds()
            ]

    # block level APIs
    def abrv_to_symbol(self, abrv):
        idx = self.abrv_to_idx(abrv)
        return None if idx is None else self.idx2block[idx][0]

    def symbol_to_abrv(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return None if idx is None else self.idx2block[idx][1]

    def abrv_to_idx(self, abrv):
        return self.abrv2idx.get(abrv, self.abrv2idx['UNK'])

    def symbol_to_idx(self, symbol):
        return self.symbol2idx.get(symbol, self.abrv2idx['UNK'])
    
    def idx_to_symbol(self, idx):
        return self.idx2block[idx][0]

    def idx_to_abrv(self, idx):
        return self.idx2block[idx][1]
    
    def get_block_dummy_idx(self):
        return self.symbol_to_idx(self.block_dummy[0])

    # atom level APIs 
    def get_atom_dummy_idx(self):
        return self.atom2idx[self.atom_dummy]
    
    def idx_to_atom(self, idx):
        return self.idx2atom[idx]

    def atom_to_idx(self, atom):
        return self.atom2idx.get(atom, self.atom2idx[self.atom_dummy])

    # canonical order
    def abrv_to_atoms(self, abrv):
        symbol = self.abrv_to_symbol(abrv)
        return self.atom_canonical_orders.get(symbol, [])
    
    def abrv_to_elements(self, abrv):
        symbol = self.abrv_to_symbol(abrv)
        return self.element_canonical_orders.get(symbol, [])
    
    def abrv_to_bonds(self, abrv):
        symbol = self.abrv_to_symbol(abrv)
        return self.chemical_bonds.get(symbol, [])
    
    # sizes
    def get_num_atom_type(self):
        return len(self.idx2atom)
    
    def get_num_block_type(self):
        return len(self.idx2block)

    def __len__(self):
        return len(self.symbol2idx)

    # others
    @property
    def ca_channel_idx(self):
        return const.backbone_atoms.index('CA')


VOCAB = MoleculeVocab()

