# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import warnings
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem, rdDetermineBonds
from biotite.structure import BondType as BT

from . import const
from .hierarchy import Atom, Block, Molecule, Complex, BondType, Bond


def is_aa(block: Block) -> bool:
    '''return whether this block is an amino acid'''
    return is_standard_aa(block.name)
    if block.name in const.AA_GEOMETRY: # known amino acids
        return True
    # non-canonical amino acids: with N, CA, C, O (WARN: not so reliable)
    profile = { atom_name: False for atom_name in const.backbone_atoms }
    for atom in block:
        if atom.name in profile: profile[atom.name] = True
    for atom_name in profile:
        if not profile[atom_name]: return False
    return True


def is_standard_aa(abrv: str) -> bool:
    '''3-code abbreviation'''
    return abrv in const.chi_angles_atoms


def is_standard_base(abrv: str) -> bool:
    bases = { abrv: True for _, abrv in const.bases }
    return abrv in bases


def is_standard_block(abrv: str) -> bool:
    '''abbreviation in pdb (3-code for aa, DA/T/C/G for DNA, A/G/C/U for RNA)'''
    return is_standard_aa(abrv) or is_standard_base(abrv)


def bond_type_from_rdkit(bond: rdchem.Bond) -> BondType:
    '''Convert RDKit bond type to custom BondType.'''
    if bond == rdchem.BondType.SINGLE or bond.GetBondType() == rdchem.BondType.SINGLE:
        return BondType.SINGLE
    elif bond == rdchem.BondType.DOUBLE or bond.GetBondType() == rdchem.BondType.DOUBLE:
        return BondType.DOUBLE
    elif bond == rdchem.BondType.TRIPLE or bond.GetBondType() == rdchem.BondType.TRIPLE:
        return BondType.TRIPLE
    elif bond == rdchem.BondType.AROMATIC or bond.GetBondType() == rdchem.BondType.AROMATIC:
        return BondType.AROMATIC
    else:
        return BondType.NONE
    

def bond_type_from_biotite(bond):
    if bond == BT.ANY or bond == BT.COORDINATION:
        warnings.warn(f'Unknown bond {bond} is automatically set to single bond')
        bond = BT.SINGLE # for CONECT records (mostly in cyclic peptides) and coordination bonds
    if isinstance(bond, BT):
        biotite_bond = bond
    else: 
        biotite_bond = BT(bond)
    return BondType(biotite_bond.without_aromaticity().value)


def bond_type_to_rdkit(bond: BondType):
    '''Convert custom BondType to RDKit bond type'''
    if bond == BondType.SINGLE:
        return rdchem.BondType.SINGLE
    elif bond == BondType.DOUBLE:
        return rdchem.BondType.DOUBLE
    elif bond == BondType.TRIPLE:
        return rdchem.BondType.TRIPLE
    elif bond == BondType.AROMATIC:
        return rdchem.BondType.AROMATIC
    else:
        return None
    

def extract_atom_coords(block: Block, names: List[str]) -> List:
    '''extract atom coords given the names'''
    name2coords = {}
    for atom in block:
        name2coords[atom.name] = atom.get_coord()
    coords = []
    for name in names: coords.append(name2coords.get(name, None))
    return coords


def recur_index(obj, index: tuple):
    for _id in index:
        obj = obj[_id]
    return obj


def expand_nc_block_ids(mol: Molecule, block_id: tuple):
    if block_id in mol.id2idx: return [block_id]
    assert block_id[1] == '', f'Only block ids with empty insertion code can be expanded, but got {block_id}'
    ids, ic = [], 0
    while (block_id[0], str(ic)) in mol.id2idx:
        ids.append((block_id[0], str(ic)))
        ic += 1
    return ids


def index_to_numerical_index(obj, index: tuple):
    numerical_index = []
    for i in index:
        numerical_index.append(obj.id2idx[i])
        obj = obj[i]
    return tuple(numerical_index)


def overwrite_block(cplx: Complex, index: tuple, block: Block):
    if not isinstance(index[0], int): index = index_to_numerical_index(cplx, index)
    mol = cplx[index[0]]
    mol.blocks[index[1]] = block
    # delete the original bonds related to this block
    new_bonds = []
    for bond in cplx.bonds:
        if bond.index1[0] == index[0] and bond.index1[1] == index[1]:
            continue
        elif bond.index2[0] == index[0] and bond.index2[1] == index[1]:
            continue
        else: new_bonds.append(bond)
    cplx.bonds = new_bonds


def format_standard_aa_block(block: Block) -> Block:
    assert is_standard_aa(block)
    # TODO: rename atoms according to the amino acid type
    # i.e. for atom in block: atom.name = formatted name (N, CA, C, O, CB, ...)
    return None


def renumber_res_id(res_ids: List[tuple]):
    # assume res_ids are ordered within each chain
    offset_map = {}
    new_res_ids = []
    for chain, (res_nb, insert_code) in res_ids:
        if chain not in offset_map: offset_map[chain] = 0
        if insert_code == '':
            new_res_ids.append((chain, (res_nb + offset_map[chain], '')))
        else:
            offset_map[chain] += 1
            new_res_ids.append((chain, (res_nb + offset_map[chain], '')))
    return new_res_ids


def _is_peptide_bond(cplx, start_id, end_id, bond_type):
    start_block, end_block = recur_index(cplx, start_id[:-1]), recur_index(cplx, end_id[:-1])
    start_atom, end_atom = recur_index(cplx, start_id), recur_index(cplx, end_id)

    both_aa = is_standard_aa(start_block.name) and is_standard_aa(end_block.name)
    single_bond = bond_type == BondType.SINGLE

    if not (both_aa and single_bond): return False

    # sequence distance
    seq_dist = end_block.id[0] - start_block.id[0]
    if seq_dist == 0: # same position id. different insertion code
        start_iccode = start_block.id[1]
        end_iccode = end_block.id[1]
        if len(start_iccode) > 1 or len(end_iccode) > 1: return False # weird residue id, must not be peptide
        if start_iccode == '': start_iccode = chr(ord('A') - 1)
        if end_iccode == '': end_iccode = chr(ord('A') - 1)
        seq_dist = ord(end_iccode) - ord(start_iccode)
    if abs(seq_dist) != 1: return False # not consecutive
    if seq_dist == -1: start_atom, end_atom = end_atom, start_atom  # swap

    return start_atom.name == 'C' and end_atom.name == 'N'


def _wrap_coord(val, width):
    val = str(round(val, 3))
    if len(val) > width:
        warnings.warn(f'coordinate overflow: {val}')
        val = val[:width]
        val = val.rstrip('.') # in case the dot is accidentally the last one
    return val.rjust(width)


def create_rw_mol_with_coord(lig_array) -> Chem.Mol:
    # extract ligand information
    atoms = lig_array.element.tolist()
    coordinates = list(map(tuple, lig_array.coord.tolist())) # coordinates -> list[tuple]
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atoms)) 
    rdkit_atom_mapping = {}
    for i, (atom, coord) in enumerate(zip(atoms, coordinates)):
        rd_atom = Chem.Atom(atom)
        rd_atom.SetNoImplicit(True)
        idx = mol.AddAtom(rd_atom)
        conf.SetAtomPosition(idx, coord)  
        rdkit_atom_mapping[i] = idx 

    # add conformer
    mol.AddConformer(conf,assignId=True)
    rdDetermineBonds.DetermineConnectivity(mol)
    return mol

def assign_bond(refmol: Chem.Mol, mol: Chem.Mol) -> Chem.Mol:
    refmol2 = Chem.Mol(refmol)
    mol2 = Chem.Mol(mol)    
    coords = np.array(mol2.GetConformer().GetPositions()).astype(np.float64)
    refmol3 = Chem.Mol(refmol)
    conf = mol2.GetConformer()

    try: # Assume that ref_mol and mol have exactly the same atom orders
        assert all([mol2.GetAtomWithIdx(i).GetSymbol() == refmol2.GetAtomWithIdx(i).GetSymbol() for i in range(refmol2.GetNumAtoms())]), 'Switch to substruct matching'
        conf.SetPositions(coords)
    except (AssertionError, ValueError):
        # -Backup mapping strategy - to find matched substruct
        # Set bonds to SINGLE
        for b in mol2.GetBonds():
            if b.GetBondType() != Chem.BondType.SINGLE:
                b.SetBondType(Chem.BondType.SINGLE)
                b.SetIsAromatic(False)
        for b in refmol2.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        # Set atom charges to zero;
        for a in refmol2.GetAtoms():
            a.SetFormalCharge(0)
        for a in mol2.GetAtoms():
            a.SetFormalCharge(0)
        matching = mol2.GetSubstructMatches(refmol2, uniquify=False)
        conf.SetPositions(coords[matching[0],:])

    refmol3.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(refmol3)
    return refmol3

def mapping_bond_order_from_ref_smiles(mol, smiles) -> Chem.Mol:
    # get reference molecule from smiles
    params = Chem.SmilesParserParams()
    params.removeHs = True
    ref_mol = Chem.RemoveAllHs(Chem.MolFromSmiles(smiles, params))
    Chem.Kekulize(ref_mol, clearAromaticFlags=True)
    assert ref_mol.GetNumAtoms() == mol.GetNumAtoms(), f"when assigning bond order, atom number {ref_mol.GetNumAtoms()} and is {mol.GetNumAtoms()} not equal."
    prepared_ligand = assign_bond(ref_mol, mol)
    return prepared_ligand


def assign_new_chain_id(all_chain_ids):
    id_dict = { c: True for c in all_chain_ids }
    c = 'A'
    while c in id_dict:
        c = chr(ord(c) + 1)
    if c >= 'Z':
        warnings.warn(f'All characters from A to Z are occupied during chain ID assignment. Trying lower-case characters')
        c = 'a'
        while c in id_dict:
            c = chr(ord(c) + 1)
        if c >= 'z':
            warnings.warn(f'All characters from A to Z and a to z are occupied during chain ID assignment. Have to use space')
            c = ' '
    return c


def extract_sub_complex(cplx: Complex, ids: List[tuple]):
    # assume block ids are ordered
    blocks = [recur_index(cplx, block_id) for block_id in ids]
    chain2blocks = {}
    for block, _id in zip(blocks, ids):
        chain_id = _id[0]
        if chain_id not in chain2blocks: chain2blocks[chain_id] = []
        chain2blocks[chain_id].append(block)
    mols, block_id2new_num_id = [], {}
    for i, chain_id in enumerate(chain2blocks):
        mols.append(Molecule(chain_id, chain2blocks[chain_id], chain_id))
        if chain_id not in block_id2new_num_id: block_id2new_num_id[chain_id] = {}
        for j, block in enumerate(chain2blocks[chain_id]):
            block_id2new_num_id[chain_id][tuple(block.id)] = (i, j)
    # Initialize a list to store bonds
    new_bonds = []
    # Iterate through the bonds in the original complex and select the relevant ones
    for bond in cplx.bonds:
        c1, c2 = cplx[bond.index1[0]].id, cplx[bond.index2[0]].id
        if (c1 not in chain2blocks) or (c2 not in chain2blocks): continue
        block_id1 = tuple(recur_index(cplx, bond.index1[:2]).id)
        block_id2 = tuple(recur_index(cplx, bond.index2[:2]).id)
        
        # If the bond connects atoms from two different blocks that are in the sub-complex
        if block_id1 in block_id2new_num_id[c1] and block_id2 in block_id2new_num_id[c2]:
            # Adjust the bond indices for the sub-complex
            new_index1 = block_id2new_num_id[c1][block_id1] + (bond.index1[2],)
            new_index2 = block_id2new_num_id[c2][block_id2] + (bond.index2[2],)
            
            # Add the bond to the list of new bonds
            new_bonds.append(Bond(new_index1, new_index2, bond.bond_type))
    
    # Return the new sub-complex with updated bonds
    return Complex(cplx.name, mols, new_bonds)