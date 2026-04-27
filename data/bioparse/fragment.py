# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem.BRICS import FindBRICSBonds

from .hierarchy import Complex, Molecule, Block, Bond, BondType
from .utils import is_aa, bond_type_to_rdkit
from .vocab import VOCAB
from . import const



def block_to_rdkit(block: Block, bonds: List[Bond], remove_atoms: List[int]=None):
    # Create an empty RDKit molecule
    mol = Chem.RWMol()
    
    # Add atoms to the molecule
    for i, atom in enumerate(block):
        rdkit_atom = Chem.Atom(atom.get_element())
        rdkit_atom.SetProp('original_idx', str(i))
        mol.AddAtom(rdkit_atom)

    # Add bonds to the molecule
    for bond in bonds:
        atom1, atom2 = bond.index1[-1], bond.index2[-1]
        mol.AddBond(int(atom1), int(atom2), bond_type_to_rdkit(bond.bond_type))
    
    # # Add coordinates to the molecule
    # conformer = Chem.Conformer(len(atom_indices))
    # for idx, atom in zip(atom_indices, block.atoms):
    #     conformer.SetAtomPosition(idx, np.array(atom.get_coord()))
    
    # mol.AddConformer(conformer)

    # delete atoms
    if remove_atoms is not None:
        for i in sorted(remove_atoms, reverse=True): mol.RemoveAtom(i)


    # Finalize the molecule
    mol.UpdatePropertyCache(strict=False)
    mol = mol.GetMol()
    
    return mol


def brics_molecule(mol):
    """
    adapted from https://github.com/zhangruochi/pepland/blob/993c4b6368c5823d19ee2fc64e23e9338f8f28ec/tokenizer/pep2fragments.py#L327
    Fragment a molecule using the BRICS algorithm.
    Args:
        smiles (str or RDKit mol): A SMILES string representing the molecule.
    Returns:
        list of str: A list of SMILES strings representing the fragments.
    """
    # Convert SMILES to RDKit molecule object
    if isinstance(object, str):
        mol = Chem.MolFromSmiles(mol)

    try:
        Chem.SanitizeMol(mol)
    except ValueError as e:
        return None, None, None
        return [], [mol], [Chem.MolToSmiles(mol)]
    
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0],i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol,break_bonds,addDummies=False)
    all_fragments = Chem.GetMolFrags(tmp, asMols=True)
    
    return break_bonds, all_fragments, [Chem.MolToSmiles(x, canonical = True) for x in all_fragments]


def brics_sidechain(block: Block, bonds: List[Bond], backbone_atoms: List[int]):

    rdkit_mol = block_to_rdkit(block, bonds, remove_atoms=backbone_atoms)
    # if rdkit_mol is None: # try to using existing bonds
    #     # if it is standard amino acid, use existing bonds for better robustness
    #     print(block.name)
    #     if block.name in const.AA_GEOMETRY:
    #         bonds = []
    #         name2idx = {}
    #         for i, atom in enumerate(block): name2idx[atom.name] = i
    #         known_bonds = const.backbone_bonds + const.sidechain_bonds[VOCAB.abrv_to_symbol(block.name)]
    #         if 'OXT' in name2idx: known_bonds += [('C', 'OXT', 1)]
    #         for a1, a2, b in known_bonds:
    #             bonds.append(Bond(
    #                 index1=(0, 0, name2idx[a1]),
    #                 index2=(0, 0, name2idx[a2]),
    #                 bond_type=BondType(b)
    #             ))
    #     rdkit_mol = block_to_rdkit(block, bonds, remove_atoms=backbone_atoms)

    _, frags, smis = brics_molecule(rdkit_mol)

    if frags is not None:

        # construct blocks for each fragment
        blocks = []
        for f, s in zip(frags, smis):
            atoms = []
            for atom in f.GetAtoms():
                i = int(atom.GetProp('original_idx'))
                atoms.append(block.atoms[i])
            blocks.append(Block(
                name=s,
                atoms=atoms,
                id=block.id,
                properties={'aa': block.name}
            ))
    else: # cannot handled by RDKit, directly use atoms
        blocks, smis = [], []
        for atom in block:
            blocks.append(Block(
                name=atom.element,
                atoms=[atom],
                id=block.id,
                properties={'aa': block.name}
            ))
            smis.append(atom.element)

    return blocks, smis


def brics_amino_acid(block: Block, bonds: List[Bond]):
    assert is_aa(block), f'block {block} not an amino acid'
    
    frag_blocks, frag_smis = [], []

    # Get backbone
    bb_atoms, has_OXT, bb_atom_idx = [], False, []
    for i, atom in enumerate(block):
        if atom.name in const.backbone_atoms:
            bb_atoms.append(atom)
            bb_atom_idx.append(i)
        elif atom.name == 'OXT':
            bb_atoms.append(atom)
            bb_atom_idx.append(i)
            has_OXT = True
    bb_smi = 'NCC(=O)O' if has_OXT else 'NCC=O'
    frag_blocks.append(Block(name=bb_smi, atoms=bb_atoms, id=block.id, properties={'aa': block.name}))
    frag_smis.append(bb_smi)

    # Fragmentation of sidechain
    sc_frag_blocks, sc_frag_smi = brics_sidechain(block, bonds, bb_atom_idx)
    frag_blocks.extend(sc_frag_blocks)
    frag_smis.extend(sc_frag_smi)

    return frag_blocks, frag_smis


def brics_small_molecule(block: Block, bonds: List[Bond]):
    return brics_sidechain(block, bonds, []) # no backbone


def brics_block(block: Block, bonds: List[Bond]) -> Tuple[List[Block], List[str]]:
    if is_aa(block): # amino acid
        frag_blocks, frag_smi = brics_amino_acid(block, bonds)
    else:
        frag_blocks, frag_smi = brics_small_molecule(block, bonds)
    # add another code (0, 1, 2, ...) to block id to indicate these blocks form one residue
    for i, block in enumerate(frag_blocks):
        assert len(block.id) == 2
        block.id = (block.id[0], block.id[1] + str(i)) # e.g. (1, '0'), or (1, 'A0') if already with insert code
    return frag_blocks, frag_smi


def brics_complex(cplx: Complex) -> Complex:
    '''
        Use BRICS to do fragmentation on a given complex
    '''
    idx_mapping = {} # from original tuple[int, int, int] to new ones
    molecules = []
    for mol_idx, mol in enumerate(cplx):
        blocks = []
        for block_idx, block in enumerate(mol):
            frag_blocks, _ = brics_block(block, cplx.get_block_inner_bonds((mol_idx, block_idx)))
            # change atom mapping
            atom_id2new_block_atom_id = {}
            for i, f_blk in enumerate(frag_blocks):
                for j, atom in enumerate(f_blk):
                    atom_id2new_block_atom_id[atom.id] = (len(blocks) + i, j)
            for atom_idx, atom in enumerate(block):
                idx_mapping[(mol_idx, block_idx, atom_idx)] = (mol_idx,) + atom_id2new_block_atom_id[atom.id]
            blocks.extend(frag_blocks)
        molecules.append(Molecule(
            name=mol.name,
            blocks=blocks,
            id=mol.id,
            properties=mol.properties
        ))
    
    # remap index in bonds
    bonds = []
    for bond in cplx.bonds:
        bonds.append(Bond(
            index1=idx_mapping[bond.index1],
            index2=idx_mapping[bond.index2],
            bond_type=bond.bond_type
        ))

    return Complex(
        name=cplx.name,
        molecules=molecules,
        bonds=bonds,
        properties=cplx.properties
    )


if __name__ == '__main__':
    from .hierarchy import Atom, BondType
    block = Block(
        name='ALA',
        atoms=[
            Atom('N', [-14.4, 8.8, 35.0], 'N', 1),
            Atom('CA', [-14.6, 7.3, 35.2], 'C', 2),
            Atom('C', [-13.4, 6.6, 35.7], 'C', 3),
            Atom('O', [-12.3, 7.1, 35.6], 'O', 4),
            Atom('CB', [-15.1, 6.7, 33.8], 'C', 5),
            Atom('OXT', [-12.3, 8.1, 35.6], 'O', 6),
        ],
        id=(1, '')
    )
    bonds = [
        Bond([0, 0, 0], [0, 0, 1], BondType.SINGLE),
        Bond([0, 0, 1], [0, 0, 2], BondType.SINGLE),
        Bond([0, 0, 2], [0, 0, 3], BondType.DOUBLE),
        Bond([0, 0, 1], [0, 0, 4], BondType.SINGLE),
        Bond([0, 0, 2], [0, 0, 5], BondType.SINGLE),
    ]
    mol = block_to_rdkit(block, bonds)
    print(Chem.MolToSmiles(mol))

    blocks, smis = brics_amino_acid(block, bonds)
    assert len(blocks) == len(smis)
    for b, s in zip(blocks, smis):
        print(s)
        print(b)