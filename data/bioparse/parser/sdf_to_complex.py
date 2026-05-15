# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional

from rdkit import Chem

from ..hierarchy import Bond, Atom, Block, Molecule, Complex, BondType
from ..tokenizer.tokenize_3d import ID2BOND, tokenize_3d, TOKENIZER


def sdf_to_complex(
        sdf_file: str,
        remove_Hs: bool=True,
        atom_level: bool=False,
    ) -> Complex:
    '''
        Convert SDF file to Complex.
        
        Parameters:
            sdf_file: Path to the SDF file
            remove_Hs: Whether to remove all hydrogens
            atom_level: Define blocks as single atoms

        Returns:
                A Complex instance
    '''

    # Read SDF file
    supplier = Chem.SDMolSupplier(sdf_file)

    # Molecules (chains) and bonds containers
    molecules = []
    bonds = []
    atom_cnt = 0

    for mol_idx, mol in enumerate(supplier): # RDKit mol, one molecule only has one block
        if TOKENIZER.kekulize: Chem.Kekulize(mol, clearAromaticFlags=True)
        conf = mol.GetConformer(0)
        mol_smi = Chem.MolToSmiles(mol)
        atoms = []
        rdkit_atom_idx2idx = {}
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol == 'H' and remove_Hs: continue
            rdkit_atom_idx2idx[atom.GetIdx()] = len(atoms)
            atom_cnt += 1
            atoms.append(Atom(
                name=symbol,
                coordinate=list(conf.GetAtomPosition(atom.GetIdx())),
                element=symbol,
                id=str(atom_cnt)
            ))

        block_bonds = []
        for bond in mol.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if begin not in rdkit_atom_idx2idx or end not in rdkit_atom_idx2idx: continue
            block_bonds.append((
                rdkit_atom_idx2idx[begin],
                rdkit_atom_idx2idx[end],
                int(ID2BOND.index(bond.GetBondType()))
            ))

        if atom_level:
            rdkit_frags, rdkit_atom_idxs = [], []
            for atom in mol.GetAtoms():
                rdkit_frags.append(f'[{atom.GetSymbol()}]')
                rdkit_atom_idxs.append([atom.GetIdx()])
        else:
            rdkit_frags, rdkit_atom_idxs = tokenize_3d(None, None, rdkit_mol=mol)
        frags, atom_idxs = [], []
        for f, idxs in zip(rdkit_frags, rdkit_atom_idxs):
            idxs = [rdkit_atom_idx2idx[i] for i in idxs if i in rdkit_atom_idx2idx]
            if len(idxs) > 0:
                frags.append(f)
                atom_idxs.append(idxs)
        # frags, atom_idxs = tokenize_3d(
        #     [atom.get_element() for atom in atoms],
        #     [atom.get_coord() for atom in atoms],
        #     bonds=block_bonds
        # )

        blocks, idx_mapping = [], {}
        for frag_idx, (smi, atom_idx) in enumerate(zip(frags, atom_idxs)):
            blocks.append(Block(
                name=smi,
                atoms=[atoms[i] for i in atom_idx],
                id=(1, str(frag_idx)),
                properties={'original_name': mol_smi}
            ))
            for local_i, i in enumerate(atom_idx): idx_mapping[i] = (frag_idx, local_i)
        
        # add bonds
        for begin, end, bond_type in block_bonds:
            bonds.append(Bond(
                index1=(mol_idx,) + idx_mapping[begin],
                index2=(mol_idx,) + idx_mapping[end],
                bond_type=BondType(bond_type)
            ))

        props = {'smiles': mol_smi}
        props.update(mol.GetPropsAsDict())

        mol_name = mol.GetProp('_Name') if mol.HasProp('_Name') else str(mol_idx)

        molecules.append(Molecule(
            name=mol_name,
            blocks=blocks,
            properties=props,
            id=mol_name[0]
        ))
        
    # Create and return the Complex
    cplx = Complex(name=sdf_file.strip('.sdf'), molecules=molecules, bonds=bonds)
    return cplx 


if __name__ == '__main__':
    import sys
    complex = sdf_to_complex(sys.argv[1])
    print(f'{sys.argv[1]} parsed')
    print(f'number of chains: {len(complex)}, number of bonds: {len(complex.bonds)}')
    for molecule in complex:
        print(molecule.id, ', number of blocks: ', len(molecule))
        print([block.name for block in molecule])

    for i, bond in enumerate(complex.bonds[:5]):
        print(f'Bond {i}:')
        print(complex.get_atom(bond.index1), complex.get_atom(bond.index2), bond.bond_type)
        print(complex.get_block(bond.index1), complex.get_block(bond.index2))