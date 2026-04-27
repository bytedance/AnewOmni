# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import tempfile
from typing import Dict, List, Optional, Union

import biotite
from Bio import PDB
from rdkit import Chem
from biotite.structure.io.pdb import PDBFile
from biotite.structure import BondType as BT

from ._biotite_to_complex import atom_array_to_complex

from ..hierarchy import Bond, Atom, Block, Molecule, Complex
from ..utils import is_aa, bond_type_from_rdkit, is_standard_aa, bond_type_from_biotite
from ..tokenizer.tokenize_3d import ID2BOND, tokenize_3d, TOKENIZER


SOLVENTS = ['HOH', 'EDO', 'BME']


def sort_pdb_residues(input_pdb, output_pdb):
    parser = PDB.PDBParser(QUIET=True)
    io = PDB.PDBIO()
    structure = parser.get_structure("structure", input_pdb)

    # Create a new structure to hold sorted residues
    new_structure = PDB.Structure.Structure("sorted_structure")
    for model in structure:
        new_model = PDB.Model.Model(model.id)
        for chain in model:
            new_chain = PDB.Chain.Chain(chain.id)
            sorted_residues = sorted(chain, key=lambda res: res.id[1])  # Sort by residue number
            for residue in sorted_residues:
                new_chain.add(residue)
            new_model.add(new_chain)
        new_structure.add(new_model)

    io.set_structure(new_structure)
    io.save(output_pdb)


def pdb_to_complex(
        pdb_file: str,
        selected_chains: Optional[List[str]]=None,
        remove_Hs: bool=True,
        remove_sol: bool=True,
        remove_het: bool=False
    ) -> Complex:
    '''
        Convert pdb file to Complex.
        Each chain will be a Molecule.
        
        Parameters:
            pdb: Path to the pdb file
            selected_chains: List of selected chain ids. The returned list will be ordered
                according to the ordering of chain ids in this parameter. If not specified,
                all chains will be returned. e.g. ['A', 'B']
            remove_Hs: Whether to remove all hydrogens
            remove_sol: Whether to remove all solvent molecules
            remove_het: Whether to remove all HETATM

        Returns:
                A Complex instance
    '''

    try:
        file = PDBFile.read(pdb_file)
        struct = file.get_structure(include_bonds=True, extra_fields=['atom_id', 'b_factor', 'occupancy'])
    except biotite.InvalidFileError:
        # reorder the pdb
        temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".pdb")
        sort_pdb_residues(pdb_file, temp_file.name)
        file = PDBFile.read(temp_file.name)
        struct = file.get_structure(include_bonds=True, extra_fields=['atom_id', 'b_factor', 'occupancy'])
        temp_file.close()
        # os.remove(temp_file.name)

    assert TOKENIZER.kekulize

    return atom_array_to_complex(
        struct, pdb_file.rstrip('.pdb'), selected_chains, remove_Hs, remove_sol, remove_het
    )

    # Molecules (chains) and bonds containers
    molecules = []
    bonds = []

    # Step 1: Group atoms into blocks and molecules
    residue_atoms = {}  # Dict to collect atoms by residue
    residue_names = {}
    chain_residues = {}  # Dict to collect residues by chain

    atomid2biotiteidx = {}

    for i in range(struct.array_length()):
        atom = struct[0][i]
        chain_id = str(atom.chain_id)
        if selected_chains is not None and chain_id not in selected_chains:
            continue
        if atom.hetero and remove_het: continue
        res_name = str(atom.res_name)
        if res_name in SOLVENTS and remove_sol:
            continue
        res_number = int(atom.res_id)
        insert_code = str(atom.ins_code).strip()
        res_id = (res_number, insert_code)

        if atom.element == 'H' and remove_Hs: continue

        # Create an Atom instance
        atom_instance = Atom(
            name = atom.atom_name.strip(),
            coordinate = atom.coord.tolist(),
            element = atom.element,
            id = str(struct.atom_id[i]),
            properties = {
                'bfactor': float(struct.b_factor[i]),
                'occupancy': float(struct.occupancy[i])
            })
        atomid2biotiteidx[atom_instance.id] = i
        
        # Group atoms by residue (res_number, insert_code) and chain (chain_id)
        if (chain_id, res_id) not in residue_atoms:
            residue_atoms[(chain_id, res_id)] = []
            residue_names[(chain_id, res_id)] = res_name
        residue_atoms[(chain_id, res_id)].append(atom_instance)
        assert residue_names[(chain_id, res_id)] == res_name        

    # Step 2: Create Blocks (residues) and group them into Molecules (chains)
    # For non standard residues (e.g. non-canonical amino acids and small molecules),
    # use principal subgraphs divide them into fragments
    for (chain_id, res_id), atoms in residue_atoms.items():
        res_name = residue_names[(chain_id, res_id)]
        block = Block(name=res_name, atoms=atoms, id=res_id)
        if chain_id not in chain_residues:
            chain_residues[chain_id] = []
        if is_standard_aa(res_name):
            chain_residues[chain_id].append(block)
        else: # fragmentation
            # get all bonds
            block_bonds = []
            biotite_atom_id2block_atom_id = {}
            for block_atom_id, atom in enumerate(atoms):
                biotite_atom_id2block_atom_id[atomid2biotiteidx[atom.id]] = block_atom_id
            for atom in atoms:
                for end_idx, bond_type in zip(*struct.bonds.get_bonds(atomid2biotiteidx[atom.id])):
                    begin_idx = atomid2biotiteidx[atom.id]
                    if end_idx <= begin_idx: continue # avoid repeating bonds
                    if end_idx not in biotite_atom_id2block_atom_id: continue # not in this block
                    bond_type_int = BT(bond_type).without_aromaticity().value
                    if bond_type_int == 0: bond_type_int = 1 # TODO: bonds in CONECT records
                    block_bonds.append((
                        biotite_atom_id2block_atom_id[begin_idx],
                        biotite_atom_id2block_atom_id[end_idx],
                        bond_type_int
                    ))
                    assert bond_type_int < 4
            frags, atom_idxs = tokenize_3d(
                [atom.get_element() for atom in atoms],
                [atom.get_coord() for atom in atoms],
                bonds=block_bonds
            )
            for frag_idx, (smi, atom_idx) in enumerate(zip(frags, atom_idxs)):
                chain_residues[chain_id].append(Block(
                    name=smi,
                    atoms=[atoms[i] for i in atom_idx],
                    id=(res_id[0], res_id[1] + str(frag_idx)),
                    properties={'original_name': res_name}
                ))

    # Create Molecules from Blocks
    for chain_id, blocks in chain_residues.items():
        # non-amino acid residues are actually small molecules in PDB format (e.g. PDB ID: 6ueg)
        new_blocks = []
        for block in blocks:
            if len(block) > 0: new_blocks.append(block)
        new_blocks = sorted(new_blocks, key=lambda block: block.id) # sorted by (res_number, insert_code)
        molecule = Molecule(name=chain_id, blocks=new_blocks, id=chain_id)
        molecules.append(molecule)

    # create mapping
    atom_to_molecule_block_atom = {}  # RDKit atom index -> (mol_idx, block_idx, atom_idx)
    for mol_idx, molecule in enumerate(molecules):
        for block_idx, block in enumerate(molecule):
            for atom_idx, atom in enumerate(block):
                atom_to_molecule_block_atom[atomid2biotiteidx[atom.id]] = (mol_idx, block_idx, atom_idx)

    # Step 3: Detect bonds and store them
    end_atoms, bond_types = struct.bonds.get_all_bonds()
    for begin_idx in range(len(end_atoms)):
        for end_idx, bond_type in zip(end_atoms[begin_idx], bond_types[begin_idx]):
            if end_idx < 0: continue
            if end_idx <= begin_idx: continue   # avoid repeating bonds
            if begin_idx not in atom_to_molecule_block_atom or end_idx not in atom_to_molecule_block_atom:
                continue
            index1 = atom_to_molecule_block_atom[begin_idx]
            index2 = atom_to_molecule_block_atom[end_idx]

            # Create Bond instance
            bond_instance = Bond(index1=index1, index2=index2, bond_type=bond_type_from_biotite(bond_type))
            bonds.append(bond_instance)

    # Step 4: Create and return the Complex
    cplx = Complex(name=pdb_file.strip('.pdb'), molecules=molecules, bonds=bonds)
    return cplx 


if __name__ == '__main__':
    import sys
    complex = pdb_to_complex(sys.argv[1])
    print(f'{sys.argv[1]} parsed')
    print(f'number of chains: {len(complex)}, number of bonds: {len(complex.bonds)}')
    for molecule in complex:
        print(molecule.id, len(molecule))

    for i, bond in enumerate(complex.bonds[:5]):
        print(f'Bond {i}:')
        print(complex.get_atom(bond.index1), complex.get_atom(bond.index2), bond.bond_type)
        print(complex.get_block(bond.index1), complex.get_block(bond.index2))

    print(complex['U'])