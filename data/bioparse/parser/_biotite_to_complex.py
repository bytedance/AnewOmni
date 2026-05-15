# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List, Optional

from biotite.structure import BondType as BT

from ..hierarchy import Bond, Atom, Block, Molecule, Complex
from ..utils import is_standard_block, bond_type_from_biotite
from ..tokenizer.tokenize_3d import tokenize_3d


SOLVENTS = ['HOH', 'EDO', 'BME']


def atom_array_to_complex(
        struct,
        name,
        selected_chains: Optional[List[str]]=None,
        remove_Hs: bool=True,
        remove_sol: bool=True,
        remove_het: bool=False,
        unknown_bond_default: int=1 # default single
    ):
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
        if is_standard_block(res_name):
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
                    if bond_type_int == 0: bond_type_int = unknown_bond_default # bonds in CONECT records without known type
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
    cplx = Complex(name=name, molecules=molecules, bonds=bonds)
    return cplx 


def protenix_atom_array_to_complex(
    atom_array,
    name,
    selected_chains: Optional[List[str]]=None,
    unknown_bond_default: int=1 # default single bond
):
    # Molecules (chains) and bonds containers
    molecules = []
    bonds = []
    
    # Step 1: Group atoms into blocks and molecules
    residue_atoms = {}  # Dict to collect atoms by residue
    residue_names = {}
    chain_residues = {}  # Dict to collect residues by chain

    atomid2biotiteidx = {}
    
    for i in range(len(atom_array)):
        _atom = atom_array[i]
        chain_id = str(_atom.chain_id)
        if selected_chains is not None and chain_id not in selected_chains: continue
        res_name = str(_atom.res_name)
        if res_name in ['H', 'HOH', 'EDO', 'BME']: continue # hydrogen or solvent
        res_number = int(_atom.res_id)
        insert_code = str(_atom.ins_code).strip()
        res_id = (res_number, insert_code)
        
        # Create an Atom instance
        atom_instance = Atom(
            name = _atom.atom_name.strip(),
            coordinate = _atom.coord.tolist(),
            element = _atom.element,
            id = str(i)
            )
        
        atomid2biotiteidx[atom_instance.id] = i # exactly the same
        
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
        if is_standard_block(res_name):
            chain_residues[chain_id].append(block)
        else: # fragmentation
            # get all bonds
            block_bonds = []
            biotite_atom_id2block_atom_id = {}
            for block_atom_id, atom in enumerate(atoms):
                biotite_atom_id2block_atom_id[atomid2biotiteidx[atom.id]] = block_atom_id
            for atom in atoms:
                for end_idx, bond_type in zip(*atom_array.bonds.get_bonds(atomid2biotiteidx[atom.id])):
                    begin_idx = atomid2biotiteidx[atom.id]
                    if end_idx <= begin_idx: continue # avoid repeating bonds
                    if end_idx not in biotite_atom_id2block_atom_id: continue # not in this block
                    bond_type_int = BT(bond_type).without_aromaticity().value
                    if bond_type_int == 0: bond_type_int = unknown_bond_default # bonds in CONECT records without known type
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
    end_atoms, bond_types = atom_array.bonds.get_all_bonds()
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
    cplx = Complex(name=name, molecules=molecules, bonds=bonds)
    return cplx 
        
        