# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from enum import Enum
from copy import copy, deepcopy
from dataclasses import dataclass
from typing import List, Iterator, Optional, Tuple, Union, Dict


class BondType(Enum):
    NONE = 0
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4

    @classmethod
    def to_valence(cls, bond):
        if bond == cls.AROMATIC:
            return 1.5
        return bond.value


@dataclass
class Bond:
    index1: Tuple[int, int, int] # numerical index for molecule, block, atom
    index2: Tuple[int, int, int] # numerical index for molecule, block, atom
    bond_type: BondType

    def to_tuple(self):
        return (self.index1, self.index2, self.bond_type.value)

    @classmethod
    def from_tuple(cls, tup):
        return Bond(tuple(tup[0]), tuple(tup[1]), BondType(tup[2]))


class Atom:
    def __init__(self, name: str, coordinate: List[float], element: str, id: str, properties: Optional[dict]=None):
        self.name = name
        self.coordinate = coordinate
        self.element = element
        self.id = id  # usually a global atom id in PDB file (but transformed to str to differentiate with index)
        self.properties: dict = {} if properties is None else deepcopy(properties)

    def get_element(self):
        return self.element
    
    def get_coord(self):
        return copy(self.coordinate)
    
    def get_property(self, name: str, default_value=None):
        return self.properties.get(name, default_value)
    
    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Atom ({self.name}, {self.id}): {self.element} [{','.join(['{:.4f}'.format(num) for num in self.coordinate])}], {self.properties}"
    
    def to_tuple(self):
        return (
            self.name,
            self.coordinate,
            self.element,
            self.id,
            self.properties
        )
    
    @classmethod
    def from_tuple(cls, data):
        return Atom(
            name=data[0],
            coordinate=data[1],
            element=data[2],
            id=data[3],
            properties=data[4]
        )


class Block:
    def __init__(self, name: str, atoms: List[Atom], id: Tuple[int, str], properties: Optional[dict]=None) -> None:
        self.name: str = name 
        self.atoms: List[Atom] = atoms 
        self.id = id # e.g. (198, ''), number + insertion code, unique
        self.properties: dict = {} if properties is None else deepcopy(properties)
        
        self.id2idx = { atom.id: idx for idx, atom in enumerate(self) }

    def __len__(self) -> int:
        return len(self.atoms)
    
    def __iter__(self) -> Iterator[Atom]:
        return iter(self.atoms)
    
    def __getitem__(self, idx: Union[int, str]) -> Atom:
        if isinstance(idx, str):
            idx = self.id2idx[idx]
            return self.atoms[idx]
        elif isinstance(idx, int):
            return self.atoms[idx]
        else:
            raise ValueError(f'Index of {type(self)} can only be str or int, but {type(idx)} is given')
    
    def get_property(self, name: str):
        return self.properties.get(name, None)

    def to_tuple(self):
        return (
            self.name,
            [atom.to_tuple() for atom in self.atoms],
            self.id,
            self.properties
        )
   
    @classmethod
    def from_tuple(cls, data):
        return Block(
            name=data[0],
            atoms=[Atom.from_tuple(atom_data) for atom_data in data[1]],
            id=tuple(data[2]), # ensure it is tuple instead of list
            properties=data[3]
        )
    
    def __repr__(self) -> str:
        return f"Block ({self.name}), id {self.id}:\n\t" + '\n\t'.join([repr(at) for at in self.atoms]) + '\n' + f'properties: {self.properties}\n'
    

class Molecule:

    def __init__(self, name: str, blocks: List[Block], id: str, properties: Optional[dict]=None) -> None:
        self.name: str = name
        self.blocks: List[Block] = blocks
        self.id: str = id # chain id (unique)
        self.properties: dict = {} if properties is None else deepcopy(properties)
        
        self.id2idx = { block.id: idx for idx, block in enumerate(self) }

    def __len__(self) -> int:
        return len(self.blocks)
    
    def __iter__(self) -> Iterator[Block]:
        return iter(self.blocks)
    
    def __getitem__(self, idx: Union[int, Tuple[int, str]]) -> Block:
        if isinstance(idx, tuple):
            idx = self.id2idx[idx]
            return self.blocks[idx]
        elif isinstance(idx, int):
            return self.blocks[idx]
        else:
            raise ValueError(f'Index of {type(self)} can only be tuple<int, str> or int, but {type(idx)} is given')
    
    def get_property(self, name: str):
        return self.properties.get(name, None)
    
    def to_tuple(self):
        return (
            self.name,
            [block.to_tuple() for block in self.blocks],
            self.id,
            self.properties
        )
    
    @classmethod
    def from_tuple(cls, data):
        return Molecule(
            name=data[0],
            blocks=[Block.from_tuple(block_data) for block_data in data[1]],
            id=data[2],
            properties=data[3]
        )
    
    def __repr__(self) -> str:
        return f'Molecule ({self.name}), id {self.id}:\n\t' + \
                '\n\t'.join([repr(block) for block in self.blocks]) + '\n' + \
                f'properties: {self.properties}\n'

class Complex:

    def __init__(self, name: str, molecules: List[Molecule], bonds: List[Bond], properties: Optional[dict]=None) -> None:
        self.name: str = name
        self.molecules: List[Molecule] = molecules
        self.bonds: List[Bond] = bonds
        self.properties: dict = {} if properties is None else deepcopy(properties)

        self.id2idx = { mol.id: idx for idx, mol in enumerate(self) }
        self.block_inner_bonds: Dict[Tuple[int, int], List[Bond]] = {} # bonds within each block
        for bond_idx, bond in enumerate(self.bonds):
            block_id1, block_id2 = bond.index1[:2], bond.index2[:2]
            if block_id1[0] == block_id2[0] and block_id1[1] == block_id2[1]:
                if block_id1 not in self.block_inner_bonds: self.block_inner_bonds[block_id1] = []
                self.block_inner_bonds[block_id1].append(bond_idx)
    
    def __len__(self) -> int:
        return len(self.molecules)
    
    def __iter__(self) -> Iterator[Molecule]:
        return iter(self.molecules)
    
    def __getitem__(self, idx: Union[int, str]) -> Molecule:
        if isinstance(idx, str):
            idx = self.id2idx[idx]
            return self.molecules[idx]
        elif isinstance(idx, int):
            return self.molecules[idx]
        else:
            raise ValueError(f'Index of {type(self)} can only be str or int, but {type(idx)} is given')
    
    def get_block(self, index: Tuple[int, int]) -> Block:
        return self.molecules[index[0]].blocks[index[1]]
    
    def get_atom(self, index: Tuple[int, int, int]) -> Atom:
        return self.get_block(index).atoms[index[2]]
    
    def get_block_inner_bonds(self, index: Tuple[int, int]) -> List[Bond]:
        bond_idx = self.block_inner_bonds.get(index, [])
        return [self.bonds[i] for i in bond_idx]
        
    def get_property(self, name: str):
        return self.properties.get(name, None)

    def to_tuple(self):
        return (
            self.name,
            [molecule.to_tuple() for molecule in self.molecules],
            [bond.to_tuple() for bond in self.bonds],
            self.properties
        )
    
    @classmethod
    def from_tuple(cls, data):
        return Complex(
            name=data[0],
            molecules=[Molecule.from_tuple(mol_data) for mol_data in data[1]],
            bonds=[Bond.from_tuple(bond_data) for bond_data in data[2]],
            properties=data[3]
        )
    
    def __repr__(self) -> str:
        return f'Complex ({self.name}):\n\t' + \
                '\n\t'.join([repr(mol) for mol in self.molecules]) + '\n' + \
                f'properties: {self.properties}\n'
    


def merge_cplx(cplx1: Complex, cplx2: Complex, name: str = None) -> Complex:
    if name is None: name = cplx1.name + ' ' + cplx2.name
    merge_properties = deepcopy(cplx1.properties)
    merge_properties.update(deepcopy(cplx2.properties))

    merge_bonds = [deepcopy(bond) for bond in cplx1.bonds]
    for bond in cplx2.bonds:
        merge_bonds.append(Bond(
            index1=(bond.index1[0] + len(cplx1),) + bond.index1[1:],
            index2=(bond.index2[0] + len(cplx1),) + bond.index2[1:],
            bond_type=bond.bond_type
        ))

    return Complex(
        name=name,
        molecules=cplx1.molecules + cplx2.molecules,
        bonds=merge_bonds,
        properties=merge_properties
    )


def remove_mols(cplx: Complex, remove_ids: List[str]) -> Complex:
    cplx = deepcopy(cplx)
    mols = [mol for mol in cplx if mol.id not in remove_ids]
    mol_numerical_ids = [cplx.id2idx[_id] for _id in remove_ids if _id in cplx.id2idx]
    molid_old2new = {}
    for i, mol in enumerate(cplx):
        if mol.id in remove_ids: continue
        molid_old2new[i] = len(molid_old2new)
    bonds = []
    for bond in cplx.bonds:
        if (bond.index1[0] in mol_numerical_ids) or (bond.index2[0] in mol_numerical_ids): continue
        bonds.append(Bond(
            (molid_old2new[bond.index1[0]],) + bond.index1[1:],
            (molid_old2new[bond.index2[0]],) + bond.index2[1:],
            bond.bond_type
        ))
    return Complex(
        name=cplx.name,
        molecules=mols,
        bonds=bonds,
        properties=cplx.properties
    )


def add_dummy_mol(cplx: Complex, size: int, id: str, same_res_id: bool=False) -> Complex:
    cplx = deepcopy(cplx)
    dummy_mol = Molecule(
        name='dummy',
        blocks=[Block(
            name='UNK',
            atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
            id=(1, str(i)) if same_res_id else (i, '')
        ) for i in range(size)],
        id=id,
    )
    return Complex(
        name=cplx.name,
        molecules=cplx.molecules + [dummy_mol],
        bonds=cplx.bonds,
        properties=cplx.properties
    )