# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import deepcopy
from typing import List
import random

import torch
import numpy as np

import utils.register as R
from utils.chem_utils import find_term_and_delete_dummy
from data.bioparse import VOCAB
from data.bioparse.tools import load_cplx
from data.bioparse.utils import recur_index, index_to_numerical_index
from data.bioparse.hierarchy import Atom, Block, Bond, BondType
from data.bioparse import const
from data.file_loader import MolType, PeptideLoader
from models.modules.adapter.model import ConditionConfig

from .base import BaseTemplate, ComplexDesc
from .utils import form_block

from ..helpers.template_creator import Creator


@R.register('LinearPeptide')
class LinearPeptide(BaseTemplate):
    
    def __init__(self, size_min=8, size_max=13):
        super().__init__(size_min, size_max)
    
    @property
    def moltype(self) -> MolType:
        return MolType.PEPTIDE
    
    def default_filter_configs(self) -> List[dict]:
        return super().default_filter_configs() + [ { 'class': 'LTypeAAFilter' } ]

    def dummy_lig_block_bonds(self, cplx_desc, size):
        blocks = [Block(
            name='GLY', # use glycine instead of UNK so that is_aa can be identified as True
            atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
            id=(i + 1, '')
        ) for i in range(size)]
        bonds = []
        return blocks, bonds
    
    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        loader = PeptideLoader(
            tgt_chains=cplx_desc.tgt_chains,
            lig_chains=cplx_desc.lig_chains,
        )
        return loader.cplx_to_data(
            cplx_desc.cplx,
            cplx_desc.pocket_block_ids,
            cplx_desc.lig_block_ids
        )


@R.register('NonStandardAALinearPeptide')
class NonStandardAALinearPeptide(LinearPeptide):

    def __init__(self, num_nsaa_min, num_nsaa_max, size_min=8, size_max=13):
        super().__init__(size_min, size_max)
        self.num_nsaa_min = num_nsaa_min
        self.num_nsaa_max = num_nsaa_max # included
        assert num_nsaa_min > 0
        assert num_nsaa_min <= num_nsaa_max
    
    def dummy_lig_block_bonds(self, cplx_desc, size):
        num_nsaa_min = min(self.num_nsaa_min, size)
        num_nsaa_max = min(self.num_nsaa_max, size)
        num_nsaa = np.random.randint(num_nsaa_min, num_nsaa_max + 1)
        nsaa_idx = np.random.choice(size, num_nsaa, replace=False)
        blocks = []

        for i in range(size):
            if i in nsaa_idx: # add non-standard aa
                nsaa_block_size = 2 # TODO: sample
                for j in range(nsaa_block_size):
                    # offset += 1
                    blocks.append(Block(
                        name='UNK',
                        atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
                        id=(i + 1, str(j))
                    ))
            else:
                blocks.append(Block(
                    name='GLY',
                    atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
                    id=(i + 1, '')
                ))
        bonds = []
        return blocks, bonds
    

@R.register('DiSulfidePeptide')
class DiSulfidePeptide(LinearPeptide):

    def __init__(self, size_min=12, size_max=14, w=1.0, head_to_tail=True):
        super().__init__(size_min, size_max)
        # TODO: sample random positions for cyclization
        self.w = w
        self.head_to_tail = head_to_tail
    
    # def default_filter_configs(self) -> List[dict]:
    #     return super().default_filter_configs() + [ { 'class': 'DisulfideFilter' }]

    def dummy_lig_block_bonds(self, cplx_desc, size):

        cys_atom_names = VOCAB.abrv_to_atoms('CYS')
        cys_elements = VOCAB.abrv_to_elements('CYS')

        cystine_head = Block(
            name='CYS',
            atoms=[Atom(
                name=atom_name,
                coordinate=[0, 0, 0],
                element=element,
                id=i
            ) for i, (atom_name, element) in enumerate(zip(cys_atom_names, cys_elements))],
            id=(1, '')
        )
        cystine_tail = deepcopy(cystine_head)
        cystine_tail.id = (size, '')

        blocks = [cystine_head] + [Block(
            name='GLY',
            atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
            id=(i + 1, '')
        ) for i in range(1, size - 1)] + [cystine_tail]

        disulfide_pairs = [(0, size - 1)]
        sulfer_order = VOCAB.abrv_to_atoms('CYS').index('SG')
        bonds = []

        # add bonds for cystine
        mol_index = len(cplx_desc.cplx)
        for i, block in enumerate(blocks):
            if block.name != 'CYS':
                continue
            for bond in VOCAB.abrv_to_bonds(block.name):
                bonds.append(Bond(
                    (mol_index, i, bond[0]),
                    (mol_index, i, bond[1]),
                    bond[2]
                ))

        # S-S
        for i, j in disulfide_pairs:
            bonds.append(Bond(
                (mol_index, i, sulfer_order),
                (mol_index, j, sulfer_order),
                BondType.SINGLE
            ))

        return blocks, bonds
    
    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        data = super().to_data(cplx_desc)
        # add topo conditioning
        mask_2d = [0 for _ in cplx_desc.pocket_block_ids] + [1 if recur_index(cplx_desc.cplx, id).name == 'CYS' else 0 for id in cplx_desc.lig_block_ids]
        data['condition_config'] = ConditionConfig(
            mask_2d=torch.tensor(mask_2d, dtype=torch.bool),
            mask_3d=None,
            mask_incomplete_2d=None,
            w=self.w
        )
        return data
    

@R.register('HeadTailPeptide')
class HeadTailPeptide(LinearPeptide):

    def __init__(self, size_min=12, size_max=14, w=1.0):
        super().__init__(size_min, size_max)
        self.w = w
    
    def dummy_lig_block_bonds(self, cplx_desc, size):

        head_forbids, tail_forbids = ['PRO', 'CYS'], ['CYS']
        head_aas = [tup[1] for tup in const.aas if tup[1] not in head_forbids]
        tail_aas = [tup[1] for tup in const.aas if tup[1] not in tail_forbids]
        head_aa = np.random.choice(head_aas, size=1)[0]
        tail_aa = np.random.choice(tail_aas, size=1)[0]

        # head
        atom_names = VOCAB.abrv_to_atoms(head_aa)
        elements = VOCAB.abrv_to_elements(head_aa)

        head = Block(
            name=head_aa,
            atoms=[Atom(
                name=atom_name,
                coordinate=[0, 0, 0],
                element=element,
                id=i
            ) for i, (atom_name, element) in enumerate(zip(atom_names, elements))],
            id=(1, '')
        )
        
        # tail
        atom_names = VOCAB.abrv_to_atoms(tail_aa)
        elements = VOCAB.abrv_to_elements(tail_aa)

        tail = Block(
            name=tail_aa,
            atoms=[Atom(
                name=atom_name,
                coordinate=[0, 0, 0],
                element=element,
                id=i
            ) for i, (atom_name, element) in enumerate(zip(atom_names, elements))],
            id=(size, '')
        )

        # compose the ligand
        blocks = [head] + [Block(
            name='GLY',
            atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)],
            id=(i + 1, '')
        ) for i in range(1, size - 1)] + [tail]

        n_order = VOCAB.abrv_to_atoms(head_aa).index('N')
        c_order = VOCAB.abrv_to_atoms(tail_aa).index('C')
        bonds = []

        # add bonds for head and tail
        mol_index = len(cplx_desc.cplx)
        for i, block in enumerate(blocks):
            if (i != 0) and (i != size - 1):
                continue
            for bond in VOCAB.abrv_to_bonds(block.name):
                bonds.append(Bond(
                    (mol_index, i, bond[0]),
                    (mol_index, i, bond[1]),
                    bond[2]
                ))

        # amide connection
        bonds.append(Bond(
            (mol_index, 0, n_order),
            (mol_index, size - 1, c_order),
            BondType.SINGLE
        ))

        return blocks, bonds
    
    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        data = super().to_data(cplx_desc)
        # add topo conditioning
        pep_head_tail_mask = [1] + [0 for _ in range(len(cplx_desc.lig_block_ids) - 2)] + [1]
        mask_2d = [0 for _ in cplx_desc.pocket_block_ids] + pep_head_tail_mask
        data['condition_config'] = ConditionConfig(
            mask_2d=torch.tensor(mask_2d, dtype=torch.bool),
            mask_3d=None,
            mask_incomplete_2d=None,
            w=self.w
        )
        return data
    

@R.register('PartialSeqPeptide')
class ParitalSeqPeptide(LinearPeptide):
    def __init__(self, templates: list, w=1.0):
        super().__init__(size_min=0, size_max=1)    # do not rely on sampled size
        # e.g. [WAYXXXX, XXIALPXXXXX, XXXXXQWY], use X to indicate generation positions
        # for each generation, a random one will be drawn from the templates
        self.templates = templates
        self.w = w

        # dynamically changing
        self.current_template = None

    def dummy_lig_block_bonds(self, cplx_desc, size):
        # sample one template
        template = np.random.choice(self.templates)

        creator = Creator(len(cplx_desc.cplx))
        for i, aa in enumerate(template):
            if aa == 'X': creator.add_unk((i + 1, ''))  # peptide position ids start from 1
            else:
                creator.add_block(VOCAB.symbol_to_abrv(aa), (i + 1, ''))
                if i > 0 and template[i - 1] != 'X':    # add amide bond
                    creator.add_bond_by_atom_name(i - 1, i, 'C', 'N', BondType.SINGLE)

        self.current_template = template
                
        return creator.get_results()
    
    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        data = super().to_data(cplx_desc)
        # add topo conditioning
        mask_2d = [0 for _ in cplx_desc.pocket_block_ids] + [0 if aa == 'X' else 1 for aa in self.current_template]
        data['condition_config'] = ConditionConfig(
            mask_2d=torch.tensor(mask_2d, dtype=torch.bool),
            mask_3d=None,
            mask_incomplete_2d=None,
            w=self.w
        )
        return data


@R.register('PeptideMutant')
class PeptideMutant(LinearPeptide):
    def __init__(self, wild_type_path: str, mutate_positions: List[int], w=1.0):
        super().__init__(size_min=0, size_max=1)    # do not rely on sampled size
        self.wild_type_path = wild_type_path
        self.mutate_positions = mutate_positions    # start from zero
        self.w = w
        self.wt_cplx = load_cplx(self.wild_type_path, cleanup_first=True)

    def remove_ref_lig(self, cplx_desc):
        return  None    # do not remove

    def add_dummy_lig(self, cplx_desc):
        return deepcopy(self.wt_cplx)   # substitute with the wild type

    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        data = super().to_data(cplx_desc)
        # add topo conditioning
        pep_mut_mask = [1 for _ in range(len(cplx_desc.lig_block_ids))]
        for i in self.mutate_positions:
            if i < len(pep_mut_mask): pep_mut_mask[i] = 0
        mask_2d = [0 for _ in cplx_desc.pocket_block_ids] + pep_mut_mask
        mask_3d = [0 for _ in cplx_desc.pocket_block_ids] + [1 for _ in pep_mut_mask]   # the coordinates should be at similar places
        data['condition_config'] = ConditionConfig(
            mask_2d=torch.tensor(mask_2d, dtype=torch.bool),
            mask_3d=torch.tensor(mask_3d, dtype=torch.bool),
            mask_incomplete_2d=None,
            w=self.w
        )
        return data


@R.register('UserDefinedPeptide')
class UserDefinedPeptide(LinearPeptide):
    def __init__(self, size_min=12, size_max=14, w=1.0, ref=None):
        super().__init__(size_min, size_max)
        self.w = w
        self.ref = ref # {'PHE180': [x, y, z]}


    def dummy_lig_block_bonds(self, cplx_desc, size):
        prefixed_num = random.randint(1, min(len(self.ref), size))
        prefixed_position = random.sample(list(range(size)), prefixed_num)
        self.mased = [1 if i in prefixed_position else 0 for i in range(size)]
        prefixed_aa = random.sample(self.ref.keys(), prefixed_num)

        blocks= []
        fixed_blocks = []
        cnt = 0


        for i in range(size):
            if i in prefixed_position:
                aa_name = prefixed_aa[cnt].split('_')[0]
                aa_coord = self.ref[prefixed_aa[cnt]]
                blocks.append(form_block(aa_name, id=(i+1, ''), coord=aa_coord))
                fixed_blocks.append(form_block(aa_name, id=(i+1, ''), coord=aa_coord))
                cnt += 1
            else:
                placeholder_block = Block(
                    name='GLY', atoms=[Atom(name='C', coordinate=[0, 0, 0], element='C', id=-1)], id=(i+1, '')
                ) 
                blocks.append(placeholder_block)

        # add intra-block bonds
        mol_index = len(cplx_desc.cplx)
        bonds = []
        for block in fixed_blocks:
            i = block.id[0] - 1
            for bond in VOCAB.abrv_to_bonds(block.name):
                bonds.append(Bond(
                    (mol_index, i, bond[0]),
                    (mol_index, i, bond[1]),
                    bond[2]
                ))

        return blocks, bonds
    
    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        data = super().to_data(cplx_desc)
        # add topo conditioning
        mask_2d = [0 for _ in cplx_desc.pocket_block_ids] + self.mased
        data['condition_config'] = ConditionConfig(
            mask_2d=torch.tensor(mask_2d, dtype=torch.bool),
            mask_3d=None,
            mask_incomplete_2d=None,
            w=self.w
        )
        return data


@R.register('NSAAPeptideWithLibrary')
class NSAAPeptideWithLibrary(LinearPeptide):
    def __init__(self, nsaa_list, num_nsaa_min=1, num_nsaa_max=1, size_min=8, size_max=13, w=0.5):
        super().__init__(size_min, size_max)
        self.nsaa_list = nsaa_list  # e.g. [ "NC1=CC=C(C=C1)C=C(C#N)C2=CC=CC(C(C(=O)*)N*)=C2" ], with * denotes the N-term and the C-term
        if num_nsaa_max is None: num_nsaa_max = size_max
        self.num_nsaa_min = num_nsaa_min
        self.num_nsaa_max = num_nsaa_max # included
        assert num_nsaa_min > 0
        assert num_nsaa_min <= num_nsaa_max
        self.w = w

        # dynamic

    def dummy_lig_block_bonds(self, cplx_desc, size):
        num_nsaa_min = min(self.num_nsaa_min, size)
        num_nsaa_max = min(self.num_nsaa_max, size)
        num_nsaa = np.random.randint(num_nsaa_min, num_nsaa_max + 1)
        nsaa_idx = np.random.choice(size, num_nsaa, replace=False)
        
        creator = Creator(len(cplx_desc.cplx))
        n_terms, c_terms = {}, {}
        for i in range(size):
            if i not in nsaa_idx: creator.add_unk((i + 1, ''), is_aa=True)
            else:   # add a random nsaa
                nsaa = np.random.choice(self.nsaa_list, 1)[0]
                nsaa_mol, n_term, c_term = find_term_and_delete_dummy(nsaa)
                offset = len(creator.blocks)
                added_blocks, _ = creator.decompose_and_add_fragment(nsaa_mol, i + 1)
                for j, block in enumerate(added_blocks):
                    for k, atom in enumerate(block):
                        # as atom ids start from 1, but atom labels in RDKit start from 0
                        if int(atom.id) == n_term + 1: n_terms[i] = (offset + j, k)
                        elif int(atom.id) == c_term + 1: c_terms[i] = (offset + j, k)
                if (i - 1) in nsaa_idx:  # adjacent
                    creator.add_bond(c_terms[i - 1][0], n_terms[i][0], c_terms[i - 1][1], n_terms[i - 1][1], BondType.SINGLE)

        return creator.get_results()
    
    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        data = super().to_data(cplx_desc)
        # add topo conditioning
        # nsaa id is (chain_id, (pos_id, insertion_code))
        mask_2d = [0 for _ in cplx_desc.pocket_block_ids] + [1 if id[1][1] != '' else 0 for id in cplx_desc.lig_block_ids]
        data['condition_config'] = ConditionConfig(
            mask_2d=torch.tensor(mask_2d, dtype=torch.bool),
            mask_3d=None,
            mask_incomplete_2d=None,
            w=self.w
        )
        return data


@R.register('CovalentPeptide')
class CovalentPeptide(LinearPeptide):
    def __init__(self, nsaa, tgt_connect_atom_label, tgt_connect_bond_level, size_min=8, size_max=13, allowed_positions=None, w=0.5):
        super().__init__(size_min, size_max)
        self.nsaa = nsaa    # e.g. *C(CC(C(=O)*)N*)=O, where the original F in the FC(CC(C(=O)*)N*)=O will be kicked off by N/Os from an AA in the tgt
        self.tgt_connect_atom_label = tgt_connect_atom_label    # e.g. [S, 119, '', OG], which is [chain id, residue number, insertion code, atom name]
        self.tgt_connect_bond_level = tgt_connect_bond_level    # e.g. 1 for single, 2 for double, 3 for tripple
        self.covalent_bond = BondType(tgt_connect_bond_level)
        self.allowed_positions = list(range(size_max)) if allowed_positions is None else allowed_positions
        self.w = w

        assert nsaa.count('*') == 3, f'The provided nsAA should include three dummy atoms, two on N/C-terminals, and one on the covalent spot'

        # dynamic
        self._covalent_bond_index = None
    
    @property
    def additional_sample_opt(self) -> dict:
        return { 'vae_disable_avoid_clash': True }  # avoid implementing clash avoidance on the covalent site

    def default_filter_configs(self)-> List[dict]:
        tgt_covalent_site = (
            self.tgt_connect_atom_label[0],
            (self.tgt_connect_atom_label[1], self.tgt_connect_atom_label[2])
        )
        return [
            {'class': 'AbnormalConfidenceFilter'},
            {'class': 'TargetBinderCovalentDistanceFilter'},
            {'class': 'SimpleGeometryFilter'},
            {'class': 'SimpleClashFilter', 'inter_covalent_exist': True},
        ]

    def dummy_lig_block_bonds(self, cplx_desc, size):
        allowed_positions = []
        for i in self.allowed_positions:
            if i < 0: # from back
                i = size + i
                if i >= 0: allowed_positions.append(i)
            elif i < size: allowed_positions.append(i)
        nsaa_idx = np.random.choice(allowed_positions, 1, replace=False)[0]
        
        creator = Creator(len(cplx_desc.cplx))
        n_terms, c_terms, covalent_spot_id = {}, {}, None
        for i in range(size):
            if i != nsaa_idx: creator.add_unk((i + 1, ''), is_aa=True)  # normal AA
            else:   # add a random nsaa
                nsaa_mol, n_term, c_term, other_connects = find_term_and_delete_dummy(self.nsaa, include_non_term_dummy=False)
                assert len(other_connects) == 1 # the covalent spot
                offset = len(creator.blocks)
                added_blocks, _ = creator.decompose_and_add_fragment(nsaa_mol, i + 1)
                for j, block in enumerate(added_blocks):
                    for k, atom in enumerate(block):
                        # as atom ids start from 1, but atom labels in RDKit start from 0
                        if int(atom.id) == n_term + 1: n_terms[i] = (offset + j, k)
                        elif int(atom.id) == c_term + 1: c_terms[i] = (offset + j, k)
                        elif int(atom.id) == other_connects[0] + 1: covalent_spot_id = (offset + j, k)
                # if (i - 1) in nsaa_idx:  # adjacent
                #     creator.add_bond(c_terms[i - 1][0], n_terms[i][0], c_terms[i - 1][1], n_terms[i - 1][1], BondType.SINGLE)
        blocks, bonds = creator.get_results()
        # add the inter-block covalent bond
        tgt_block_index = (self.tgt_connect_atom_label[0], (self.tgt_connect_atom_label[1], self.tgt_connect_atom_label[2]))
        tgt_block = recur_index(cplx_desc.cplx, tgt_block_index)
        tgt_block_index_numerical = index_to_numerical_index(cplx_desc.cplx, tgt_block_index)
        for i, atom in enumerate(tgt_block):
            if atom.name == self.tgt_connect_atom_label[3]: atom_index = i
        self._covalent_bond_index = (
            (tgt_block_index_numerical[0], tgt_block_index_numerical[1], atom_index),
            (creator.mol_index, covalent_spot_id[0], covalent_spot_id[1]),
            self.covalent_bond
        )
        bonds.append(Bond(*self._covalent_bond_index))
        return blocks, bonds
    
    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        data = super().to_data(cplx_desc)
        # add topo conditioning
        # nsaa id is (chain_id, (pos_id, insertion_code))
        tgt_block_id = (self.tgt_connect_atom_label[0], (self.tgt_connect_atom_label[1], self.tgt_connect_atom_label[2]))
        mask_2d = [0 if id != tgt_block_id else 1 for id in cplx_desc.pocket_block_ids] + [1 if id[1][1] != '' else 0 for id in cplx_desc.lig_block_ids]
        data['condition_config'] = ConditionConfig(
            mask_2d=torch.tensor(mask_2d, dtype=torch.bool),
            mask_3d=None,
            mask_incomplete_2d=None,
            w=self.w
        )
        return data
    
    def set_desc_attr(self, cplx_desc: ComplexDesc):
        if cplx_desc.additional_props is None: cplx_desc.additional_props = {}
        cplx_desc.additional_props['covalent_modifications'] = [
            (self._covalent_bond_index[0], self._covalent_bond_index[1], self.tgt_connect_bond_level) # use int insteead of the BondType object for json
        ]
        return super().set_desc_attr(cplx_desc)