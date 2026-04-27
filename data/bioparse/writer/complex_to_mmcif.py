# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Optional, List, Tuple

import numpy as np
import biotite.structure as struc
from biotite.structure.io.pdbx import CIFFile, CIFCategory, set_structure

from ..utils import _is_peptide_bond, recur_index, _wrap_coord, is_standard_block
from ..utils import is_standard_aa, is_standard_base
from ..vocab import VOCAB
from ..hierarchy import Complex, Molecule, Block, Atom, BondType
from .. import const


def make_chem_comp_category(atom_array: struc.AtomArray) -> CIFCategory:
    comp_ids = []
    comp_types = []
    mon_nstd_flags = []

    seen = set()
    res_names = atom_array.res_name.tolist()
    if hasattr(atom_array, "hetor"):
        hetero = atom_array.hetor.tolist()
    elif hasattr(atom_array, "hetero"):
        hetero = atom_array.hetero.tolist()
    else:
        hetero = [False] * len(res_names)

    het_by_resn = {}
    for rn, is_het in zip(res_names, hetero):
        if rn not in het_by_resn:
            het_by_resn[rn] = bool(is_het)
        else:
            het_by_resn[rn] = het_by_resn[rn] or bool(is_het)

    for rn in res_names:
        if rn in seen:
            continue
        seen.add(rn)

        if rn in {"HOH", "WAT", "H2O"}:
            comp_type = "WATER"
            mon_nstd = "n"
        elif is_standard_aa(rn):
            comp_type = "L-PEPTIDE LINKING"
            mon_nstd = "y"
        elif is_standard_base(rn):
            comp_type = "DNA LINKING" if rn in {"DA", "DG", "DC", "DT"} else "RNA LINKING"
            mon_nstd = "y"
        else:
            is_het = het_by_resn.get(rn, True)
            if (not is_het) and len(rn) == 3:
                comp_type = "L-PEPTIDE LINKING"
            else:
                comp_type = "NON-POLYMER"
            mon_nstd = "n"

        comp_ids.append(rn)
        comp_types.append(comp_type)
        mon_nstd_flags.append(mon_nstd)

    return CIFCategory(
        {
            "id": comp_ids,
            "type": comp_types,
            "mon_nstd_flag": mon_nstd_flags,
        }
    )


def make_entity_categories(atom_array: struc.AtomArray) -> Tuple[CIFCategory, Optional[CIFCategory]]:
    chain_ids = atom_array.chain_id.tolist()
    res_ids = atom_array.res_id.tolist()
    ins_codes = atom_array.ins_code.tolist() if hasattr(atom_array, "ins_code") else [""] * len(chain_ids)
    res_names = atom_array.res_name.tolist()

    dna_res_names = {"DA", "DG", "DC", "DT"}
    rna_res_names = {"A", "G", "C", "U", "RA", "RG", "RC", "RU"}
    water_res_names = {"HOH", "WAT", "H2O"}

    chain2residues = {}
    for ch, rid, ic, rn in zip(chain_ids, res_ids, ins_codes, res_names):
        key = (int(rid), str(ic).strip(), str(rn).strip())
        if ch not in chain2residues:
            chain2residues[ch] = set()
        chain2residues[ch].add(key)

    entity_ids = []
    entity_types = []
    polymer_entity_ids = []
    polymer_types = []

    for entity_idx, ch in enumerate(chain2residues.keys(), start=1):
        aa = dna = rna = water = other = 0
        for _, _, rn in chain2residues[ch]:
            if rn in water_res_names:
                water += 1
            elif is_standard_aa(rn):
                aa += 1
            elif rn in dna_res_names:
                dna += 1
            elif rn in rna_res_names or is_standard_base(rn):
                rna += 1
            else:
                other += 1

        total = aa + dna + rna + water + other
        non_water_total = total - water
        polymer_like = aa + dna + rna
        entity_id = str(entity_idx)

        if non_water_total == 0 and water > 0:
            entity_type = "water"
        elif polymer_like > 0 and polymer_like >= max(1, int(non_water_total * 0.5)):
            entity_type = "polymer"
            if aa > 0 and dna == 0 and rna == 0:
                poly_type = "polypeptide(L)"
            elif dna > 0 and aa == 0 and rna == 0:
                poly_type = "polydeoxyribonucleotide"
            elif rna > 0 and aa == 0 and dna == 0:
                poly_type = "polyribonucleotide"
            elif dna > 0 and rna > 0 and aa == 0:
                poly_type = "polydeoxyribonucleotide/polyribonucleotide hybrid"
            else:
                poly_type = "other"
            polymer_entity_ids.append(entity_id)
            polymer_types.append(poly_type)
        else:
            entity_type = "non-polymer"

        entity_ids.append(entity_id)
        entity_types.append(entity_type)

    entity_cat = CIFCategory({"id": entity_ids, "type": entity_types})
    entity_poly_cat = (
        CIFCategory({"entity_id": polymer_entity_ids, "type": polymer_types}) if polymer_entity_ids else None
    )
    return entity_cat, entity_poly_cat


def complex_to_mmcif(
        cplx: Complex,
        mmcif_path: str,
        selected_chains: Optional[List[str]]=None,
        title: Optional[str]=None,
        explict_bonds: Optional[List[tuple]]=None
    ):
    '''
        Args:
            cplx: Complex, the complex to written into pdb file
            pdb_path: str, output path
            selected_chains: list of chain ids to write
            title: the title of the pdb file
            explict_bonds: list of bonds to write as CONECT (each bond is represented as (id1, id2, bond_type)).
                The bond_type will be ignored as pdb do not record such information. The id1 and id2 should be
                provided as numerical ids, e.g. (0, 10, 1) means the atom at cplx[0][10][1].
    '''

    assert mmcif_path.endswith('.cif')

    atom_list = []
    resn_to_hetblocks, het_blocks = {}, {}

    mol: Molecule = None
    block: Block = None
    atom: Atom = None
    atom_number = 0 # biotite start atom number from 0, which is different from complex_to_pdb
    id2atom_number = {}
    for i, mol in enumerate(cplx): # chain
        if selected_chains is not None and mol.id not in selected_chains: continue
        for j, block in enumerate(mol):
            block_name = block.name
            if not is_standard_block(block_name): # fragments
                block_name = VOCAB.abrv_to_symbol(block_name)
            insert_code = block.id[1]
            if 'original_name' in block.properties or block.id[-1].isdigit(): # fragments has an appended insertion code with digits
                block_name = block.properties.get('original_name', None)
                if block_name is None or len(block_name) > 3:
                    if block.id[0] in resn_to_hetblocks: block_name = resn_to_hetblocks[block.id[0]]
                    else:
                        block_name = 'LI' + str(len(het_blocks)) # the original name is too long, might be smiles
                        resn_to_hetblocks[block.id[0]] = block_name
                # this block is only a fragment of the ligand, so we get rid of the insertion code
                # and put it back to a whole residue
                insert_code = ''.join([s for s in insert_code if not s.isdigit()])
            if insert_code.isdigit(): insert_code = chr(ord('A') + int(insert_code))
            # sometimes fragment will lead to insert code like A0, A1 if the residue already has one insert code.
            for k, atom in enumerate(block):
                coord = [_wrap_coord(x, 8) for x in atom.coordinate]
                atom_list.append(struc.Atom(
                    coord=np.array(coord, dtype=np.float32),
                    chain_id=mol.id,
                    res_id=block.id[0],
                    ins_code=insert_code,
                    res_name=block_name,
                    hetor=(not is_standard_block(block_name)),
                    atom_name=atom.name,
                    element=atom.element,
                    occupancy=atom.get_property('occupancy', 1.0),
                    b_factor=atom.get_property('bfactor', 0.0)
                ))
                if not is_standard_block(block_name): # hetatoms
                    if block_name not in het_blocks: het_blocks[block_name] = []
                    het_blocks[block_name].append(atom_list[-1]) # pointer
                id2atom_number[(i, j, k)] = atom_number
                atom_number += 1

    # order atom name in hetblocks
    for block_name in het_blocks:
        atoms = het_blocks[block_name]
        name_cnts = {}
        for atom in atoms:
            name = atom.atom_name
            if name in name_cnts:
                atom.atom_name = f'{name}{name_cnts[name]}'
                name_cnts[name] += 1
            else: name_cnts[name] = 1

    atom_array = struc.array(atom_list)

    # Define bond type for mmCIF
    bond_order = {
        BondType.SINGLE: struc.BondType.SINGLE,
        BondType.DOUBLE: struc.BondType.DOUBLE,
        BondType.TRIPLE: struc.BondType.TRIPLE,
    }

    recorded_bonds = {}
    atom_array.bonds = struc.BondList(atom_array.array_length())

    def add_bond(start_id, end_id, bond_type):
        # if _is_peptide_bond(cplx, start_id, end_id, bond_type): return # do not record normal peptide bond
        start_atom_number = id2atom_number[start_id]
        end_atom_number = id2atom_number[end_id]
        if ((start_atom_number, end_atom_number) in recorded_bonds) or \
           ((end_atom_number, start_atom_number)) in recorded_bonds:
            return
        # add bond
        atom_array.bonds.add_bond(start_atom_number, end_atom_number, bond_order[bond_type])
        recorded_bonds[(start_atom_number, end_atom_number)] = True

    # write non-aa bonds
    for bond in cplx.bonds:
        if selected_chains is not None:
            if cplx[bond.index1[0]].id not in selected_chains: continue
            if cplx[bond.index2[0]].id not in selected_chains: continue
        add_bond(bond.index1, bond.index2, bond.bond_type)

    # write explicit bonds (drop normal peptide bond)
    if explict_bonds is not None:
        for start_id, end_id, bond_type in explict_bonds:
            add_bond(start_id, end_id, bond_type)

    file = CIFFile()
    set_structure(file, atom_array, include_bonds=True)
    file["structure"]["chem_comp"] = make_chem_comp_category(atom_array)
    entity_cat, entity_poly_cat = make_entity_categories(atom_array)
    file["structure"]["entity"] = entity_cat
    if entity_poly_cat is not None:
        file["structure"]["entity_poly"] = entity_poly_cat
    file.write(mmcif_path)
