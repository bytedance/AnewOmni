# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import tempfile

from Bio.PDB import PDBParser, MMCIFParser, PDBIO, MMCIFIO
import numpy as np

from .parser.mmcif_to_complex import mmcif_to_complex
from .parser.pdb_to_complex import pdb_to_complex
from .utils import is_standard_aa
from .vocab import VOCAB
from . import const


def load_cplx(path, selected_chains=None, cleanup_first=False, *args, **kwargs):

    if cleanup_first:
        tmpf = tempfile.NamedTemporaryFile(suffix=os.path.splitext(path)[1])
        cleanup_file(path, tmpf.name)
        path = tmpf.name

    if path.endswith('.pdb'): cplx = pdb_to_complex(path, selected_chains, *args, **kwargs)
    elif path.endswith('.cif'): cplx = mmcif_to_complex(path, selected_chains, *args, **kwargs)
    else: raise ValueError(f'File format not supported: {path}')
    return cplx


def extract_seqs(path, selected_chains):
    cplx = load_cplx(path, selected_chains)
    seqs = []
    for chain in cplx:
        seqs.append(''.join([VOCAB.abrv_to_symbol(block.name) for block in chain]))
    return seqs


def extract_CA_coords(path, selected_chains):
    cplx = load_cplx(path, selected_chains)
    coords, masks = [], []
    for chain in cplx:
        chain_coords, chain_masks = [], []
        for block in chain:
            if not is_standard_aa(block.name): continue
            x = None
            for atom in block:
                if atom.name == 'CA': x = atom.get_coord()
            chain_coords.append([0, 0, 0] if x is None else x)
            chain_masks.append(False if x is None else True)
        coords.append(np.array(chain_coords, dtype=np.float64))
        masks.append(np.array(chain_masks, dtype=bool))
    return coords, masks


def cleanup_file(input_file, output_file):
    '''
        1. Delete OXT
        2. Reorder atoms in each residue to the canonical order
        These two things usually cause bugs
    '''
    canonical_order = {
        abrv: VOCAB.abrv_to_atoms(abrv)
        for _, abrv in const.aas
    }

    def reorder_residue_atoms(residue):
        resname = residue.get_resname().strip()
        atoms = list(residue.get_atoms())
        if resname not in canonical_order:
            return atoms  # skip non-standard residues

        order = canonical_order[resname]
        # create a map of atom name -> atom
        atom_dict = {a.get_name(): a for a in atoms}

        # order atoms by canonical list, keep unrecognized ones at the end
        ordered = [atom_dict[a] for a in order if a in atom_dict]
        unordered = [a for a in atoms if a.get_name() not in order]
        return ordered + unordered

    def reorder_structure(structure):
        for model in structure:
            for chain in model:
                for residue in chain:
                    atoms = reorder_residue_atoms(residue)
                    residue.detach_child  # just to prevent duplicate adding
                    for atom in list(residue):
                        residue.detach_child(atom.id)
                    for atom in atoms:
                        residue.add(atom)

    in_ext = os.path.splitext(input_file)[1].lower()
    if in_ext == '.pdb': parser = PDBParser(QUIET=True)
    elif in_ext == '.cif': parser = MMCIFParser(QUIET=True)
    else: raise ValueError('Unsupported input format (use .pdb or .cif/.mmcif)')
    
    out_ext = os.path.splitext(output_file)[1].lower()
    if out_ext == '.pdb': io = PDBIO()
    elif out_ext == '.cif': io = MMCIFIO()
    else: raise ValueError('Unsupported output format (use .pdb or .cif/.mmcif)')

    structure = parser.get_structure("struct", input_file)
    reorder_structure(structure)
    io.set_structure(structure)
    io.save(output_file)