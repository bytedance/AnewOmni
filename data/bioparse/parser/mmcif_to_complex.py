# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import List, Optional

from biotite.structure.io.pdbx import CIFFile, get_structure

from ._biotite_to_complex import atom_array_to_complex

from ..hierarchy import Complex
from ..tokenizer.tokenize_3d import TOKENIZER


def mmcif_to_complex(
        mmcif_file: str,
        selected_chains: Optional[List[str]]=None,
        remove_Hs: bool=True,
        remove_sol: bool=True,
        remove_het: bool=False
    ) -> Complex:
    '''
        Convert mmCIF file to Complex.
        Each chain will be a Molecule.
        
        Parameters:
            pdb: Path to the mmcif file or text contents
            selected_chains: List of selected chain ids. The returned list will be ordered
                according to the ordering of chain ids in this parameter. If not specified,
                all chains will be returned. e.g. ['A', 'B']
            remove_Hs: Whether to remove all hydrogens
            remove_sol: Whether to remove all solvent molecules
            remove_het: Whether to remove all HETATM

        Returns:
                A Complex instance
    '''

    file = CIFFile.read(mmcif_file)
    struct = get_structure(file, include_bonds=True, extra_fields=['atom_id', 'b_factor', 'occupancy'])

    assert TOKENIZER.kekulize

    name = mmcif_file.rstrip('.cif') if mmcif_file.endswith('.cif') else 'anonymous'
    return atom_array_to_complex(
        struct, name, selected_chains, remove_Hs, remove_sol, remove_het,
        unknown_bond_default=0  # keep original bond definition
    )