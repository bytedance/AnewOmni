# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import ray

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np 

import utils.register as R

from .base import BaseFilter, FilterResult, FilterInput


chirality = {
    1: 'L',
    -1: 'D'
}
def get_enantiomer(n, ca, c, cb):
    '''
    reference:
    https://www.biotite-python.org/latest/examples/gallery/structure/protein/residue_chirality.html
    '''
    n = np.cross(ca - n, ca - c)
    sign = np.sign(np.dot(cb - ca, n))
    return chirality.get(sign, 'N/A')



@R.register('LTypeAAFilter')
class LTypeAAFilter(BaseFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    @property
    def name(self):
        return self.__class__.__name__ 

    
    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):

        '''
        reference:
        https://www.biotite-python.org/latest/examples/gallery/structure/protein/residue_chirality.html
        '''
        # read ligand pdb
        pdb_file = strucio.load_structure(input.path_prefix+'.pdb')
        array = pdb_file[np.isin(pdb_file.chain_id, input.lig_chains)]
        seq_dict = {'chirality':[]} # L, D, N/A(e.g. Glycine)

        # Filter backbone + CB
        array = array[struc.filter_amino_acids(array)]
        array = array[(array.atom_name == "CB") | (struc.filter_peptide_backbone(array))]
        # Iterate over each residue
        ids, names = struc.get_residues(array)
        for i, id in enumerate(ids):
            coord = array.coord[array.res_id == id]
            if len(coord) != 4:
                seq_dict['chirality'].append('N/A') # Glyine -> no chirality
            else:
                seq_dict['chirality'].append(get_enantiomer(coord[0], coord[1], coord[2], coord[3]))

        
        if 'D' not in seq_dict['chirality']:
            return FilterResult.PASSED, seq_dict
        else:
            return FilterResult.FAILED, seq_dict
    
