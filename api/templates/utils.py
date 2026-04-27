# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from data.bioparse import VOCAB
from data.bioparse.hierarchy import Atom, Block, Bond, BondType


def form_block(abrv, id):
    '''
    Args:
        abrv: abbrevation of the block (three-code letters for amino acids, and smiles for fragments)
    '''
    atom_names = VOCAB.abrv_to_atoms(abrv)
    elements = VOCAB.abrv_to_elements(abrv)

    block = Block(
        name=abrv,
        atoms=[Atom(
            name=atom_name,
            coordinate=[0, 0, 0],
            element=element,
            id=i
        ) for i, (atom_name, element) in enumerate(zip(atom_names, elements))],
        id=id
    )

    return block
