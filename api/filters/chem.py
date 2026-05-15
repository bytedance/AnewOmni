# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import ray
import yaml
from rdkit import Chem
from pathlib import Path
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

import utils.register as R

from .base import BaseFilter, FilterResult, FilterInput


@R.register('MolWeightFilter')
class MolWeightFilter(BaseFilter):
    def __init__(self, weight_min: float=None, weight_max: float=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        Args:
            weight_min: minimal molecular weight in Da (inclusive)
            weight_max: maximal molecular weight in Da (inclusive)
        '''
        self.weight_min, self.weight_max = weight_min, weight_max
    
    @property
    def name(self):
        return f'MolWeightFilter(weight_min={self.weight_min}, weight_max={self.weight_max})'
    
    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        try:
            mol = Chem.MolFromSmiles(input.smiles) # Actually, the default behavior is adding hydrogens, so the next line might not be necessary
            mol = Chem.AddHs(mol)
            mol_weight = Descriptors.MolWt(mol)
        except Exception as e:
            return FilterResult.ERROR, { 'error': str(e) }
        flag = FilterResult.PASSED
        if self.weight_min is not None and mol_weight < self.weight_min: flag = FilterResult.FAILED
        if self.weight_max is not None and mol_weight > self.weight_max: flag = FilterResult.FAILED
        return flag, { 'mol_weight': mol_weight }


@R.register('ChiralCentersFilter')
class ChiralCentersFilter(BaseFilter):
    def __init__(self, center_max: float=6, ring_mode: bool=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        Args:
            center_max: maximal chiral centers (default: 6).
            ring_mode: additionally exclude molecule containing at least one ring where over half of the atoms are chiral.
        '''
        self.center_max, self.ring_mode = center_max, ring_mode
    
    @property
    def name(self):
        return f'ChiralCentersFilter(center_max={self.center_max}, ring_mode:{self.ring_mode})'
    
    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        try:
            mol = Chem.MolFromSmiles(input.smiles) # Actually, the default behavior is adding hydrogens, so the next line might not be necessary
            mol = Chem.AddHs(mol)
            # generate 3D conformation and assign `_ChiralityPossible` property to chiral centers
            AllChem.EmbedMolecule(mol)
            Chem.AssignAtomChiralTagsFromStructure(mol)
            # get chiral atom
            chiral_atoms = [atom.GetIdx() for atom in mol.GetAtoms() 
                   if atom.HasProp('_ChiralityPossible')]
            _chirality_valid = len(chiral_atoms) <= self.center_max
            # extra check in ring system
            if _chirality_valid and self.ring_mode:
                ring_info = mol.GetRingInfo()
                ring_atoms = ring_info.AtomRings()
                _chirality_valid = all(len([idx for idx in chiral_atoms if idx in ring]) <= len(ring)/2 for ring in ring_atoms)
        except Exception as e:
            return FilterResult.ERROR, { 'error': str(e) }
        flag = FilterResult.PASSED

        if self.center_max is not None and not _chirality_valid: flag = FilterResult.FAILED
        return flag, { 'num_chiral_centers': chiral_atoms }


@R.register('MolSMARTSFilter')
class MolSMARTSFilter(BaseFilter):
    def __init__(self, SMARTS_config: str | Path='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        Args:
            SMARTS_list: list of SMARTS strings to exclude.
        '''
        default_smarts_config = Path(__file__).parent/'_SMARTS.yaml'
        if SMARTS_config == '': SMARTS_config = default_smarts_config
        # Load config
        with open(SMARTS_config) as f:
            SMARTS = yaml.safe_load(f)
        self.SMARTS_list = [item['pattern'] for item in SMARTS]
        self.patterns = [Chem.MolFromSmarts(smarts) for smarts in self.SMARTS_list]
    
    @property
    def name(self):
        return f"MolSMARTSFilter(SMARTS_list:{','.join(self.SMARTS_list)})"
    
    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        try:
            mol = Chem.MolFromSmiles(input.smiles)
            mol = Chem.AddHs(mol)
            match = any(mol.HasSubstructMatch(pattern) for pattern in self.patterns)
        except Exception as e:
            return FilterResult.ERROR, { 'error': str(e) }
        flag = FilterResult.PASSED
        if match:
            flag = FilterResult.FAILED
        return flag, {}


@R.register('RotatableBondsFilter')
class RotatableBondsFilter(BaseFilter):
    def __init__(self, max_num_rot_bonds: int=7):
        '''
        Args:
            max_num_rot_bonds: maximum rotatable bonds (inclusive)
        '''
        self.max_num_rot_bonds = max_num_rot_bonds

    @property
    def name(self):
        return self.__class__.__name__ + f'(max_num_rot_bonds={self.max_num_rot_bonds})'

    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        try:
            mol = Chem.MolFromSmiles(input.smiles)
            num_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol, strict=True)
        except Exception as e:
            return FilterResult.ERROR, { 'error': str(e) }
        flag = FilterResult.PASSED if num_rot_bonds <= self.max_num_rot_bonds else FilterResult.FAILED
        return flag, { 'num_rot_bonds': num_rot_bonds }


if __name__ == '__main__':
    SMARTS_config = Path(__file__).parent/'_SMARTS.yaml'
    f_mw = MolWeightFilter(100, 600)
    f_smarts = MolSMARTSFilter(SMARTS_config)
    f_chiral = ChiralCentersFilter()
    import sys
    smiles = sys.argv[1]
    print(f_mw(FilterInput(None, [], [], smiles)))
    print(f_smarts(FilterInput(None, [], [], smiles)))
    print(f_chiral(FilterInput(None, [], [], smiles)))
    