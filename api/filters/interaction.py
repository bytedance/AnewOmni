# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # prevent warnings from prolif

import re
import ray
import yaml
from typing import List
from rdkit import Chem
from pathlib import Path
import numpy as np
import tempfile
import biotite.structure as struc
import biotite.structure.io as strucio
from biotite.structure.io.pdbx import CIFFile, get_structure
import biotite.structure.io.pdb as pdb

import utils.register as R
import prolif as plf
from data.bioparse.interface import compute_interacting_pairs
from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
from data.bioparse.tools import load_cplx
from .base import BaseFilter, FilterResult, FilterInput


@R.register('InteractionFilter')
class InteractionFilter(BaseFilter):
    def __init__(self, interaction_config: str | Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load config
        self.load_plf_parameters(interaction_config)


    @property
    def name(self):
        return f'InteractionFilter(protein_path={self.protein_path}, interaction_dict={self.region_interaction_dict})'
    

    def load_plf_parameters(self, config_path):
        """
        Load and process PLF (Protein-Ligand Interaction Fingerprint) parameters from a YAML config file.
        Args:
            config_path (str): Path to the YAML configuration file.
        """

        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        # Load and process configuration sections
        self.protein_path = config.get('protein_path', '')
        self.interactions = config.get('interactions', [])
        
        self.region_interaction_dict = config.get('region_interaction_dict', {})
        self.region_interaction_dict = {
            key: [tuple(item) for item in value] 
            for key, value in self.region_interaction_dict.items()
        }
        
    def get_protein_obj(self):
        if self.protein_path.endswith('mae'):
            protein_obj = Chem.MaeMolSupplier(self.protein_path, removeHs=False)[0]
            for i, atom in enumerate(protein_obj.GetAtoms()):
                atom.SetAtomMapNum(i + 1)
            self.protein_plf = plf.Molecule.from_rdkit(protein_obj)
        elif self.protein_path.endswith('pdb'):
            protein_obj = Chem.MolFromPDBFile(self.protein_path, removeHs=False)
            self.protein_plf = plf.Molecule.from_rdkit(protein_obj)
        elif self.protein_path == '':
            self.protein_path = self.input.path_prefix+'.pdb'
            protein_obj = Chem.MolFromPDBFile(self.protein_path, removeHs=False)
            self.protein_plf = plf.Molecule.from_rdkit(protein_obj)
        else:
            print('protein for calculation should be pdb or mae format')

    def cal_interactions(self):
        """
        Returns:
            interactions: [(amino acid, interaction)]
        """
        try:
            sdf_path = self.input.path_prefix + '.sdf'
            mol = Chem.SDMolSupplier(sdf_path, removeHs=False)[0]
            mol = Chem.AddHs(mol, addCoords=True)
            lig_mol = plf.Molecule.from_rdkit(mol)
            
        except Exception as e:
            print({ 'error': str(e)})
            return []  # no interactions

        fp = plf.Fingerprint(interactions=self.interactions)
        ifp = fp.generate(lig_mol, self.protein_plf, metadata=True)

        if ifp != {}:
            interaction_df = plf.to_dataframe({0: ifp}, fp.interactions)
            interaction_df = interaction_df.droplevel('ligand', axis=1)
            interactions = list(interaction_df.columns)
            self.interactions_hit = [(interaction[0].replace(" ", ""), interaction[1]) for interaction in interactions]
        else:
            print('Caution. No ineraction dectect')
            self.interactions_hit = []
        


    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        self.input = input 
        self.get_protein_obj()
        self.cal_interactions()

        for key, value in self.region_interaction_dict.items():
            values = self.region_interaction_dict[key]
            found = False
            for value in values:
                if value in self.interactions_hit:
                    found = True
                    break
            if not found:
                return FilterResult.FAILED, self.interactions_hit
            
        return FilterResult.PASSED, self.interactions_hit


def split_mmcif_to_sdf(mmcif_path, tgt_chains, lig_chains, tgt_out_path, lig_out_path):
    """
    Load a mmCIF file and split its chains into separate SDF files.
    
    Parameters
    ----------
    mmcif_path : str
        Path to the mmCIF file
    tgt/lig_out_path : str, optional
        Paths for output SDF files (default: "chain")
    """
    # Load the mmCIF file
    mmcif_file = CIFFile.read(mmcif_path)
    atom_array = get_structure(mmcif_file, model=1, include_bonds=True, extra_fields=['atom_id', 'b_factor', 'occupancy'])
    
    # Split by chain and save each as SDF
    # target
    chain_mask = (atom_array.chain_id == tgt_chains[0])
    for c in tgt_chains[1:]:
        chain_mask = chain_mask | (atom_array.chain_id == c)
    chain_atoms = atom_array[chain_mask]
    strucio.save_structure(tgt_out_path, chain_atoms)

    # ligand
    chain_mask = (atom_array.chain_id == lig_chains[0])
    for c in lig_chains[1:]:
        chain_mask = chain_mask | (atom_array.chain_id == c)
    chain_atoms = atom_array[chain_mask]
    strucio.save_structure(lig_out_path, chain_atoms)


def _load_mol_from_sdf(sdf):
    mol = Chem.SDMolSupplier(sdf, removeHs=False)[0]
    mol = Chem.AddHs(mol, addCoords=True)
    mol = plf.Molecule.from_rdkit(mol)
    return mol


DEFAULT_INTERACTIONS = [
    'Cationic',
    'HBAcceptor',
    'HBDonor',
    'CationPi',
    'PiCation',
    'EdgeToFace',
    'FaceToFace'
]


@R.register('NumInteractionFilter')
class NumInteractionFilter(BaseFilter):
    def __init__(self, minimum_num: int, interaction_types: List[str] = DEFAULT_INTERACTIONS, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
            Reject the candidate if number of interactions is less than "minimum_num"
        '''
        self.minimum_num = minimum_num
        self.interactions = interaction_types

    @property
    def name(self):
        return self.__class__.__name__ + f'(interaction_types={self.interactions})'

    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        # split mmcif into two SDFs
        tgt_out_file = tempfile.NamedTemporaryFile(suffix='.sdf')
        lig_out_file = tempfile.NamedTemporaryFile(suffix='.sdf')
        split_mmcif_to_sdf(
            input.path_prefix + '.cif',
            input.tgt_chains, input.lig_chains,
            tgt_out_file.name, lig_out_file.name
        )
        # load
        try:
            prot = _load_mol_from_sdf(tgt_out_file.name)
            lig = _load_mol_from_sdf(lig_out_file.name)
        except Exception as e:
            return FilterResult.ERROR, { 'error': str(e) }
        
        # calculate interactions
        fp = plf.Fingerprint(interactions=self.interactions)
        ifp = fp.generate(lig, prot, metadata=True)

        if ifp != {}:
            interaction_df = plf.to_dataframe({0: ifp}, fp.interactions)
            interactions = list(interaction_df.columns)
            interactions_hit = [(interaction[0].replace(" ", ""), interaction[1].replace(" ", ""), interaction[2]) for interaction in interactions]
        else: interactions_hit = []

        tgt_out_file.close()
        lig_out_file.close()

        flag = FilterResult.FAILED if len(interactions_hit) < self.minimum_num else FilterResult.PASSED

        return flag, { 'interaction_hit': interactions_hit }


@R.register('ContactFilter')
class ContactFilter(BaseFilter):
    def __init__(self, minimum_num: int, tgt_hotspots: list, contact_dist: float=3.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
            Reject the candidate if number of interactions is less than "minimum_num"
        '''
        self.minimum_num = minimum_num
        self.tgt_hotspots = [(id[0], tuple(id[1])) for id in tgt_hotspots]
        self.contact_dist = contact_dist

    @property
    def name(self):
        return self.__class__.__name__ + f'(minimum_num={self.minimum_num}, tgt_hotspots={self.tgt_hotspots}), contact_dist={self.contact_dist})'

    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        try:
            cplx = mmcif_to_complex(input.path_prefix + '.cif', selected_chains=input.tgt_chains + input.lig_chains)
        except Exception as e:
            return FilterResult.ERROR, { 'error': str(e) }
        pairs = compute_interacting_pairs(cplx, input.tgt_chains, input.lig_chains, dist_th=self.contact_dist, efficient=True)
        satisfied_pairs, all_pairs = set(), []
        for pair in pairs:
            src, dst = pair[0], pair[1]
            src_id = (src[0], (src[1][0], re.sub(r'\d', '', src[1][1])))    # get rid of fragmentations, e.g. (E, (1, '0')) and (E, (1, '1')) both belong to (E, (1, ''))
            dst_id = (dst[0], (dst[1][0], re.sub(r'\d', '', dst[1][1])))
            if src_id in self.tgt_hotspots:
                satisfied_pairs.add((src_id, dst_id))
            all_pairs.append((pair[0], pair[1], float(pair[-1])))
        flag = FilterResult.FAILED if len(satisfied_pairs) < self.minimum_num else FilterResult.PASSED
        
        return flag, { 'all_contacts': all_pairs, 'satisfied_contacts': list(satisfied_pairs), 'num_satisfied_interactions': len(satisfied_pairs) }


@R.register('ContactRecoveryFilter')
class ContactRecoveryFilter(BaseFilter):
    def __init__(self, ref_cplx_path: str, ref_lig_chains: list, recovery_th: float=0.5, contact_dist: float=4.0, normalize_mode: int=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
            Reject the candidate if the recovery rate of the contacts is below recovery_th
            for the contact_dist, 4.0 angstrom is a safe threshold for interactions, but if you
            are focused more on hydrogen bonds, 3.5 or 3.0 might be a better threshold
            normalize_mode: different normalizing length of the recovered contacts
                0: num_recover / min(num_ref, num_model)
                1: num_recover / num_ref
                2: num_recover / num_model
        '''
        self.ref_cplx_path = ref_cplx_path  # target chains should be the same as the generated model
        self.ref_lig_chains = ref_lig_chains
        self.ref_cplx = load_cplx(ref_cplx_path)
        self.recovery_th = recovery_th
        self.contact_dist = contact_dist
        self.normalize_mode = normalize_mode

        # lazy-init
        self.ref_interactions = None

    @property
    def name(self):
        return self.__class__.__name__ + f'(ref_cplx_path={self.ref_cplx_path}, recovery_th={self.recovery_th}, contact_dist={self.contact_dist})'

    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        if self.ref_interactions is None:
            ref_pairs = compute_interacting_pairs(self.ref_cplx, input.tgt_chains, self.ref_lig_chains, dist_th=self.contact_dist, efficient=True)
            self.ref_interactions = {}
            for pair in ref_pairs:
                src = pair[0]
                if src not in self.ref_interactions: self.ref_interactions[src] = 0
                self.ref_interactions[src] += 1
        # load generated complex
        try:
            cplx = mmcif_to_complex(input.path_prefix + '.cif', selected_chains=input.tgt_chains + input.lig_chains)
        except Exception as e:
            return FilterResult.ERROR, { 'error': str(e) }
        pairs = compute_interacting_pairs(cplx, input.tgt_chains, input.lig_chains, dist_th=self.contact_dist, efficient=True)
        model_interactions = {}
        for pair in pairs:
            src = pair[0]
            if src not in model_interactions: model_interactions[src] = 0
            model_interactions[src] += 1
        hits = 0
        for resid in self.ref_interactions:
            hits += min(self.ref_interactions[resid], model_interactions.get(resid, 0))

        num_ref = sum(list(self.ref_interactions.values()))
        num_model = sum(list(model_interactions.values()))
        if self.normalize_mode == 0: recovery_rate = hits / (min(num_ref, num_model) + 1e-10)
        elif self.normalize_mode == 1: recovery_rate = hits / (num_ref + 1e-10)
        elif self.normalize_mode == 2: recovery_rate = hits / (num_model + 1e-10)

        flag = FilterResult.FAILED if recovery_rate < self.recovery_th else FilterResult.PASSED
        
        return flag, { 'contact_recovery': recovery_rate }


if __name__ == '__main__':
    import sys
    # f = ContactFilter(2, [('E', (4, ''))])
    f = ContactRecoveryFilter(sys.argv[1], ['L'])
    print(f.name)
    print(f.run(FilterInput(sys.argv[2], ['A', 'R'], ['L'], None, None, None, None)))
