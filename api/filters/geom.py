# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import ray
import numpy as np
from rdkit import Chem


import utils.register as R
from evaluation.posebuster import denovo_validity
from evaluation.clash import detect_atom_clash
from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
from data.bioparse.parser.sdf_to_complex import sdf_to_complex
from data.bioparse.utils import recur_index, extract_sub_complex
from data.bioparse.interface import compute_pocket
from data.bioparse.writer.complex_to_pdb import complex_to_pdb

from .base import BaseFilter, FilterResult, FilterInput


@R.register('AbnormalConfidenceFilter')
class AbnormalConfidenceFilter(BaseFilter):

    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        if np.isnan(input.confidence): return FilterResult.FAILED, { 'confidence': input.confidence }
        if input.likelihood is not None and input.likelihood < 0: return FilterResult.FAILED, { 'likelihood': input.likelihood }
        return FilterResult.PASSED, { 'confidence': input.confidence, 'likelihood': input.likelihood }


@R.register('ConfidenceThresholdFilter')
class ConfidenceThresholdFilter(BaseFilter):

    def __init__(self, th: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.th = th

    @property
    def name(self):
        return self.__class__.__name__ + f'(th={self.th})'

    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        flag = FilterResult.PASSED if input.confidence < self.th else FilterResult.FAILED
        return flag, { 'confidence': input.confidence }


@R.register('PhysicalValidityFilter')
class PhysicalValidityFilter(BaseFilter):

    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        # save new pockets (generated ligand might clash with unseen parts)
        pocket_pdb = input.path_prefix + '_tmp_pocket.pdb'
        try:
            cplx = mmcif_to_complex(input.path_prefix + '.cif', selected_chains=input.tgt_chains + input.lig_chains)
            pocket_block_ids, _ = compute_pocket(cplx, input.tgt_chains, input.lig_chains)
            pocket_cplx = extract_sub_complex(cplx, pocket_block_ids)
            complex_to_pdb(pocket_cplx, pocket_pdb, selected_chains=input.tgt_chains)
        except Exception as e:
            if os.path.exists(pocket_pdb): os.remove(pocket_pdb)
            return FilterResult.ERROR, { 'error': str(e) }
        # pocket_pdb = os.path.join(os.path.dirname(input.path_prefix), 'pocket.pdb')
        mol_sdf = input.path_prefix + '.sdf'
        # Load the SDF file using RDKit
        supplier = Chem.SDMolSupplier(mol_sdf)
        for idx, mol in enumerate(supplier):
            # Create a temporary filename for each molecule
            temp_filename = input.path_prefix + f'_tmp{idx}.sdf'

            # Write the molecule to the temporary SDF file
            with Chem.SDWriter(temp_filename) as writer:
                writer.write(mol)
            validity, details = denovo_validity(pocket_pdb, temp_filename, remove_energy_term=True, loose_th=True)
            assert len(validity) == 1
            os.remove(temp_filename)
            if not validity[0]: return FilterResult.FAILED, details
        if os.path.exists(pocket_pdb): os.remove(pocket_pdb)
        return FilterResult.PASSED, {}


@R.register('SimpleGeometryFilter')
class SimpleGeometryFilter(BaseFilter):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # bond length
        self.bond_len_cutoff = 2.0  # a relatively loose criterion
        self.atom_sets = { 6, 7, 8 } # C, N, O

    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        
        try:
            supplier = Chem.SDMolSupplier(input.path_prefix + '.sdf')
        except Exception as e:
            return FilterResult.ERROR, { 'error': str(e) }
        
        # very long bonds
        for mol in supplier:
            if mol is None: continue
            defs, dists = bond_length_from_mol(mol)
            for (a1, a2, _), d in zip(defs, dists):
                if a1 in self.atom_sets and a2 in self.atom_sets:
                    if d > self.bond_len_cutoff:
                        return FilterResult.FAILED, { f'{a1}-{a2}': d }

        return FilterResult.PASSED, {}
    

@R.register('SimpleClashFilter')
class SimpleClashFilter(BaseFilter):

    def __init__(self, inter_covalent_exist=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clash_rates = 0.1  # a relatively loose criterion (10% clash)
        # self.skip_tgt_res = [] if skip_tgt_res is None else [tuple(i) for i in skip_tgt_res]    # [chain id, (residue number, insertion code)]
        self.inter_covalent_exist = inter_covalent_exist

    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):

        skip_residues = []
        try:
            if self.inter_covalent_exist:
                cplx = mmcif_to_complex(input.path_prefix + '.cif', selected_chains=input.tgt_chains + input.lig_chains)
                for bond in cplx.bonds:
                    mol1, mol2 = cplx[bond.index1[0]], cplx[bond.index2[0]]
                    if (mol1.id in input.tgt_chains and mol2.id in input.lig_chains) or \
                       (mol1.id in input.lig_chains and mol2.id in input.tgt_chains):
                        skip_residues.append((mol1.id, mol1[bond.index1[1]].id))
                        skip_residues.append((mol2.id, mol2[bond.index2[1]].id))
                target = [cplx[c] for c in input.tgt_chains]
                ligand = [cplx[c] for c in input.lig_chains]
            else:
                target = mmcif_to_complex(input.path_prefix + '.cif', selected_chains=input.tgt_chains)
                ligand = sdf_to_complex(input.path_prefix + '.sdf') # only the generated part (e.g. antibody CDRs)
        except Exception as e:
            return FilterResult.ERROR, { 'error': str(e) }

        
        # clash
        def get_pos_elements(cplx, skip_blocks):
            pos, elements = [], []
            for mol in cplx:
                for block in mol:
                    if (mol.id, block.id) in skip_blocks:
                        continue
                    for atom in block:
                        pos.append(atom.get_coord())
                        elements.append(atom.get_element())
            return pos, elements
        
        pos_tgt, element_tgt = get_pos_elements(target, skip_residues)
        pos_lig, element_lig = get_pos_elements(ligand, skip_residues)

        clash_atoms = detect_atom_clash(pos_tgt, pos_lig, element_tgt, element_lig)

        clash_rate = len(clash_atoms) / len(pos_lig)

        stats = { 'clash_rate': clash_rate }

        if clash_rate < self.clash_rates: return FilterResult.PASSED, stats
        else: return FilterResult.FAILED, stats


@R.register('ChainBreakFilter')
class ChainBreakFilter(BaseFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cn_th = 2.0    # loose
        self.ca_th = 4.0

    @property
    def name(self):
        return self.__class__.__name__ + f'(cn_th={self.cn_th}, ca_th={self.ca_th})'
    
    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        cplx = mmcif_to_complex(input.path_prefix + '.cif', selected_chains=input.lig_chains)
        for mol in cplx:
            c_coords, n_coords, ca_coords = [], [], []
            for block in mol:
                has_C, has_N, has_CA = False, False, False
                for atom in block:
                    if atom.name == 'C':
                        c_coords.append(atom.get_coord())
                        has_C = True
                    elif atom.name == 'N':
                        n_coords.append(atom.get_coord())
                        has_N = True
                    elif atom.name == 'CA':
                        ca_coords.append(atom.get_coord())
                        has_CA = True
                if not has_C: c_coords.append([np.nan] * 3)
                if not has_N: n_coords.append([np.nan] * 3)
                if not has_CA: ca_coords.append([np.nan] * 3)
            c_coords, n_coords, ca_coords = np.array(c_coords), np.array(n_coords), np.array(ca_coords)
            cn_dist = np.linalg.norm(n_coords[1:] - c_coords[:-1], axis=-1)
            ca_dist = np.linalg.norm(ca_coords[1:] - ca_coords[:-1], axis=-1)
            cn_break = (cn_dist[np.where(~np.isnan(cn_dist))] > self.cn_th).sum()
            ca_break = (ca_dist[np.where(~np.isnan(ca_dist))] > self.ca_th).sum()
            if cn_break > 0 or ca_break > 0:
                return FilterResult.FAILED, { 'cn_break': int(cn_break), 'ca_break': int(ca_break) }
        return FilterResult.PASSED, {}


@R.register('TargetBinderCovalentDistanceFilter')
class TargetBinderCovalentDistanceFilter(BaseFilter):

    def __init__(self, cutoff=3.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cutoff = cutoff
    
    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        try: cplx = mmcif_to_complex(input.path_prefix + '.cif', selected_chains=input.tgt_chains + input.lig_chains)
        except Exception as e: return FilterResult.ERROR, { 'error': str(e) }

        stats = { 'bonds': [] }
        for bond in cplx.bonds:
            index1, index2 = bond.index1, bond.index2
            mol1, mol2 = cplx[index1[0]], cplx[index2[0]]
            if (mol1.id in input.tgt_chains and mol2.id in input.lig_chains) or \
               (mol1.id in input.lig_chains and mol2.id in input.tgt_chains):
                block1, block2 = recur_index(cplx, index1[:-1]), recur_index(cplx, index2[:-1])
                atom1, atom2 = block1[index1[-1]], block2[index2[-1]]
                # a covalent bond identified
                dist = np.linalg.norm(np.array(atom1.get_coord()) - np.array(atom2.get_coord()))
                stats['bonds'].append((index1, index2, dist))
                if dist > self.cutoff: return FilterResult.FAILED, stats

        return FilterResult.PASSED, stats


@R.register('DisulfideFilter')
class DisulfideFilter(BaseFilter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cutoff = 3.0   # 3.0 angstrom is taken as disulfide cutoff in PDB
    
    @property
    def name(self):
        return self.__class__.__name__
    
    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        try: cplx = mmcif_to_complex(input.path_prefix + '.cif', selected_chains=input.lig_chains)
        except Exception as e: return FilterResult.ERROR, { 'error': str(e) }

        stats = { 'bonds': [] }
        for bond in cplx.bonds:
            index1, index2 = bond.index1, bond.index2
            block1, block2 = recur_index(cplx, index1[:-1]), recur_index(cplx, index2[:-1])
            if block1.name != 'CYS' or block2.name != 'CYS': continue
            atom1, atom2 = block1[index1[-1]], block2[index2[-1]]
            if atom1.name != 'SG' or atom2.name != 'SG': continue
            # a disulfide bond identified
            dist = np.linalg.norm(np.array(atom1.get_coord()) - np.array(atom2.get_coord()))
            stats['bonds'].append((index1, index2, dist))
            if dist > self.cutoff: return FilterResult.FAILED, stats

        return FilterResult.PASSED, stats


@R.register('HeadTailAmideFilter')
class HeadTailAmideFilter(BaseFilter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cutoff = 2.0
    
    @property
    def name(self):
        return self.__class__.__name__
    
    @ray.remote(num_cpus=1)
    def run(self, input: FilterInput):
        try:
            assert len(input.lig_chains) == 1, f'Only cyclic peptides requires this filter, but got several ligand chains: {input.ligand_chains}'
            cplx = mmcif_to_complex(input.path_prefix + '.cif', selected_chains=input.lig_chains)
        except Exception as e: return FilterResult.ERROR, { 'error': str(e) }

        n_term, c_term = None, None
        for atom in cplx[0][0]:
            if atom.name == 'N': n_term = atom
        for atom in cplx[0][len(cplx[0]) - 1]:
            if atom.name == 'C': c_term = atom
        if n_term is None: return FilterResult.ERROR, { 'error': 'N terminal not found'}
        if c_term is None: return FilterResult.ERROR, { 'error': 'C terminal not found'}
        dist = np.linalg.norm(np.array(n_term.get_coord()) - np.array(c_term.get_coord()))
        if dist > self.cutoff: return FilterResult.FAILED, { 'nc_dist': dist }

        return FilterResult.PASSED, { 'nc_dist': dist }


if __name__ == '__main__':
    import sys
    # smiles = 'CCCCCC'
    # f = TwistedGeometryFilter()
    # print(f.name)
    # print(f.run(FilterInput(sys.argv[1], [], [], smiles, None, None, None)))
    # f = DisulfideFilter()
    # f = HeadTailAmideFilter()
    # f = SimpleGeometryFilter()
    # f = SimpleClashFilter()
    # f = ChainBreakFilter()
    # f = TargetBinderCovalentDistanceFilter()
    f = PhysicalValidityFilter()
    print(f.name)
    print(f.run(FilterInput(sys.argv[1], ['A'], ['B'], None, None, None, None)))
    