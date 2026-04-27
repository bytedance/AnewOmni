# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import math
from copy import deepcopy
from typing import List, Optional
from dataclasses import dataclass

from tqdm import tqdm
from rdkit import Chem
import numpy as np

from data.bioparse import Complex, Block, Atom, BondType, VOCAB
from data.bioparse.utils import overwrite_block, is_standard_aa, index_to_numerical_index, bond_type_to_rdkit, bond_type_from_rdkit, recur_index
from data.bioparse.tokenizer.tokenize_3d import TOKENIZER
from data.bioparse.hierarchy import remove_mols, add_dummy_mol
from data.bioparse.writer.complex_to_mmcif import complex_to_mmcif
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from data.bioparse.writer.rdkit_mol_to_sdf import rdkit_mol_to_sdf
from data.file_loader import MolLoader, PeptideLoader, AntibodyLoader, MolType
# from evaluation.geom.check_twisted_bond import check_twisted_bond, validate_geometry
from evaluation.geom.check_twisted_bond import validate_geometry
from models.confidence.model import ConfidenceReturn
from utils.chem_utils import valence_check, cycle_check, connect_fragments
from utils.logger import print_log


@dataclass
class OverwriteTask:

    cplx: Complex
    # necessary information
    select_indexes: list
    generate_mask: list
    target_chain_ids: list
    ligand_chain_ids: list
    # generated part
    S: list
    X: list
    A: list
    ll: list
    inter_bonds: tuple
    intra_bonds: list
    confidence: ConfidenceReturn
    likelihood: float
    # output
    out_path: str
    save_cif: bool
    # modifications
    covalent_modifications: Optional[List]=None # list of tuples as ((a1, b1, c1), (a2, b2, c2), bond type) where a, b, c denote the numerical index of chain, block, and atom

    def _get_num_atoms(self):
        cnt = 0
        for a in self.A: cnt += len(a)
        return cnt

    def get_generated_seq(self):
        gen_seq = ''.join([VOCAB.idx_to_symbol(block_S) for block_S in self.S])
        return gen_seq

    def get_total_likelihood(self):
        if self.confidence is not None:
            return self.confidence.confidence
        elif self.likelihood is not None:
            return self.likelihood  # diffusion likelihood
        flat_ll = []
        for block_ll in self.ll: flat_ll.extend(block_ll)
        return sum(flat_ll) / len(flat_ll)

    def get_normalized_likelihood(self):
        if self.likelihood is not None:
            # e^{1/50 * (-ll / N)}
            if self.likelihood < 0: return 0 # abnormal
            return min(1.0, math.exp(-self.likelihood / self._get_num_atoms() * 145))  # max 1.0
        return None
    
    def get_overwritten_results(
            self,
            check_validity: bool=False,             # whether to check validity of small molecules
            expect_atom_num: int=None,              # discard very small molecules when checking validity
            filters: list=None,                     # other filters (using the cplx as input)
            struct_pred: bool=False                 # if true, use the original naming for atoms
        ):

        cplx = deepcopy(self.cplx)

        overwrite_indexes = [i for i, is_gen in zip(self.select_indexes, self.generate_mask) if is_gen]
        if len(overwrite_indexes) != len(self.X): # length change, need to modify the complex
            cplx, overwrite_indexes = modify_gen_length(cplx, len(self.X), self.ligand_chain_ids)

        assert len(overwrite_indexes) == len(self.X)
        assert len(self.X) == len(self.A)
        assert len(self.A) == len(self.ll)

        has_nan = False
    
        explicit_bonds, atom_idx_map, gen_mol, all_atom_coords = [], {}, Chem.RWMol(), []
        cif_intra_block_bonds = []
        for i, index in enumerate(overwrite_indexes):
            block_S, block_X, block_A, block_ll, block_bonds = self.S[i], self.X[i], self.A[i], self.ll[i], self.intra_bonds[i]
            block_name = 'UNK' if block_S is None else VOCAB.idx_to_abrv(block_S)
            # construct a new block
            atoms, local2global = [], {}
            if struct_pred:
                gt_block = recur_index(cplx, index)
                canonical_order = [atom.name for atom in gt_block]
            elif block_bonds is None or is_standard_aa(block_name): # use canonical order
                canonical_order = VOCAB.abrv_to_atoms(block_name)
            else:
                canonical_order = []    # use the input order, which might be different from the canonical order
            for atom_order, (x, a, l) in enumerate(zip(block_X, block_A, block_ll)):
                if np.isnan(x).any():
                    print_log(f'NaN encountered for overwriting task on {self.out_path}; set to [0, 0, 0]', level='WARN')
                    x = [0.0, 0.0, 0.0]
                    has_nan = True
                atom_element = VOCAB.idx_to_atom(a)
                atoms.append(Atom(
                    # TODO: for structure prediction, the input order might be different from the canonical order
                    # However, atom names are not passed to the model
                    # Therefore, the only solution is that, for structure prediction, if there is a standard
                    # amino acid, then the input order of the atoms should align with the canonical order,
                    # which should be handled in the data processing logic, as the users only need to input the
                    # amino acid sequences
                    name=canonical_order[atom_order] if atom_order < len(canonical_order) else atom_element,
                    coordinate=x,
                    element=atom_element,
                    id=-1,  # normally this should be positive integer, set to -1 for later renumbering
                    properties={'bfactor': round(l, 2), 'occupancy': 1.0 }
                ))
                # update atom global index mapping to (block index, intra-block order)
                atom_idx_map[len(atom_idx_map)] = (index, atom_order)
                # update RWMol
                gen_mol.AddAtom(Chem.Atom(atom_element))
                all_atom_coords.append(x)
                # update local2global
                local2global[atom_order] = len(atom_idx_map) - 1
            # overwrite block
            overwrite_block(cplx, index, Block(
                name=block_name,
                atoms=atoms,
                id=index[1],
            ))
            # prepare intra-block bonds
            if block_bonds is None: block_bonds = VOCAB.abrv_to_bonds(block_name)
            else: block_bonds = [(src, dst, BondType(t)) for src, dst, t in zip(*block_bonds)]
            # add explicit bonds to record in PDB if the block is not a canonical amino acid
            if not is_standard_aa(block_name):
                numerical_index = index_to_numerical_index(cplx, index)
                for bond in block_bonds:
                    explicit_bonds.append((
                        (numerical_index[0], numerical_index[1], bond[0]),
                        (numerical_index[0], numerical_index[1], bond[1]),
                        bond[2]
                    ))
            else:
                numerical_index = index_to_numerical_index(cplx, index)
                for bond in block_bonds:
                    cif_intra_block_bonds.append((
                        (numerical_index[0], numerical_index[1], bond[0]),
                        (numerical_index[0], numerical_index[1], bond[1]),
                        bond[2]
                    ))
            # update bonds for RWMol
            for bond in block_bonds:
                begin, end = local2global[bond[0]], local2global[bond[1]]
                gen_mol.AddBond(begin, end, bond_type_to_rdkit(bond[2]))

        # processing inter-block bonds
        def format_prob_tuple(prob):
            conf, dist = prob
            dist_level = int(dist / 0.5) # [0, 0.5) - 0, [0.5, 1.0) - 1, [1.0, 1.5) - 2
            uncertainty = 1 - conf
            return (dist_level, uncertainty)

        # using model predicted bonds
        if self.inter_bonds is not None:
            bond_tuples = []
            for atom_idx1, atom_idx2, prob, bond_type in zip(*self.inter_bonds): # idxs are global idxs
                prob = format_prob_tuple(prob)
                if bond_type == 4 and TOKENIZER.kekulize: continue # no aromatic bonds
                bond_tuples.append((atom_idx1, atom_idx2, prob, BondType(bond_type))) # prob: confidence, distance
            bond_tuples = sorted(bond_tuples, key=lambda tup: tup[2]) # sorted by confidence
            for atom_idx1, atom_idx2, prob, bond_type in bond_tuples:
                rdkit_bond = bond_type_to_rdkit(bond_type)
                # bond_len = np.linalg.norm(np.array(all_atom_coords[atom_idx1]) - np.array(all_atom_coords[atom_idx2]))
                if valence_check(gen_mol, atom_idx1, atom_idx2, rdkit_bond) and cycle_check(gen_mol.GetMol(), atom_idx1, atom_idx2, bond_type_to_rdkit(bond_type)): # and sp2_check(gen_mol.GetMol(), atom_idx1, atom_idx2, all_atom_coords):
                    # pass valence check and cycle check
                    gen_mol.AddBond(atom_idx1, atom_idx2, rdkit_bond)
                    # add to explicit bonds
                    index1, atom_order1 = atom_idx_map[atom_idx1]
                    numerical_index1 = index_to_numerical_index(cplx, index1)
                    index2, atom_order2 = atom_idx_map[atom_idx2]
                    numerical_index2 = index_to_numerical_index(cplx, index2)
                    explicit_bonds.append((
                        (numerical_index1[0], numerical_index1[1], atom_order1),
                        (numerical_index2[0], numerical_index2[1], atom_order2),
                        bond_type
                    ))
                    
        # connect disconnected fragments
        gen_mol, added_bonds = connect_fragments(gen_mol, all_atom_coords)
        for atom_idx1, atom_idx2, rdkit_bond in added_bonds:
            # add to explicit bonds
            index1, atom_order1 = atom_idx_map[atom_idx1]
            numerical_index1 = index_to_numerical_index(cplx, index1)
            index2, atom_order2 = atom_idx_map[atom_idx2]
            numerical_index2 = index_to_numerical_index(cplx, index2)
            explicit_bonds.append((
                (numerical_index1[0], numerical_index1[1], atom_order1),
                (numerical_index2[0], numerical_index2[1], atom_order2),
                bond_type_from_rdkit(rdkit_bond)
            ))

        gen_mol = gen_mol.GetMol()

        try: Chem.SanitizeMol(gen_mol)
        except Exception: pass
        smiles = Chem.MolToSmiles(gen_mol)

        if check_validity and ((not validate_small_mol(gen_mol, smiles, all_atom_coords, expect_atom_num)) or has_nan):
            return None, None, None
        
        if filters is not None:
            for func in filters:
                if not func(cplx):
                    return None, None, None

        # add covalent modifications
        if self.covalent_modifications is not None:
            for a, b, bt in self.covalent_modifications:
                explicit_bonds.append((a, b, BondType(bt)))

        if self.save_cif:
            # save mmcif
            complex_to_mmcif(cplx, self.out_path.rstrip('.pdb') + '.cif', selected_chains=self.target_chain_ids + self.ligand_chain_ids, explict_bonds=explicit_bonds + cif_intra_block_bonds)
        # save pdb
        complex_to_pdb(cplx, self.out_path, selected_chains=self.target_chain_ids + self.ligand_chain_ids, explict_bonds=explicit_bonds)
        # save sdf
        gen_mol.SetProp('_Name', self.ligand_chain_ids[0])
        rdkit_mol_to_sdf(gen_mol, all_atom_coords, self.out_path.rstrip('.pdb') + '.sdf')
        # save confidence results
        self.save_confidence(cplx, overwrite_indexes)
    
        return cplx, gen_mol, overwrite_indexes

    def save_confidence(self, cplx, overwrite_indexes):
        save_path = self.out_path.rstrip('.pdb') + '_confidence.json'
        if self.confidence is None: # only save likelihood
            with open(save_path, 'w') as fout:
                json.dump({
                    'likelihood': self.likelihood,
                    'normalized_likelihood': self.get_normalized_likelihood(),
                }, fout)
            return
        context_atom_residue_ids = []
        generate_atom_residue_ids = []
        for resid in overwrite_indexes:
            block = recur_index(cplx, resid)
            generate_atom_residue_ids.extend([resid for _ in block])
        for resid, is_gen in zip(self.select_indexes, self.generate_mask):
            if is_gen: continue
            block = recur_index(cplx, resid)
            if 'original_name' in block.properties: # maybe need to rectify resid (e.g. ('C', (121, '0')))
                chain_id, (resnum, icode) = resid
                if icode.isdigit(): icode = ''  # this is a fragment of the original residue
                resid = (chain_id, (resnum, icode))
            context_atom_residue_ids.extend([resid for atom in block if atom.get_element() != 'H'])

        lig_PDE = self.confidence.lig_pde.cpu().tolist()
        cplx_PDE = self.confidence.cplx_pde.cpu().tolist()
        with open(save_path, 'w') as fout:
            json.dump({
                'context_atom_residue_ids': context_atom_residue_ids,
                'generate_atom_residue_ids': generate_atom_residue_ids,
                'lig_PDE': lig_PDE,
                'cplx_PDE': cplx_PDE,
                'likelihood': self.likelihood,
                'normalized_likelihood': self.get_normalized_likelihood(),
                'cplx_PDE': cplx_PDE,
                'lig_PDE_local': self.confidence.lig_pde_local.cpu().tolist(),
                'cplx_PDE_local_lig': self.confidence.cplx_pde_local_lig.cpu().tolist(),
                'cplx_PDE_local_pocket': self.confidence.cplx_pde_local_pocket.cpu().tolist()
            }, fout)


def modify_gen_length(cplx: Complex, new_len: int, replace_ids: List[str]):
    cplx = remove_mols(cplx, replace_ids)
    cplx = add_dummy_mol(cplx, new_len, replace_ids[0])
    indexes = [(replace_ids[0], block.id) for block in cplx[replace_ids[0]]]
    return cplx, indexes


def validate_small_mol(mol, smiles, coords, expect_atom_num=None):
    if '.' in smiles: return False
    mol_size = mol.GetNumAtoms()
    if expect_atom_num is not None:
        if mol_size < expect_atom_num - 5:
            print_log(f'mol size {mol_size}, far below expectation {expect_atom_num}')
            return False # sometimes the model will converge to single blocks (like one indole)
    # if mol_size < 15: return False  # sometimes the model will converge to single blocks (like one indole)
    # validate bond length and angles. As we are predicting bonds by model,
    # there might be a few failed cases with many abnormal bond length and angles
    # between fragments. Such results should be discarded.
    # (num_twist_bond, num_total_bond), (num_twist_angle, num_total_angle) = check_twisted_bond(mol, coords)
    geometry_profile = validate_geometry(mol, coords)
    num_twist_bond, num_twist_angle = len(geometry_profile['bond_length_problems']), len(geometry_profile['bond_angle_problems'])
    num_total_bond, num_total_angle = geometry_profile['bond_cnt'], geometry_profile['angle_cnt']
    rel_bond, rel_angle = num_twist_bond / mol_size, num_twist_angle / mol_size
    # rel_bond = num_twist_bond / (num_total_bond + 1e-10)
    # rel_angle = num_twist_angle / (num_total_angle + 1e-10)
    print_log(f'twist bond: {num_twist_bond}/{num_total_bond}, twist angle: {num_twist_angle}/{num_total_angle}, mol size: {mol_size}', level='DEBUG')
    return (rel_bond + rel_angle) < 0.1
    # return (rel_bond < 0.05) & (rel_angle < 0.05)


def _get_item_multitype(pocket_path, sdf_path, cplx_path, tgt_chains, lig_chains, cdr_type, fr_len, mol_type: MolType):
    if mol_type == MolType.MOLECULE:
        if cplx_path.endswith('.cif'): # mmcif records everything including chemical bonds
            loader = MolLoader(
                cplx_file=cplx_path,
                tgt_chains=tgt_chains,
                lig_chains=lig_chains
            )
        else:
            loader = MolLoader( # otherwise use pocket pdb + molecule sdf
                tgt_file=pocket_path,
                lig_file=sdf_path
            )
    elif mol_type == MolType.PEPTIDE:
        loader = PeptideLoader(
            cplx_file=cplx_path,
            tgt_chains=tgt_chains,
            lig_chains=lig_chains
        )
    elif mol_type == MolType.ANTIBODY:
        loader = AntibodyLoader(
            cplx_file=cplx_path,
            tgt_chains=tgt_chains,
            lig_chains=lig_chains,
            cdr_type=cdr_type,
            fr_len=fr_len
        )
    else: raise NotImplementedError(f'type {mol_type} not implemented')
    cplx, pocket_block_ids, ligand_block_ids = loader.load_cplx()
    data, pocket_block_ids, ligand_block_ids = loader.cplx_to_data(cplx, pocket_block_ids, ligand_block_ids, return_ids=True)
    return data, cplx, pocket_block_ids, ligand_block_ids # selected indexes


class Recorder:
    def __init__(self, test_set, n_samples, save_dir, max_retry=None, verbose=True):
        self.pbar = tqdm(total=n_samples * len(test_set)) if verbose else None
        self.verbose = verbose
        self.waiting_list = [(i, n) for n in range(n_samples) for i in range(len(test_set))]
        self.num_generated, self.num_failed = 0, 0
        self.max_retry = max_retry

        self.tried_times = {}
        self.fout = open(os.path.join(save_dir, 'results.jsonl'), 'w')

    def is_finished(self):
        return len(self.waiting_list) == 0

    def get_next_batch_list(self, batch_size):
        batch_list = self.waiting_list[:batch_size]
        self.waiting_list = self.waiting_list[batch_size:]
        return batch_list

    def check_and_save(self, log, item_idx, n, struct_only=False):
        self.num_generated += 1
        if log is None:
            self.num_failed += 1
            uid = (item_idx, n)
            if uid not in self.tried_times: self.tried_times[uid] = 0
            self.tried_times[uid] += 1
            if self.max_retry is None or self.tried_times[uid] < self.max_retry:
                self.waiting_list.append((item_idx, n))
            else:
                print_log(f'test set index {uid[0]}, candidate {uid[1]} failed for {self.tried_times[uid]} times, skip this one', level='DEBUG')
        else:
            log.update({
                'n': n,
                'struct_only': struct_only
            })
            self.fout.write(json.dumps(log) + '\n')
            self.fout.flush()
            if self.verbose: self.pbar.update(1)

    def __del__(self):
        self.fout.close()