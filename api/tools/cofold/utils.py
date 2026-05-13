# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import json
import warnings
warnings.filterwarnings("ignore", module="Bio") # suppress warnings of pairwise2
import tempfile
from copy import deepcopy

import numpy as np
from Bio import pairwise2
from Bio.Align import substitution_matrices
from Bio.Data.IUPACData import protein_letters_3to1
from biotite.structure.io.pdbx import CIFFile, get_structure, set_structure
from biotite.structure import residue_iter

from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
from data.bioparse.writer.complex_to_mmcif import complex_to_mmcif
from data.bioparse.hierarchy import remove_mols
from evaluation.rmsd import kabsch, compute_rmsd


def load_confidences(json_path, tgt_chains, lig_chains, iptm_row_cols=None):
    tgt_chains, lig_chains = set(tgt_chains), set(lig_chains)

    # from confidence summary
    item_summary = json.load(open(json_path, 'r'))

    # from full details
    full_detail_json_path = json_path.replace('summary_', '')
    item = json.load(open(full_detail_json_path, 'r'))
    # plddt
    atom_chain_ids = item['atom_chain_ids']
    atom_plddts = item['atom_plddts']
    binder_plddts = [plddt for plddt, c in zip(atom_plddts, atom_chain_ids) if c in lig_chains]

    # ipAE
    # bugs: in atom_chain_ids, the chain ids are the same as given,
    # but in token_chain_ids, the chain ids are A, B, C, ...
    ac2i, tc2i = {}, {}
    for c in item['atom_chain_ids']:
        if c not in ac2i: ac2i[c] = len(ac2i)
    for c in item['token_chain_ids']:
        if c not in tc2i: tc2i[c] = len(tc2i)
    i2ac = [None for _ in ac2i]
    for c in ac2i: i2ac[ac2i[c]] = c
    tc2ac = { tc: i2ac[tc2i[tc]] for tc in tc2i }
    token_chain_ids, token_pae = item['token_chain_ids'], item['pae']
    ipae = []
    for i, row in enumerate(token_pae):
        for j, val in enumerate(row):
            ci, cj = tc2ac[token_chain_ids[i]], tc2ac[token_chain_ids[j]]
            if (ci in tgt_chains and cj in lig_chains) or (ci in lig_chains and cj in tgt_chains):
                ipae.append(val) 

    # iptm
    if iptm_row_cols is not None:
        chain_pair_iptm = item_summary['chain_pair_iptm']
        iptm = [chain_pair_iptm[row][col] for row, col in iptm_row_cols]
        iptm = sum(iptm) / len(iptm)
    else: iptm = item_summary['iptm']   # overall iptm

    return {
        'iptm': item_summary['iptm'],
        'ptm': item_summary['ptm'],
        'ranking_score': item_summary['ranking_score'],
        'plddt': sum(atom_plddts) / len(atom_plddts) if len(atom_plddts) > 0 else None,
        'binder_plddt': sum(binder_plddts) / len(binder_plddts) if len(binder_plddts) > 0 else None,
        'ipae': sum(ipae) / len(ipae) if len(ipae) > 0 else None
    }


def get_scRMSD(ref_path, model_path, tgt_chains, lig_chains, gen_mask=None, align_by_target=True):
    ref_cplx = mmcif_to_complex(ref_path, selected_chains=tgt_chains + lig_chains)
    model_cplx = mmcif_to_complex(model_path, selected_chains=tgt_chains + lig_chains)

    # get CA coordinates
    def get_ca_coords(mol):
        coords = []
        for block in mol:
            for atom in block:
                if atom.name == 'CA': coords.append(atom.get_coord())
        return coords

    # get CA coordinates of the target
    ref_tgt_ca, model_tgt_ca = [], []
    for c in tgt_chains: ref_tgt_ca.extend(get_ca_coords(ref_cplx[c]))
    for c in tgt_chains: model_tgt_ca.extend(get_ca_coords(model_cplx[c]))
    ref_tgt_ca, model_tgt_ca = np.array(ref_tgt_ca), np.array(model_tgt_ca)

    # get CA coordinates of the ligand
    ref_lig_ca, model_lig_ca = [], []
    for c in lig_chains:
        ref_lig_ca.extend(get_ca_coords(ref_cplx[c]))
        model_lig_ca.extend(get_ca_coords(model_cplx[c]))
    ref_lig_ca, model_lig_ca = np.array(ref_lig_ca), np.array(model_lig_ca)
    
    if align_by_target:
        # get transformation matrix
        _, rotation, t = kabsch(model_tgt_ca, ref_tgt_ca)
        # transform
        model_lig_ca_aligned = np.dot(model_lig_ca, rotation) + t
    else: model_lig_ca_aligned, _, _ = kabsch(model_lig_ca, ref_lig_ca)

    sc_rmsd = compute_rmsd(ref_lig_ca, model_lig_ca_aligned)
    if gen_mask is not None:
        gen_mask = np.array(gen_mask, dtype=bool)
        assert len(gen_mask) == len(model_lig_ca_aligned)
        gen_sc_rmsd = compute_rmsd(ref_lig_ca[gen_mask], model_lig_ca_aligned[gen_mask])
    else: gen_sc_rmsd = None
    
    return sc_rmsd, gen_sc_rmsd


def align_sequences(sequence_A, sequence_B, **kwargs):
    """
    Performs a global pairwise alignment between two sequences
    using the BLOSUM62 matrix and the Needleman-Wunsch algorithm
    as implemented in Biopython. Returns the alignment, the sequence
    identity and the residue mapping between both original sequences.

    The choices of gap_open and gap_extend are domain conventions which
    relate to the usage of BLOSUM62
    """

    matrix = kwargs.get('matrix', substitution_matrices.load("BLOSUM62"))
    gap_open = kwargs.get('gap_open', -10.0)
    gap_extend = kwargs.get('gap_extend', -0.5)

    alns = pairwise2.align.globalds(sequence_A, sequence_B,
                                    matrix, gap_open, gap_extend,
                                    penalize_end_gaps=(False, False) )
    
    best_aln = alns[0]
    aligned_A, aligned_B, score, begin, end = best_aln

    return aligned_A, aligned_B


def extract_seq_from_biotite_atom_array(atom_array):
    seq, indices = [], []
    for i, residue in enumerate(residue_iter(atom_array)):
        if residue[0].hetero: continue
        resname = residue[0].res_name
        aa = protein_letters_3to1.get(resname.capitalize(), "X")
        if aa == 'X': continue
        seq.append(aa)
        indices.append(i)
    return "".join(seq), indices


def get_template(seq, c, cif_path, out_path):
    '''
    Args:
        seq: sequence to predict
        c: chain id in the cif file
        cif_path: the template file
        out_path: output path (add marker to the cif file)
    '''
    # load the original cif file
    file = CIFFile.read(cif_path)
    struct = get_structure(file, include_bonds=True, extra_fields=['atom_id', 'b_factor', 'occupancy'])[0]
    struct = struct[struct.res_name != 'HOH']   # get rid of water
    
    file = CIFFile()
    chain_struct = struct[struct.chain_id == c].copy()
    unique_res_ids = np.unique(chain_struct.res_id)
    res_id_map = {old: new for new, old in enumerate(unique_res_ids, start=1)}
    chain_struct.res_id = np.array([res_id_map[r] for r in chain_struct.res_id])
    set_structure(file, chain_struct, include_bonds=True)
    file.write(out_path)
    struct_seq, struct_indices = extract_seq_from_biotite_atom_array(chain_struct)

    aligned_seq, aligned_struct_seq = align_sequences(seq, struct_seq)
    query_idxs, template_idxs = [], []
    k, j = 0, 0
    for i, (a, b) in enumerate(zip(aligned_seq, aligned_struct_seq)):
        if (a != '-') and (b != '-'):
            query_idxs.append(k)
            template_idxs.append(struct_indices[j])
        if a != '-': k += 1
        if b != '-': j += 1
    # add release date
    with open(out_path, 'a+') as fout: fout.write('_pdbx_audit_revision_history.revision_date 2012-12-19\n#')
    return (
        out_path,
        query_idxs,
        template_idxs
    )


def _get_chain_str(cplx, chain):
    cplx = deepcopy(cplx)
    cplx = remove_mols(cplx, [mol.id for mol in cplx if mol.id != chain])
    # renumber to remove insertion codes
    for i, block in enumerate(cplx[chain]):
        block.id = (i + 1, '')

    with tempfile.NamedTemporaryFile(suffix=".cif") as tmp:
        complex_to_mmcif(cplx, tmp.name)
        cif_str = tmp.read().decode()
        # don't know why, but some cofolding models require the template to have a date mark
        cif_str += '_pdbx_audit_revision_history.revision_date 2012-12-19\n#'
    return cif_str