# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import shutil

import numpy as np

from data.bioparse.interface import compute_interacting_pairs
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
from data.bioparse.writer.complex_to_mmcif import complex_to_mmcif
from data.bioparse.numbering import assign_pos_ids
from evaluation.rmsd import kabsch, compute_rmsd


def get_binding_site(path, tgt_chains, lig_chains):
    tgt_chains, lig_chains = list(tgt_chains), list(lig_chains)
    if path.endswith('.cif'): cplx = mmcif_to_complex(path, tgt_chains + lig_chains)
    elif path.endswith('.pdb'): cplx = pdb_to_complex(path, tgt_chains + lig_chains)
    pairs = compute_interacting_pairs(cplx, list(tgt_chains), list(lig_chains))
    bs_residues = set([tup[0] for tup in pairs])
    return bs_residues


def get_contact_is_cdr_ratio(path, tgt_chains, hchain, hmark, lchain, lmark, specify_cdrs=None):
    
    hcdr_allowed, lcdr_allowed = [], []
    if specify_cdrs is not None:
        for cdr in specify_cdrs:
            if cdr == 'HCDR1': hcdr_allowed.append('1')
            elif cdr == 'HCDR2': hcdr_allowed.append('2')
            elif cdr == 'HCDR3': hcdr_allowed.append('3')
            elif cdr == 'LCDR1': lcdr_allowed.append('1')
            elif cdr == 'LCDR2': lcdr_allowed.append('2')
            elif cdr == 'LCDR3': lcdr_allowed.append('3')
    else:
        hcdr_allowed = ['1', '2', '3']
        lcdr_allowed = ['1', '2', '3']

    tgt_chains = list(tgt_chains)
    lig_chains = []
    if hchain is not None: lig_chains.append(hchain)
    if lchain is not None: lig_chains.append(lchain)
    if path.endswith('.cif'): cplx = mmcif_to_complex(path, tgt_chains + lig_chains)
    elif path.endswith('.pdb'): cplx = pdb_to_complex(path, tgt_chains + lig_chains)
    hchain_len = 0 if hchain is None else len(hmark)
    pairs = compute_interacting_pairs(cplx, list(tgt_chains), list(lig_chains), efficient=True)
    record, cdr_res_cnt = {}, 0
    for _, resid, _, j, _ in pairs:
        if resid in record: continue
        else: record[resid] = 1
        if (resid[0] == hchain) and hmark[j] in hcdr_allowed: cdr_res_cnt += 1
        elif (resid[0] == lchain) and lmark[j - hchain_len] in lcdr_allowed: cdr_res_cnt += 1
    if len(record) == 0: return 0
    return cdr_res_cnt / len(record)


def get_scRMSD(ref_path, model_path, tgt_chains, lig_chains, hmark, lmark):
    ref_cplx = mmcif_to_complex(ref_path, selected_chains=tgt_chains + lig_chains)
    model_cplx = mmcif_to_complex(model_path, selected_chains=tgt_chains + lig_chains)

    # get CA coordinates
    def get_ca_coords(mol):
        coords = []
        for block in mol:
            for atom in block:
                if atom.name == 'CA': coords.append(atom.get_coord())
        return coords

    ref_tgt_ca = []
    for c in tgt_chains: ref_tgt_ca.extend(get_ca_coords(ref_cplx[c]))
    model_tgt_ca = []
    for c in tgt_chains: model_tgt_ca.extend(get_ca_coords(model_cplx[c]))
    ref_tgt_ca, model_tgt_ca = np.array(ref_tgt_ca), np.array(model_tgt_ca)

    # get transformation matrix
    _, rotation, t = kabsch(model_tgt_ca, ref_tgt_ca)

    # get CA coordinates of the ligand
    ref_lig_ca, model_lig_ca = [], []
    for c in lig_chains:
        ref_lig_ca.extend(get_ca_coords(ref_cplx[c]))
        model_lig_ca.extend(get_ca_coords(model_cplx[c]))
    ref_lig_ca, model_lig_ca = np.array(ref_lig_ca), np.array(model_lig_ca)

    # transform
    model_lig_ca_aligned = np.dot(model_lig_ca, rotation) + t

    # get CDR mark
    is_cdr = []
    if hmark is not None:
        for m in hmark: is_cdr.append(int(m != '0'))
    if lmark is not None:
        for m in lmark: is_cdr.append(int(m != '0'))
    assert len(is_cdr) == len(model_lig_ca_aligned)
    is_cdr = np.array(is_cdr)

    # calculate rmsd
    return compute_rmsd(ref_lig_ca, model_lig_ca_aligned), compute_rmsd(ref_lig_ca[is_cdr], model_lig_ca_aligned[is_cdr])


def renumber_ab(in_path, out_path, tgt_chains, hchain, hmark, lchain, lmark):
    tgt_chains = list(tgt_chains)
    lig_chains = []
    if hchain is not None: lig_chains.append(hchain)
    if lchain is not None: lig_chains.append(lchain)
    cplx = mmcif_to_complex(in_path, selected_chains=tgt_chains + lig_chains)

    if hchain is not None:
        pos_ids = assign_pos_ids(hmark, 'H')
        for block, _id in zip(cplx[hchain], pos_ids):
            block.id = _id
    if lchain is not None:
        pos_ids = assign_pos_ids(lmark, 'L')
        for block, _id in zip(cplx[lchain], pos_ids):
            block.id = _id

    complex_to_mmcif(cplx, out_path, tgt_chains + lig_chains)


def cleanup_cofold_server(proj_dir):
    """
    Delete all cofold task artifacts while keeping the root directory layout.

    This keeps the root directory layout stable for the running scan loop while
    removing everything that can accumulate disk usage:
      - all task inputs and sidecar files/directories directly under `proj_dir`
      - everything under `proj_dir/logs/`
      - everything under `proj_dir/output/`
    """
    if not os.path.isdir(proj_dir):
        return

    # Remove everything in proj_dir except the logs/output roots.
    for fn in os.listdir(proj_dir):
        if fn in ('logs', 'output'):
            continue
        path = os.path.join(proj_dir, fn)
        try:
            if os.path.isdir(path): shutil.rmtree(path)
            else: os.remove(path)
        except Exception:
            continue

    # Remove everything under logs/ and output/ (but keep the directories).
    for sub in ('logs', 'output'):
        root = os.path.join(proj_dir, sub)
        os.makedirs(root, exist_ok=True)
        for child in os.listdir(root):
            p = os.path.join(root, child)
            try:
                if os.path.isdir(p): shutil.rmtree(p)
                else: os.remove(p)
            except Exception: pass


def cleanup_candidates(save_dir, del_ids):
    for _id in del_ids:
        d = os.path.join(save_dir, _id)
        if os.path.exists(d): shutil.rmtree(d)
