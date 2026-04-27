# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import os
from copy import deepcopy

import yaml
import torch
from rdkit import Chem

import models
from utils.config_utils import overwrite_values
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from data.bioparse import Complex, Block, Atom, VOCAB, BondType
from data.base import Summary
from data.file_loader import MolType
from data import create_dataloader, create_dataset
from utils.logger import print_log
from utils.random_seed import setup_seed
from models.LDM.data_utils import Recorder, OverwriteTask, _get_item_multitype
from models.modules.adapter.model import ConditionConfig


def get_best_ckpt(ckpt_dir):
    with open(os.path.join(ckpt_dir, 'checkpoint', 'topk_map.txt'), 'r') as f:
        ls = f.readlines()
    ckpts = []
    for l in ls:
        k,v = l.strip().split(':')
        k = float(k)
        v = v.split('/')[-1]
        ckpts.append((k,v))

    # ckpts = sorted(ckpts, key=lambda x:x[0])
    best_ckpt = ckpts[0][1]
    return os.path.join(ckpt_dir, 'checkpoint', best_ckpt)


def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def clamp_coord(coord):
    # some models (e.g. diffab) will output very large coordinates (absolute value >1000) which will corrupt the pdb file
    new_coord = []
    for val in coord:
        if abs(val) >= 1000:
            val = 0
        new_coord.append(val)
    return new_coord


def generate_wrapper(model, sample_opt={}, struct_pred=False, w=2.0, conf_model=None):
    if isinstance(model, models.CondIterAutoEncoderEdge):
        def wrapper(batch):
            batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = model.generate(**batch, confidence_model=conf_model)
            return batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds
    elif isinstance(model, models.LDMMolDesignClean):
        def wrapper(batch):
            if struct_pred:
                condition_config = ConditionConfig(
                    mask_2d=batch['generate_mask'].clone(),
                    mask_3d=torch.zeros_like(batch['generate_mask']),
                    mask_incomplete_2d=torch.zeros_like(batch['generate_mask']),
                    w=w
                )
            else: condition_config = None
            batch['condition_config'] = condition_config
            res_tuple = model.sample(sample_opt=sample_opt, confidence_model=conf_model, **batch)
            if len(res_tuple) == 6:
                batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = res_tuple
            else:
                batch_S, batch_X, batch_A, batch_ll, batch_bonds = res_tuple
                batch_intra_bonds = []
                for s in batch_S:
                    batch_intra_bonds.append([None for _ in s])
            return batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds
    else:
        raise NotImplementedError(f'Wrapper for {type(model)} not implemented')
    return wrapper


def overwrite(
        cplx: Complex, summary: Summary, S: list, X: list, A: list, ll: list, bonds: tuple, intra_bonds: list, out_path: str,
        check_validity: bool=True, expect_atom_num: float=None, struct_pred: bool=False
    ):
    '''
        Args:
            bonds: [row, col, prob, type], row and col are atom index, prob has confidence and distance
    '''

    # if isinstance(ll, tuple): ll, confidence = ll
    # else: confidence = None

    task = OverwriteTask(
        cplx = cplx,
        select_indexes = summary.select_indexes,
        generate_mask = summary.generate_mask,
        target_chain_ids = summary.target_chain_ids,
        ligand_chain_ids = summary.ligand_chain_ids,
        S = S,
        X = X,
        A = A,
        ll = ll['fm_ll'],
        inter_bonds = bonds,
        intra_bonds = intra_bonds,
        confidence = ll.get('confidence', None),
        likelihood = ll.get('likelihood', None),
        out_path = out_path,
        save_cif = False    # TODO: some bugs with original bonds after overwriting
    )

    cplx, gen_mol, overwrite_indexes = task.get_overwritten_results(
        check_validity = check_validity,
        expect_atom_num = expect_atom_num,
        struct_pred=struct_pred
    )

    if cplx is None or gen_mol is None:
        return None

    return {
        'id': summary.id,
        'pmetric': task.get_total_likelihood(),
        'smiles': Chem.MolToSmiles(gen_mol),
        'gen_seq': task.get_generated_seq(),
        'target_chains_ids': summary.target_chain_ids,
        'ligand_chains_ids': summary.ligand_chain_ids,
        'gen_block_idx': overwrite_indexes, # TODO: in pdb, (1, '0') will be saved as (1, 'A')
        'gen_pdb': os.path.abspath(out_path),
        'ref_pdb': os.path.abspath(summary.ref_pdb),
    }


def format_id(summary: Summary):
    # format saving id for cross dock
    # e.g. BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pdb|BSD_ASPTE_1_130_0/2z3h_A_rec_1wn6_bst_lig_tt_docked_3.sdf
    if '|' in summary.id:
        summary.id = summary.id.split('|')[0].strip('.pdb')


def main(args, opt_args):
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)
    mode = config.get('sample_opt', {}).get('mode', 'codesign')
    struct_only = mode == 'fixseq'
    
    assert not ((args.ckpt is None) and (args.confidence_ckpt is None)), f'At least one of ckpt and confidence_ckpt should be provided'
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    # load confidence model
    if args.confidence_ckpt is not None:
        conf_ckpt = args.confidence_ckpt
        if not conf_ckpt.endswith('.ckpt'): conf_ckpt = get_best_ckpt(conf_ckpt)
        print(f'Using confidence model from {conf_ckpt}')
        conf_model = torch.load(conf_ckpt, map_location='cpu', weights_only=False)
        conf_model.to(device)
        conf_model.eval()
        print(f'Using the base model saved with the confidence model')
        model = conf_model.base_model
    else:
        conf_model = None
        # load model
        b_ckpt = args.ckpt if args.ckpt.endswith('.ckpt') else get_best_ckpt(args.ckpt)
        ckpt_dir = os.path.split(os.path.split(b_ckpt)[0])[0]
        print(f'Using checkpoint {b_ckpt}')
        model = torch.load(b_ckpt, map_location='cpu', weights_only=False)
        model.to(device)
        model.eval()

    # load data
    _, _, test_set = create_dataset(config['dataset'])
    
    # save path
    if args.save_dir is None:
        if args.struct_pred: save_dir = os.path.join(ckpt_dir, 'results_struct_pred')
        else: save_dir = os.path.join(ckpt_dir, 'results')
    else:
        save_dir = args.save_dir
    ref_save_dir = os.path.join(save_dir, 'references')
    cand_save_dir = os.path.join(save_dir, 'candidates')
    tmp_cand_save_dir = os.path.join(save_dir, 'tmp_candidates')
    for directory in [ref_save_dir, cand_save_dir, tmp_cand_save_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    

    # fout = open(os.path.join(save_dir, 'results.jsonl'), 'w')

    n_samples = config.get('n_samples', 1)
    n_cycles = config.get('n_cycles', 0)

    recorder = Recorder(test_set, n_samples, save_dir, args.max_retry)
    
    batch_size = config['dataloader']['batch_size']

    while not recorder.is_finished():
        batch_list = recorder.get_next_batch_list(batch_size)
        batch = [test_set[i] for i, _ in batch_list]
        batch = test_set.collate_fn(batch)
        batch = to_device(batch, device)
        
        with torch.no_grad():
            batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = generate_wrapper(
                model, deepcopy(config.get('sample_opt', {})), args.struct_pred, args.w,
                conf_model=conf_model if n_cycles==0 else None
            )(batch)
            likelihoods = [ll.get('likelihood', None) for ll in batch_ll]   # saved here as it is derived from the LDM

        vae_batch_list = []
        for S, X, A, ll, bonds, intra_bonds, (item_idx, n) in zip(batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds, batch_list):
            cplx: Complex = deepcopy(test_set.get_raw_data(item_idx))
            summary: Summary = deepcopy(test_set.get_summary(item_idx))
            # revise id
            format_id(summary)
            summary.ref_pdb = os.path.join(ref_save_dir, summary.ref_pdb)
            if n == 0: # the first round
                os.makedirs(os.path.dirname(summary.ref_pdb), exist_ok=True)
                complex_to_pdb(cplx, summary.ref_pdb, summary.target_chain_ids + summary.ligand_chain_ids)
                os.makedirs(os.path.join(cand_save_dir, summary.id), exist_ok=True)
                os.makedirs(os.path.join(tmp_cand_save_dir, summary.id), exist_ok=True)
                complex_to_pdb(cplx, os.path.join(tmp_cand_save_dir, summary.id, 'pocket.pdb'), summary.target_chain_ids)
            if n_cycles == 0: save_path = os.path.join(cand_save_dir, summary.id, f'{n}.pdb')
            else: save_path = os.path.join(tmp_cand_save_dir, summary.id, f'{n}.pdb')
            log = overwrite(cplx, summary, S, X, A, ll, bonds, intra_bonds, save_path, check_validity=False, struct_pred=args.struct_pred)
            if n_cycles == 0: recorder.check_and_save(log, item_idx, n, struct_only)
            else:
                data, cplx, pocket_block_ids, lig_block_ids = _get_item_multitype(
                        os.path.join(tmp_cand_save_dir, summary.id, 'pocket.pdb'),
                        save_path.rstrip('.pdb') + '.sdf',
                        save_path,
                        summary.target_chain_ids,
                        summary.ligand_chain_ids,
                        cdr_type=getattr(test_set, 'cdr_type', [None])[0],
                        fr_len=getattr(test_set, 'fr_len', 3), # WARN: depends on the processing of the data
                        mol_type=test_set.mol_type
                    )
                vae_batch_list.append((data, cplx, pocket_block_ids + lig_block_ids))
        for cyc_i in range(n_cycles):
            print_log(f'Cycle: {cyc_i}', level='DEBUG')
            final_cycle = cyc_i == n_cycles - 1
            batch = test_set.collate_fn([tup[0] for tup in vae_batch_list])
            batch = to_device(batch, device)
            ori_vae_batch_list, vae_batch_list = vae_batch_list, []
            model_autoencoder = getattr(model, 'autoencoder', model)
            with torch.no_grad():
                if final_cycle: batch['topo_generate_mask'] = torch.zeros_like(batch['generate_mask'])
                batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = generate_wrapper(
                    model_autoencoder, deepcopy(config.get('sample_opt', {})), conf_model=conf_model)(batch)
            for S, X, A, ll, bonds, intra_bonds, likelihood, (item_idx, n), (data, cplx, sel_indexes) in zip(batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds, likelihoods, batch_list, ori_vae_batch_list):
                ll['likelihood'] = likelihood
                # cplx: Complex = deepcopy(test_set.get_raw_data(item_idx))
                summary: Summary = deepcopy(test_set.get_summary(item_idx))
                # revise id
                format_id(summary)
                # change selected indexes and generate mask since the pocket is re-located
                summary.select_indexes = sel_indexes
                summary.generate_mask = data['generate_mask']
                if final_cycle: save_path = os.path.join(cand_save_dir, summary.id, f'{n}.pdb')
                else: save_path = os.path.join(tmp_cand_save_dir, summary.id, f'{n}_cyc{cyc_i}.pdb')
                # get expect atom number
                if hasattr(test_set, 'get_expected_atom_num'):
                    expect_atom_num = test_set.get_expected_atom_num(item_idx)
                else: expect_atom_num = None
                log = overwrite(
                    cplx, summary, S, X, A, ll, bonds, intra_bonds, save_path, 
                    check_validity=final_cycle and test_set.mol_type == MolType.MOLECULE,
                    expect_atom_num=None if args.struct_pred else expect_atom_num,
                    struct_pred=args.struct_pred
                )
                if final_cycle: recorder.check_and_save(log, item_idx, n, struct_only)
                else:
                    data, cplx, pocket_block_ids, lig_block_ids = _get_item_multitype(
                            os.path.join(tmp_cand_save_dir, summary.id, 'pocket.pdb'),
                            save_path.rstrip('.pdb') + '.sdf',
                            save_path,
                            summary.target_chain_ids,
                            summary.ligand_chain_ids,
                            cdr_type=getattr(test_set, 'cdr_type', [None])[0],
                            fr_len=getattr(test_set, 'fr_len', 3), # WARN: depends on the processing of the data
                            mol_type=test_set.mol_type
                        )
                    vae_batch_list.append((data, cplx, pocket_block_ids + lig_block_ids))
                    # vae_batch_list.append(
                    #     _get_item_multitype(
                    #         os.path.join(tmp_cand_save_dir, summary.id, 'pocket.pdb'),
                    #         save_path.rstrip('.pdb') + '.sdf',
                    #         save_path,
                    #         summary.target_chain_ids,
                    #         summary.ligand_chain_ids,
                    #         cdr_type=getattr(test_set, 'cdr_type', [None])[0],
                    #         fr_len=getattr(test_set, 'fr_len', 3), # WARN: depends on the processing of the data
                    #         mol_type=test_set.mol_type
                    #     )
                    # )

        print_log(f'Failed rate: {recorder.num_failed / recorder.num_generated}', level='DEBUG')
    return    


def parse():
    parser = argparse.ArgumentParser(description='Generate peptides given epitopes')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--confidence_ckpt', type=str, default=None, help='Path to the confidence model')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save generated peptides')
    parser.add_argument('--struct_pred', action='store_true', help='Structure prediction mode')
    parser.add_argument('--w', type=float, default=1.0, help='Structure prediction topo controlling strength')
    parser.add_argument('--max_retry', type=int, default=None, help='Maximum number of retries')

    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    parser.add_argument('--n_cpu', type=int, default=4, help='Number of CPU to use (for parallelly saving the generated results)')
    return parser.parse_known_args()


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(12)
    main(args, opt_args)