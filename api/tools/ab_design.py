# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import sys
import time
import json
import gzip
import shlex
import random
import argparse
import shutil
import threading
import subprocess
from typing import List
from copy import deepcopy

import ray
import yaml
import torch

from api.renumber import renumber_seq
from data.bioparse.numbering import Chothia
from utils.logger import print_log

from .cofold.server import scan_tasks
from .cofold import utils as cofold_utils
from .data_defs import *
from .funcs import *


def parse():
    default_ckpt = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'checkpoints', 'model.ckpt'))
    parser = argparse.ArgumentParser(description='pipeline for antibody design')
    parser.add_argument('--config', type=str, required=True, help='Path to the yaml config')
    parser.add_argument('--ckpt', type=str, default=default_ckpt, help='Path to the confidence model')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--max_num_iterations', type=int, default=60, help='Maximum number of iterations')

    # cofold configurations
    # Default to protenix.
    parser.add_argument('--cofold_model', type=str, default='protenix', choices=['alphafold3', 'boltz2', 'protenix'])
    # Backend-specific configs:
    # - alphafold3: repo_dir/env/db/param are used by alphafold3_predict.sh
    # - boltz2: env should be a conda prefix containing `bin/boltz`; param is used as boltz cache/weights dir
    # - protenix: env should be a conda prefix containing `bin/protenix`; param can be a model name
    parser.add_argument('--cofold_repo_dir', type=str, default='')
    parser.add_argument('--cofold_env', type=str, default='')
    parser.add_argument('--cofold_db', type=str, default='')
    parser.add_argument('--cofold_param', type=str, default='')

    default_gpus = list(range(torch.cuda.device_count()))
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=default_gpus, help='GPUs to use')

    # parameters for searching algorithm
    parser.add_argument('--n_beam', type=int, default=10, help='Number of beams')
    parser.add_argument('--n_loser_up', type=int, default=2, help='Number of randomly sampled losers in the beams')
    
    # maintain all results and stop cleaning up, for debug usage
    parser.add_argument('--disable_cleanup', action='store_true', help='Do not cleanup')
    parser.add_argument('--retain_topk', type=int, default=100, help='Clean up and maintain only topk')

    # required type
    parser.add_argument('--allowed_type', type=str, nargs='+',
                        default=['antibody', 'nanobody'], choices=['antibody', 'nanobody'],
                        help='Allowed types of candidates, used when loading from framework library')
    args = parser.parse_args()
    # add default roots
    if args.cofold_model in ['protenix', 'boltz2']: # change environment
        default_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'cofold', args.cofold_model))
        if args.cofold_env == '': args.cofold_env = os.path.join(default_root, 'env')
        if args.cofold_param == '': args.cofold_param = os.path.join(default_root, 'params')
    return args


class DesignScheduler:
    def __init__(self, ckpt, save_dir, template, cofold_model='protenix', batch_size=16, n_samples=50, verbose=True):
        self.ckpt = ckpt
        self.save_dir = save_dir
        self.template = deepcopy(template)
        self.cofold_model = cofold_model
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.verbose = verbose
        
        # recording
        self.ray_tasks = []
        self.waiting_list = []

    def launch(self, candidate: Candidate, subdir_name='design', cdr_use_subsets=False):
        # candidate from cofold prediction
        save_dir = os.path.join(self.save_dir, candidate.id, subdir_name)
        os.makedirs(save_dir, exist_ok=True)
        # complex path
        path = os.path.join(self.save_dir, candidate.id, 'model.cif')
        # configuration for generation
        template = deepcopy(self.template)
        gen_config = {
            'dataset': {
                'pdb_paths': [os.path.abspath(path)],
                'tgt_chains': [candidate.get_tgt_chains()],
                'lig_chains': [candidate.get_lig_chains()],
            },
            'templates': [template],
            'batch_size': self.batch_size,
            'n_samples': self.n_samples,
            'trial_size': self.batch_size,
            'patience': 0,
            'filter_print_freq': min(20, self.n_samples)
        }
        input_kwargs = {
            'config': gen_config,
            'ckpt': self.ckpt,
            'save_dir': save_dir
        }
        candidate.set_candidate_dir(os.path.join(
            save_dir, template['class'], 'candidates'
        ))
        if self._find_gpu_launch(candidate, input_kwargs): return True
        else: self.waiting_list.append((candidate, input_kwargs))

    def _find_gpu_launch(self, candidate, input_kwargs, low_pass_rate_th=0.05):
        save_dir = input_kwargs['save_dir']
        path = input_kwargs['config']['dataset']['pdb_paths'][0]
        # (id, ray task, start time)
        task = design_worker.remote(
            **input_kwargs,
            candidate_dir=candidate.candidate_dir,
            low_pass_rate_th=low_pass_rate_th
        )
        self.ray_tasks.append((candidate, task, time.time()))
        print_log(f'Launched generation task at {save_dir}, using template from {path}')
        print_log(f'Configuration:')
        print()
        print(input_kwargs['config'])
        print()

        return True

    def check_finished_and_launch_wl(self) -> List[Candidate]:
        finished, failed, killed = [], [], []
        # collect all running ray tasks
        running_tasks = []
        task_info = {}
        for i, (candidate, task, start_time) in enumerate(self.ray_tasks):
            running_tasks.append(task)
            task_info[task] = (i, candidate, start_time)
        
        # check finished tasks
        if running_tasks:
            done_ids, _ = ray.wait(running_tasks, timeout=0)
            for task in done_ids:
                i, candidate, _ = task_info[task]
                try:
                    # Get the result (returns the exit code and elapsed time)
                    exit_code, elapsed_time = ray.get(task)
                    if exit_code == 0:
                        print_log(f'{candidate.id} generation task finished, elapsed {elapsed_time:.2f} seconds')
                        finished.append(candidate)
                    elif exit_code == 2:
                        print_log(f'{candidate.id} with too low passrate, killed, elapsed {elapsed_time:.2f} seconds', level='WARN')
                        killed.append(candidate)
                    else:
                        print_log(f'{candidate.id} exited abnormally with exit code {exit_code}, elapsed {elapsed_time:.2f} seconds', level='ERROR')
                        failed.append(candidate)
                except Exception as e:
                    print_log(f'{candidate.id} exited with exception: {e}', level='ERROR')
                    failed.append(candidate)
                # Remove the task from the list
                self.ray_tasks.pop(i)
        
        # check waiting list
        while len(self.waiting_list) > 0:
            next_candidate, next_input_kwargs = self.waiting_list.pop(0)
            self._find_gpu_launch(next_candidate, next_input_kwargs)
        return finished, failed, killed
    

def check_pass_rate(candidate_dir, low_pass_rate_th, log_path=None):
    if not candidate_dir:
        return False
    if log_path is None:
        log_path = os.path.join(candidate_dir, '..', '..', 'log.txt')
    if not os.path.exists(log_path):
        return False
    with open(log_path, 'r') as fin:
        text = fin.read()
    pattern = r'#checked:\s*(?P<checked>\d+).*?#passed:\s*(?P<passed>\d+)'
    matches = re.findall(pattern, text)
    if len(matches) < 3:
        return False  # pass rate is not stable for the first batches
    checked, passed = matches[-1]
    checked, passed = int(checked), int(passed)
    return passed / checked < low_pass_rate_th


@ray.remote(num_cpus=2, num_gpus=1)
def design_worker(config, ckpt, save_dir, rank_criterion='confidence', candidate_dir=None, low_pass_rate_th=0.05):
    start_time = time.time()
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, 'config.yaml')
    with open(config_path, 'w') as fout:
        yaml.dump(config, fout, default_flow_style=False, indent=2)
    log_path = os.path.join(save_dir, 'log.txt')
    log_err_path = os.path.join(save_dir, 'log_err.txt')
    # Use the current interpreter to ensure torch and repo deps are available
    # in the Ray worker environment.
    cmd = f'{sys.executable} -m api.generate_with_rank --config {config_path} --ckpt {ckpt} --save_dir {save_dir} --rank_criterion {rank_criterion}'
    env = os.environ.copy()
    gpu_ids = [str(i) for i in ray.get_gpu_ids()]
    assert len(gpu_ids) == 1
    gpu = gpu_ids[0]
    print_log(f'Generation (GPU={gpu}) with: {cmd}')
    env['CUDA_VISIBLE_DEVICES'] = gpu
    # Use subprocess.Popen to capture output in real-time
    p = subprocess.Popen(
        shlex.split(cmd),
        stdout=open(log_path, 'w'),
        stderr=open(log_err_path, 'w'),
        env=env,
        text=True
    )
    
    # Monitor the process and check pass rate
    while p.poll() is None:
        if check_pass_rate(candidate_dir, low_pass_rate_th):
            print_log(f'Low pass rate detected, killing process...')
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
            elapsed_time = time.time() - start_time
            return 2, elapsed_time  # Return special exit code for OOD
        time.sleep(30)  # Check every 30 seconds
    
    elapsed_time = time.time() - start_time
    return p.returncode, elapsed_time


def launch_cofold_server(cofold_work_dir, cofold_model, cofold_repo_dir, cofold_env, cofold_db, cofold_param):
    print_log(f'cofold monitoring {cofold_work_dir}, configurations are as follows:')
    print_log(f'\tcofold_repo_dir: {cofold_repo_dir}')
    print_log(f'\tcofold_env: {cofold_env}')
    print_log(f'\tcofold_db: {cofold_db}')
    print_log(f'\tcofold_param: {cofold_param}')
    # check resources
    print_log(f'ray resources: {ray.available_resources()}')
    # Start cofold task scanning in a separate thread
    def scan_cofold_tasks():
        visited = {}
        while True:
            try:
                tasks = scan_tasks(
                    cofold_work_dir,
                    visited,
                    cofold_model,
                    cofold_repo_dir,
                    cofold_env,
                    cofold_db,
                    cofold_param,
                    verbose=False
                )
                if len(tasks) > 0:
                    print_log(f'Scanned {cofold_work_dir} and {len(tasks)} cofold tasks appended...')
            except Exception as e:
                error_msg = f'Error in cofold scan: {e}\n'
                print_log(error_msg, level='ERROR')
            time.sleep(10)
    
    cofold_scan_thread = threading.Thread(target=scan_cofold_tasks, daemon=True)
    cofold_scan_thread.start()



def get_chain_mark(seq, is_hchain):
    res = renumber_seq(seq, 'chothia')
    if res is None:
        print_log(f'Renumber failed for sequence: {seq}', level='WARN')
        return None
    cut_seq, ids, chain_type = res
    if len(cut_seq) != len(seq):
        print_log(f'sequence {seq} ajusted by renumbering, the result is {cut_seq}')
    if is_hchain: mark = Chothia.mark_heavy_seq(ids)
    else: mark = Chothia.mark_light_seq(ids)
    return cut_seq, ids, mark


def _load_tgt_data(data: List[dict], template_dir: str):
    msa_keys = ['unpairedMsa', 'pairedMsa']
    tgt_data = []
    for item in data:
        # load msa specified by file path
        for key in msa_keys:
            if (key in item) and isinstance(item[key], str) and item[key].endswith('.a3m'): item[key] = open(item[key], 'r').read()
            elif key not in item: item[key] = ""    # do not search MSA because it is too slow
        # load templates specified by file path
        if 'template' in item:
            template_config = item.pop('template')
            out_path = os.path.join(template_dir, f'chain_{item["chain_id"]}.cif')
            cif_path, query_idx, template_idx = cofold_utils.get_template(
                item['sequence'], template_config['template_chain_id'], template_config['cif'], out_path
            )
            item['templates'] = [{
                'mmcifPath': cif_path,
                'queryIndices': query_idx,
                'templateIndices': template_idx,
                # preserve the original template chain id for backends that
                # require explicit chain mapping (e.g., boltz `template_id`)
                'templateChainId': template_config['template_chain_id'],
            }]
        else: item['templates'] = []    # do not search templates

        tgt_data.append(ChainData(
            id=item['chain_id'],
            sequence=item['sequence'],
            modifications=item.get('modifications', []),
            unpairedMsa=item.get('unpairedMsa', ''),
            pairedMsa=item.get('pairedMsa', ''),
            templates=item.get('templates', []),
            type=item['type']
        ))
    return tgt_data


def _load_framework_lib_initials(tgt_data, config, template_dir, allowed_type=['antibody', 'nanobody']):
    candidates = []
    default_hchain = config.get('heavy_chain_id', 'H')
    default_lchain = config.get('light_chain_id', 'L')
    path = config['path']
    template_data_dir = config['template']
    # load initial tasks from the framework library
    with open(path, 'r') as fin: frameworks = json.load(fin)
    for _id in frameworks:
        template_path = os.path.join(template_data_dir, _id.split('_')[0] + '_template.cif.gz')
        data = frameworks[_id]
        # decide type
        if 'heavychain' in data and 'lightchain' in data: _type = 'antibody'
        elif 'heavychain' in data: _type = 'nanobody'
        if _type not in allowed_type: continue
        if 'heavychain' in data:
            hseq, ids, _ = renumber_seq(data['heavychain'], 'chothia')
            hmark = Chothia.mark_heavy_seq([i[0] for i in ids])
            # add template
            template_out_path = os.path.join(template_dir, f'init_{_id.split("_")[0]}_{default_hchain}.cif')
            with gzip.open(template_path, 'rb') as fin:
                with open(template_out_path, 'wb') as fout:
                    fout.write(fin.read())  # save to the target path
            cif_path, query_idx, template_idx = cofold_utils.get_template(
                hseq, 'H', template_out_path, template_out_path # in-place overwrite
            )
            hdata = ChainData(
                id=default_hchain,
                sequence=hseq,
                modifications=[],
                unpairedMsa='',
                pairedMsa='',
                templates=[{
                    'mmcifPath': cif_path,
                    'queryIndices': query_idx,
                    'templateIndices': template_idx
                }]
            )
        else: hdata, hmark = None, None
        if 'lightchain' in data:
            lseq, ids, _ = renumber_seq(data['lightchain'], 'chothia')
            lmark = Chothia.mark_light_seq([i[0] for i in ids])
            # add template
            template_out_path = os.path.join(template_dir, f'init_{_id.split("_")[0]}_{default_lchain}.cif')
            with gzip.open(template_path, 'rt') as fin:
                with open(template_out_path, 'w') as fout:
                    fout.write(fin.read())  # save to the target path
            cif_path, query_idx, template_idx = cofold_utils.get_template(
                lseq, 'L', template_out_path, template_out_path
            )           
            ldata = ChainData(
                id=default_lchain,
                sequence=lseq,
                modifications=[],
                unpairedMsa='',
                pairedMsa='',
                templates=[{
                    'mmcifPath': cif_path,
                    'queryIndices': query_idx,
                    'templateIndices': template_idx
                }]
            )
        else: ldata, lmark = None, None

        candidates.append(Candidate(
            id=f'init_{_id}',
            tgt_data=tgt_data,
            hdata=hdata,
            hmark=hmark,
            ldata=ldata,
            lmark=lmark,
            cplx_confidences=CofoldConfidences(),
            lig_only_confidences=CofoldConfidences(),
            lig_temp_confidences=CofoldConfidences(),
            generative_confidences=GenerativeConfidences()
        ))
    return candidates


def _load_manual_binder_initials(tgt_data, config):
    candidates = []

    def _extract_data(chain_data, mark_func):
        if 'mark' in chain_data: seq, mark = chain_data['sequence'], chain_data['mark']
        else:
            seq, ids, _ = renumber_seq(chain_data['seq'], 'chothia')
            mark = mark_func([i[0] for i in ids])
        data = ChainData(
            id=chain_data['chain_id'],
            sequence=seq,
            modifications=[]
        )
        if not chain_data.get('need_msa', True): data.set_null_msa()
        return data, mark

    for _id in config:
        hdata, hmark, ldata, lmark = None, None, None, None
        for chain in config[_id]:
            if chain['type'] == 'heavychain':
                hdata, hmark = _extract_data(chain, Chothia.mark_heavy_seq)
            elif chain['type'] == 'lightchain':
                ldata, lmark = _extract_data(chain, Chothia.mark_light_seq)
        candidates.append(Candidate(
            id=_id,
            tgt_data=tgt_data,
            hdata=hdata,
            hmark=hmark,
            ldata=ldata,
            lmark=lmark,
            cplx_confidences=CofoldConfidences(),
            lig_only_confidences=CofoldConfidences(),
            lig_temp_confidences=CofoldConfidences(),
            generative_confidences=GenerativeConfidences()
        ))

    return candidates


def save_states(candidates, save_dir):
    with open(os.path.join(save_dir,'meta_data.json'), 'w') as fout:
        json.dump(MetaData().to_dict(), fout, indent=2)
    for c in candidates: c.save_full_data(save_dir)

def load_states(save_dir):
    candidates = []
    if not os.path.exists(save_dir): return None
    for _id in os.listdir(save_dir):
        d = os.path.join(save_dir, _id)
        data_path = os.path.join(d, 'full_data.json')
        if not os.path.exists(data_path): continue
        candidates.append(Candidate.from_dict(json.load(open(data_path, 'r'))))
    if len(candidates) == 0: return None
    candidates = sorted(candidates, key=lambda cand: cand.get_val_for_ranking(), reverse=True)  # metrics the larger, the better
    meta_data_path = os.path.join(save_dir, 'meta_data.json')
    if os.path.exists(meta_data_path):
        with open(meta_data_path, 'r') as fin:
            meta_data = json.load(fin)
        MetaData().update_dict(meta_data)
        print_log(f'Loaded meta data: {MetaData().to_dict()}')
    else: print_log(f'Meta data not found in {meta_data_path}', level='WARN')

    return candidates


def main(args):
    global RANKING_WEIGHTS

    # preparation
    # load the configuration
    config = yaml.safe_load(open(args.config, 'r'))
    
    if args.name is not None and config['name'] != args.name:   # overwriting the name of this experiments
        print_log(f'Overwritten original name ({config["name"]}) with {args.name}')
        config['name'] = args.name
    if args.save_dir is None:
        args.save_dir = os.path.join(os.path.dirname(args.config), config['name'], 'results')
        print_log(f'Setting saving directory to {args.save_dir}')
    ranking_weights = config.get('ranking_weights', None)
    if ranking_weights is not None:
        print_log(f'Using specified weights for ranking criterion: {ranking_weights}')
        RANKING_WEIGHTS = ranking_weights
    need_renumber = config.get('need_renumber', True)
    print_log(f'Enabled renumbering cofold-predicted binders: {need_renumber}')
    # alphafold 3 working directory
    cofold_work_dir = os.path.join(args.save_dir, 'cofold')
    if os.path.exists(cofold_work_dir):
        print_log(f'Directory for cofold exists. Removing it: {cofold_work_dir}', level='WARN')
        shutil.rmtree(cofold_work_dir)
    proj_dir = os.path.join(cofold_work_dir, config['name'])
    cofold_n_seeds = config.get('cofold_n_seeds', 1)
    print_log(f'Alphafold 3 working directory: {proj_dir}, number of seeds for each prediction: {cofold_n_seeds}')
    os.makedirs(proj_dir, exist_ok=True)

    # Initialize Ray
    ray.init()
    
    launch_cofold_server(
        cofold_work_dir,
        args.cofold_model,
        args.cofold_repo_dir,
        args.cofold_env,
        args.cofold_db,
        args.cofold_param
    )

    custom_metrics = {}


    # get reference to define binding residues
    assert ('reference' in config) or ('epitope' in config), 'Either a reference complex or the residue ids of the epitope should be provided to define the epitope'
    if 'epitope' in config:
        ref_bs_residues = []
        for res_id in config['epitope']:
            if len(res_id) == 2: ref_bs_residues.append((res_id[0], (res_id[1], '')))
            else: ref_bs_residues.append((res_id[0], (res_id[1], res_id[2])))
        ref_bs_residues = set(ref_bs_residues)
    else:
        ref_bs_residues = get_binding_site(config['reference']['path'], config['reference']['tgt_chains'], config['reference']['lig_chains'])
    print(f'Specified epitope: {ref_bs_residues}')

    # load target protein
    template_dir = os.path.abspath(os.path.join(args.save_dir, 'templates'))
    os.makedirs(template_dir, exist_ok=True)
    tgt_data = _load_tgt_data(config['target'], template_dir)

    # check whether states can be loaded
    candidates = load_states(args.save_dir)
    if candidates is not None:
        for c in candidates: c.cofold_model = args.cofold_model # in case the user decides to use new cofolding models
    if (candidates is not None) and MetaData().finish_init:
        print_log(f'Loaded states from {args.save_dir}, global count: {MetaData().global_count}')
    else:
        if candidates is None:
            candidates: List[Candidate] = []
            print_log(f'States not found. From ground up.')
        else: print_log(f'Loaded {len(candidates)} initialization from {args.save_dir}, keep initializing.')
        loaded_ids = { c.id: True for c in candidates }
        if 'framework_lib' in config:   # loading frameworks from the given library
            candidates.extend([c for c in _load_framework_lib_initials(tgt_data, config['framework_lib'], template_dir, args.allowed_type) if c.id not in loaded_ids])
        if 'binders' in config: # append manually specified binder description
            candidates.extend([c for c in _load_manual_binder_initials(tgt_data, config['binders']) if c.id not in loaded_ids])

        # initial guess of cofold
        for candidate in candidates:
            if candidate.id in loaded_ids: continue
            candidate.cofold_model = args.cofold_model
            candidate.launch_cofold(proj_dir, 1)  # only use one seed for initial guess
        finish_cnt = len(loaded_ids)
        while finish_cnt < len(candidates):
            for candidate in candidates:
                if candidate.cofold_status is not None: continue  # already done
                candidate.check_cofold_finished(proj_dir)
                if candidate.cofold_status == 2:
                    print_log(f'Initial cofold prediction failed for {candidate.id}. Please check its logs.')
                    finish_cnt += 1
                elif candidate.cofold_status == 1:
                    print_log(f'Initial cofold prediction for {candidate.id} finished')
                    candidate.update_metrics(ref_bs_residues, config.get('interaction_specify_cdrs', None), cofold_mode='cplx', custom_metrics=custom_metrics)       # update metrics
                    candidate.set_msa_from_cofold_results(proj_dir)    # update msa
                    candidate.save_results(args.save_dir, need_renumber, cofold_mode='cplx')
                    candidate.save_full_data(args.save_dir)
                    finish_cnt += 1
        MetaData().finish_init = True
        # sort
        candidates = sorted(candidates, key=lambda cand: cand.get_val_for_ranking(), reverse=True)  # metrics the larger, the better
        # save states
        save_states(candidates, args.save_dir)

    # start looping
    design_scheduler = DesignScheduler(
        args.ckpt, args.save_dir, config['template_options'], args.cofold_model,
        batch_size=config.get('batch_size', 8), n_samples=config.get('batch_n_samples', 10)
    )
    iteration = 0
    while iteration < args.max_num_iterations:

        # 1. select top n_beam - n_loser_up and sample n_loser_up from the rest
        top_k = args.n_beam - args.n_loser_up
        no_ood_candidates = [c for c in candidates if not c.ood_flag]
        selected_candidates = no_ood_candidates[:top_k]
        if len(no_ood_candidates) - top_k >= args.n_loser_up:  # have enough remaining candidates
            selected_candidates.extend(random.sample(no_ood_candidates[top_k:], k=args.n_loser_up))
        print_log(f'Use top-{top_k} complexes and sample {args.n_loser_up} from the rest for design:')

        # 2. initiate design tasks
        for i, c in enumerate(selected_candidates):
            print(i, c.get_summary())
        for c in selected_candidates:
            design_scheduler.launch(c)

        # 3. watching the status of the design tasks, if done, launch cofold tasks
        unfinished_new_candidates: List[Candidate] = []
        while (len(selected_candidates) > 0) or (len(unfinished_new_candidates) > 0):
            # check finished cofold tasks
            for candidate in unfinished_new_candidates:
                if candidate.cofold_status is not None: continue  # already done
                candidate.check_cofold_finished(proj_dir)
                if candidate.cofold_status == 2:
                    print_log(f'cofold prediction failed for {candidate.id}. Please check its logs.')
                elif candidate.cofold_status == 1:
                    print_log(f'{candidate.id} finished')
                    candidate.update_metrics(ref_bs_residues, config.get('interaction_specify_cdrs', None), custom_metrics=custom_metrics)       # update metrics
                    candidate.set_msa_from_cofold_results(proj_dir)    # update msa
                    candidate.save_results(args.save_dir, need_renumber)
                    candidate.save_full_data(args.save_dir)
                    with open(os.path.join(args.save_dir,'meta_data.json'), 'w') as fout:
                        json.dump(MetaData().to_dict(), fout, indent=2)
                    candidates.append(candidate)
            unfinished_new_candidates = [c for c in unfinished_new_candidates if c.cofold_status is None]
            if (len(unfinished_new_candidates) > 0) and (len(selected_candidates) == 0):
                print_log(f'Waiting... {len(selected_candidates)} design tasks in progressing, {len(unfinished_new_candidates)} cofold tasks in progressing.')
                time.sleep(60)

            if len(selected_candidates) > 0:
                # get finished generation task
                finished, failed, killed = design_scheduler.check_finished_and_launch_wl()
                if len(killed) != 0:
                    killed_ids = { c.id for c in killed }
                    selected_candidates = [ c for c in selected_candidates if c.id not in killed_ids ]
                    # further remove it from global candidate list
                    print_log(f'Detected OOD complexes with low pass rates: {killed_ids}. Removed from the candidate list.')
                    has_good_sample = False
                    for c in candidates:
                        if c.id in killed_ids: c.ood_flag = True    # mark as OOD sample
                        elif not c.ood_flag: has_good_sample = True
                    assert has_good_sample, f'All candidates are OOD!!!'
                if len(failed) != 0:
                    failed_ids = { c.id for c in failed }
                    selected_candidates = [ c for c in selected_candidates if c.id not in failed_ids ]
                if len(finished) == 0:
                    print_log(f'Waiting... {len(selected_candidates)} design tasks in progressing, {len(unfinished_new_candidates)} cofold tasks in progressing.')
                    time.sleep(60)
                    continue
                # launch new cofold tasks
                for candidate in finished:
                    candidate.update_design_confidence()
                    candidate.save_results(args.save_dir, need_renumber, update_only=True)
                    new_candidates: List[Candidate] = candidate.get_new_candidates(args.save_dir)
                    for c in new_candidates: c.launch_cofold(proj_dir, cofold_n_seeds)
                    unfinished_new_candidates.extend(new_candidates)
                # update waiting candidates
                finished_ids = { c.id for c in finished }
                selected_candidates = [c for c in selected_candidates if c.id not in finished_ids]

        # add iteration count
        iteration += 1
        # sort
        candidates = sorted(candidates, key=lambda cand: cand.get_val_for_ranking(), reverse=True)  # metrics the larger, the better
        # cleanup
        if not args.disable_cleanup:
            cleanup_cofold_server(proj_dir)    # delete cofold server cache
            del_ids = [c.id for c in candidates[args.retain_topk:]]
            print_log(f'Deleting candidates: {del_ids}')
            cleanup_candidates(args.save_dir, del_ids)    # delete low-ranking candidates
            candidates = candidates[:args.retain_topk]
        # save states
        save_states(candidates, args.save_dir)
        # save topk results
        with open(os.path.join(args.save_dir, 'top_results.json'), 'w') as fout:
            json.dump({c.id: c.get_briefs() for c in candidates[:args.retain_topk]}, fout, indent=2)
    
    print_log(f'Finished pipeline. Clean up.')
    ray.shutdown()


if __name__ == '__main__':
    main(parse())
