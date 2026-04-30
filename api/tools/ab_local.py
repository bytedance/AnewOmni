# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import time
import yaml
import json
import argparse
from typing import List

from utils.logger import print_log

from .ab_design import Candidate, DesignScheduler, get_binding_site


def load_states(root_dir, ids):
    candidates = []
    if not os.path.exists(root_dir): return None
    for _id in ids:
        d = os.path.join(root_dir, str(_id))
        data_path = os.path.join(d, 'full_data.json')
        if not os.path.exists(data_path):
            print_log(f'Required starting candidate is corrupted: {_id}')
            continue
        candidates.append(Candidate.from_dict(json.load(open(data_path, 'r'))))
        print_log(f'Candidate {_id} added.')
    
    return candidates


def parse():
    parser = argparse.ArgumentParser(description='Local optimization of given IDs')
    parser.add_argument('--config', type=str, required=True, help='Path to the yaml config')
    parser.add_argument('--confidence_ckpt', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--af3_work_dir', type=str, required=True, help='Directory for launching and outputting AF3 tasks')
    parser.add_argument('--root_dir', type=str, required=True, help='Directory to seeds')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='GPUs to use')
    return parser.parse_args()


def main(args):
    # preparation
    # load the configuration
    config = yaml.safe_load(open(args.config, 'r'))
    need_renumber = config.get('need_renumber', True)
    print_log(f'Enabled renumbering AF3-predicted binders: {need_renumber}')
    # alphafold 3 working directory
    proj_dir = os.path.join(args.af3_work_dir, config['name'])
    af3_n_seeds = config.get('af3_n_seeds', 1)
    print_log(f'Alphafold 3 working directory: {proj_dir}, number of seeds for each prediction: {af3_n_seeds}')
    os.makedirs(proj_dir, exist_ok=True)
    low_pass_rate_th = config.get('low_pass_rate_th', 0.2)
    print_log(f'Setting low pass rate threshold as {low_pass_rate_th}')
    enable_af3_consistency = config.get('enable_af3_consistency', True)
    print_log(f'AF3 consistency enabled: {enable_af3_consistency}')

    # get reference to define binding residues
    ref_bs_residues = get_binding_site(config['reference']['path'], config['reference']['tgt_chains'], config['reference']['lig_chains'])

    # load candidates
    candidates: List[Candidate] = load_states(args.root_dir, config['seed_ids'])

    # design schedular
    design_scheduler = DesignScheduler(
        args.confidence_ckpt, args.root_dir, config['template_options'],
        batch_size=config.get('batch_size', 16), n_samples=config.get('batch_n_samples', 100)
    )

    subdir_name = config['name'] + '_local_sampling'
    for c in candidates: design_scheduler.launch(c, subdir_name=subdir_name)    # results in subdir_name under /path/to/candidate/
    af3_tasks, id2save_dir = [], {}    

    while (len(candidates) > 0) or (len(af3_tasks) > 0):
        
        # check AF3 results
        for candidate in af3_tasks:
            if candidate.af3_status is not None: continue  # already done
            candidate.check_af3_finished(proj_dir)
            if candidate.af3_status == 2:
                print_log(f'AF3 prediction failed for {candidate.id}. Please check its logs.')
            elif candidate.af3_status == 1:
                print_log(f'{candidate.id} finished')
                candidate.update_metrics(ref_bs_residues, config.get('interaction_specify_cdrs', None))       # update metrics
                candidate.set_msa_from_af3_results(proj_dir)    # update msa
                candidate.save_results(id2save_dir[candidate.id], need_renumber)
        af3_tasks = [c for c in af3_tasks if c.af3_status is None]

        finished, failed, killed = design_scheduler.check_finished_and_launch_wl(low_pass_rate_th=low_pass_rate_th)
        if len(killed) != 0:
            killed_ids = { c.id for c in killed }
            candidates = [ c for c in candidates if c.id not in killed_ids ]
            # further remove it from global candidate list
            print_log(f'Detected OOD complexes with low pass rates: {killed_ids}. Removed from the candidate list.')
        if len(failed) != 0:
            failed_ids = { c.id for c in failed }
            candidates = [ c for c in candidates if c.id not in failed_ids ]
        if len(finished) == 0:
            print_log(f'Waiting... {len(candidates)} design tasks in progressing, {len(af3_tasks)} AF3 tasks in progressing.')
            time.sleep(60)
            continue

        if enable_af3_consistency:
            # launch new AF3 tasks
            for candidate in finished:
                new_candidates: List[Candidate] = candidate.get_new_candidates(candidate.candidate_dir, n=None, use_global_id=False)
                for c in new_candidates:
                    c.launch_af3(proj_dir, af3_n_seeds)
                    id2save_dir[c.id] = candidate.candidate_dir
                af3_tasks.extend(new_candidates)
        
        finished_ids = { c.id for c in finished }
        candidates = [ c for c in candidates if c.id not in finished_ids ]
        



if __name__ == '__main__':
    main(parse())
