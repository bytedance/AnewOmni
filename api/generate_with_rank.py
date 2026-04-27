# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import yaml
import json
import shutil
import argparse
from time import time, sleep
from copy import deepcopy
from typing import List

# enable TF32
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '1'
import torch
import numpy as np

from utils.config_utils import overwrite_values
from utils.logger import print_log
from utils.random_seed import setup_seed
import utils.register as R

from .pdb_dataset import PDBDataset
from .helpers.gen_utils import load_model, generate_for_one_template, generate_multiple_cdrs
from .filters.runner import AsyncFilterRunner
from .templates import AntibodyMultipleCDR


def parse():
    default_ckpt = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'model.ckpt')
    parser = argparse.ArgumentParser(description='Generate peptides given epitopes')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration')
    parser.add_argument('--ckpt', type=str, default=default_ckpt, help='Path to the confidence model')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save generated peptides')

    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    parser.add_argument('--n_cpus', type=int, default=4, help='Number of CPU to use (for parallelly saving the generated results)')

    # criterion
    parser.add_argument('--rank_criterion', type=str, default='confidence', choices=['confidence', 'likelihood', 'none', 'normalized_likelihood'],
                        help='Criterion of ranking: confidence | likelihood | normalized_likelihood | none, default confidence')

    # for filters
    parser.add_argument('--filter_num_cpus', type=int, default=4, help='Default using four cpus')
    parser.add_argument('--filter_num_gpus', type=int, default=None, help='Default using all available gpus')
    
    parser.add_argument('--seed', type=int, default=None, help='seed')

    return parser.parse_known_args()


def _load_items(input_dir) -> List[dict]:
    result_path = os.path.join(input_dir, 'results.jsonl')
    with open(result_path, 'r') as fin: lines = fin.readlines()

    items = [json.loads(line) for line in lines]
    return items


def _binary_search(values, target, increasing=True): # assuming values are sorted from small ones to large ones if given increasing=True
    if not increasing: values = values[::-1]    # reverse
    l, r = 0, len(values) - 1
    mid = (l + r) // 2
    while (r - l) >= 0:
        if values[mid] < target: l = mid + 1
        elif values[mid] > target: r = mid - 1
        else: return mid
        mid = (l + r) // 2
    if not increasing: l = len(values) - l
    return l


class TopKMaintainer:

    def __init__(self, topk: int, patience: int, out_dir: str, rank_criterion: str=None):
        self.topk = topk
        self.patience = patience
        self.out_dir = out_dir
        self.rank_criterion = 'pmetric' if rank_criterion is None else rank_criterion # use 'none' to stop ranking

        self.candidates = {} # <id>: list of (path, confidence)
        self.finish_flag = {}
        self.stable_steps = {}
        self.n_tried = {}
        self.start_time = time()

    def finished(self):
        if len(self.finish_flag) == 0: return False
        finished = True
        for _id in self.finish_flag: finished = finished and self.finish_flag[_id]
        return finished
    
    def print_final_confidences(self):
        print_log('=' * 20 + f'Final Confidences (Top {self.topk})' + '=' * 20)
        for _id in self.candidates:
            cfds = [tup[1] for tup in self.candidates[_id]]
            print_log(f'{_id}: {cfds}')

    def update_topk(self, input_dir, items):
        print()
        print_log('=' * 26 + ' BEGIN UPDATION ' + '=' * 26)
        print_log(f'Output directory: {self.out_dir}')

        updating_flag = False
        for item in items:
            _id = item['id']
            os.makedirs(os.path.join(self.out_dir, _id), exist_ok=True)
            if self.finish_flag.get(_id, False): continue # this target is already finished
            inserted = self._insert(item, input_dir)
            updating_flag = updating_flag or inserted
            if _id not in self.n_tried: self.n_tried[_id] = 0
            self.n_tried[_id] += 1

        if not updating_flag:
            print_log('Nothing changed')

        self._write_summary()
        self._print_statistics()

        print_log('=' * 27 + ' END UPDATION ' + '=' * 27)
        print()

    def _insert(self, item, root_dir):
        _id = item['id']
        if _id not in self.candidates: self.candidates[_id] = []
        if self.rank_criterion == 'none': confidence = len(self.candidates[_id]) # do not use ranking
        else: confidence = item[self.rank_criterion]
        if np.isnan(confidence) or np.isinf(confidence):
            print_log('nan/inf encountered!', level='WARN')
            return False
        
        path = os.path.join(root_dir, item['id'], str(item['n']))
        increasing = {
            'likelihood': True, # TODO: the direction
            'confidence': True,
            'none': True,
            'normalized_likelihood': False
        }
        insert_pos = _binary_search([tup[1] for tup in self.candidates[_id]], confidence, increasing=increasing[self.rank_criterion])
        if insert_pos < self.topk:
            print_log(f'Inserting item with {self.rank_criterion} {round(confidence, 3)} into rank {insert_pos} for {_id}')
            self._replace_files(_id, insert_pos, path)
            self.candidates[_id].insert(insert_pos, (item, confidence))
            self.candidates[_id] = self.candidates[_id][:self.topk] # discard the last one
            self.stable_steps[_id] = 0
        else:
            self.stable_steps[_id] += 1 # initialized in the if branch
            if self.stable_steps[_id] >= self.patience: self.finish_flag[_id] = True # already stable

        return insert_pos < self.topk   # whether inserted
    
    def _replace_files(self, _id, insert_pos, path):
        suffix = ['_confidence.json', '.cif', '.pdb', '.sdf']
        for i in range(self.topk - insert_pos - 1):
            j = self.topk - i - 2
            if j >= len(self.candidates[_id]): continue 
            for s in suffix:
                src_path = os.path.join(self.out_dir, _id, str(j) + s)
                dst_path = os.path.join(self.out_dir, _id, str(j + 1) + s)
                if os.path.exists(src_path): os.system(f'mv {src_path} {dst_path}')  # overwrite
        # copy new path
        for s in suffix:
            dst_path = os.path.join(self.out_dir, _id, str(insert_pos) + s)
            if os.path.exists(f'{path}{s}'): os.system(f'cp {path}{s} {dst_path}')

    def _write_summary(self):
        fout = open(os.path.join(self.out_dir, 'results.jsonl'), 'w')
        for _id in self.candidates:
            for i, (item, cfd) in enumerate(self.candidates[_id]):
                summary = deepcopy(item)
                summary['n'] = i
                fout.write(json.dumps(summary) + '\n')
        fout.close()

    def _print_statistics(self):
        print_log('=' * 30 + ' STATS ' + '=' * 30)
        for _id in self.n_tried:
            print_log(_id)
            print_log(f'\tchecked {self.n_tried[_id]} samples')
            cfds = [tup[1] for tup in self.candidates[_id]]
            _min, _max = round(min(cfds), 3), round(max(cfds), 3)
            _mean = round(sum(cfds) / len(cfds), 3)
            _median = round(cfds[len(cfds) // 2], 3)
            print_log(f'\t{self.rank_criterion} for top {self.topk}: min({_min}), mean({_mean}), median({_median}), max({_max})')
            print_log(f'\tnot updated for {self.stable_steps[_id]} generations (patience {self.patience})')
            print_log('')
        print_log(f'Overall {int(time() - self.start_time)}s elapsed')


def main(args, opt_args):
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)

    # load model
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model, conf_model = load_model(None, args.ckpt, device)

    final_n_samples = config['n_samples']
    sample_opt = config.get('sample_opt', {})
    patience = config.get('patience', 0)    # default zero, stop when the desired number of samples is reached
    batch_size = config.get('batch_size', 16)
    # trial_size = config.get('trial_size', min(final_n_samples, 100))
    trial_size = config.get('trial_size', batch_size)   # launch filter every batch
    filter_print_freq = config.get('filter_print_freq', trial_size) # get statistics each trial

    print_log(f'Start generation with ranking, so "n_samples" ({final_n_samples}) indicate the maintained top K candidates ranked by {args.rank_criterion}.')
    print_log(f'The generation will stop when the top K candidates have not changed for {patience} generations.')

    # initialize the filter runner
    filter_runner = AsyncFilterRunner(num_cpus=args.filter_num_cpus, num_gpus=args.filter_num_gpus, print_freq=filter_print_freq)
    # load dataset and dataloader
    for template in config['templates']:
        custom_filter_configs = template.pop('filters', [])
        template_instance = R.construct(template)
        if hasattr(template_instance, '__len__'): template_factory = template_instance  # factory
        else: template_factory = [template_instance]

        for i in range(len(template_factory)):
            template_instance = template_factory[i]
            filter_runner.reset()
            filter_configs = template_instance.default_filter_configs() + custom_filter_configs

            print_log('Start generation for the following template configuration:')
            print()
            print(template)
            print()
            print_log('Filter settings are as follows:')
            print()
            print(filter_configs)
            print()
            if template_instance.resample_mode:
                print_log(f'Resampling mode enabled!')
                print()

            out_dir = os.path.join(args.save_dir, template_instance.name, 'candidates')
            trial_dir = os.path.join(args.save_dir, template_instance.name, 'trials')
            os.makedirs(out_dir, exist_ok=True)
            n_trials = 0

            dataset = PDBDataset(**config['dataset'], template_config=template_instance)

            maintainer = TopKMaintainer(final_n_samples, patience, out_dir, args.rank_criterion)

            try:
                while True:
                    # generation for one trial
                    cur_trial_dir = os.path.join(trial_dir, str(n_trials))
                    if os.path.exists(cur_trial_dir): shutil.rmtree(cur_trial_dir)
                    os.makedirs(cur_trial_dir)
                    print_log(f'Starting generation for {trial_size} samples under {cur_trial_dir}')
                    if isinstance(template_instance, AntibodyMultipleCDR): gen_func = generate_multiple_cdrs
                    else: gen_func = generate_for_one_template
                    cur_sample_opt = deepcopy(sample_opt)
                    cur_sample_opt.update(template_instance.additional_sample_opt)
                    gen_func(
                        model, dataset, trial_size, batch_size, cur_trial_dir, device, cur_sample_opt,
                        n_cycles=1, conf_model=conf_model, max_retry=0, verbose=False,
                        check_validity=False, resample_mode=template_instance.resample_mode) # do not retry anymore
                    print_log(f'Finished generation')

                    items = _load_items(cur_trial_dir)
                    if len(filter_configs) == 0: # no filter required
                        # replace top candidates
                        maintainer.update_topk(cur_trial_dir, items)
                        shutil.rmtree(cur_trial_dir)
                    else:
                        while len(filter_runner) > 0:
                            _has_finished = False
                            for folder in filter_runner.check_finished():
                                finished_items = filter_runner.get_results(folder)
                                maintainer.update_topk(folder, finished_items)
                                shutil.rmtree(folder)
                                _has_finished = True
                            if not _has_finished:
                                sleep(10)
                                print_log(f'Waiting for the previous batch to finish filtering...')
                        if not maintainer.finished():
                            filter_runner.launch(cur_trial_dir, items, filter_configs)
                            print_log(f'Launched {len(items)} tasks for filtering')

                    n_trials += 1
                    if maintainer.finished():
                        maintainer.print_final_confidences()
                        break
                print_log(f'Finished the generations at {out_dir}')
                if len(filter_configs) > 0:
                   filter_runner.print_info()
            except KeyboardInterrupt:
                print_log('Stopping')

            os.system(f'rm -rf {trial_dir}')

    del filter_runner
    print_log('All finished')

if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    if args.seed is not None: seed = args.seed
    else: seed = np.random.randint(0, 4294967295)
    print_log(f'Global seed set to {seed}')
    setup_seed(seed)
    main(args, opt_args)
