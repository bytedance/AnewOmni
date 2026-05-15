# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
from typing import List
from time import time
from dataclasses import dataclass

import ray

import utils.register as R
from utils.logger import print_log

from .base import FilterInput, FilterResult


@dataclass
class Task:
    input_dir: str
    item: dict
    filters: list
    filter_results: list = None


@ray.remote
def run_filters(task: Task):
    # filters = []
    inputs = FilterInput(
        path_prefix=os.path.join(task.input_dir, task.item['id'], str(task.item['n'])),
        tgt_chains=task.item['tgt_chains'],
        lig_chains=task.item['lig_chains'],
        smiles=task.item['smiles'],
        seq=task.item['gen_seq'],
        confidence=task.item['confidence'],
        likelihood=task.item['likelihood']
    )
    filter_results = []
    for f in task.filters:
        res = f.run.remote(f, inputs)
        res = ray.get(res)
        filter_results.append(res)
        if res[0] != FilterResult.PASSED: break # no need to run later filters
    task.filter_results = filter_results
    return task


class AsyncFilterRunner:
    def __init__(self, num_cpus=None, num_gpus=None, print_freq=100):
        self.print_freq = print_freq
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

        # recording
        self.futures = {}
        self.future_results = {}
        self.filter_stats = {}
        self.num_checked = 0
        self.num_passed = 0

    def launch(self, input_dir: str, items: List[dict], filter_configs: List[dict]):
        '''
        Start runing filters on the inputs
        '''
        assert input_dir not in self.futures

        self.futures[input_dir] = []
        self.future_results[input_dir] = []
        for item in items:
            self.futures[input_dir].append(run_filters.remote(Task(
                input_dir = input_dir,
                item = item,
                filters = [R.construct(c) for c in filter_configs]
            )))
    
    def check_finished(self):
        '''
        Return the list of folders whose tasks have been finished
        '''
        finished_folders = []
        for folder in self.futures:
            done, pending = ray.wait(self.futures[folder], num_returns=len(self.futures[folder]), timeout=1.0)
            self.futures[folder] = pending
            for done_id in done:
                self.future_results[folder].append(ray.get(done_id))
            if len(pending) == 0: finished_folders.append(folder)
        return finished_folders
    
    def check_item_success(self, future_res: Task):
        filters, results = future_res.filters, future_res.filter_results
        suc_flag, metrics = True, {}
        for f, f_res in zip(filters, results):
            if f.name not in self.filter_stats: self.filter_stats[f.name] = [0, 0, 0] # passed/failed/error
            self.filter_stats[f.name][f_res[0].value - 1] += 1 # enum starts from 1
            metrics.update(f_res[1])
            if f_res[0] != FilterResult.PASSED:
                suc_flag = False
                break

        self.num_checked += 1
        if suc_flag: self.num_passed += 1
        if self.num_checked % self.print_freq == 0: self.print_info()

        return suc_flag, metrics
        
    def get_results(self, folder):
        # overwrite results.jsonl
        suc_items = []
        for i, future_res in enumerate(self.future_results[folder]):
            suc_flag, metrics = self.check_item_success(future_res)
            if not suc_flag: continue
            item = future_res.item
            item.update(metrics)
            suc_items.append(item)
        del self.futures[folder]
        del self.future_results[folder]
        return suc_items

    def reset(self):
        self.futures = {}
        self.future_results = {}
        self.filter_stats = {}
        self.num_checked = 0
        self.num_passed = 0

    def print_info(self):
        print()
        print_log('=' * 30 + ' Filters ' + '=' * 30)

        # basic information
        print_log(f'#checked: {self.num_checked}, #passed: {self.num_passed}')
        # elapsed_time = time() - self.launch_time
        # if self.num_passed == 0:
        #     print_log(f'Elapsed time: {int(elapsed_time)}s')
        # else:
        #     remain_time = elapsed_time / self.num_passed * (self.total_samples - self.num_passed)
        #     print_log(f'Elapsed time: {int(elapsed_time)}s, expected remaining time: {int(remain_time)}s')

        # filter information
        print_log('Filter statistics: Passed/Failed/Error/Total, %')
        for f_name in self.filter_stats:
            cnts = self.filter_stats[f_name]
            s = f'\t{f_name}: ' + '/'.join([str(c) for c in cnts] + [str(sum(cnts))])
            s += ', ' + '/'.join([str(int((c / (sum(cnts) + 1e-10) + 1e-5) * 100)) + '%' for c in cnts])
            s += '/100%' # total
            print_log(s)
        print_log('=' * 69)
        print()

    def __del__(self):
        ray.shutdown()

    def __len__(self):
        return len(self.futures)