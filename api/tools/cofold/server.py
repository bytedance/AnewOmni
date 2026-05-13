# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import sys
import time
import argparse
import subprocess
from dataclasses import dataclass
from typing import List

import ray

from utils.logger import print_log

from .backends import get_backend, BackendTaskConfig


@dataclass
class Task:
    id: str
    input_path: str
    log_dir: str
    output_dir: str
    status_file: str
    exit_code: int = 0
    elapsed_time: int = 0

    model: str = 'protenix'
    repo_dir: str = ''
    env: str = ''
    db: str = ''
    param: str = ''


def get_task_name(path: str, model: str) -> str:
    return get_backend(model).get_task_name(path)


@ray.remote(num_cpus=2, num_gpus=1)
def run(task: Task):
    start = time.time()
    gpu_ids = [str(i) for i in ray.get_gpu_ids()]
    input_path = os.path.abspath(task.input_path)
    out_dir = os.path.abspath(task.output_dir)
    backend = get_backend(task.model)
    cfg = BackendTaskConfig(repo_dir=task.repo_dir, env=task.env, db=task.db, param=task.param)
    try:
        with open(task.status_file, 'w') as fout: fout.write('PROCESSING\n')

        # Backend-specific preparation (e.g. boltz template conversion).
        backend.preprocess_input(cfg=cfg, input_path=input_path)

        cmd = backend.build_command(cfg=cfg, gpu_ids=gpu_ids, input_path=input_path, out_dir=out_dir)
        print_log(f'command initiated: {cmd}')
        sys.stdout.flush()

        # Run command and capture combined stdout/stderr.
        proc = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        text = proc.stdout
        status = proc.returncode
        name = backend.get_task_name(input_path)
        log_out = os.path.join(os.path.abspath(task.log_dir), name)
        os.makedirs(log_out, exist_ok=True)
        with open(os.path.join(log_out, 'log.txt'), 'w') as fout:
            fout.write(text)
        task.exit_code = status

        if task.exit_code == 0:
            backend.postprocess_outputs(out_dir=out_dir, input_name=name, input_path=input_path)

        # add file marker
        marker = 'SUCCEEDED' if task.exit_code == 0 else 'FAILED'
        with open(task.status_file, 'w') as fout: fout.write(marker + '\n')
    except Exception as e:
        print_log(f'cofold run error: {e}', level='ERROR')
        task.exit_code = 1
    # add time
    task.elapsed_time = time.time() - start
    return task


def _candidate_suffixes(model: str) -> List[str]:
    return get_backend(model).input_suffixes()


def scan_tasks(dir, visited, model, repo_dir, env, db, param, verbose=True):
    tasks = []
    for proj in os.listdir(dir):
        d = os.path.join(dir, proj)
        if not os.path.isdir(d): continue
        if verbose: print_log(f'scanning {d}...')
        sys.stdout.flush()
        try:
            os.makedirs(os.path.join(d, 'output'), exist_ok=True)
        except Exception: continue  # might be in the process of being deleted
        suffixes = _candidate_suffixes(model)
        for fn in os.listdir(d):
            if not any(fn.endswith(s) for s in suffixes):
                continue
            f = os.path.join(d, fn)
            try:
                if (f in visited) and (os.path.getmtime(f) == visited[f]): continue # already visited and the file has not been modified
                name = get_task_name(f, model)
            except Exception: continue  # might be in the processing of writing this file
            status_file = os.path.join(d, 'logs', name, '_STATUS')
            if os.path.exists(status_file): continue    # already added
            try:
                os.makedirs(os.path.join(d, 'logs', name), exist_ok=True)
                with open(status_file, 'w') as fout: fout.write('ADDED\n')
            except Exception: continue  # this folder might be in the process of being deleted
            tasks.append(run.remote(Task(
                id=f,
                input_path=f,
                log_dir=os.path.join(d, 'logs'),
                output_dir=os.path.join(d, 'output'),
                status_file=status_file,
                model=model,
                repo_dir=repo_dir,
                env=env,
                db=db,
                param=param
            )))
            if verbose: print_log(f'task {f} added')
            sys.stdout.flush()
            visited[f] = os.path.getmtime(f)
    # cleanup visited and delete those deleted path
    if verbose: print_log(f'Current number of visited path: {len(visited)}')
    del_keys = [key for key in visited if not os.path.exists(key)]
    for key in del_keys: visited.pop(key)
    if verbose: print_log(f'After cleaning, number of visited path: {len(visited)}')
    return tasks


def main(args):
    ray.init()
    try:
        futures, visited = [], {}
        while True:
            tasks = scan_tasks(args.task_dir, visited, args.model, args.repo_dir, args.env, args.db, args.param)
            # for task in tasks: futures.append(run.remote(task))
            futures.extend(tasks)
            if len(tasks) > 0:
                print_log(f'Scanned {args.task_dir} and {len(tasks)} tasks appended...')
                sys.stdout.flush()
            else: print_log('No new tasks identified.')
            time.sleep(10)
            while len(futures) > 0:
                done_ids, futures = ray.wait(futures, num_returns=1, timeout=1)
                if len(done_ids) == 0: break    # not any tasks finished yet
                for done_id in done_ids:
                    task = ray.get(done_id)
                    print_log(f'{task.id} finished. Elapsed {round(task.elapsed_time, 2)}s. Exit code: {task.exit_code}.')
                    sys.stdout.flush()
            if len(futures) == 0: print_log('Idling...')
            sys.stdout.flush()

    except KeyboardInterrupt:
        print_log('Stopping...')
    ray.shutdown()



def parse():
    parser = argparse.ArgumentParser(description='cofold server')
    parser.add_argument('--task_dir', type=str, required=True, help='Directory to store tasks')
    parser.add_argument('--model', type=str, required=True, choices=['alphafold3', 'boltz2', 'protenix'])
    parser.add_argument('--repo_dir', type=str, default='', help='Directory of the model repo (AF3 only)')
    parser.add_argument('--env', type=str, default='', help='Environment (AF3: conda env name/prefix; boltz2: conda prefix)')
    parser.add_argument('--db', type=str, default='', help='Database dir (AF3 only)')
    parser.add_argument('--param', type=str, default='', help='Model parameters dir (AF3) or boltz cache/weights dir (boltz2)')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())
