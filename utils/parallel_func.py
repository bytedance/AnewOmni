# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from multiprocessing import Process, Queue

import ray

from .logger import print_log


########## Parallel Function Generator by Multiprocessing ##########

def func_wrapper(func, task_queue, result_queue):
    while True:
        (i, inputs) = task_queue.get()
        if inputs is None: break
        try:
            outputs = func(*inputs)
        except Exception as e:
            outputs = None
            print_log(f'{e}, {i}, inputs {inputs} failed', level='WARN')
        result_queue.put((i, outputs))


def parallel_func(func, inputs, n_cpus, unordered=False, result_queue_maxsize=1000):
    task_queue = Queue()
    result_queue = Queue(maxsize=result_queue_maxsize)

    # create worker process
    processes = []
    for _ in range(n_cpus):
        p = Process(target=func_wrapper, args=(func, task_queue, result_queue))
        processes.append(p)
        p.start()

    # Distribute tasks to workers
    for i, args in enumerate(inputs): task_queue.put((i, args))

    # Add a sentinel (None) to signal workers to exit
    for _ in range(n_cpus): task_queue.put((-1, None)) # end

    # Collect results from workers
    if unordered: # don't care ordering
        for _ in inputs:
            _, outputs = result_queue.get()
            # print_log(f'queue size: {result_queue.qsize()}')
            yield outputs
    else: # the same ordering as inputs
        stored_outputs, current = {}, 0
        for _ in inputs:
            i, outputs = result_queue.get()
            stored_outputs[i] = outputs
            if current in stored_outputs:
                yield stored_outputs.pop(current)
                current += 1
        while len(stored_outputs):
            yield stored_outputs.pop(current)
            current += 1
    
    # Ensure all processes have finished
    for p in processes:
        p.join()


########## Parallel Function Generator by Ray ##########

@ray.remote(num_cpus=1)
def ray_func_wrapper(inputs):
    func, args = inputs
    try:
        outputs = func(*args)
    except Exception as e:
        outputs = None
        print_log(f'{e}, inputs {args} failed', level='WARN')
    return outputs


def parallel_func_ray(func, inputs, n_cpus, unordered=False, result_queue_maxsize=1000, verbose=False):
    ray.init(num_cpus=n_cpus, ignore_reinit_error=True, log_to_driver=verbose)
    futures = [ray_func_wrapper.remote((func, args)) for args in inputs[:result_queue_maxsize]]
    inputs = inputs[result_queue_maxsize:]  # remaining inputs
    if unordered:
        while len(futures) > 0:
            done_ids, futures = ray.wait(futures, num_returns=1)
            for done_id in done_ids:
                outputs = ray.get(done_id)
                if len(inputs) > 0:
                    futures.append(ray_func_wrapper.remote((func, inputs.pop(0))))
                yield outputs
    else:
        while len(futures) > 0:
            outputs = ray.get(futures.pop(0))
            if len(inputs) > 0:
                futures.append(ray_func_wrapper.remote((func, inputs.pop(0))))
            yield outputs
    ray.shutdown()