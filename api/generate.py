# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse
from copy import deepcopy

import yaml
import torch

from utils.config_utils import overwrite_values
from utils.logger import print_log
from utils.random_seed import setup_seed
import utils.register as R

from .pdb_dataset import PDBDataset
from .helpers.gen_utils import load_model, generate_for_one_template, generate_multiple_cdrs
from .templates import AntibodyMultipleCDR


def parse():
    parser = argparse.ArgumentParser(description='Generate peptides given epitopes')
    parser.add_argument('--config', type=str, required=True, help='Path to the test configuration')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--confidence_ckpt', type=str, default=None, help='Path to the confidence model')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save generated peptides')
    parser.add_argument('--max_retry', type=int, default=None, help='Maximum number of retries')

    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    parser.add_argument('--n_cpus', type=int, default=4, help='Number of CPU to use (for parallelly saving the generated results)')
    
    parser.add_argument('--seed', type=int, default=None, help='seed')
    return parser.parse_known_args()


def main(args, opt_args):
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)

    # load model
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model, conf_model = load_model(args.ckpt, args.confidence_ckpt, device)

    sample_opt = config.get('sample_opt', {})
    
    # load dataset and dataloader
    batch_size = config.get('batch_size', 32)
    for template in config['templates']:
        template_instance = R.construct(template)
        if hasattr(template_instance, '__len__'): # factory
            template_factory = template_instance
        else: template_factory = [template_instance]
        for i in range(len(template_factory)):
            template_instance = template_factory[i]
            out_dir = os.path.join(args.save_dir, template_instance.name, 'candidates')
            os.makedirs(out_dir, exist_ok=True)
            dataset = PDBDataset(**config['dataset'], template_config=template_instance)
            if isinstance(template_instance, AntibodyMultipleCDR): gen_func = generate_multiple_cdrs
            else: gen_func = generate_for_one_template
            cur_sample_opt = deepcopy(sample_opt)
            cur_sample_opt.update(template_instance.additional_sample_opt)
            gen_func(
                model, dataset, config['n_samples'], batch_size, out_dir, device, cur_sample_opt,
                n_cycles=1, conf_model=conf_model, max_retry=args.max_retry)


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    if args.seed is not None: setup_seed(12)
    main(args, opt_args)