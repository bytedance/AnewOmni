#!/usr/bin/python
# -*- coding:utf-8 -*-
from .iterae_trainer import IterAETrainer
from .ldm_trainer import LDMTrainer
from .confidence_trainer import ConfidenceTrainer

import utils.register as R

def create_trainer(config, model, train_loader, valid_loader):
    return R.construct(
        config['trainer'],
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_config=config)


