# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from torch_scatter import scatter_mean

from utils import register as R
from utils.gnn_utils import length_to_batch_id
from .abs_trainer import Trainer


@R.register('ConfidenceTrainer')
class ConfidenceTrainer(Trainer): # latent diffusion trainer

    def __init__(self, model, train_loader, valid_loader, criterion: str, config: dict, save_config: dict):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.criterion = criterion

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx)

    def valid_step(self, batch, batch_idx):
        loss = self.share_step(batch, batch_idx, val=True)
        return loss

    def _train_epoch_end(self, device):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'update_epoch'):
            dataset.update_epoch()
        return super()._train_epoch_end(device)

    def _valid_epoch_begin(self, device):
        self.rng_state = torch.random.get_rng_state()
        torch.manual_seed(12) # each validation epoch uses the same initial state
        return super()._valid_epoch_begin(device)

    def _valid_epoch_end(self, device):
        torch.random.set_rng_state(self.rng_state)
        return super()._valid_epoch_end(device)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        loss_dict = self.model(**batch)
        if self.is_oom_return(loss_dict):
            return loss_dict
        loss = loss_dict['total']# loss_dict['X'] + loss_dict['S'] + loss_dict['A']

        log_type = 'Validation' if val else 'Train'

        for key in loss_dict:
            self.log(f'{key}/{log_type}', loss_dict[key], batch_idx, val)

        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)
            self.log('epoch', self.epoch, batch_idx, val)

        return loss