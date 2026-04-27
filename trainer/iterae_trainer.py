# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
from torch_scatter import scatter_mean

from utils import register as R
from utils.gnn_utils import length_to_batch_id
from .abs_trainer import Trainer


@R.register('IterAETrainer')
class IterAETrainer(Trainer): # autoregressive autoencoder trainer

    def __init__(self, model, train_loader, valid_loader, criterion: str, config: dict, save_config: dict):
        super().__init__(model, train_loader, valid_loader, config, save_config)
        self.criterion = criterion

    def train_step(self, batch, batch_idx):
        if self.config.warmup != 0:
            batch['warmup_progress'] = min(self.global_step / self.config.warmup, 1.0)
            self.log('warmup_progress', batch['warmup_progress'], batch_idx, val=False)
        return self.share_step(batch, batch_idx)

    def valid_step(self, batch, batch_idx):
        loss = self.share_step(batch, batch_idx, val=True)

        if self.local_rank != -1: # DDP mode
            model = self.model.module
        else:
            model = self.model
        ########### structure prediction validation ##########
        generate_mask = batch['generate_mask']
        batch_ids = length_to_batch_id(batch['lengths'])
        block_ids = length_to_batch_id(batch['block_lengths'])
        true_X = batch['X'][generate_mask[block_ids]]
        # fix sequence for structure prediction
        recon_X = model.generate(topo_generate_mask=torch.zeros_like(generate_mask), return_x_only=True, **batch)

        recon_X = recon_X[generate_mask[block_ids]]

        rmsd = torch.sqrt(scatter_mean(((recon_X - true_X) ** 2).sum(-1), batch_ids[block_ids][generate_mask[block_ids]])) # [bs]
        rmsd = rmsd.mean()

        self.log('RMSD_atom/Validation', rmsd, batch_idx, val=True, batch_size=len(batch['lengths']))

        if self.criterion == 'RMSD_atom':
            return rmsd.detach()
        else:
            return loss.detach()

        X_gt, S_gt, A_gt = batch['X'].clone(), batch['S'].clone(), batch['A'].clone()
        ########### encoding ##########
        Zh, Zx, Zh_kl_loss, Zx_kl_loss = self.model.encode(
            batch['X'], batch['S'], batch['A'], batch['position_ids'],
            batch['segment_ids'], batch['block_lengths'], batch['lengths']
        ) # [Nblock, d_latent], [Nblock, 3]


        ########### codesign validation ##########
        X_mask, S_mask = self.model._codesign_mask(batch['ligand_mask'])
        self.share_valid_decode(
            batch, batch_idx, Zh, Zx, X_mask, S_mask, X_gt, S_gt, A_gt,
            'Codesign', ['BlockRecovery', 'AtomRecovery', 'RMSD', 'RMSD_CA']
        )

        ########### structure prediction validation ##########
        X_mask, S_mask = self.model._structure_prediction_mask(batch['ligand_mask'])
        self.share_valid_decode(
            batch, batch_idx, Zh, Zx, X_mask, S_mask, X_gt, S_gt, A_gt,
            'Struct', ['RMSD', 'RMSD_CA']
        )
        
        ########### inverse folding validation ##########
        X_mask, S_mask = self.model._inverse_folding_mask(batch['ligand_mask'])
        self.share_valid_decode(
            batch, batch_idx, Zh, Zx, X_mask, S_mask, X_gt, S_gt, A_gt,
            'IF', ['BlockRecovery', 'AtomRecovery', 'RMSD']
        )

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
    

    def share_valid_decode(self, batch, batch_idx, Zh, Zx, X_mask, S_mask, X_gt, S_gt, A_gt, suffix, metrics):
        batch_ids = length_to_batch_id(batch['lengths'])
        cond_Zh, cond_Zx, new_block_lengths, (x_loss_mask, gt_x_loss_mask, a_loss_mask, gt_a_loss_mask) = \
            self.model.condition_latent(
            Zh, Zx, batch['X'], batch['S'], batch['A'], batch['position_ids'], batch['block_lengths'],
            batch['is_protein'], X_mask, S_mask
        )

        # decode back
        X_rec, S_rec, A_rec = self.model.decode(cond_Zh, cond_Zx, batch['segment_ids'], new_block_lengths, batch['lengths']) # [Natom', 3], [Natom', num_atom_type]

        # calculate loss: block type, atom type, coordinates, pairwise distance
        block_ids = length_to_batch_id(batch['block_lengths'])
        new_block_ids = length_to_batch_id(new_block_lengths)
        A_gt_new = torch.ones(A_rec.shape[0], dtype=torch.long, device=A_rec.device) * self.model.dummy_atom_idx
        A_gt_new[gt_a_loss_mask] = A_gt[S_mask[block_ids]]
        metric_dict = {}
        type_loss_dict = self.model.cal_type_loss(
            S_rec[S_mask], S_gt[S_mask], A_rec[a_loss_mask], A_gt_new[a_loss_mask],
            batch_ids[S_mask], batch_ids[new_block_ids][a_loss_mask])
        metric_dict.update(type_loss_dict)
        struct_loss_dict = self.model.cal_struct_loss(
            X_rec[x_loss_mask], X_gt[gt_x_loss_mask], batch['is_ca'][gt_x_loss_mask], batch_ids[block_ids][gt_x_loss_mask])
        metric_dict.update(struct_loss_dict)
        for metric in metrics:
            self.log(f'{metric}/Validation_{suffix}', metric_dict[metric], batch_idx, val=True, batch_size=len(batch['lengths']))