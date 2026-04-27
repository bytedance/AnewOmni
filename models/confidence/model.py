# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
from random import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.register as R
from data.bioparse import VOCAB, const
from models.modules.adapter.model import ConditionConfig
from models.modules.nn import MLP
from utils.oom_decorator import oom_decorator
from utils.nn_utils import graph_to_batch_nx
from utils.gnn_utils import length_to_batch_id

from ..modules.create_net import create_net


@dataclass
class ConfidenceReturn:
    lig_pde: torch.Tensor                   # [N, N]
    cplx_pde: torch.Tensor                  # [N, M]
    atom_confidence: torch.Tensor           # [N]
    lig_pde_avg: torch.Tensor               # [1]
    cplx_pde_avg: torch.Tensor              # [1]
    lig_pde_local: torch.Tensor             # [N]
    cplx_pde_local_lig: torch.Tensor        # [N]
    cplx_pde_local_pocket: torch.Tensor     # [M]
    confidence: torch.Tensor                # [1]


def _clean_nan_and_inf(vals):
    vals = torch.where(torch.isinf(vals), torch.zeros_like(vals), vals)
    vals = torch.where(torch.isnan(vals), torch.zeros_like(vals), vals)
    return vals


@R.register('Confidence')
class Confidence(nn.Module):
    def __init__(
            self,
            base_model_path: str,
            hidden_size: int,
            edge_size: int,
            encoder_opt: dict,
            encoder_type: str='EPT',
            error_bin_max=16,
            error_bin_min=0,
            error_bin_num=32,
            num_atom_type=VOCAB.get_num_atom_type(),
            use_vae_dec_embedding=True,
            recycle=0,
            use_bonds=False,
            pairwise_loss_weight=None,  # None for not enable, otherwise (0, 1)
        ):
        super().__init__()
        self.base_model = torch.load(base_model_path, map_location='cpu', weights_only=False)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        
        self.use_vae_dec_embedding = use_vae_dec_embedding
        self.recycle = recycle
        self.use_bonds = use_bonds
        self.pairwise_loss_weight = pairwise_loss_weight
        if self.use_bonds:
            assert self.recycle > 0 # only recycle has access to ground-truth chemical bonds
            self.atom_edge_embedding = nn.Embedding(5, edge_size) # [None, single, double, triple, aromatic]
        
        if self.use_vae_dec_embedding:
            input_size = self.base_model.autoencoder.hidden_size
            self.input_linear = nn.Linear(input_size, hidden_size, bias=False)
        else: # use new embeddings
            latent_size = self.base_model.autoencoder.latent_size
            self.atom_embedding = nn.Embedding(num_atom_type, hidden_size)
            self.input_linear = nn.Linear(hidden_size + latent_size, hidden_size, bias=False)

        self.encoder = create_net(encoder_type, hidden_size, edge_size, encoder_opt)
        if self.encoder.require_block_edges:
            self.edge_embedding = nn.Embedding(3, edge_size) # [intra, inter, topo]a

        self.register_buffer('bins', torch.linspace(error_bin_min, error_bin_max, error_bin_num + 1))
        self.bin_num = error_bin_num + 1
        
        # lig pde and cplx pde (*2), each logit contains dot product of 16-dimensional embedding
        self.pde_mlp = MLP(hidden_size, hidden_size, self.bin_num * 2 * 16, n_layers=3)

    def _generate_data(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3), single-directional
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for used to calculate complex center of mass
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
        ):
        X_gt = X.clone()
        # sample results
        struct_pred_config = ConditionConfig(
            mask_2d=generate_mask,
            mask_3d=torch.zeros_like(generate_mask),
            mask_incomplete_2d=torch.zeros_like(generate_mask),
            w=random() * 3.0 # controling strengths max 3.0
        )
        base_model_sample_opt = {'ddim': True, 'ddim_steps': 10, 'cal_likelihood': False} # accelerating LDM
        if self.recycle > 0: base_model_sample_opt['vae_decode_n_iter'] = 5 # accelerating VAE
        Zh, Zx, H_sample, X_sample = self.base_model.sample(
            X, S, A, bonds, position_ids, chain_ids, generate_mask, center_mask,
            block_lengths, lengths, is_aa, condition_config=struct_pred_config,
            return_tensor=True, sample_opt=base_model_sample_opt, _overwrite_inf_nan=True
        ) # [Natom, h], [Natom, 3]
        for _ in range(self.recycle):
            _, _, H_sample, X_sample = self.base_model.autoencoder.generate(
                X_sample, S, A, bonds, position_ids, chain_ids, generate_mask,
                block_lengths, lengths, is_aa, return_tensor=True,
                topo_generate_mask=torch.zeros_like(S, dtype=torch.bool)
            )
        H_sample, X_sample = H_sample.detach(), X_sample.detach()

        block_ids = length_to_batch_id(block_lengths)
        batch_ids = length_to_batch_id(lengths)

        # calculate ligand PDE and cplx PDE and PAE
        dist_error = self._get_metrics(
            X_gt, H_sample, X_sample, generate_mask, block_ids, batch_ids
        )
        # dist_error_bins = self._discretize_values(dist_error)

        # get mask
        lig_pde_mask, cplx_pde_mask = self._get_mask(
            H_sample, X_sample, generate_mask, block_ids, batch_ids
        )

        # clean inputs
        H_sample = _clean_nan_and_inf(H_sample)
        X_sample = _clean_nan_and_inf(X_sample)

        # form edges
        if self.encoder.require_block_edges:
            edges, edge_type = self.base_model.autoencoder.get_edges(batch_ids, chain_ids, X_sample, block_ids, None, True, True)
        else: edges, edge_type = None, None

        return H_sample, X_sample, Zh, edges, edge_type, batch_ids, block_ids, dist_error, lig_pde_mask, cplx_pde_mask
    
    def _cal_pde(self, H_sample, X_sample, Zh, A, bonds, batch_ids, block_ids, edges, edge_type):
        # calculate confidence
        # transform input hidden states
        if self.use_vae_dec_embedding: H_in = self.input_linear(H_sample)
        else:
            atom_embed = self.atom_embedding(A)
            H_in = self.input_linear(torch.cat([atom_embed, Zh[block_ids]], dim=-1))
        # chemical bonds
        if self.use_bonds:
            # make bonds bidirectional
            bond_row, bond_col, bond_type = bonds.T
            topo_edges = torch.stack([
                torch.cat([bond_row, bond_col], dim=0),
                torch.cat([bond_col, bond_row], dim=0)
            ], dim=0)
            topo_edge_attr = self.atom_edge_embedding(torch.cat([bond_type, bond_type], dim=0))
        else: topo_edges, topo_edge_attr = None, None
        if edge_type is not None: edge_attr = self.edge_embedding(edge_type)
        else: edge_attr = None
        H_conf, _ = self.encoder(H_in, X_sample, block_ids, batch_ids, edges, edge_attr, topo_edges, topo_edge_attr) # [Natom', hidden_size], [Natom', 3]

        # logits ([bs, Natom_max, Natom_max])
        pde_logits = self._get_pair_logits(self.pde_mlp(H_conf), batch_ids, block_ids) # [bs, Natom_max, Natom_max, bins * 2]
        lig_pde_logits, cplx_pde_logits = pde_logits[:, :, :, 0::2], pde_logits[:, :, :, 1::2] # [bs, Natom_max, Natom_max, bins]
        return lig_pde_logits, cplx_pde_logits

    @oom_decorator
    def forward(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3), single-directional
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for used to calculate complex center of mass
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
        ):
        
        with torch.no_grad():
            H_sample, X_sample, Zh, edges, edge_type, batch_ids, block_ids, dist_error1, lig_pde_mask1, cplx_pde_mask1 = self._generate_data(
                X.clone(), S.clone(), A.clone(), bonds.clone(), position_ids.clone(), chain_ids.clone(), generate_mask.clone(), center_mask.clone(),
                block_lengths.clone(), lengths.clone(), is_aa.clone()
            )
            dist_error_bins1 = self._discretize_values(dist_error1)
        # calculate ligand PDE and cplx PDE
        lig_pde_logits1, cplx_pde_logits1 = self._cal_pde(
            H_sample, X_sample, Zh, A.clone(), bonds.clone(), batch_ids, block_ids, edges, edge_type
        ) # [bs, Natom_max, Natom_max, bins]
        
        ############### No Pairwise Loss ###############
        if self.pairwise_loss_weight is None:   # no pairwise loss
        
            # loss
            lig_pde_loss = F.cross_entropy(lig_pde_logits1[lig_pde_mask1], dist_error_bins1[lig_pde_mask1])
            cplx_pde_loss = F.cross_entropy(cplx_pde_logits1[cplx_pde_mask1], dist_error_bins1[cplx_pde_mask1])
        
            loss = lig_pde_loss + cplx_pde_loss
        
            loss_dict = {
                'lig_pde_loss': lig_pde_loss,
                'cplx_pde_loss': cplx_pde_loss,
                'total': loss
            }

            # evaluation
            with torch.no_grad():
                # continuous value
                lig_pde_pred1 = self._logits_to_continuous(lig_pde_logits1)
                cplx_pde_pred1 = self._logits_to_continuous(cplx_pde_logits1)
                loss_dict['lig_pde_corr'] = self._get_corr(lig_pde_pred1[lig_pde_mask1], dist_error1[lig_pde_mask1])
                loss_dict['cplx_pde_corr'] = self._get_corr(cplx_pde_pred1[cplx_pde_mask1], dist_error1[cplx_pde_mask1])

            return loss_dict
        
        ############### With Pairwise Loss ###############

        with torch.no_grad():
            H_sample, X_sample, Zh, edges, edge_type, batch_ids, block_ids, dist_error2, lig_pde_mask2, cplx_pde_mask2 = self._generate_data(
                X.clone(), S.clone(), A.clone(), bonds.clone(), position_ids.clone(), chain_ids.clone(), generate_mask.clone(), center_mask.clone(),
                block_lengths.clone(), lengths.clone(), is_aa.clone()
            )
            dist_error_bins2 = self._discretize_values(dist_error2)
        # calculate ligand PDE and cplx PDE
        lig_pde_logits2, cplx_pde_logits2 = self._cal_pde(
            H_sample, X_sample, Zh, A.clone(), bonds.clone(), batch_ids, block_ids, edges, edge_type
        ) # [bs, Natom_max, Natom_max, bins]
        
        # continuous value
        lig_pde_pred1 = self._logits_to_continuous(lig_pde_logits1)
        cplx_pde_pred1 = self._logits_to_continuous(cplx_pde_logits1)
        lig_pde_pred2 = self._logits_to_continuous(lig_pde_logits2)
        cplx_pde_pred2 = self._logits_to_continuous(cplx_pde_logits2)
        
        # loss
        lig_pde_loss = (F.cross_entropy(lig_pde_logits1[lig_pde_mask1], dist_error_bins1[lig_pde_mask1]) + \
                        F.cross_entropy(lig_pde_logits2[lig_pde_mask2], dist_error_bins2[lig_pde_mask2])) * 0.5
        cplx_pde_loss = (F.cross_entropy(cplx_pde_logits1[cplx_pde_mask1], dist_error_bins1[cplx_pde_mask1]) + \
                        F.cross_entropy(cplx_pde_logits2[cplx_pde_mask2], dist_error_bins2[cplx_pde_mask2])) * 0.5
        
        # pairwise loss
        lig_pde_share_mask = lig_pde_mask1 & lig_pde_mask2
        cplx_pde_share_mask = cplx_pde_mask1 & cplx_pde_mask2
        lig_pde_pairwise_loss = F.huber_loss((lig_pde_pred1 - lig_pde_pred2)[lig_pde_share_mask], (dist_error1 - dist_error2)[lig_pde_share_mask], delta=0.5)
        cplx_pde_pairwise_loss = F.huber_loss((cplx_pde_pred1 - cplx_pde_pred2)[cplx_pde_share_mask], (dist_error1 - dist_error2)[cplx_pde_share_mask], delta=0.5)

        loss = (lig_pde_loss + cplx_pde_loss) * (1.0 - self.pairwise_loss_weight) + (lig_pde_pairwise_loss + cplx_pde_pairwise_loss) * self.pairwise_loss_weight
        
        loss_dict = {
            'lig_pde_loss': lig_pde_loss,
            'cplx_pde_loss': cplx_pde_loss,
            'lig_pde_pairwise_loss': lig_pde_pairwise_loss,
            'cplx_pde_pairwise_loss': cplx_pde_pairwise_loss,
            'total': loss
        }

        # evaluation
        with torch.no_grad():
            loss_dict['lig_pde_corr'] = self._get_corr(lig_pde_pred1[lig_pde_mask1], dist_error1[lig_pde_mask1])
            loss_dict['cplx_pde_corr'] = self._get_corr(cplx_pde_pred1[cplx_pde_mask1], dist_error1[cplx_pde_mask1])
            loss_dict['lig_pde_corr_double'] = self._get_corr(
                torch.cat([lig_pde_pred1[lig_pde_mask1], lig_pde_pred2[lig_pde_mask2]], dim=0),
                torch.cat([dist_error1[lig_pde_mask1], dist_error2[lig_pde_mask2]], dim=0)
            )
            loss_dict['cplx_pde_corr_double'] = self._get_corr(
                torch.cat([cplx_pde_pred1[cplx_pde_mask1], cplx_pde_pred2[cplx_pde_mask2]], dim=0),
                torch.cat([dist_error1[cplx_pde_mask1], dist_error2[cplx_pde_mask2]], dim=0)
            )

        return loss_dict
    
    def inference(self, H, X, chain_ids, generate_mask, block_ids, batch_ids, local_dist_th=6.0):

        # form edges
        if self.encoder.require_block_edges:
            edges, edge_type = self.base_model.autoencoder.get_edges(batch_ids, chain_ids, X, block_ids, None, True, True)
            edge_attr = self.edge_embedding(edge_type)
        else: edges, edge_attr = None, None

        H_in = self.input_linear(H)
        H_conf, _ = self.encoder(H_in, X, block_ids, batch_ids, edges, edge_attr) # [Natom', hidden_size], [Natom', 3]

        # logits ([bs, Natom_max, Natom_max])
        pde_logits = self._get_pair_logits(self.pde_mlp(H_conf), batch_ids, block_ids) # [bs, Natom_max, Natom_max, bins * 2]
        lig_pde_logits, cplx_pde_logits = pde_logits[:, :, :, 0::2], pde_logits[:, :, :, 1::2] # [bs, Natom_max, Natom_max, bins]

        # continous
        lig_pde = self._logits_to_continuous(lig_pde_logits)   # [bs, Natom_max, Natom_max]
        cplx_pde = self._logits_to_continuous(cplx_pde_logits) # [bs, Natom_max, Natom_max]

        # get mask
        lig_pde_mask, cplx_pde_mask = self._get_mask(
            torch.zeros_like(H), torch.zeros_like(X), # regardless of nan
            generate_mask, block_ids, batch_ids
        ) # [bs, Natom_max, Natom_max]

        # get distances
        atom_batch_ids = batch_ids[block_ids]
        X_batch, _ = graph_to_batch_nx(X, atom_batch_ids, mask_is_pad=False) # [bs, Natom_max, 3], [bs, Natom_max]
        pair_dist = torch.norm(X_batch[:, :, None, :] - X_batch[:, None, :, :], dim=-1) # [bs, Natom_max, Natom_max]
        local_mask = pair_dist < local_dist_th

        # get results for each item
        confidences = []
        atom_generate_mask = generate_mask[block_ids]
        for i in range(batch_ids.max() + 1):
            lig_natom = atom_generate_mask[atom_batch_ids == i].long().sum()
            lig_pde_item = lig_pde[i][lig_pde_mask[i]].reshape(lig_natom, -1)
            cplx_pde_item = cplx_pde[i][cplx_pde_mask[i]].reshape(lig_natom, -1)
            lig_pde_avg, cplx_pde_avg = lig_pde_item.mean(), cplx_pde_item.mean()

            lig_pde_local = lig_pde_item.clone()
            lig_local_mask = local_mask[i][lig_pde_mask[i]].reshape(lig_natom, -1)  # [n_lig_atom, n_lig_atom]
            lig_pde_local[~lig_local_mask] = 0.0
            lig_pde_local = lig_pde_local.sum(-1) / lig_local_mask.sum(-1)  # [n_lig_atom]
            cplx_pde_local = cplx_pde_item.clone()
            cplx_local_mask = local_mask[i][cplx_pde_mask[i]].reshape(lig_natom, -1)
            cplx_pde_local[~cplx_local_mask] = 0.0
            cplx_pde_local_lig = cplx_pde_local.sum(-1) / cplx_local_mask.sum(-1)   # [n_lig_atom]
            cplx_pde_local_pocket = cplx_pde_local.sum(0) / cplx_local_mask.sum(0)  # [n_pocket_atom]
         
            confidences.append(ConfidenceReturn(
                lig_pde = lig_pde_item.cpu(),
                cplx_pde = cplx_pde_item.cpu(),
                atom_confidence = 0.5 * (lig_pde_item.mean(dim=-1) + cplx_pde_item.mean(dim=-1)).cpu(),
                lig_pde_avg = lig_pde_avg.cpu().item(),
                cplx_pde_avg = cplx_pde_avg.cpu().item(),
                confidence = 0.5 * (lig_pde_avg + cplx_pde_avg).cpu().item(),
                lig_pde_local = lig_pde_local.cpu(),
                cplx_pde_local_lig = cplx_pde_local_lig.cpu(),
                cplx_pde_local_pocket = cplx_pde_local_pocket.cpu()
            ))

        return confidences
    
    def _get_mask(self, H, X, ligand_mask, block_ids, batch_ids):
        atom_batch_ids = batch_ids[block_ids]
        H, mask = graph_to_batch_nx(H, atom_batch_ids, mask_is_pad=False) # [bs, Natom_max, d], [bs, Natom_max]
        X, _ = graph_to_batch_nx(X, atom_batch_ids, mask_is_pad=False) # [bs, Natom_max, 3], [bs, Natom_max]

        # discard nan
        nan_mask = (torch.isnan(X).sum(-1) > 0) | (torch.isnan(H).sum(-1) > 0) # [bs, Natom_max]
        inf_mask = (torch.isinf(X).sum(-1) > 0) | (torch.isinf(H).sum(-1) > 0) # [bs, Natom_max]
        mask = mask & (~nan_mask) & (~inf_mask)
        mask = mask[:, :, None] & mask[:, None, :]  # [bs, Natom, Natom]

        # masks
        ligand_mask, _ = graph_to_batch_nx(ligand_mask[block_ids], atom_batch_ids, mask_is_pad=False, padding_value=False) # [bs, Natom]
        lig_pde_mask = ligand_mask[:, :, None] & ligand_mask[:, None, :] & mask     # [bs, Natom, Natom]
        cplx_pde_mask = ligand_mask[:, :, None] & (~ligand_mask[:, None, :]) & mask # [bs, Natom, Natom]

        return lig_pde_mask, cplx_pde_mask

    def _get_metrics(self, X_gt, H_sample, X_sample, ligand_mask, block_ids, batch_ids):
        atom_batch_ids = batch_ids[block_ids]
        X_gt, mask = graph_to_batch_nx(X_gt, atom_batch_ids, mask_is_pad=False) # [bs, Natom_max, 3], [bs, Natom_max]
        pair_dist_gt = torch.norm(X_gt[:, :, None, :] - X_gt[:, None, :, :], dim=-1) # [bs, Natom_max, Natom_max]
        X_sample, mask = graph_to_batch_nx(X_sample, atom_batch_ids, mask_is_pad=False)
        H_sample, _ = graph_to_batch_nx(H_sample, atom_batch_ids, mask_is_pad=False)
        pair_dist_sample = torch.norm(X_sample[:, :, None, :] - X_sample[:, None, :, :], dim=-1) # [bs, Natom_max, Natom_max]
        dist_error = torch.abs(pair_dist_gt - pair_dist_sample) # [bs, Natom, Natom]
        
        return dist_error

        # # discard nan
        # nan_mask = (torch.isnan(X_sample).sum(-1) > 0) | (torch.isnan(H_sample).sum(-1) > 0) # [bs, Natom_max]
        # inf_mask = (torch.isinf(X_sample).sum(-1) > 0) | (torch.isinf(H_sample).sum(-1) > 0) # [bs, Natom_max]
        # mask = mask & (~nan_mask) & (~inf_mask)
        # mask = mask[:, :, None] & mask[:, None, :]  # [bs, Natom, Natom]

        # # masks
        # ligand_mask, _ = graph_to_batch_nx(ligand_mask[block_ids], atom_batch_ids, mask_is_pad=False, padding_value=False) # [bs, Natom]
        # lig_pde_mask = ligand_mask[:, :, None] & ligand_mask[:, None, :] & mask     # [bs, Natom, Natom]
        # cplx_pde_mask = ligand_mask[:, :, None] & (~ligand_mask[:, None, :]) & mask # [bs, Natom, Natom]

        # return dist_error, lig_pde_mask, cplx_pde_mask
    
    def _get_pair_logits(self, H, batch_ids, block_ids):
        # pairwise confidence
        H_conf, mask = graph_to_batch_nx(H, batch_ids[block_ids], mask_is_pad=False) # [bs, Natom_max, h], [bs, Natom_max]
    
        # bins
        batch_size, natom, hidden_size = H_conf.shape
        H_conf = H_conf.view(batch_size, natom, self.bin_num * 2, -1).transpose(1, 2) # [bs, n_bins * 2, Naton_max, h//d]

        # dot
        scale = math.sqrt(hidden_size // (self.bin_num * 2))
        dot_prod = torch.matmul(H_conf, H_conf.transpose(-1, -2)) / scale # [bs, n_bins * 2, Natom_max, Natom_max]

        # summation of different heads
        return dot_prod.transpose(1, 2).transpose(-1, -2) # [bs, Natom_max, Natom_max, n_bins * 2]
    
    def _discretize_values(self, vals):
        # discretize
        vals = torch.bucketize(vals, self.bins) # [bs, Natom, Natom], int
        # put overflow bins into the last bin
        vals = torch.where(vals == self.bin_num, torch.ones_like(vals) * (self.bin_num - 1), vals)
        return vals

    def _logits_to_continuous(self, logits):
        probs = F.softmax(logits, dim=-1) # [bs, Natom, Natom, bin_num]
        return torch.matmul(probs, self.bins.unsqueeze(-1)).squeeze(-1) # [bs, Natom, Natom]
    
    def _get_corr(self, pred_vals, gt_vals):
        pred_vals, gt_vals = pred_vals.flatten(), gt_vals.flatten()
        return torch.corrcoef(torch.stack([pred_vals, gt_vals], dim=0))[0][1]