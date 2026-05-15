# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from torch_scatter import scatter_mean, scatter_sum

from data.bioparse import VOCAB

import utils.register as R
from utils.oom_decorator import oom_decorator
from utils.nn_utils import SinusoidalPositionEmbedding
from utils.gnn_utils import length_to_batch_id, std_conserve_scatter_mean, variadic_meshgrid
from utils.logger import print_log

from .diffusion.dpm_full import FullDPM
from ..IterVAE.model_edge import CondIterAutoEncoderEdge
from ..modules.nn import GINEConv, MLP
from ..modules.adapter.model import Conditions, ConditionConfig, ConditionSampler, config_to_condition


@R.register('LDMMolDesignClean')
class LDMMolDesignClean(nn.Module):

    def __init__(
            self,
            autoencoder_ckpt,
            latent_deterministic,
            hidden_size,
            num_steps,
            h_loss_weight=None,
            std=10.0,
            max_cond_ratio_2d=1.0,
            max_cond_ratio_3d=1.0,
            is_aa_corrupt_ratio=0.1,
            num_block_type = VOCAB.get_num_block_type(),
            diffusion_opt={},
            use_condition_sampler=False,
            condition_sampler_opt={}
        ):
        super().__init__()
        self.latent_deterministic = latent_deterministic

        self.autoencoder: CondIterAutoEncoderEdge = torch.load(autoencoder_ckpt, map_location='cpu', weights_only=False)
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()
        
        latent_size = self.autoencoder.latent_size

        # topo embedding
        self.bond_embed = nn.Embedding(5, hidden_size) # [None, single, double, triple, aromatic]
        self.atom_embed = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.topo_gnn = GINEConv(hidden_size, hidden_size, hidden_size, hidden_size)

        self.position_encoding = SinusoidalPositionEmbedding(hidden_size)
        self.block_type_embedding = nn.Embedding(num_block_type, hidden_size)
        self.topo_cond_linear = nn.Linear(hidden_size * 2, hidden_size)
        self.is_aa_embed = nn.Embedding(2, hidden_size) # is or is not standard amino acid

        # condition embedding MLP
        self.cond_mlp = MLP(
            input_size=2 * hidden_size, # [position, is_aa]
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=3,
            dropout=0.1
        )

        self.diffusion = FullDPM(
            latent_size=latent_size,
            hidden_size=hidden_size,
            num_steps=num_steps,
            **diffusion_opt
        )
        if h_loss_weight is None:
            self.h_loss_weight = 3 / latent_size  # make loss_X and loss_H about the same size
        else:
            self.h_loss_weight = h_loss_weight
        self.register_buffer('std', torch.tensor(std, dtype=torch.float))
        self.max_cond_ratio_2d = max_cond_ratio_2d
        self.max_cond_ratio_3d = max_cond_ratio_3d
        self.is_aa_corrupt_ratio = is_aa_corrupt_ratio
        self.use_condtion_sampler = use_condition_sampler
        self.condition_sampler = ConditionSampler(**condition_sampler_opt)

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
        '''
            L: [bs, 3, 3], cholesky decomposition of the covariance matrix \Sigma = LL^T
        '''

        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            # encoding
            Zh, Zx, _, _, _, _, _, _ = self.autoencoder.encode(
                X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
            ) # [Nblock, d_latent], [Nblock, 3]

            batch_ids = length_to_batch_id(lengths)
            block_ids = length_to_batch_id(block_lengths)

            # CoM for 3D conditions
            com = scatter_mean(X, block_ids, dim=0, dim_size=block_lengths.shape[0])
            com[center_mask] = Zx[center_mask] # to ensure the same normalization as Zx
            com, _ = self._normalize_position(com, batch_ids, center_mask)

        position_embedding = self.position_encoding(position_ids)

        # normalize
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)
        
        if self.use_condtion_sampler:
            conditions = self._random_conditions_sampler(S, com, A, bonds, generate_mask, block_lengths, lengths)
        else:
            conditions = self._random_conditions(S, com, A, bonds, generate_mask, block_lengths, lengths)

        # is aa embedding (sample 50% for generation part)
        topo_generate_mask = (~conditions.mask_2d) & generate_mask
        corrupt_mask = topo_generate_mask & (torch.rand_like(is_aa, dtype=torch.float) < self.is_aa_corrupt_ratio)
        is_aa_embedding = self.is_aa_embed(
            torch.where(corrupt_mask, torch.zeros_like(is_aa), is_aa).long()
        )

        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, is_aa_embedding], dim=-1))

        loss_dict = self.diffusion.forward(
            H_0=Zh,
            X_0=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            conditions=conditions
        )

        # loss
        loss_dict['total'] = loss_dict['H'] * self.h_loss_weight + loss_dict['X']

        return loss_dict

    def _random_conditions_sampler(self, S, com, A, bonds, generate_mask, block_lengths, lengths) -> Conditions:

        with torch.no_grad():
            config = self.condition_sampler(generate_mask, lengths)
            block_ids = length_to_batch_id(block_lengths)
        atom_topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), config.mask_2d | (~generate_mask))
        block_topo_embedding = scatter_sum(atom_topo_embedding, block_ids, dim=0, dim_size=block_lengths.shape[0])
        block_embedding = self.block_type_embedding(S)
        block_embedding = self.topo_cond_linear(torch.cat([block_topo_embedding, block_embedding], dim=-1))
        return config_to_condition(config, com, block_lengths, bonds, atom_topo_embedding, block_embedding)

    def _random_conditions(self, S, com, A, bonds, generate_mask, block_lengths, lengths) -> Conditions:
        with torch.no_grad():
            batch_ids = length_to_batch_id(lengths)
            block_ids = length_to_batch_id(block_lengths)

            cond_mask = torch.rand(lengths.shape[0], device=lengths.device) < 0.5 # [batch_size]
            cond_mask = cond_mask[batch_ids] # [Nblock]

            # topo condition
            # topo embedding for structure prediction (sample topo ratio [0, 1] in training)
            cond_ratio = torch.rand(lengths.shape[0], device=lengths.device)[batch_ids] * self.max_cond_ratio_2d
            cond_mask_2d = (torch.rand_like(generate_mask, dtype=torch.float) < cond_ratio) & generate_mask & cond_mask # [Nblock]

            # random drop topology with 50% probability
            # find atoms that should not be dropped (those connecting atoms in different blocks)
            inter_bond_mask = block_ids[bonds[:, 0]] != block_ids[bonds[:, 1]]
            retain_mask = torch.zeros_like(A, dtype=torch.bool)
            row, col, _ = bonds[inter_bond_mask].T
            retain_mask[row] = True
            retain_mask[col] = True
            # sample 50% blocks for topology dropout
            drop_ratio = 0.5
            block_topo_drop_mask = torch.rand_like(batch_ids, dtype=torch.float) < drop_ratio
            block_topo_drop_ratio = torch.rand_like(batch_ids, dtype=torch.float)
            atom_drop_mask = torch.rand_like(block_ids, dtype=torch.float) < block_topo_drop_ratio[block_ids]
            atom_drop_mask = cond_mask_2d[block_ids] & block_topo_drop_mask[block_ids] & atom_drop_mask & (~retain_mask)
            keep_bond_mask = ~(atom_drop_mask[bonds[:, 0]] | atom_drop_mask[bonds[:, 1]])
            bonds = bonds[keep_bond_mask]

        atom_topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), cond_mask_2d | (~generate_mask))
        block_topo_embedding = scatter_sum(atom_topo_embedding[~atom_drop_mask], block_ids[~atom_drop_mask], dim=0, dim_size=block_lengths.shape[0])
        block_embedding = self.block_type_embedding(S)
        block_embedding = torch.where(block_topo_drop_mask.unsqueeze(-1), torch.zeros_like(block_embedding), block_embedding) # for incomplete blocks, do not use block type embedding
        block_embedding = self.topo_cond_linear(torch.cat([block_topo_embedding, block_embedding], dim=-1))

        # H_2d = std_conserve_scatter_mean(atom_topo_embedding, block_ids, dim=0) # [Nblock, h]
        # H_2d = H_2d + self.block_type_embedding(S)
        # H_2d = torch.where(cond_mask_2d.unsqueeze(-1), H_2d, torch.zeros_like(H_2d))
        # block_row, block_col = block_ids[bonds[0]], block_ids[bonds[1]]
        # select_mask = (block_row != block_col) & cond_mask_2d[block_row] & cond_mask_2d[block_col]
        # row, col = bonds[0][select_mask], bonds[1][select_mask]
        # row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
        # E_2d = (
        #     torch.stack([block_ids[row], block_ids[col]], dim=0), # [2, E2d]
        #     torch.cat([atom_topo_embedding[row], atom_topo_embedding[col]], dim=-1).T # [2*h, E2d]
        # )

        # coordinate condition
        cond_ratio = torch.rand(lengths.shape[0], device=lengths.device)[batch_ids] * self.max_cond_ratio_3d
        cond_mask_3d = (torch.rand_like(generate_mask, dtype=torch.float) < cond_ratio) & generate_mask & cond_mask # 1 for given as condition
        # Z_3d = torch.where(cond_mask_3d.unsqueeze(-1), Z, torch.zeros_like(Z))

        # # pairwise distance condition
        # com = scatter_mean(X, block_ids, dim=0, dim_size=Z.shape[0]) # [Nblock, 3], center of mass
        # row, col = variadic_meshgrid(
        #     input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
        #     size1=lengths,
        #     input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
        #     size2=lengths,
        # ) # (row, col)
        # select_mask = generate_mask[row] & generate_mask[col] & (row < col)
        # row, col = row[select_mask], col[select_mask]
        # dist = torch.norm(com[row] - com[col]) # [Edist]
        # E_dist = (torch.zeros(2, 0, device=Z.device, dtype=torch.long), torch.zeros(0, device=Z.device, dtype=torch.float))

        config = ConditionConfig(
            mask_2d=cond_mask_2d,
            mask_3d=cond_mask_3d,
            mask_incomplete_2d=block_topo_drop_mask,
            w=None
        )
        
        return config_to_condition(config, com, block_lengths, bonds, atom_topo_embedding, block_embedding)

        # return Conditions(
        #     H_2d=H_2d,
        #     mask_2d=cond_mask_2d,
        #     E_2d=E_2d, # bidirectional
        #     Z_3d=Z_3d,
        #     mask_3d=cond_mask_3d,
        #     E_dist=E_dist
        # )

    def topo_embedding(self, A, bonds, block_ids, ctx_mask):
        ctx_mask = ctx_mask[block_ids]

        # only retain bonds in the context
        bond_select_mask = ctx_mask[bonds[:, 0]] & ctx_mask[bonds[:, 1]]
        bonds = bonds[bond_select_mask]

        # embed bond type
        edge_attr = self.bond_embed(bonds[:, 2])
        
        # embed atom type
        H = self.atom_embed(A)

        # get topo embedding
        topo_embedding = self.topo_gnn(H, bonds[:, :2].T, edge_attr) # [Natom]

        return topo_embedding

        # aggregate to each block
        atom_topo_embedding = topo_embedding
        topo_embedding = std_conserve_scatter_mean(topo_embedding, block_ids, dim=0) # [Nblock]

        # set generation part to zero
        topo_embedding = torch.where(
            generate_mask[:, None].expand_as(topo_embedding),
            torch.zeros_like(topo_embedding),
            topo_embedding
        )

        return topo_embedding, atom_topo_embedding

    def _normalize_position(self, X, batch_ids, center_mask):
        # TODO: pass in centers from dataset, which might be better for antibody (custom center)
        centers = scatter_mean(X[center_mask], batch_ids[center_mask], dim=0, dim_size=batch_ids.max() + 1) # [bs, 3]
        centers = centers[batch_ids] # [N, 3]
        X = (X - centers) / self.std
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids):
        X = X_norm * self.std + centers
        return X

    @torch.no_grad()
    def sample(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for calculating complex mass center
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            condition_config=None,# conditions on 2D topo, 3D coords, and pairwise distances
            sample_opt={
                'pbar': False,
                'ddim': False,
                'ddim_steps': None,
                'use_jacobian': False, # for likelihood module
                'cal_likelihood': True
                # 'energy_func': None,
                # 'energy_lambda': 0.0,
            },
            # topo_generate_mask=None,
            return_tensor=False,
            confidence_model=None,
            _overwrite_inf_nan=False
        ):

        vae_decode_n_iter = sample_opt.pop('vae_decode_n_iter', 10)
        vae_disable_avoid_clash = sample_opt.pop('vae_disable_avoid_clash', False)

        batch_ids = length_to_batch_id(lengths)
        block_ids = length_to_batch_id(block_lengths)

        if condition_config is None: topo_non_cond_mask = generate_mask
        else: topo_non_cond_mask = (~condition_config.mask_2d) & generate_mask
        # if topo_generate_mask is None: topo_generate_mask = generate_mask
        if condition_config is None: coord_non_cond_mask = generate_mask
        else: coord_non_cond_mask = (~condition_config.mask_3d) & generate_mask

        # ensure there is no data leakage
        S[topo_non_cond_mask] = 0
        X[coord_non_cond_mask[block_ids]] = 0
        A[topo_non_cond_mask[block_ids]] = 0
        ctx_atom_mask = ~topo_non_cond_mask[block_ids]
        bonds = bonds[ctx_atom_mask[bonds[:, 0]] & ctx_atom_mask[bonds[:, 1]]]

        # encoding context
        self.autoencoder.eval()
        Zh, Zx, _, signed_Zx_log_var, _, _, _, _ = self.autoencoder.encode(
            X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
        ) # [Nblock, d_latent], [Nblock, 3]
        
        # CoM for 3D conditions
        com = scatter_mean(X, block_ids, dim=0, dim_size=block_lengths.shape[0])
        com[center_mask] = Zx[center_mask] # to ensure the same normalization as Zx
        com, _ = self._normalize_position(com, batch_ids, center_mask)

        # normalize
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # topo embedding for structure prediction
        atom_topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), ~topo_non_cond_mask)
        
        # position embedding
        position_embedding = self.position_encoding(position_ids)

        # is aa embedding
        is_aa_embedding = self.is_aa_embed(is_aa.long())
        
        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, is_aa_embedding], dim=-1))

        # form conditions
        if condition_config is not None:
            block_topo_embedding = scatter_sum(atom_topo_embedding, block_ids, dim=0, dim_size=block_lengths.shape[0])
            block_type_embedding = self.block_type_embedding(S)
            block_type_embedding = torch.where(condition_config.mask_incomplete_2d.unsqueeze(-1), torch.zeros_like(block_type_embedding), block_type_embedding)
            block_embedding = self.topo_cond_linear(torch.cat([block_topo_embedding, block_type_embedding], dim=-1))
            conditions = config_to_condition(condition_config, com, block_lengths, bonds, atom_topo_embedding, block_embedding)
        else: conditions = None
        
        # calculate ddim likelihood or not 
        cal_likelihood = sample_opt.pop('cal_likelihood', True)
        use_jacobian = sample_opt.pop('use_jacobian', False) 

        if sample_opt.pop('ddim', False):
            traj = self.diffusion.sample_ddim(
                H=Zh,
                X=Zx,
                cond_embedding=cond_embedding,
                chain_ids=chain_ids,
                generate_mask=generate_mask,
                lengths=lengths,
                conditions=conditions,
                **sample_opt
            )
        else:
            traj = self.diffusion.sample(
                H=Zh,
                X=Zx,
                cond_embedding=cond_embedding,
                chain_ids=chain_ids,
                generate_mask=generate_mask,
                lengths=lengths,
                conditions=conditions,
                **sample_opt
            )
        X_0, H_0 = traj[0]
        X_0 = torch.where(generate_mask[:, None].expand_as(X_0), X_0, Zx)
        H_0 = torch.where(generate_mask[:, None].expand_as(H_0), H_0, Zh)

        # unnormalize
        X_0 = self._unnormalize_position(X_0, centers, batch_ids)
        
        if _overwrite_inf_nan:  # stop assertion if only tensor is required
            # statistics of number of nan and inf in X_0 and H_0
            X_0 = torch.where(torch.isnan(X_0), torch.zeros_like(X_0), X_0)
            H_0 = torch.where(torch.isnan(H_0), torch.zeros_like(H_0), H_0)
            X_0 = torch.where(torch.isinf(X_0), torch.zeros_like(X_0), X_0)
            H_0 = torch.where(torch.isinf(H_0), torch.zeros_like(H_0), H_0)
        else:
            # assertion check, sometimes generation will fail with large CFG w
            assert not torch.any(torch.isnan(X_0))
            assert not torch.any(torch.isnan(H_0))
            assert not torch.any(torch.isinf(X_0))
            assert not torch.any(torch.isinf(H_0))

        # autodecoder decode
        if condition_config is None: topo_generate_mask = generate_mask
        else: topo_generate_mask = ((~condition_config.mask_2d) | (condition_config.mask_incomplete_2d)) & generate_mask
        res = self.autoencoder.generate(
            X=X, S=S, A=A, bonds=bonds, position_ids=position_ids,
            chain_ids=chain_ids, generate_mask=generate_mask, block_lengths=block_lengths,
            lengths=lengths, is_aa=is_aa, given_latent=(H_0, X_0, None),
            n_iter=vae_decode_n_iter, topo_generate_mask=topo_generate_mask,
            return_tensor=return_tensor, confidence_model=confidence_model,
            disable_avoid_clash=vae_disable_avoid_clash
        )

        if cal_likelihood:    # by default calculate likelihood
            with torch.set_grad_enabled(True):
                if use_jacobian:
                    ddim_likelihood = self.cal_sample_likelihood_jacobian(
                        traj=traj,
                        cond_embedding=cond_embedding,
                        chain_ids=chain_ids,
                        generate_mask=generate_mask,
                        lengths=lengths,
                        conditions=conditions
                    )
                else:
                    ddim_likelihood = self.cal_sample_likelihood(
                        traj=traj,
                        cond_embedding=cond_embedding,
                        chain_ids=chain_ids,
                        generate_mask=generate_mask,
                        lengths=lengths,
                        conditions=conditions
                    )
            for ll_dict, ddim_ll in zip(res[3], ddim_likelihood): # res[3] is batch_ll, which records ranking values
                ll_dict['likelihood'] = ddim_ll
        
        return res
    
    ########## Below Are Functions for Likelihoods ##########

    def cal_sample_likelihood(
        self,
        traj,
        cond_embedding,
        chain_ids,
        generate_mask,
        lengths,
        conditions
    ):
        starts = torch.where(generate_mask & ~generate_mask.roll(1))[0]  
        ends = torch.where(generate_mask & ~generate_mask.roll(-1))[0] 
        delta_log_prob_delta = torch.tensor([0.0 for i in range(len(starts))])
        delta_log_prob_delta = delta_log_prob_delta.to(cond_embedding.device)
        for t in traj.keys():
            if t == 0:
                continue
            ## 2.1 get model pred
            x_t, h_t = traj[t]
            x_t = x_t.to(cond_embedding.device)
            h_t = h_t.to(cond_embedding.device)
            x_t.requires_grad_(True)  
            h_t.requires_grad_(True)  
            
            eps_H, eps_X = self.diffusion.perform_single_step_diffusion(
                h_t=h_t, 
                x_t=x_t, 
                t=t, 
                cond_embedding=cond_embedding,
                chain_ids=chain_ids,
                generate_mask=generate_mask,
                lengths=lengths,
                conditions=conditions
            )
            assert eps_H.requires_grad and eps_X.requires_grad, "Diffusion output has no gradients!"

            ## 2.2 get the output of score fn, i.e  \nabla_x log p_t(x_t)
            ## pred_type=eps
            vs = self.diffusion.trans_h.var_sched
            alpha = vs.alphas[t].clamp_min(
                vs.alphas[-2]
            )

            beta = vs.betas[t]
            alpha_bar = vs.alpha_bars[t]
            alpha_bar_pre = vs.alpha_bars[t-1]
            sigma = vs.sigmas[t]

            xt_u_value = - 1/2 * beta * x_t + 1/2 * beta / torch.sqrt(1-alpha_bar) * eps_X 
            ht_u_value = - 1/2 * beta * h_t + 1/2 * beta / torch.sqrt(1-alpha_bar) * eps_H 

            ## 2.4 calculate div u_value
            sample_times = 5
            
            for i in range(sample_times):
                z_x = torch.randn_like(x_t)
                z_x[~generate_mask, :] = 0
                z_h = torch.randn_like(h_t)
                z_h[~generate_mask, :] = 0
                 # print(z_x.shape, z_h.shape) torch.Size([36, 14, 3]) torch.Size([36, 8])
    
                sum_output = torch.sum(ht_u_value * z_h) + torch.sum(xt_u_value * z_x) 
                sum_output.backward(retain_graph=True)

                grad_x = x_t.grad 
                div_value_x = torch.mean(grad_x * z_x, dim=tuple(range(1, x_t.dim())))  
                grad_h = h_t.grad 
                div_value_h = torch.mean(grad_h * z_h, dim=tuple(range(1, h_t.dim()))) 
                for kk, (start, end) in enumerate(zip(starts, ends)):
                    x_segment_sum = div_value_x[start:end+1].sum(0)  
                    h_segment_sum = div_value_h[start:end+1].sum(0)  
                    delta_log_prob_delta[kk] += (torch.sum(x_segment_sum + h_segment_sum).item()) / sample_times 

        delta_log_prob_delta = [item.item() / (len(traj)-1) for item in delta_log_prob_delta] 
        return tuple(delta_log_prob_delta)
    
    def cal_sample_likelihood_jacobian(
        self,
        traj,
        cond_embedding,
        chain_ids,
        generate_mask,
        lengths,
        conditions
    ):
        starts = torch.where(generate_mask & ~generate_mask.roll(1))[0] 
        ends = torch.where(generate_mask & ~generate_mask.roll(-1))[0] 
        delta_log_prob_delta = torch.tensor([0.0 for i in range(len(starts))])
        delta_log_prob_delta = delta_log_prob_delta.to(cond_embedding.device)
        for t in traj.keys():
            if t == 0:
                continue
            ## 2.1 get model pred
            x_t, h_t = traj[t]
            x_t = x_t.to(cond_embedding.device)
            h_t = h_t.to(cond_embedding.device)
            x_t.requires_grad_(True)  
            h_t.requires_grad_(True)

        def get_velocity(h_t: torch.Tensor, x_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            eps_H, eps_X = self.diffusion.perform_single_step_diffusion(
                h_t=h_t, 
                x_t=x_t, 
                t=t, 
                cond_embedding=cond_embedding,
                chain_ids=chain_ids,
                generate_mask=generate_mask,
                lengths=lengths,
                conditions=conditions
            )
            assert eps_H.requires_grad and eps_X.requires_grad, "Diffusion output has no gradients!"

            ## 2.2 get the output of score fn, i.e  \nabla_x log p_t(x_t)
            ## pred_type=eps
            vs = self.diffusion.trans_h.var_sched
            alpha = vs.alphas[t].clamp_min(
                vs.alphas[-2]
            )
            alpha_bar = vs.alpha_bars[t]
            alpha_bar_pre = vs.alpha_bars[t-1]
            sigma = vs.sigmas[t]
            xt_u_value = torch.sqrt(alpha_bar) * eps_X - torch.sqrt(1 - alpha_bar) * x_t
            ht_u_value = torch.sqrt(alpha_bar) * eps_H - torch.sqrt(1 - alpha_bar) * h_t
            return ht_u_value, xt_u_value
        ## 2.4 calculate div u_value
        jacobian_h_x = jacobian(get_velocity, inputs=(h_t, x_t), vectorize=True)
        jacobian_h_h = jacobian_h_x[0][0]
        jacobian_x_x = jacobian_h_x[1][1]
        for kk, (start, end) in enumerate(zip(starts, ends)):
            jacobian_h_h_seg = jacobian_h_h[start:end + 1, :, start:end + 1, :]
            jacobian_x_x_seg = jacobian_x_x[start:end + 1, :, start:end + 1, :]
            
            jacobian_h_h_seg_flatten = jacobian_h_h_seg.flatten(0, 1).flatten(-2, -1)
            jacobian_x_x_seg_flatten = jacobian_x_x_seg.flatten(0, 1).flatten(-2, -1)
            div_h_h = jacobian_h_h_seg_flatten.diagonal().sum()
            div_x_x = jacobian_x_x_seg_flatten.diagonal().sum()
            delta_log_prob_delta[kk] += div_h_h + div_x_x 

        delta_log_prob_delta = [item.item() for item in delta_log_prob_delta] 
        return tuple(delta_log_prob_delta)
