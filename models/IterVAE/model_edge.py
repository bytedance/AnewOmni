# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean, scatter_sum, scatter_min

from data.bioparse import VOCAB, const
from utils.nn_utils import SinusoidalPositionEmbedding, expand_like, SinusoidalTimeEmbeddings, graph_to_batch_nx
from utils.gnn_utils import length_to_batch_id, std_conserve_scatter_mean, scatter_sort
import utils.register as R
from utils.oom_decorator import oom_decorator

from .map import block_to_atom_map
from .tools import _avoid_clash

from ..modules.GET.tools import fully_connect_edges, knn_edges
from ..modules.nn import BlockEmbedding, MLP
from ..modules.create_net import create_net
from ..modules.metrics import batch_accu


def _random_mask(batch_ids, sigma=0.15):
    '''
        Get random mask position, with mask ratio 68% within 1 sigma, 95% within 2 sigma
    '''
    w = torch.empty(batch_ids.max() + 1, device=batch_ids.device) # [batch_size]
    # 68% within 1sigma (0.15), 95% within 2sigma (0.30)
    if sigma < 1e-5: # eps, near zero
        mask_ratio = w * 0.0
    else:
        mask_ratio = torch.abs(nn.init.trunc_normal_(w, mean=0.0, std=sigma, a=-1.0, b=1.0))
    mask = torch.rand_like(batch_ids, dtype=torch.float) < mask_ratio[batch_ids]
    return mask


def _contrastive_loss(x_repr, y_repr, reduction='none'):
    '''
        Args:
            x_repr: [bs, hidden_size]
            y_repr: [bs, hidden_size]
        Returns:
            loss: [bs] if reduction == 'none' else [1] (e.g. sum, mean)
    '''
    if x_repr.shape[0] == 0: return 0.0
    x2y = torch.matmul(x_repr, y_repr.T) # [bs, bs], from x to y
    label = torch.arange(x_repr.shape[0], device=x_repr.device)
    x2y_loss = F.cross_entropy(input=x2y, target=label, reduction=reduction) # [bs] or [1]
    y2x_loss = F.cross_entropy(input=x2y.T, target=label, reduction=reduction) # [bs] or [1]
    return 0.5 * (x2y_loss + y2x_loss)


def _contrastive_accu(x_repr, y_repr):
    if x_repr.shape[0] == 0: return 1.0, 1.0
    x2y = torch.matmul(x_repr, y_repr.T) # [bs, bs], from x to y
    label = torch.arange(x_repr.shape[0], device=x_repr.device)
    x2y_accu = (torch.argmax(x2y, dim=-1) == label).long().sum() / len(label)
    y2x_accu = (torch.argmax(x2y.T, dim=-1) == label).long().sum() / len(label)
    return x2y_accu, y2x_accu


def _local_distance_loss(X_gt, X_t, t, batch_ids, block_ids, generate_mask, dist_th=6.0, t_th=0.25):
    with torch.no_grad():
        row, col = fully_connect_edges(batch_ids[block_ids])

        # at least one end is in generation part, and don't repeat the same pair
        select_mask = (generate_mask[block_ids[row]] | generate_mask[block_ids[col]]) & (row < col)
        row, col = row[select_mask], col[select_mask]

        # get distance within 6.0 A
        dist = torch.norm(X_gt[row] - X_gt[col], dim=-1) # [E]
        select_mask = (dist < dist_th)

        row, col, dist = row[select_mask], col[select_mask], dist[select_mask]

    # MSE
    dist_t = torch.norm(X_t[row] - X_t[col], dim=-1)
    loss = F.smooth_l1_loss(dist, dist_t, reduction='none')
    
    # only add loss on t steps below 0.25 (near data state)
    loss = loss[(t[row] < t_th)] # t[row] should be equal to t[col] since row and col are in the same batch
    return loss.mean() if len(loss) else 0


def _bond_length_loss(X_gt, X_t, t, bonds, block_ids, generate_mask, t_th=0.25):
    with torch.no_grad():
        generate_mask = generate_mask[block_ids]
        bond_mask = generate_mask[bonds[:, 0]] & generate_mask[bonds[:, 1]]
        bonds = bonds[bond_mask]
        row, col = bonds[:, 0], bonds[:, 1]
        dist = torch.norm(X_gt[row] - X_gt[col], dim=-1)

    # MSE
    dist_t = torch.norm(X_t[row] - X_t[col], dim=-1)
    loss = F.smooth_l1_loss(dist, dist_t, reduction='none')
    
    # only add loss on t steps below 0.25 (near data state)
    loss = loss[(t[row] < t_th)] # t[row] should be equal to t[col] since row and col are in the same batch
    return loss.mean() if len(loss) else 0


# conditional autoregressive autoencoder
@R.register('CondIterAutoEncoderEdge')
class CondIterAutoEncoderEdge(nn.Module):
    def __init__(
            self,
            encoder_type: str,
            decoder_type: str,
            embed_size,
            hidden_size,
            latent_size,
            edge_size,
            num_block_type = VOCAB.get_num_block_type(),
            num_atom_type = VOCAB.get_num_atom_type(),
            num_bond_type = 5, # [None, single, double, triple, aromatic]
            max_num_atoms = const.aa_max_n_atoms,
            k_neighbors = 9,
            encoder_opt = {},
            decoder_opt = {},
            loss_weights = {
                'Zh_kl_loss': 0.3,
                'Zx_kl_loss': 0.5,
                'atom_coord_loss': 1.0,
                'block_type_loss': 1.0,
                'contrastive_loss': 0.3,
                'local_distance_loss': 0.5,
                'bond_loss': 0.5,
                'bond_length_loss': 0.0
            },
            # auxiliary parameters
            prior_coord_std=1.0,
            coord_noise_scale=0.0,
            pocket_mask_ratio=0.05,     # cannot be zero when kl_on_pocket=False, otherwise the latent state of the pocket will explode
            decode_mask_ratio=0.0,
            kl_on_pocket=False,         # whether to exert kl regularization also on pocket nodes
            discrete_timestep=False,
            default_num_steps=10,
            encode_atom_only=False
        ) -> None:
        super().__init__()
        self.encoder = create_net(encoder_type, hidden_size, edge_size, encoder_opt)
        self.decoder = create_net(decoder_type, hidden_size, edge_size, decoder_opt)

        self.embedding = BlockEmbedding(num_block_type, num_atom_type, embed_size, encode_atom_only)
        self.ctx_embedding = nn.Embedding(2, embed_size) # [context, generation]
        if self.encoder.require_block_edges or self.decoder.require_block_edges:
            self.edge_embedding = nn.Embedding(3, edge_size) # [intra, inter, topo]
        self.atom_edge_embedding = nn.Embedding(5, edge_size) # [None, single, double, triple, aromatic]
        self.enc_embed2hidden = nn.Linear(embed_size, hidden_size)
        self.dec_atom_embedding = nn.Embedding(num_atom_type, hidden_size)
        self.dec_pos_embedding = SinusoidalPositionEmbedding(hidden_size)
        self.dec_latent2hidden = nn.Linear(latent_size, hidden_size)
        self.dec_time_embedding = SinusoidalTimeEmbeddings(hidden_size)
        self.dec_input_linear = nn.Linear(hidden_size * 3 + latent_size, hidden_size) # atom, time, position, latent

        self.mask_embedding = nn.Parameter(torch.randn(latent_size), requires_grad=True)

        # For KL divergence
        self.Wh_mu = nn.Linear(hidden_size, latent_size)
        self.Wh_log_var = nn.Linear(hidden_size, latent_size)
        self.Wx_log_var = nn.Linear(hidden_size, 1) # has to be isotropic gaussian to maintain equivariance

        # For output
        self.block_type_mlp = MLP(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=num_block_type,
            n_layers=3,
            dropout=0.1
        )
        self.bond_type_mlp = MLP(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=num_bond_type,
            n_layers=3,
            dropout=0.1
        )

        self.max_num_atoms = max_num_atoms
        self.k_neighbors = k_neighbors
        self.loss_weights = loss_weights
        self.prior_coord_std = prior_coord_std
        self.coord_noise_scale = coord_noise_scale
        self.pocket_mask_ratio = pocket_mask_ratio
        self.decode_mask_ratio = decode_mask_ratio
        self.kl_on_pocket = kl_on_pocket
        self.discrete_timestep = discrete_timestep
        self.default_num_steps = default_num_steps

    @property
    def latent_size(self):
        return self.mask_embedding.shape[0]
    
    @property
    def hidden_size(self):
        return self.enc_embed2hidden.weight.shape[0]
    
    @oom_decorator
    def forward(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3) (single-directional)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            warmup_progress=1.0, # increasing KL from 0% to 100% for training stability
            **kwargs
        ):
        # backup ground truth
        X_gt, S_gt, A_gt = X.clone(), S.clone(), A.clone()
        block_ids = length_to_batch_id(block_lengths)
        batch_ids = length_to_batch_id(lengths)
        batch_size = lengths.shape[0]

        # sample binding site mask prediction (0% to 30%)
        binding_site_gen_mask = _random_mask(batch_ids, sigma=self.pocket_mask_ratio) & (~generate_mask)

        # encoding
        Zh, Zx, Zx_mu, signed_Zx_log_var, Zh_kl_loss, Zx_kl_loss, bind_site_repr, ligand_repr = self.encode(
            X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, binding_site_gen_mask=binding_site_gen_mask
        ) # [Nblock, d_latent], [Nblock, 3], [1], [1]
        # Zx_log_var = -torch.abs(signed_Zx_log_var).expand_as(Zx_mu)

        if self.training: # add noise on Zx to improve robustness
            noise = torch.randn_like(Zx) * self.coord_noise_scale
            noise = torch.where(expand_like(generate_mask, Zx), noise, torch.zeros_like(noise))
            Zx = Zx + noise

        # random mask some Zh for contextual prediction
        Zh = self._random_mask(Zh, generate_mask, batch_ids)

        # decode block types from latent graph
        pred_block_logits = self.decode_block_type(Zh, Zx, chain_ids, lengths)

        # sample a timestep and decode structure
        # sample a timestep
        if self.discrete_timestep:
            t = torch.randint(1, self.default_num_steps + 1, size=(batch_size,), device=X.device)[batch_ids][block_ids]
            t = t.float() / self.default_num_steps
        else:
            t = torch.rand(batch_size, device=X.device)[batch_ids][block_ids]
        # sample an initial state
        X_t, vector = self._sample_from_prior(X, Zx, block_ids, generate_mask | binding_site_gen_mask, t)
        # get topo edges
        topo_edges, topo_edge_type = self._get_topo_edges(bonds, block_ids, generate_mask)
        # sample some ground-truth inter-block topo edges
        inter_topo_edges, inter_topo_edge_type, inter_topo_edge_select_mask = self._unmask_inter_topo_edges(bonds, batch_ids, block_ids, generate_mask)
        topo_edges = torch.cat([topo_edges, inter_topo_edges.T[inter_topo_edge_select_mask].T], dim=1)
        topo_edge_type = torch.cat([topo_edge_type, inter_topo_edge_type[inter_topo_edge_select_mask]], dim=0)

        topo_edge_attr = self.atom_edge_embedding(topo_edge_type)
        # decode structure
        H_t, X_next = self.decode_structure(Zh, X_t, A, position_ids, topo_edges, topo_edge_attr, chain_ids, batch_ids, block_ids, t)
        pred_vector = X_next - X_t
        X_final = X_t + pred_vector * t[:, None] # for local neighborhood
        # decode inter-block bonds
        given_bond_num = (~inter_topo_edge_select_mask).sum() // 2 # bidirectional to one-directional
        bond_to_pred, bond_label = self._get_bond_to_pred(
            X_t, bonds, batch_ids, block_ids, generate_mask,
            given_gt=(
                inter_topo_edges.T[~inter_topo_edge_select_mask][:given_bond_num].T,
                inter_topo_edge_type[~inter_topo_edge_select_mask][:given_bond_num]
            ))
        pred_bond_logits = self.bond_type_mlp(H_t[bond_to_pred[0]] + H_t[bond_to_pred[1]]) # commutative property
        bond_loss = F.cross_entropy(input=pred_bond_logits, target=bond_label)
        if torch.isnan(bond_loss): bond_loss = 0.0
        # TODO: maybe we need to add distance supervision on inter-block bonds

        # loss
        loss_mask = generate_mask[block_ids]
        binding_site_loss_mask = binding_site_gen_mask[block_ids]
        contrastive_loss_mask = scatter_sum(~generate_mask, batch_ids, dim=0) > 0 # has target
        loss_dict = {
            'Zh_kl_loss': Zh_kl_loss,
            'Zx_kl_loss': Zx_kl_loss,
            'block_type_loss': F.cross_entropy(
                input=pred_block_logits[generate_mask | binding_site_gen_mask],
                target=S_gt[generate_mask | binding_site_gen_mask]
            ).clamp_max(10.),
            'atom_coord_loss': F.mse_loss(
                pred_vector[loss_mask | binding_site_loss_mask],
                vector[loss_mask | binding_site_loss_mask], reduction='none'
            ).sum(-1).mean(), # sum the xyz dimension and average between atoms
            'contrastive_loss': _contrastive_loss(bind_site_repr[contrastive_loss_mask], ligand_repr[contrastive_loss_mask], reduction='mean'),
            'local_distance_loss': _local_distance_loss(X_gt, X_final, t, batch_ids, block_ids, generate_mask | binding_site_gen_mask),
            'bond_loss': bond_loss,
            'bond_length_loss': _bond_length_loss(X_gt, X_final, t, bonds, block_ids, generate_mask)
        }

        total = 0
        for name in self.loss_weights:
            weight = self.loss_weights[name]
            if 'kl_loss' in name: weight *= min(warmup_progress, 1.0)
            total = total + loss_dict[name] * weight
        loss_dict['total'] = total

        # for evaluation
        with torch.no_grad():
            loss_dict.update({
                'block_type_accu': batch_accu(
                    pred_block_logits[generate_mask],
                    S_gt[generate_mask],
                    batch_ids[generate_mask],
                    reduction='mean'
                ),
            })
            x2y_accu, y2x_accu = _contrastive_accu(bind_site_repr[contrastive_loss_mask], ligand_repr[contrastive_loss_mask])
            loss_dict.update({
                'bind_site_to_ligand_accu': x2y_accu,
                'ligand_to_bind_site_accu': y2x_accu,
            })
            loss_dict.update({
                'bond_accu': (torch.argmax(pred_bond_logits, dim=-1) == bond_label).long().sum() / (len(bond_label) + 1e-10)
            })
            # record deviation of Zx and centers
            block_centers = scatter_mean(X, block_ids, dim=0, dim_size=block_ids.max() + 1) # [Nblock, 3]
            zx_rmsd = ((block_centers - Zx_mu) ** 2).sum(-1) # [Nblock]
            loss_dict.update({
                'pocket_zx_center_rmsd': torch.sqrt(scatter_mean(zx_rmsd[~generate_mask], batch_ids[~generate_mask], dim=0)).mean(),
                'ligand_zx_center_rmsd': torch.sqrt(scatter_mean(zx_rmsd[generate_mask], batch_ids[generate_mask], dim=0)).mean(),
            })
            # record norm of Zh
            zh_norm = torch.norm(Zh, dim=-1) # [Nblock]
            loss_dict.update({
                'pocket_zh_norm': zh_norm[~generate_mask].mean(),
                'ligand_zh_norm': zh_norm[generate_mask].mean()
            })
            # record std of Zh and Zx
            zx_std = torch.exp(-torch.abs(signed_Zx_log_var) / 2) # sigma
            loss_dict.update({
                'pocket_zx_std': zx_std[~generate_mask].mean(),
                'ligand_zx_std': zx_std[generate_mask].mean()
            })
        # print(loss_dict)
        return loss_dict
    
    def encode(self, X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=False, binding_site_gen_mask=None):

        batch_ids = length_to_batch_id(lengths)
        block_ids = length_to_batch_id(block_lengths)

        # H = self.embedding(S, A, block_ids) #+ self.rel_pos_embedding(position_ids)[block_ids] # [Natom, embed_size]
        H = self.embedding(S, A, block_ids) + self.ctx_embedding(generate_mask[block_ids].long())
        H = self.enc_embed2hidden(H) # [Natom, hidden_size]
        if self.encoder.require_block_edges:
            edges, edge_type = self.get_edges(batch_ids, chain_ids, X, block_ids, generate_mask, allow_gen_to_ctx=False, allow_ctx_to_gen=False)
            edge_attr = self.edge_embedding(edge_type)
        else: edges, edge_attr = None, None
        attn_mask = self.get_attn_mask(batch_ids, block_ids, generate_mask) # forbid context attending to generated part during encoding

        # make bonds bidirectional
        bond_row, bond_col, bond_type = bonds.T

        Zh_atom, Zx_atom = self.encoder(
            H, X, block_ids, batch_ids, edges, edge_attr,
            topo_edges=torch.stack([
                torch.cat([bond_row, bond_col], dim=0),
                torch.cat([bond_col, bond_row], dim=0)
            ], dim=0),
            topo_edge_attr=self.atom_edge_embedding(torch.cat([bond_type, bond_type], dim=0)),
            attn_mask=attn_mask
        ) # [Natom, hidden_size], [Natom, 3]
        Zh = std_conserve_scatter_mean(Zh_atom, block_ids, dim=0) # [Nblock, hidden_size]
        Zx = scatter_mean(Zx_atom, block_ids, dim=0) # [Nblock, 3], not use std_conserve_scatter_mean for equivariance
        # assert not torch.any(torch.isnan(Zh)), 'encoding Zh nan'
        # print(Zh[generate_mask][0], Zx[generate_mask][0])

        # contrastive learning between Binding Site and Ligand
        batch_size = batch_ids.max() + 1
        bind_site_repr = scatter_mean(Zh[~generate_mask], batch_ids[~generate_mask], dim=0, dim_size=batch_size) # [bs, hidden_size]
        ligand_repr = scatter_mean(Zh[generate_mask], batch_ids[generate_mask], dim=0, dim_size=batch_size) # [bs, hidden_size]

        # rsample: for coordinates, prior at the center of each block? or do not add constraints?
        Zx_prior_mu = scatter_mean(X, block_ids, dim=0) # [Nblock, 3]
        # Zx_prior_mu = None # do not add constraints
        Zx_mu, signed_Zx_log_var = Zx.clone(), self.Wx_log_var(Zh)
        Zh, Zx, Zh_kl_loss, Zx_kl_loss = self.rsample(Zh, Zx, generate_mask, Zx_prior_mu, deterministic, binding_site_gen_mask)

        return Zh, Zx, Zx_mu, signed_Zx_log_var, Zh_kl_loss, Zx_kl_loss, bind_site_repr, ligand_repr
    
    def decode_block_type(self, Zh, Zx, chain_ids, lengths):
        '''
            Args:
                Zh: [Nblock, latent_size]
                Zx: [Nblock, 3]
                chain_ids: [Nblock]
                lengths: [batch_size]
            Returns:
                pred_block_logits: [Nblock, n_block_type]
        '''
        batch_ids = length_to_batch_id(lengths)
        latent_block_ids = torch.ones_like(batch_ids).cumsum(dim=-1) - 1
        if self.decoder.require_block_edges:
            edges, edge_type = self.get_edges(batch_ids, chain_ids, Zx, latent_block_ids, None, True, True)
            edge_attr = self.edge_embedding(edge_type)
        else: edges, edge_attr = None, None
        H = self.dec_latent2hidden(Zh)
        block_h, _ = self.decoder(H, Zx, latent_block_ids, batch_ids, edges, edge_attr)
        pred_block_logits = self.block_type_mlp(block_h)
        return pred_block_logits

    def decode_structure(self, Zh, X, A, position_ids, topo_edges, topo_edge_attr, chain_ids, batch_ids, block_ids, t):
        '''
            Args:
                Zh: [Nblock, latent_size]
                X: [Natom, 3]
                A: [Natom]
                position_ids: [Nblock], only work for proteins/peptides. For small molecules, they are the same
                topo_edges: [2, Etopo]
                topo_edge_attr: [Etopo, edge_size]
                chain_ids: [Nblock]
                batch_ids: [Nblock]
                block_ids: [Natom]
                t: [Natom]
        '''
        # decode atom-level structures
        if self.decoder.require_block_edges:
            edges, edge_type = self.get_edges(batch_ids, chain_ids, X, block_ids, None, True, True)
            edge_attr = self.edge_embedding(edge_type)
        else: edges, edge_attr = None, None
        H = self.dec_input_linear(torch.cat([
            self.dec_atom_embedding(A), self.dec_time_embedding(t), self.dec_pos_embedding(position_ids[block_ids]), Zh[block_ids]
        ], dim=-1)) # [Natom, hidden_size]
        H_t, X_next = self.decoder(H, X, block_ids, batch_ids, edges, edge_attr, topo_edges, topo_edge_attr) # [Natom', hidden_size], [Natom', 3]
        
        return H_t, X_next
    
    def _random_mask(self, Zh, generate_mask, batch_ids):
        Zh = Zh.clone()

        mask = _random_mask(batch_ids, sigma=self.decode_mask_ratio)
        mask = mask & generate_mask

        Zh[mask] = self.mask_embedding
        return Zh

    def _sample_from_prior(self, X, Zx_mu, block_ids, generate_mask, t):
        atom_generate_mask = expand_like(generate_mask[block_ids], X)
        Zx_mu = Zx_mu[block_ids]

        # sample random noise (directly use gaussian)
        noise = torch.randn_like(X) * self.prior_coord_std
        
        # sample each atom from block prior (only the generation part)
        X_init = torch.where(atom_generate_mask, Zx_mu + noise, X)

        # vector
        vector = X - X_init

        # state at timestep t (0.0 is the data, 1.0 is the prior)
        X_t = torch.where(atom_generate_mask, X_init + vector * (1 - t)[..., None], X)

        return X_t, vector
    
    @torch.no_grad()
    def _get_inter_block_nbh(self, X_t, batch_ids, block_ids, topo_generate_mask, dist_th):
        # local neighborhood for negative bonds
        row, col = fully_connect_edges(batch_ids[block_ids])

        # inter-block and at least one end is in topo generation part, and row < col to avoid repeated bonds
        select_mask = (block_ids[row] != block_ids[col]) & (topo_generate_mask[block_ids[row]] | topo_generate_mask[block_ids[col]]) * (row < col)
        row, col = row[select_mask], col[select_mask]

        # get edges within 3.5A
        select_mask = torch.norm(X_t[row] - X_t[col], dim=-1) < dist_th
        row, col = row[select_mask], col[select_mask]

        return torch.stack([row, col], dim=0) # [2, E]

    @torch.no_grad()
    def _get_bond_to_pred(self, X_t, gt_bonds, batch_ids, block_ids, generate_mask, neg_dist_th=3.5, neg_to_pos_ratio=1.0, given_gt=None):

        if given_gt is None:
            # get ground truth
            gt_row, gt_col, gt_type = gt_bonds[:, 0], gt_bonds[:, 1], gt_bonds[:, 2]
            # inter-block and at least one end is in generation part
            select_mask = (block_ids[gt_row] != block_ids[gt_col]) & (generate_mask[block_ids[gt_row]] | generate_mask[block_ids[gt_col]])
            gt_row, gt_col, gt_type = gt_row[select_mask], gt_col[select_mask], gt_type[select_mask]
        else:
            gt_row, gt_col = given_gt[0]
            gt_type = given_gt[1]

        # local neighborhood for negative bonds
        row, col = self._get_inter_block_nbh(X_t, batch_ids, block_ids, generate_mask, neg_dist_th)

        # negative sampling from local neighborhood (low possibility to coincide with postive bonds)
        if len(row) == 0: ratio = 0.1
        else: ratio = len(gt_row) / len(row) * neg_to_pos_ratio # neg:pos ~ 2:1
        select_mask = torch.rand_like(row, dtype=torch.float) < ratio
        row, col = row[select_mask], col[select_mask]
        neg_type = torch.zeros_like(row, dtype=torch.long)

        bonds_to_pred = torch.stack([
            torch.cat([gt_row, row], dim=0),
            torch.cat([gt_col, col], dim=0),
        ], dim=0)
        labels = torch.cat([gt_type, neg_type])

        return bonds_to_pred, labels
    
    @torch.no_grad()
    def get_edges(self, batch_ids, segment_ids, Z, block_ids, generate_mask, allow_gen_to_ctx, allow_ctx_to_gen):
        row, col = fully_connect_edges(batch_ids)
        if not allow_gen_to_ctx: # forbid message passing from generated part to context
            select_mask = generate_mask[row] | (~generate_mask[col]) # src is in generated part or dst is not in generated part
            row, col = row[select_mask], col[select_mask]
        if not allow_ctx_to_gen: # forbid message passing from context to generated part
            select_mask = (~generate_mask[row]) | (generate_mask[col])
            row, col = row[select_mask], col[select_mask]
        is_intra = segment_ids[row] == segment_ids[col]
        intra_edges = torch.stack([row[is_intra], col[is_intra]], dim=0)
        inter_edges = torch.stack([row[~is_intra], col[~is_intra]], dim=0)
        intra_edges = knn_edges(block_ids, batch_ids, Z.unsqueeze(1), self.k_neighbors, intra_edges)
        inter_edges = knn_edges(block_ids, batch_ids, Z.unsqueeze(1), self.k_neighbors, inter_edges)
        
        edges = torch.cat([intra_edges, inter_edges], dim=1)
        edge_type = torch.cat([torch.zeros_like(intra_edges[0]), torch.ones_like(inter_edges[0])])

        return edges, edge_type
    
    @torch.no_grad()
    def get_attn_mask(self, batch_ids, block_ids, generate_mask):
        '''
            Args:
                batch_ids: [Nblock]
                block_ids: [Natom]
                generate_mask: [Nblock]
            Returns:
                attn_mask: [bs, max_n_atom, max_n_atom]
        '''
        generate_mask = generate_mask[block_ids] # [Natom]
        mask, _ = graph_to_batch_nx(generate_mask, batch_ids[block_ids], padding_value=True, factor_req=8) # [bs, max_n]
        bs, N = mask.shape
        
        # Create base attention mask allowing all tokens to attend to all other tokens
        attention_mask = torch.ones(bs, N, N, dtype=torch.bool, device=mask.device)

        # Create symmetric restriction: no attention between context and generated tokens
        context_to_generated = (mask == 0).unsqueeze(2) & (mask == 1).unsqueeze(1)
        generated_to_context = (mask == 1).unsqueeze(2) & (mask == 0).unsqueeze(1)
    
        # Remove attention from context to generated and vice versa
        attention_mask = attention_mask & ~(context_to_generated | generated_to_context)

        return attention_mask

    @torch.no_grad()
    def _get_topo_edges(self, bonds, block_ids, generate_mask):
        '''
        Only used in training
            bonds: [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            block_ids: [N]
            generate_mask: [Nblock]
        '''
        row, col, bond_type = bonds.T
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0) # bidirectional
        bond_type = torch.cat([bond_type, bond_type], dim=-1) #[2*Nbonds, edge_size]

        # # TODO: for block-level: two blocks are connected if there are two atoms having chemical bonds
        # # TODO: if this is enabled, there must also be an block edge prediction module
        # inter_block = block_ids[row] != block_ids[col]
        # row_block, col_block = block_ids[row][inter_block], block_ids[col][inter_block]
        # block_topo_edges = torch.unique(torch.stack([row_block, col_block], dim=0), dim=1) # [2, Eblock]

        # for atom-level chemical bonds: only give intra-block ones for generated part, but all for context
        # intra_block = block_ids[row] == block_ids[col]
        # row, col, bond_type = row[intra_block], col[intra_block], bond_type[intra_block]
        row_block, col_block = block_ids[row], block_ids[col]
        select_mask = (row_block == col_block) | ((~generate_mask[row_block]) & (~generate_mask[col_block]))
        row, col, bond_type = row[select_mask], col[select_mask], bond_type[select_mask]
        topo_edges = torch.stack([row, col], dim=0)

        return topo_edges, bond_type
    
    @torch.no_grad()
    def _unmask_inter_topo_edges(self, bonds, batch_ids, block_ids, generate_mask):
        atom_batch_ids = batch_ids[block_ids]
        row, col, bond_type = bonds.T

        # get inter-block bonds
        row_block, col_block = block_ids[row], block_ids[col]
        select_mask = (row_block != col_block) & (generate_mask[row_block] | generate_mask[col_block])
        row, col, bond_type = row[select_mask], col[select_mask], bond_type[select_mask]

        # # sample some to provide as contexts, others for prediction
        # unmask_ratio = torch.rand(batch_ids.max() + 1, device=bonds.device)
        # select_mask = torch.rand_like(atom_batch_ids[row], dtype=torch.float) < unmask_ratio[atom_batch_ids[row]]
        # 50% cases for structure prediction, others for design
        unmask = torch.rand(batch_ids.max() + 1, device=bonds.device) < 0.5
        select_mask = unmask[atom_batch_ids[row]]

        # bi-directional
        row, col = torch.cat([row, col], dim=0), torch.cat([col, row])
        bond_type = torch.cat([bond_type, bond_type], dim=0)
        select_mask = torch.cat([select_mask, select_mask], dim=0)

        return torch.stack([row, col], dim=0), bond_type, select_mask

    def rsample(self, Zh, Zx, generate_mask, Zx_prior_mu=None, deterministic=False, binding_site_gen_mask=None):
        '''
            Zh: [Nblock, latent_size]
            Zx: [Nblock, 3]
            Zx_prior_mu: [Nblock, 3]
        '''

        if binding_site_gen_mask is not None: generate_mask = generate_mask | binding_site_gen_mask

        # if hasattr(self, 'kl_on_pocket') and self.kl_on_pocket: # also exert kl regularizations on latent points of pocket
        if self.kl_on_pocket: # also exert kl regularizations on latent points of pocket
            generate_mask = torch.ones_like(generate_mask)

        # data_size = Zh.shape[0]
        data_size = generate_mask.long().sum()

        # invariant latent features
        Zh_mu = self.Wh_mu(Zh)
        Zh_log_var = -torch.abs(self.Wh_log_var(Zh)) #Following Mueller et al., z_log_var is log(\sigma^2))
        Zh_kl_loss = -0.5 * torch.sum((1.0 + Zh_log_var - Zh_mu * Zh_mu - torch.exp(Zh_log_var))[generate_mask]) / (data_size * Zh_mu.shape[-1])
        Zh_sampled = Zh_mu if deterministic else Zh_mu + torch.exp(Zh_log_var / 2) * torch.randn_like(Zh_mu)

        # equivariant latent features
        if Zx_prior_mu is None:
            Zx_sampled, Zx_kl_loss = Zx + torch.randn_like(Zx), 0 # fix as standard gaussian
        else:
            Zx_mu_delta = Zx - Zx_prior_mu # [Nblock, 3], if perfectly from prior, the expectation should be zero
            Zx_log_var = -torch.abs(self.Wx_log_var(Zh)).expand_as(Zx)
            Zx_kl_loss = -0.5 * torch.sum((1.0 + Zx_log_var - Zx_mu_delta * Zx_mu_delta - torch.exp(Zx_log_var))[generate_mask]) / (data_size * Zx.shape[-1])
            Zx_sampled = Zx if deterministic else Zx + torch.exp(Zx_log_var / 2) * torch.randn_like(Zx)
        
        return Zh_sampled, Zx_sampled, Zh_kl_loss, Zx_kl_loss
    
    def _init_atoms(self, pred_block_type, X, A, bonds, Zx_mu, block_ids, generate_mask, topo_generate_mask=None):
        
        gt_bonds = bonds.clone()
        if topo_generate_mask is None: topo_generate_mask = generate_mask
        # # get intra-block bonds of context
        # ctx_intra_block = (block_ids[bonds[:, 0]] == block_ids[bonds[:, 1]]) & (~generate_mask[block_ids[bonds[:, 0]]])
        # bonds = bonds[ctx_intra_block] # [Ectx, 3]

        # get bonds of context
        ctx_block = (~generate_mask[block_ids[bonds[:, 0]]]) & (~generate_mask[block_ids[bonds[:, 1]]])
        bonds = bonds[ctx_block] # [Ectx, 3]

        # extract context
        atom_ctx_mask = ~generate_mask[block_ids]
        ctx_X, ctx_A, ctx_block_ids = X[atom_ctx_mask], A[atom_ctx_mask], block_ids[atom_ctx_mask]

        # order mapping of context bonds
        ctx_atom_order_map = -torch.ones_like(block_ids, dtype=torch.long) # -1 will be non-context atoms
        ctx_atom_order_map[atom_ctx_mask] = torch.arange(atom_ctx_mask.long().sum(), device=atom_ctx_mask.device)
        bonds = torch.stack([
            ctx_atom_order_map[bonds[:, 0]], ctx_atom_order_map[bonds[:, 1]], bonds[:, 2]
        ], dim=-1) # [E, 3]

        # generate atoms/bonds based on predicted blocks
        gen_A, gen_block_ids, gen_bonds = block_to_atom_map(pred_block_type[generate_mask], torch.nonzero(generate_mask).squeeze(-1))
        # gen_block_ids is mononically increasing
        # replace topo-fix atoms
        topo_fix_mask = (~topo_generate_mask) & generate_mask
        # print(gen_block_ids.shape, block_ids.shape)
        # print(gen_A.shape, A.shape)
        # print(gen_A[topo_fix_mask[gen_block_ids]].shape, A[topo_fix_mask[block_ids]].shape)
        gen_A[topo_fix_mask[gen_block_ids]] = A[topo_fix_mask[block_ids]]
        # extract topo-fix bonds (between fixed blocks in the generated part, or between one fixed block and one context block)
        topo_fix_bonds = gt_bonds[topo_fix_mask[block_ids][gt_bonds[:, 0]] & topo_fix_mask[block_ids][gt_bonds[:, 1]]]
        # extract ctx to binder (fixed part) bonds
        ctx_to_binder_bonds = gt_bonds[(~generate_mask)[block_ids][gt_bonds[:, 0]] & topo_fix_mask[block_ids][gt_bonds[:, 1]]]
        ctx_to_binder_bonds[:, 0] = ctx_atom_order_map[ctx_to_binder_bonds[:, 0]]
        # extract binder (fixed part) to ctx bonds
        binder_to_ctx_bonds = gt_bonds[topo_fix_mask[block_ids][gt_bonds[:, 0]] & (~generate_mask)[block_ids][gt_bonds[:, 1]]]
        binder_to_ctx_bonds[:, 1] = ctx_atom_order_map[binder_to_ctx_bonds[:, 1]]
        # topo_fix_bonds = gt_bonds[
        #     (topo_fix_mask[block_ids][gt_bonds[:, 0]] & topo_fix_mask[block_ids][gt_bonds[:, 1]]) | \
        #     (topo_fix_mask[block_ids][gt_bonds[:, 0]] & (~generate_mask)[block_ids][gt_bonds[:, 1]]) | \
        #     ((~generate_mask)[block_ids][gt_bonds[:, 0]] & topo_fix_mask[block_ids][gt_bonds[:, 1]])]
        # print(((topo_fix_mask[block_ids][gt_bonds[:, 0]] & (~generate_mask)[block_ids][gt_bonds[:, 1]]) | \
        #     ((~generate_mask)[block_ids][gt_bonds[:, 0]] & topo_fix_mask[block_ids][gt_bonds[:, 1]])).sum())
        # from global index to local index
        block_offsets = scatter_sum(torch.ones_like(block_ids), block_ids, dim=0).cumsum(dim=0) # [Nblock]
        block_offsets = F.pad(block_offsets[:-1], pad=(1, 0), value=0) # [Nblock]
        row_block_ids, col_block_ids = block_ids[topo_fix_bonds[:, 0]], block_ids[topo_fix_bonds[:, 1]]
        topo_fix_bonds[:, 0] = topo_fix_bonds[:, 0] - block_offsets[row_block_ids]
        topo_fix_bonds[:, 1] = topo_fix_bonds[:, 1] - block_offsets[col_block_ids]
        c2b_col_block_ids, b2c_row_block_ids = block_ids[ctx_to_binder_bonds[:, 1]], block_ids[binder_to_ctx_bonds[:, 0]]
        ctx_to_binder_bonds[:, 1] = ctx_to_binder_bonds[:, 1] - block_offsets[c2b_col_block_ids]
        binder_to_ctx_bonds[:, 0] = binder_to_ctx_bonds[:, 0] - block_offsets[b2c_row_block_ids]
        # from local index to global index defined by the generated atoms
        block_offsets = scatter_sum(torch.ones_like(gen_block_ids), gen_block_ids, dim=0).cumsum(dim=0) # [Nblock]
        block_offsets = F.pad(block_offsets[:-1], pad=(1, 0), value=0) # [Nblock]
        topo_fix_bonds[:, 0] = topo_fix_bonds[:, 0] + block_offsets[row_block_ids]
        topo_fix_bonds[:, 1] = topo_fix_bonds[:, 1] + block_offsets[col_block_ids]
        ctx_to_binder_bonds[:, 1] = ctx_to_binder_bonds[:, 1] + block_offsets[c2b_col_block_ids]
        binder_to_ctx_bonds[:, 0] = binder_to_ctx_bonds[:, 0] + block_offsets[b2c_row_block_ids]
        # delete the original bonds from block_to_atom_map
        gen_bonds = gen_bonds[~topo_fix_mask[gen_block_ids][gen_bonds[:, 0]]] # since these bonds are only intra-block
        # add the topo fix bonds
        gen_bonds = torch.cat([gen_bonds, topo_fix_bonds], dim=0) # [E1+E2, 3]
        gen_X = Zx_mu[gen_block_ids] + torch.randn_like(Zx_mu[gen_block_ids]) * self.prior_coord_std

        # concat context and generated part
        # atoms
        X, A, block_ids = torch.cat([ctx_X, gen_X], dim=0), torch.cat([ctx_A, gen_A], dim=0), torch.cat([ctx_block_ids, gen_block_ids], dim=0)
        # bonds
        ctx_row, ctx_col, ctx_bond_type = bonds[:, 0], bonds[:, 1], bonds[:, 2]
        gen_row, gen_col, gen_bond_type = gen_bonds[:, 0] + len(ctx_A), gen_bonds[:, 1] + len(ctx_A), gen_bonds[:, 2]
        c2b_row, c2b_col, c2b_bond_type = ctx_to_binder_bonds[:, 0], ctx_to_binder_bonds[:, 1] + len(ctx_A), ctx_to_binder_bonds[:, 2]
        b2c_row, b2c_col, b2c_bond_type = binder_to_ctx_bonds[:, 0] + len(ctx_A), binder_to_ctx_bonds[:, 1], binder_to_ctx_bonds[:, 2]
        bonds = torch.stack([
            torch.cat([ctx_row, ctx_col, gen_row, gen_col, c2b_row, c2b_col, b2c_row, b2c_col], dim=0), # bidirectional
            torch.cat([ctx_col, ctx_row, gen_col, gen_row, c2b_col, c2b_row, b2c_col, b2c_row], dim=0), # bidirectional
            torch.cat([ctx_bond_type, ctx_bond_type, gen_bond_type, gen_bond_type, c2b_bond_type, c2b_bond_type, b2c_bond_type, b2c_bond_type], dim=0)
        ], dim=-1) # [Ectx*2 + Egen*2, 3]
        
        # sorting
        block_ids, perm = scatter_sort(block_ids, block_ids, dim=0)
        X, A = X[perm], A[perm]
        # atom order mapping
        atom_order_map = torch.ones_like(A, dtype=torch.long)
        atom_order_map[perm] = torch.arange(len(A), device=A.device)
        bonds = torch.stack([
            atom_order_map[bonds[:, 0]],
            atom_order_map[bonds[:, 1]],
            bonds[:, 2]
        ], dim=-1)
        
        return X, A, block_ids, bonds

    def _bond_length_guidance(self, t, H_t, X_t, batch_ids, block_ids, generate_mask, dist_th=3.5, bond_th=0.9):
        
        # get inter-block bonding distribution
        row, col = self._get_inter_block_nbh(X_t, batch_ids, block_ids, generate_mask, dist_th=dist_th)
        pred_bond_logits = self.bond_type_mlp(H_t[row] + H_t[col]) # [E, 5], commutative property
        pred_bond_probs = F.softmax(pred_bond_logits, dim=-1) # [E, 5]
        has_bond_mask = torch.argmax(pred_bond_probs, dim=-1) != 0
        pred_bond_probs = pred_bond_probs[has_bond_mask] # [E', 5], not None bond
        row, col = row[has_bond_mask], col[has_bond_mask]
        bond_prob, bond_type = torch.max(pred_bond_probs, dim=-1)
        
        bond_select_mask = (bond_prob > bond_th) & (row < col)
        row, col, bond_type = row[bond_select_mask], col[bond_select_mask], bond_type[bond_select_mask]
        bond_prob = bond_prob[bond_select_mask]

        # get approaching vector
        BOND_DIST = 1.6
        relative_x = X_t[col] - X_t[row]   # [E, 3]
        relative_dist = torch.norm(relative_x, dim=-1) # [E]
        relative_x = relative_x / (relative_dist[:, None] + 1e-10)
        approaching_speed = (relative_dist - BOND_DIST) * 0.5 # a->b and b->a, therefore 0.5
        approaching_speed = approaching_speed * bond_prob
        v = torch.where(approaching_speed > 0, approaching_speed, torch.zeros_like(approaching_speed))[:, None] * relative_x

        # aggregation
        block_row = block_ids[torch.cat([row, col], dim=0)]
        v = torch.cat([v, -v], dim=0)
        aggr_v = scatter_sum(v, block_row, dim=0, dim_size=block_ids.max() + 1)   # [Nblock]
        aggr_v = aggr_v[block_ids]  # [Natom]

        # weights
        w = min(t / (1 - t + 1e-10), 10)

        return w * aggr_v
    
    def generate(
        self,
        X,                  # [Natom, 3], atom coordinates     
        S,                  # [Nblock], block types
        A,                  # [Natom], atom types
        bonds,              # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
        position_ids,       # [Nblock], block position ids
        chain_ids,          # [Nblock], split different chains
        generate_mask,      # [Nblock], 1 for generation, 0 for context
        block_lengths,      # [Nblock], number of atoms in each block
        lengths,            # [batch_size]
        is_aa,              # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
        n_iter=10,          # number of iterations
        return_x_only=False,# return x only (used in validation)
        return_tensor=False,# return H and X
        topo_generate_mask=None,
        confidence_model=None,
        resample_mode=False,
        disable_avoid_clash=False,
        # encoder_deterministic=True,
        # decoder_mask=False,
        **kwargs
    ):
        
        # if self.discrete_timestep: assert n_iter == self.default_num_steps
        if 'given_latent' in kwargs:
            Zh, Zx, _ = kwargs.pop('given_latent')
            # Zx_log_var = -torch.abs(self.Wx_log_var(Zh)).view(*Zx.shape)
        else:
            # encoding
            Zh, Zx, _, _, _, _, _, _ = self.encode(
                X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=not resample_mode
            ) # [Nblock, d_latent], [Nblock, 3], [1], [1]
        # Zx_log_var = -torch.abs(signed_Zx_log_var).expand_as(Zx)
        block_ids = length_to_batch_id(block_lengths)
        batch_ids = length_to_batch_id(lengths)
        if resample_mode: Zh = self._random_mask(Zh, generate_mask, batch_ids)

        if topo_generate_mask is None: topo_generate_mask = generate_mask

        # start_t = 0.5
        if topo_generate_mask.sum() != 0:
            # decode block types from latent graph
            pred_block_logits = self.decode_block_type(Zh, Zx, chain_ids, lengths)
            # mask non aa positions if is_aa == True
            non_aa_mask = ~torch.tensor(VOCAB.aa_mask, dtype=torch.bool, device=is_aa.device)
            pred_block_logits = pred_block_logits.masked_fill(non_aa_mask[None, :] & is_aa[:, None], float('-inf'))
            pred_block_prob = torch.softmax(pred_block_logits, dim=-1) # [Nblock, num_block_type]
            if resample_mode: pred_block_type = torch.multinomial(pred_block_prob, num_samples=1).squeeze(-1)
            else: prob, pred_block_type = torch.max(pred_block_prob, dim=-1) # [Nblock]
            pred_block_type[~topo_generate_mask] = S[~topo_generate_mask]
        
            # initialize (append atoms and sample coordinates)
            X_t, A, block_ids, bonds = self._init_atoms(pred_block_type, X, A, bonds, Zx, block_ids, generate_mask, topo_generate_mask)
        else: # purely structure prediction
            pred_block_type = S
            # only need to initialize atoms
            random_X = Zx[block_ids] + torch.randn_like(Zx[block_ids]) * self.prior_coord_std
            X_t = torch.where(expand_like(generate_mask[block_ids], X), random_X, X)
            # bidirectional
            _row, _col, _type = bonds[:, 0], bonds[:, 1], bonds[:, 2]
            bonds = torch.stack([
                torch.cat([_row, _col], dim=0),
                torch.cat([_col, _row], dim=0),
                torch.cat([_type, _type], dim=0)
            ], dim=1)
        
        # concat context bonds and generated bonds
        topo_edge_type = bonds[:, 2]
        topo_edges, topo_edge_attr = bonds[:, :2].T, self.atom_edge_embedding(topo_edge_type)

        # iterative
        X_init = X_t.clone()
        all_vectors, span = [], 1.0 / n_iter
        X_gen_mask = expand_like(generate_mask[block_ids], X_t)
        # for i in range(int((1 - start_t) * n_iter), n_iter):

        topo_edges_add, topo_edge_attr_add = topo_edges, topo_edge_attr

        for i in range(n_iter):
            t = (1.0 - i * span) * torch.ones_like(block_ids, dtype=torch.float)
            H_t, X_next = self.decode_structure(Zh, X_t, A, position_ids, topo_edges_add, topo_edge_attr_add, chain_ids, batch_ids, block_ids, t)
            pred_vector = torch.where(
                X_gen_mask,
                # X_next - X_t, # + self._bond_length_guidance((1.0 - i * span), H_t, X_t, batch_ids, block_ids, generate_mask),
                X_next - X_t, # + _inter_clash_guidance((1.0 - i * span), A, X_t, batch_ids, block_ids, chain_ids, generate_mask),
                torch.zeros_like(X_t))
            X_t = X_t + pred_vector * span # update
            if not disable_avoid_clash:
                X_t = _avoid_clash(A, X_t, batch_ids, block_ids, chain_ids, generate_mask, is_aa)
            all_vectors.append(pred_vector)

        X = X_t

        if return_x_only:
            return X
        if return_tensor:
            return Zh, Zx, H_t, X
        # VLB for iterative process (the smaller, the better)
        ll = ((X_t - X_init).unsqueeze(0) - torch.stack(all_vectors, dim=0)) ** 2 # [T, Natom, 3]
        ll = ll.sum(-1).mean(0) # [Natom]

        # bonds
        row, col = self._get_inter_block_nbh(X, batch_ids, block_ids, topo_generate_mask, dist_th=3.5)
        pred_bond_logits = self.bond_type_mlp(H_t[row] + H_t[col]) # [E, 5], commutative property
        pred_bond_probs = F.softmax(pred_bond_logits, dim=-1) # [E, 5]
        has_bond_mask = torch.argmax(pred_bond_probs, dim=-1) != 0
        pred_bond_probs = pred_bond_probs[has_bond_mask] # [E', 5], not None bond
        # predicted bonds
        row, col = row[has_bond_mask], col[has_bond_mask]
        bond_prob, bond_type = torch.max(pred_bond_probs, dim=-1)
        # topo-fix bonds (ctx to binder or binder to ctx bonds are not recorded in the output, and should be saved at the additional props of the ComplexDesc)
        topo_fix_mask = (~topo_generate_mask) & generate_mask
        topo_inter_mask = block_ids[bonds[:, 0]] != block_ids[bonds[:, 1]]
        topo_fix_bonds = bonds[topo_inter_mask & topo_fix_mask[block_ids[bonds[:, 0]]] & topo_fix_mask[block_ids[bonds[:, 1]]]] # [Efix, 3]
        topo_fix_bonds = topo_fix_bonds[topo_fix_bonds[:, 0] < topo_fix_bonds[:, 1]] # avoid repeated bonds
        # concat
        row = torch.cat([row, topo_fix_bonds[:, 0]], dim=0)
        col = torch.cat([col, topo_fix_bonds[:, 1]], dim=0)
        bond_type = torch.cat([bond_type, topo_fix_bonds[:, 2]], dim=0)
        bond_prob = torch.cat([bond_prob, torch.ones_like(topo_fix_bonds[:, 2], dtype=torch.float)], dim=0)
        # concat prob and distance
        bond_prob = torch.stack([bond_prob, torch.norm(X[row] - X[col], dim=-1)], dim=-1) # [E, 2]

        # intra block bonds for generated part
        intra_block_bond_mask = generate_mask[block_ids[topo_edges[0]]] & generate_mask[block_ids[topo_edges[1]]] # in generation
        intra_block_bond_mask = intra_block_bond_mask & (block_ids[topo_edges[0]] == block_ids[topo_edges[1]]) # in the same block
        intra_block_bond_mask = intra_block_bond_mask & (topo_edges[0] < topo_edges[1])  # avoid redundance
        intra_row, intra_col = topo_edges[0][intra_block_bond_mask], topo_edges[1][intra_block_bond_mask]
        intra_bond_type = topo_edge_type[intra_block_bond_mask]

        # get results
        batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds = [], [], [], [], [], []
        batch_ids = length_to_batch_id(lengths)
        for i, l in enumerate(lengths):
            batch_X.append([])
            batch_A.append([])
            batch_ll.append({ 'fm_ll': [] })
            batch_intra_bonds.append([])
            cur_mask = (batch_ids == i) & generate_mask # [Nblock]
            cur_mask = cur_mask[block_ids] # [Natom]
            cur_atom_type, cur_atom_coord, cur_atom_ll = A[cur_mask], X[cur_mask], ll[cur_mask]
            cur_block_ids = block_ids[cur_mask] # [Natom']

            for j in range(cur_block_ids.min(), cur_block_ids.max() + 1):
                batch_X[-1].append(cur_atom_coord[cur_block_ids == j].tolist())
                batch_A[-1].append(cur_atom_type[cur_block_ids == j].tolist())
                batch_ll[-1]['fm_ll'].append(cur_atom_ll[cur_block_ids == j].tolist())

            batch_S.append(pred_block_type[generate_mask & (batch_ids == i)].tolist())

            # get bonds (inter-block)
            global2local = -torch.ones_like(cur_mask, dtype=torch.long) # set non-related to -1 for later check
            global2local[cur_mask] = torch.arange(cur_mask.long().sum(), device=cur_mask.device) # assume atoms sorted by block ids
            bond_mask = cur_mask[row] & cur_mask[col]
            local_row, local_col = global2local[row[bond_mask]], global2local[col[bond_mask]]
            assert not torch.any(local_row == -1)
            assert not torch.any(local_col == -1)
            batch_bonds.append((local_row.tolist(), local_col.tolist(), bond_prob[bond_mask].tolist(), bond_type[bond_mask].tolist()))
            # get bonds (intra-block)
            block_offsets = scatter_sum(torch.ones_like(block_ids), block_ids, dim=0).cumsum(dim=0)
            block_offsets = F.pad(block_offsets[:-1], pad=(1, 0), value=0)
            for j in range(cur_block_ids.min(), cur_block_ids.max() + 1):
                bond_mask = cur_mask[intra_row] & cur_mask[intra_col] & (block_ids[intra_row] == j)
                local_row = intra_row[bond_mask] - block_offsets[block_ids[intra_row[bond_mask]]]
                local_col = intra_col[bond_mask] - block_offsets[block_ids[intra_col[bond_mask]]]
                batch_intra_bonds[-1].append((local_row.tolist(), local_col.tolist(), intra_bond_type[bond_mask].tolist()))

        # confidence
        if confidence_model is not None:
            # use the metrics from the confidence model
            confidence = confidence_model.inference(H_t, X, chain_ids, generate_mask, block_ids, batch_ids)
            for ll_dict, conf in zip(batch_ll, confidence):
                ll_dict['confidence'] = conf
            # batch_ll = [(ll, conf) for ll, conf in zip(batch_ll, confidence)]
        
        return batch_S, batch_X, batch_A, batch_ll, batch_bonds, batch_intra_bonds # inter-block bonds and intra-block bonds