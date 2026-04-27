# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn

from utils.gnn_utils import length_to_batch_id, std_conserve_scatter_mean

from .layer import AdapterLayer
from ..EPT.radial_basis import RadialBasis


@dataclass
class Conditions:
    H_2d: torch.Tensor      # [N, cond_embedding]
    mask_2d: torch.Tensor   # [N]
    E_2d: torch.Tensor      # [2 + cond_embedding * 2, E2d]
    Z_3d: torch.Tensor      # [N, 3]
    mask_3d: torch.Tensor   # [N]
    E_dist: torch.Tensor    # [3, E3d]
    w: float = None         # only used in inference


@dataclass
class ConditionConfig:
    mask_2d: torch.Tensor   # [N], for 2D conditioning
    mask_3d: torch.Tensor   # [N], for 3D conditioning (seems weired, only COM can be used during inference, so training should also use CoM)
    mask_incomplete_2d: torch.Tensor # [N], 1 for incomplete topo
    w: float = None         # CFG weight

    def to(self, device):
        self.mask_2d = self.mask_2d.to(device)
        self.mask_3d = self.mask_3d.to(device)
        self.mask_incomplete_2d = self.mask_incomplete_2d.to(device)
        return self

    @classmethod
    def batchify(cls, config_list):
        mask_2d, mask_3d, mask_incomplete_2d = [], [], []
        for config in config_list:
            assert not (config.mask_2d is None and config.mask_3d is None)
            mask_template = config.mask_3d if config.mask_2d is None else config.mask_2d
            if config.mask_2d is None: mask_2d.append(torch.zeros_like(mask_template))
            else: mask_2d.append(config.mask_2d)
            if config.mask_3d is None: mask_3d.append(torch.zeros_like(mask_template))
            else: mask_3d.append(config.mask_3d)
            if config.mask_incomplete_2d is None: mask_incomplete_2d.append(torch.zeros_like(mask_template))
            else: mask_incomplete_2d.append(config.mask_incomplete_2d)
        mask_2d = torch.cat(mask_2d, dim=0)
        mask_3d = torch.cat(mask_3d, dim=0)
        mask_incomplete_2d = torch.cat(mask_incomplete_2d, dim=0)
        for config in config_list: assert config.w == config_list[0].w
        return ConditionConfig(
            mask_2d=mask_2d,
            mask_3d=mask_3d,
            mask_incomplete_2d=mask_incomplete_2d,
            w=config_list[0].w
        )
    

def config_to_condition(config: ConditionConfig, Z, block_lengths, bonds, atom_topo_embedding, block_embedding):
    block_ids = length_to_batch_id(block_lengths)
    bonds = bonds.T # [3, E]

    # topo condition
    cond_mask_2d = config.mask_2d
    H_2d = torch.where(cond_mask_2d.unsqueeze(-1), block_embedding, torch.zeros_like(block_embedding))
    block_row, block_col = block_ids[bonds[0]], block_ids[bonds[1]]
    select_mask = (block_row != block_col) & cond_mask_2d[block_row] & cond_mask_2d[block_col]
    row, col = bonds[0][select_mask], bonds[1][select_mask]
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    E_2d = (
        torch.stack([block_ids[row], block_ids[col]], dim=0), # [2, E2d]
        torch.cat([atom_topo_embedding[row], atom_topo_embedding[col]], dim=-1).T # [2*h, E2d]
    )

    # coordinate condition
    cond_mask_3d = config.mask_3d
    Z_3d = torch.where(cond_mask_3d.unsqueeze(-1), Z, torch.zeros_like(Z))

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
    E_dist = (torch.zeros(2, 0, device=Z.device, dtype=torch.long), torch.zeros(0, device=Z.device, dtype=torch.float))
    
    return Conditions(
        H_2d=H_2d,
        mask_2d=cond_mask_2d,
        E_2d=E_2d, # bidirectional
        Z_3d=Z_3d,
        mask_3d=cond_mask_3d,
        E_dist=E_dist,
        w=config.w
    )


class ConditionSampler:

    UNCOND = 0
    STRUCT_PRED = 1
    INV_FOLD = 2
    IMPAINTING = 3
    PARTIAL_2D = 4
    OTHER = 5

    def __init__(
        self,
        p_uncond = 0.5,
        p_struct_pred = 0.15,
        p_inv_fold = 0.1,
        p_impainting = 0.1,
        p_partial_2D = 0.15,
    ):
        self.p_uncond = p_uncond
        self.p_struct_pred = p_struct_pred
        self.p_inv_fold = p_inv_fold
        self.p_impainting = p_impainting
        self.p_partial_2D = p_partial_2D
        self.p_other = max(0, 1.0 - self.p_uncond - self.p_struct_pred - self.p_inv_fold - self.p_impainting - self.p_partial_2D)

    def _sample_types(self, batch_size, device):
        probs = torch.tensor(
            [self.p_uncond, self.p_struct_pred, self.p_inv_fold, self.p_impainting, self.p_partial_2D, self.p_other],
            dtype=torch.float, device=device
        )
        probs = probs / probs.sum()
        return torch.multinomial(probs, num_samples=batch_size, replacement=True)

    def __call__(self, generate_mask, lengths) -> ConditionConfig:
        batch_ids = length_to_batch_id(lengths)

        cond_types = self._sample_types(lengths.shape[0], lengths.device)
        
        # structure prediction
        sp_mask_2d = (cond_types == self.STRUCT_PRED)[batch_ids] # [Nblock]
        sp_mask_3d = torch.zeros_like(sp_mask_2d) # [Nblock], do not give 3D condition for structure prediction

        # inverse folding
        if_mask_3d = (cond_types == self.INV_FOLD)[batch_ids] # [Nblock]
        if_mask_2d = torch.zeros_like(if_mask_3d)

        # impainting
        cond_ratio = torch.rand(lengths.shape[0], device=lengths.device)[batch_ids] # [Nblock]
        imp_mask_2d = (cond_types == self.IMPAINTING)[batch_ids] & (torch.rand_like(cond_ratio) < cond_ratio) # [Nblock]
        imp_mask_3d = imp_mask_2d

        # partial 2D
        cond_ratio = torch.rand(lengths.shape[0], device=lengths.device)[batch_ids] # [Nblock]
        p2d_mask_2d = (cond_types == self.PARTIAL_2D)[batch_ids] & (torch.rand_like(cond_ratio) < cond_ratio)
        p2d_mask_3d = torch.zeros_like(p2d_mask_2d)

        # others
        other_mask = (cond_types == self.OTHER)[batch_ids]
        cond_ratio = torch.rand(lengths.shape[0], device=lengths.device)[batch_ids]
        other_mask_2d = other_mask & (torch.rand_like(cond_ratio) < cond_ratio)    # 1 for given as condition
        cond_ratio = torch.rand(lengths.shape[0], device=lengths.device)[batch_ids]
        other_mask_3d = other_mask & (torch.rand_like(cond_ratio) < cond_ratio)    # 1 for given as condition
        
        mask_2d = (sp_mask_2d | if_mask_2d | imp_mask_2d | p2d_mask_2d | other_mask_2d) & generate_mask
        mask_3d = (sp_mask_3d | if_mask_3d | imp_mask_3d | p2d_mask_3d | other_mask_3d) & generate_mask

        return ConditionConfig(
            mask_2d = mask_2d,
            mask_3d = mask_3d,
            mask_incomplete_2d = torch.zeros_like(mask_2d),
            w = None
        )


class Adapter(nn.Module):
    def __init__(self,
                 hidden_size,
                 node_embed_size_2d,
                 edge_embed_size_2d,
                 edge_feat_size,
                 edge_type_size,
                 n_layers,
                 n_rbf=64,
                 cutoff=6.0):
        super().__init__()
        self.edge_feat_size = edge_feat_size

        self.virtual_node_embed = nn.Parameter(torch.randn(1, node_embed_size_2d), requires_grad=True)
        self.edge_type_embed = nn.Embedding(3, edge_type_size) # topo/coord(virtual)/dist
        self.dist_rbf = RadialBasis(n_rbf, cutoff)
        self.rbf_linear = nn.Linear(n_rbf, edge_feat_size, bias=False)
        self.edge2d_linear = nn.Linear(edge_embed_size_2d, edge_feat_size, bias=False)

        self.i_linear = nn.Linear(hidden_size + node_embed_size_2d, hidden_size, bias=False)
        self.gnns = nn.ModuleList([AdapterLayer(hidden_size, edge_feat_size + edge_type_size, n_rbf, cutoff) for _ in range(n_layers)])

    def forward(self, H, V, Z, layer_i, H_2d, mask_2d, E_2d, Z_3d, mask_3d, E_dist):
        '''
        Args:
            H: [N, h1],
            V: [N, h1, 3],
            Z: [N, 3],
            H_2d: [N, h2],       topo condition (all zero for no condition)
            mask_2d: [N],        1 for topo condition, 0 for no condition
            E_2d: ([2, E1], [d, E1]),   topo condition, first two dimensions are row and col (bidirectional)
            Z_3d: [N, 3],        coordinate condition
            mask_3d: [N],        1 for coordinate condition, 0 for no condition
            E_dist: ([2, E2], [E2]),     pairwise distance condition, row/col/distance (bidirectional)


        Returns:
            H_add: [N, h]
            V_add: [N, h, 3]
        '''
        # create adapter graph
        # nodes
        n_cond_3d = mask_3d.long().sum()
        nodes_real = torch.cat([H, H_2d], dim=-1)   # [N, h1 + h2]
        nodes_virtual = torch.cat([H[mask_3d], self.virtual_node_embed.repeat(n_cond_3d, 1)], dim=-1) # [N3d, h1 + h2]
        v_real, v_virtual = V, torch.zeros_like(V[mask_3d])
        nodes = torch.cat([nodes_real, nodes_virtual], dim=0)
        nodes = self.i_linear(nodes) # [N+N3d, h]
        v = torch.cat([v_real, v_virtual], dim=0) # [N+N3d, h, 3]
        z = torch.cat([Z, Z_3d[mask_3d]], dim=0)   # [N + N3d, 3]
        # edges
        row_3d = torch.arange(nodes_real.shape[0], device=nodes.device)[mask_3d]
        col_3d = torch.arange(n_cond_3d, device=nodes.device) + nodes_real.shape[0]
        E_3d = torch.stack([
            torch.cat([row_3d, col_3d], dim=0),
            torch.cat([col_3d, row_3d], dim=0)
        ], dim=0) # [2, E_3d], bidirectional
        edges = torch.cat([E_2d[0], E_3d, E_dist[0]], dim=-1)
        edge_type = torch.cat([
            torch.zeros_like(E_2d[0][0]),
            torch.zeros_like(E_3d[0]) + 1,
            torch.zeros_like(E_dist[0][0]) + 2
        ], dim=0)
        edge_type = self.edge_type_embed(edge_type)
        edge_attr = torch.cat([
            torch.cat([
                self.edge2d_linear(E_2d[1].T),
                torch.zeros(E_3d.shape[1], self.edge_feat_size, dtype=torch.float, device=E_3d.device),
                self.rbf_linear(self.dist_rbf(E_dist[1]))
            ], dim=0),
            edge_type
        ], dim=-1)
        # message passing with the specified layer
        H_add, V_add = self.gnns[layer_i](nodes, v, z, edges, edge_attr)
        H_add, V_add = H_add[:nodes_real.shape[0]], V_add[:v_real.shape[0]] # discard virtual nodes
        # get updates
        mask_dist = torch.zeros_like(mask_3d)
        mask_dist[E_dist[0][0]] = True
        mask_dist[E_dist[0][1]] = True
        update_mask = mask_2d | mask_3d | mask_dist

        if not self.training:
            # clamp to avoid OOD problem during inference
            H_add = torch.clamp(H_add, min=-100, max=100)
            V_add = torch.clamp(V_add, min=-100, max=100)
        
        H_add = torch.where(update_mask.unsqueeze(-1), H_add, torch.zeros_like(H_add))
        V_add = torch.where(update_mask.unsqueeze(-1).unsqueeze(-1), V_add, torch.zeros_like(V_add))
        return H_add, V_add

    def wrap_func(self, conditions: Conditions):
        if conditions is None: return None
        return partial(
            self.forward,
            H_2d=conditions.H_2d,
            mask_2d=conditions.mask_2d,
            E_2d=conditions.E_2d,
            Z_3d=conditions.Z_3d,
            mask_3d=conditions.mask_3d,
            E_dist=conditions.E_dist
        )