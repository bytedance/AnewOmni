# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_mean, scatter_sum, scatter_std

import utils.register as R

from utils.nn_utils import stable_norm, std_conserve_scatter_sum, graph_to_batch_nx

from ..GET.tools import _unit_edges_from_block_edges
from .radial_basis import RadialBasis

try:
    from xformers.ops import memory_efficient_attention as attn_func
    xformers_enable = True
except ImportError:
    xformers_enable = False


@R.register('XTransEncoderAct')
class XTransEncoderAct(nn.Module):
    def __init__(self, hidden_size, ffn_size, n_rbf, cutoff=7.0, z_requires_grad=False, 
                 edge_size=16, n_layers=3, n_head=4, pre_norm=False, use_edge_feat=False, sparse_k=3, local_mask=False, attn_bias=True,
                 efficient=False, vector_act='none', 
                 # use_ieconv=False, zero_conv=False, efficient_ieconv=False, ieconv_share_edge_feat=False
        ) -> None:
        super().__init__()

        self.encoder = Transformer(
            d_hidden = hidden_size, d_ffn = ffn_size, n_heads = n_head, n_layers = n_layers,
            n_rbf = n_rbf, d_edge = edge_size, cutoff = cutoff, use_edge_feat = use_edge_feat, local_mask = local_mask, attn_bias = attn_bias,
            layer_norm = 'pre' if pre_norm else 'post', sparse_k = sparse_k, efficient = efficient,
            vector_act = vector_act, 
        )

    @property
    def require_block_edges(self):
        return True

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None, adapter=None):
        H, V = self.encoder(H, Z, block_id, batch_id, edges, edge_attr, topo_edges, topo_edge_attr, attn_mask, adapter)
        block_repr = std_conserve_scatter_sum(H, block_id, dim=0)
        graph_repr = std_conserve_scatter_sum(block_repr, batch_id, dim=0)
        # return H, block_repr, graph_repr, V.reshape(Z.shape) + Z
        return H, V.reshape(Z.shape) + Z


class Transformer(nn.Module):
    '''Equivariant Adaptive Block Transformer'''

    def __init__(
            self,
            d_hidden,
            d_ffn,
            n_heads,
            n_layers,
            n_rbf,
            d_edge,
            cutoff=7.0,
            act_fn=nn.SiLU(),
            layer_norm = 'pre',
            residual = True,
            use_edge_feat = False,
            local_mask = False,
            attn_bias = True,
            sparse_k=None,
            efficient=False,
            vector_act='none',
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        self.use_edge_feat = use_edge_feat
        self.sparse_k = sparse_k
        self.efficient = efficient
        self._local_mask = local_mask
        if self.efficient and not xformers_enable:
            print("xformers are not downloaded, change into custom attention mechanism. "
                  "Please install xformers via 'pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121',"
                  "or seek 'https://github.com/facebookresearch/xformers' for more details.")
            self.efficient = False

        self.edge_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2 + d_edge + n_rbf, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden * 2)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden)
        )

        self.final_v = GVPFFNLayer(
            d_hidden, d_ffn, act_fn, d_output=1
        )

        self.n_rbf = n_rbf
        if n_rbf > 1:
            self.rbf = RadialBasis(num_radial=n_rbf, cutoff=cutoff)

        if self.use_edge_feat:
            self.rbf_mapping = nn.Linear(n_rbf + d_edge, n_layers)    
        else:
            self.rbf_mapping = nn.Linear(n_rbf, n_layers)

        if self.layer_norm == 'pre':
            self.ln = nn.LayerNorm(d_hidden)

        for i in range(0, n_layers):
            self.add_module(f'layer_{i}', EPTLayer(
                d_hidden, d_ffn, n_heads, i, act_fn, layer_norm, residual, self.efficient, vector_act, attn_bias
            ))

    def forward(self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None, adapter=None):
        with torch.no_grad():
            if topo_edges is not None:
                # first delete self-loop of 3D edges. Otherwise there might be two same atom-level edges overwriting each other
                not_self_loop = edges[0] != edges[1]
                edges = edges.T[not_self_loop].T
                if edge_attr is not None: edge_attr = edge_attr[not_self_loop]
            (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = _unit_edges_from_block_edges(block_id, edges.T, Z, k=self.sparse_k) # [Eu], Eu = \sum_{i, j \in E} n_i * n_j
        
        if edge_attr is not None: edge_attr = edge_attr[block_edge_id]
        
        # concat 3D and 2D edges
        if topo_edges is not None:
            unit_row = torch.cat([unit_row, topo_edges[0]], dim=0)
            unit_col = torch.cat([unit_col, topo_edges[1]], dim=0)
        if topo_edge_attr is not None:
            edge_attr = torch.cat([edge_attr, topo_edge_attr], dim=0) # [E1 + E2, d]

        # vector init
        Z = Z.view(-1, 3)
        edge_vec = Z[unit_row] - Z[unit_col] # [Ne, 3]
        edge_dis = torch.norm(edge_vec, dim=-1)
        dis_feat = self.rbf(edge_dis)
        edge_feat = torch.cat([H[unit_row], H[unit_col], dis_feat, edge_attr], dim=-1)
        edge_scaler = self.edge_mlp(edge_feat) # [Ne, d_hidden]
        inv_feat, equiv_feat = torch.split(edge_scaler, self.d_hidden, dim=-1)
        edge_scas = H[unit_col] * inv_feat
        edge_vecs = edge_vec.unsqueeze(-1) * equiv_feat.unsqueeze(-2) # [Ne, 3, d_hidden]
        H = self.node_mlp(torch.cat([H, scatter_sum(edge_scas, unit_row, dim_size=H.shape[0], dim=0)], dim=-1))
        V = scatter_mean(edge_vecs, unit_row, dim_size=H.shape[0], dim=0)

        # graph to batch
        batch_to_nodes = batch_id[block_id]
        H_batch, H_mask = graph_to_batch_nx(H, batch_to_nodes, mask_is_pad=False, factor_req=8)
        bs, max_n = H_batch.shape[0], H_batch.shape[1]
        V_batch = torch.zeros((bs, max_n, *V.shape[1:]), dtype=V.dtype, device=V.device)
        V_batch[H_mask] = V
        Z_batch = torch.zeros((bs, max_n, *Z.shape[1:]), dtype=Z.dtype, device=Z.device)
        Z_batch[H_mask] = Z

        # rbf to all layer & heads
        if self.use_edge_feat:
            dis_feat = torch.cat([dis_feat, edge_attr], dim=-1)
        rbf_feat = self.rbf_mapping(dis_feat)
        lengths = torch.zeros(bs, dtype=batch_id.dtype, device=batch_id.device)
        lengths[1:] = torch.cumsum(scatter_sum(torch.ones_like(batch_to_nodes), batch_to_nodes), dim=-1)[:-1]  # [bs]
        lengths = lengths[batch_to_nodes]
        tot_idx = torch.cumsum(torch.ones_like(batch_to_nodes), dim=-1) - 1
        self_idx = tot_idx - lengths
        if self._local_mask:
            rbf_feat_batch = torch.ones((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device) * float('-inf')
            rbf_feat_batch[~H_mask] = 0.0 # to prevent nan in padding which will lead to 0 * nan = nan (broadcast to other positions)
        else:
            rbf_feat_batch = torch.zeros((bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device)
        if attn_mask is not None:
            rbf_feat_batch[~attn_mask] = float('-inf')
        rbf_feat_batch[batch_to_nodes[unit_row], self_idx[unit_row], self_idx[unit_col]] = rbf_feat
        rbf_feat_batch = rbf_feat_batch.reshape(bs, max_n, max_n, self.n_layers, -1).permute(3,0,4,1,2) # [l, bs, h, n, n]

        # svd init
        D_batch = torch.norm(Z_batch.unsqueeze(1) - Z_batch.unsqueeze(2), dim=-1) # [bs, n, n]
        D_batch = -D_batch

        cached_info = (D_batch.detach(), rbf_feat_batch, H_mask)    

        for i in range(self.n_layers):
            H_batch, V_batch = self._modules[f'layer_{i}'](
                H_batch, V_batch, cached_info
            )
            if adapter is not None:
                H_add, V_add = adapter(H_batch[H_mask], V_batch[H_mask], Z_batch[H_mask], i)
                H_batch = H_batch + graph_to_batch_nx(H_add, batch_to_nodes, mask_is_pad=False, factor_req=8)[0]
                V_batch = V_batch + graph_to_batch_nx(V_add, batch_to_nodes, mask_is_pad=False)[0]
        
        if self.layer_norm == 'pre':
            H_batch = self.ln(H_batch)

        H_graph = H_batch[H_mask]
        V_graph = V_batch[H_mask]

        V_graph = V_graph / (V_graph.norm(dim=-2, keepdim=True) + 1e-5)

        _, V_graph = self.final_v(H_graph, V_graph)
        return H_graph, V_graph


class EPTLayer(nn.Module):
    def __init__(
            self,
            d_hidden,
            d_ffn,
            n_heads,
            layer_idx=-1,
            act_fn=nn.SiLU(),
            layer_norm = 'pre',
            residual = True,
            efficient = False,
            vector_act = 'none',
            attn_bias = True
        ):
        super(EPTLayer, self).__init__()
        self.attn_layer = SubLayerWrapper(
            SelfAttnLayer(d_hidden, n_heads, layer_idx, efficient, attn_bias = attn_bias),
            d_hidden,
            layer_norm,
            residual
        )
        self.ffn_layer = SubLayerWrapper(
            GVPFFNLayer(d_hidden, d_ffn, act_fn, vector_act = vector_act),
            d_hidden,
            layer_norm,
            residual
        )
        self.layer_idx = layer_idx

    def forward(self, H, V, cached_info=None):

        H, V = self.attn_layer(H, V, cached_info=cached_info)
        H, V = self.ffn_layer(H, V)

        return H, V

        
class SelfAttnLayer(nn.Module):

    def __init__(self, d_hidden, n_heads, layer_idx=-1, efficient=False, attn_bias=True):

        super(SelfAttnLayer, self).__init__()

        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.d_head = self.d_hidden // self.n_heads
        self.layer_idx = layer_idx
        self.factor = 0.5 / math.sqrt(self.d_head)
        self.efficient = efficient
        self.scaler_q = nn.Linear(d_hidden, d_hidden * 4, bias=attn_bias)
        self.scaler_k = nn.Linear(d_hidden, d_hidden * 4, bias=attn_bias)
        self.scaler_v = nn.Linear(d_hidden, d_hidden, bias=attn_bias)
        self.vector_v = nn.Linear(d_hidden, d_hidden, bias = False)
        self.scaler_o = nn.Linear(d_hidden, d_hidden)
        self.vector_o = nn.Linear(d_hidden, d_hidden, bias = False)

    def forward(self, H, V, cached_info=None):

        # H : [B, N, d_hidden]
        # V : [B, N, 3, d_hidden]

        batch_size, num_nodes = H.shape[0], H.shape[1]

        D_batch, rbf_feat_batch, H_mask = cached_info  

        H_q = self.scaler_q(H).view(batch_size, num_nodes, self.n_heads, -1)
        H_k = self.scaler_k(H).view(batch_size, num_nodes, self.n_heads, -1)
        H_v = self.scaler_v(H).view(batch_size, num_nodes, self.n_heads, -1)
        V_v = self.vector_v(V).view(batch_size, num_nodes, 3, self.n_heads, -1).transpose(-2, -3).flatten(start_dim=-2) 
        V_attn = torch.cat([H_v, V_v], dim=-1)

        bias = rbf_feat_batch[self.layer_idx] + D_batch.unsqueeze(1)
        mask = H_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        bias = bias.masked_fill(mask == 0, float("-inf"))

        if not self.efficient:
            attn = torch.einsum('bnhd, bmhd -> bhnm', H_q, H_k)
            attn = F.softmax(attn * self.factor + bias, dim=-1)
            res = torch.einsum('bhnm, bmhd -> bnhd', attn, V_attn)

        else:
            res = attn_func(
                query = H_q,
                key = H_k,
                value = V_attn,
                attn_bias = bias.expand(-1, self.n_heads, -1, -1)
            )


        H_res = res[:, :, :, :self.d_head].reshape(batch_size, num_nodes, self.d_hidden)
        V_res = res[:, :, :, self.d_head:].reshape(batch_size, num_nodes, self.n_heads, 3, self.d_head).transpose(-2, -3).reshape(batch_size, num_nodes, 3, self.d_hidden)
        
        H_o = self.scaler_o(H_res) 
        V_o = self.vector_o(V_res)

        return H_o, V_o


def cross_feats(V, scale_down=0.1):
    D = V.shape[-1]
    channelswarp = V[..., list(range(1, D)) + [0]]
    aftercross = torch.cross(V, channelswarp, dim=-2) * scale_down
    V = torch.cat([V, aftercross], dim=-1).contiguous()
    return V


class GVPFFNLayer(nn.Module):

    def __init__(self, d_hidden, d_ffn, act_fn=nn.SiLU(), d_output=None, vector_act='none', enable_cross_product=True):

        super(GVPFFNLayer, self).__init__()

        self.d_hidden = d_hidden
        self.d_ffn = d_ffn
        self.act_fn = act_fn
        self.d_output = d_hidden if d_output is None else d_output
        self.enable_cross_product = enable_cross_product

        self.linear_v = nn.Linear((d_hidden * 2) if self.enable_cross_product else d_hidden, d_hidden + self.d_output, bias=False)
        self.ffn_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_ffn),
            act_fn,
            nn.Linear(d_ffn, d_hidden + self.d_output)
        )

        self.vector_act = vector_act
        if self.vector_act == 'layernorm':
            self.vector_layernorm = nn.LayerNorm(self.d_output)

    @property
    def se3_equivariance(self):
        return getattr(self, 'enable_cross_product', False) # for compatibility

    def vector_act_func(self, Vs):
        if self.vector_act == 'none':
            return Vs
        elif self.vector_act == 'sigmoid':
            return F.sigmoid(Vs)
        elif self.vector_act == 'tanh':
            return F.tanh(Vs)
        elif self.vector_act == 'layernorm':
            return self.vector_layernorm(Vs)
        elif self.vector_act == 'one':
            return torch.ones_like(Vs)

    def forward(self, H, V):
        if self.se3_equivariance: V = cross_feats(V)
        V_proj = self.linear_v(V)
        V1, V2 = V_proj[...,:self.d_hidden], V_proj[...,self.d_hidden:]
        scaler = torch.cat([H, V1.norm(dim=-2)], dim=-1)
        scaler_out = self.ffn_mlp(scaler)
        H_out, V_update = scaler_out[...,:self.d_hidden], scaler_out[...,self.d_hidden:]
        V_out = self.vector_act_func(V_update).unsqueeze(-2) * V2
        return H_out, V_out    
    

class SubLayerWrapper(nn.Module):

    def __init__(self, sub_layer, d_hidden, layer_norm = 'pre', residual = True):

        super(SubLayerWrapper, self).__init__()
        self.sub_layer = sub_layer
        self.d_hidden = d_hidden
        self.layer_norm = layer_norm
        self.ln = nn.LayerNorm(d_hidden)
        self.residual = residual

    def forward(self, H, V, **kwargs):
        H0, V0 = H, V
        if self.layer_norm == 'pre':
            H = self.ln(H0)
        H, V = self.sub_layer(H, V, **kwargs)
        if self.residual:
            H = H + H0
            V = V + V0
        if self.layer_norm == 'post':
            H = self.ln(H)
        return H, V



if __name__ == '__main__':
    d_hidden = 64
    d_ffn = 16
    d_edge = 16
    n_rbf = 16
    n_heads = 4
    n_layers = 3
    device = torch.device('cuda:0')

    # d_hidden, d_ffn, n_heads, n_layers, n_rbf, d_edge, cutoff=7.0, act_fn=nn.SiLU(), layer_norm = 'pre', residual = True, sparse_k=3, svd_k=128

    model = Transformer(d_hidden, d_ffn, n_heads, n_layers, n_rbf, d_edge=d_edge, use_ieconv=True, use_edge_feat=True, efficient_ieconv=True, ieconv_share_edge_feat=False)
    model.to(device)
    model.eval()
    
    block_id = torch.tensor([0,0,1,1,1,1,2,2,2,3,4,4,5,6,6,6,6,7,7], dtype=torch.long).to(device)
    batch_id = torch.tensor([0,0,0,0,0,1,1,1], dtype=torch.long).to(device)
    src_dst = torch.tensor([[0,1], [2,3], [1,3], [2,4], [3, 0], [3, 3], [5,7], [7,6], [5,6], [6,7]], dtype=torch.long).to(device)
    src_dst = src_dst.T
    edge_attr = torch.randn(len(src_dst[0]), d_edge).to(device)
    n_unit = block_id.shape[0]

    H = torch.randn(n_unit, d_hidden, device=device)
    Z = torch.randn(n_unit, 3, device=device)

    H1, V1 = model(H, Z, block_id, batch_id, src_dst, edge_attr)

    # random rotaion matrix
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q1, t1 = U.mm(V), torch.randn(3, device=device)
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q2, t2 = U.mm(V), torch.randn(3, device=device)

    unit_batch_id = batch_id[block_id]
    Z[unit_batch_id == 0] = torch.matmul(Z[unit_batch_id == 0], Q1) + t1
    Z[unit_batch_id == 1] = torch.matmul(Z[unit_batch_id == 1], Q2) + t2
    # Z = torch.matmul(Z, Q) + t

    H2, V2 = model(H, Z, block_id, batch_id, src_dst, edge_attr)

    print(f'invariant feature: {torch.abs(H1 - H2).sum()}')
    V1[unit_batch_id == 0] = torch.einsum('nih, ij -> njh', V1[unit_batch_id == 0], Q1)
    V1[unit_batch_id == 1] = torch.einsum('nih, ij -> njh', V1[unit_batch_id == 1], Q2)
    print(f'equivariant feature: {torch.abs(V1 - V2).sum()}')