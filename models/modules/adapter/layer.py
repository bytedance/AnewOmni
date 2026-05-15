# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
    Adapted from https://github.com/torchmd/torchmd-net/blob/1deecd1d8777b9d0d1ff9b63c38d10c3873f06f9/torchmdnet/models/torchmd_et.py#L245
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum

from ..nn import MLP
from ..EPT.radial_basis import RadialBasis

from utils.nn_utils import stable_norm


class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, edge_size, n_rbf, cutoff=1.0):
        super().__init__()
        self.hidden_size = hidden_size

        self.rbf = RadialBasis(n_rbf, cutoff=cutoff)
        self.vec_proj = nn.Linear(
            hidden_size, hidden_size * 3, bias=False
        )
        self.mlp = MLP(
            input_size=hidden_size * 3 + n_rbf + edge_size,
            hidden_size=hidden_size,
            output_size=hidden_size * 3,
            n_layers=3)
        
        self.o_proj = nn.Linear(hidden_size, hidden_size * 3)

    def forward(self, h, vec, coord, edge_index, edge_attr):
        '''
        Args:
            vec: [N, 3, h]
        '''
        row, col = edge_index   # [E]
        x_ij = coord[row] - coord[col] # [E, 3]
        d_ij = self.rbf(stable_norm(x_ij, dim=-1)) # [E, n_rbf]

        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_size, dim=-1) # [E, 3, h]
        vec_dot = (vec1 * vec2).sum(dim=1)  # [N, h]

        msg = self.mlp(torch.cat([h[row], h[col], d_ij, (vec1[row] * vec2[col]).sum(dim=1), edge_attr], dim=-1)) # [E, h * 3]
        msg_h, msg_v, msg_x = torch.split(msg, self.hidden_size, dim=1) # [E, h]
        h_aggr = scatter_sum(msg_h, row, dim=0, dim_size=h.shape[0]) # [N, h]
        vec_aggr = scatter_sum(vec3[col] * msg_v.unsqueeze(1) + msg_x.unsqueeze(1) * x_ij.unsqueeze(2), row, dim=0, dim_size=vec.shape[0]) # [N, 3, h]

        o1, o2, o3 = torch.split(self.o_proj(h_aggr), self.hidden_size, dim=1) #[E, h]

        dh = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_aggr

        return dh, dvec
    

if __name__ == '__main__':
    # test equivariance
    hidden_size = 10
    edge_size = 5
    n_rbf = 5
    n = 8
    device = torch.device('cuda:0')
    model = AdapterLayer(hidden_size, edge_size, n_rbf)
    model.eval()
    model.to(device)

    h = torch.randn(8, hidden_size, device=device)
    vec = torch.randn(8, hidden_size, 3, device=device)
    coord = torch.randn(8, 3, device=device)
    edge_index = torch.tensor([[0, 4, 6, 2, 3, 5], [2, 4, 0, 1, 3, 6]], dtype=torch.long, device=device)
    edge_attr = torch.randn(edge_index.shape[1], edge_size, device=device)

    dh1, dvec1 = model(h, vec.transpose(-1, -2), coord, edge_index, edge_attr)
    dvec1 = dvec1.transpose(-1, -2)

    # random rotation and translation
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q1, t1 = U.mm(V), torch.randn(3, device=device)

    coord = coord.mm(Q1) + t1
    vec = torch.matmul(vec, Q1.unsqueeze(0))

    dh2, dvec2 = model(h, vec, coord, edge_index, edge_attr)
    dvec2 = dvec2.transpose(-1, -2)
    
    print(f'invariant feature: {torch.abs(dh1 - dh2).sum()}')
    ideal_dvec2 = torch.matmul(dvec1, Q1.unsqueeze(0))
    print(f'equivariant feature: {torch.abs(ideal_dvec2 - dvec2).sum()}')