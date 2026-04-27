# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_sum


# embedding of blocks (for proteins, it is residue).
class BlockEmbedding(nn.Module):
    '''
    [atom embedding + block embedding]
    '''
    def __init__(self, num_block_type, num_atom_type, embed_size, atom_only=False):
        super().__init__()
        if not atom_only:
            self.block_embedding = nn.Embedding(num_block_type, embed_size)
        self.atom_embedding = nn.Embedding(num_atom_type, embed_size)
    
    @property
    def atom_only(self):
        return not hasattr(self, 'block_embedding')

    def forward(self, S, A, block_id):
        '''
        :param S: [Nb], block (residue) types
        :param A: [Nu], unit (atom) types
        :param block_id: [Nu], block id of each unit
        '''
        atom_embed = self.atom_embedding(A)
        if self.atom_only: return atom_embed
        block_embed = self.block_embedding(S[block_id])
        return atom_embed + block_embed
    

class AtomTopoEmbedding(nn.Module):
    '''
    TODO: atom embedding based on 2D chemical interactions
    '''
    def __init__(self, num_atom_type, num_bond_type, embed_size) -> None:
        super().__init__()


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, act_fn=nn.SiLU(), end_with_act=False, dropout=0.0):
        super().__init__()
        assert n_layers >= 2, f'MLP should have at least two layers (input/output)'
        self.input_linear = nn.Linear(input_size, hidden_size)
        medium_layers = [act_fn]
        for i in range(n_layers):
            medium_layers.append(nn.Linear(hidden_size, hidden_size))
            medium_layers.append(act_fn)
            medium_layers.append(nn.Dropout(dropout))
        self.medium_layers = nn.Sequential(*medium_layers)
        if end_with_act:
            self.output_linear = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                act_fn
            )
        else:
            self.output_linear = nn.Linear(hidden_size, output_size)

    def forward(self, H):
        '''
        Args:
            H: [..., input_size]
        Returns:
            H: [..., output_size]
        '''
        H = self.input_linear(H)
        H = self.medium_layers(H)
        H = self.output_linear(H)
        return H
    

class GINEConv(nn.Module):
    r"""The modified :class:`GINConv` operator from the `"Strategies for
    Pre-training Graph Neural Networks" <https://arxiv.org/abs/1905.12265>`_
    paper.

    .. math::
        \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathrm{ReLU}
        ( \mathbf{x}_j + \mathbf{e}_{j,i} ) \right)

    that is able to incorporate edge features :math:`\mathbf{e}_{j,i}` into
    the aggregation procedure.

    Args:
        input_size: dimension of the input size
        hidden_size: dimension of the hidden variable
        eps (float, optional): (Initial) :math:`\epsilon`-value.
            (default: :obj:`0.`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        edge_size (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, input_size, hidden_size, out_size, edge_size, n_layers: int=3, eps: float=0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_size = edge_size
        self.n_layers = n_layers
        self.eps = eps

        self.linear_input = nn.Linear(input_size, hidden_size)
        self.linear_edge = nn.Linear(edge_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, out_size)

        mlps = []
        for i in range(n_layers):
            mlps.append(MLP(
                hidden_size, hidden_size, hidden_size,
                n_layers=2, act_fn=nn.ReLU(),
                end_with_act=True
            ))
        self.mlps = nn.ModuleList(mlps)

    def forward(self, H, E, edge_attr):
        '''
        Args:
            H: [N, input_size]
            E: [2, E] src/dst
            edge_attr: [E, edge_size]
        '''
        # prepare
        src, dst = E # message passing aggregates dst nodes to src nodes
        H = self.linear_input(H)  # [N, hidden_size]
        edge_attr = self.linear_edge(edge_attr) # [E, hidden_size]

        for i in range(self.n_layers):

            # get message
            msg = F.relu(H[dst] + edge_attr) # [E, hidden_size]
            aggr = scatter_sum(msg, src, dim=0, dim_size=H.shape[0]) # [N, hidden_size]

            # update
            updated = (1 + self.eps) * H + aggr
            H = self.mlps[i](updated)

        return self.linear_out(H)