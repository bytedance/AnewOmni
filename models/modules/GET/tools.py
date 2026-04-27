# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_min


def _unit_edges_from_block_edges(unit_block_id, block_src_dst, Z=None, k=None):
    '''
    :param unit_block_id [N], id of block of each unit. Assume X is sorted so that block_id starts from 0 to Nb - 1
    :param block_src_dst: [Eb, 2], all edges (block level), represented in (src, dst)
    '''
    block_n_units = scatter_sum(torch.ones_like(unit_block_id), unit_block_id)  # [Nb], number of units in each block
    block_offsets = F.pad(torch.cumsum(block_n_units[:-1], dim=0), (1, 0), value=0)  # [Nb]
    edge_n_units = block_n_units[block_src_dst]  # [Eb, 2], number of units at two end of the block edges
    edge_n_pairs = edge_n_units[:, 0] * edge_n_units[:, 1]  # [Eb], number of unit-pairs in each edge

    # block edge id for unit pairs
    edge_id = torch.zeros(edge_n_pairs.sum(), dtype=torch.long, device=edge_n_pairs.device)  # [Eu], which edge each unit pair belongs to
    edge_start_index = torch.cumsum(edge_n_pairs, dim=0)[:-1]  # [Eb - 1], start index of each edge (without the first edge as it starts with 0) in unit_src_dst
    edge_id[edge_start_index] = 1
    edge_id = torch.cumsum(edge_id, dim=0)  # [Eu], which edge each unit pair belongs to, start from 0, end with Eb - 1

    # get unit-pair src-dst indexes
    unit_src_dst = torch.ones_like(edge_id)  # [Eu]
    unit_src_dst[edge_start_index] = -(edge_n_pairs[:-1] - 1)  # [Eu], e.g. [1,1,1,-2,1,1,1,1,-4,1], corresponding to edge id [0,0,0,1,1,1,1,1,2,2]
    del edge_start_index  # release memory
    if len(unit_src_dst) > 0:
        unit_src_dst[0] = 0 # [Eu], e.g. [0,1,1,-2,1,1,1,1,-4,1], corresponding to edge id [0,0,0,1,1,1,1,1,2,2]
    unit_src_dst = torch.cumsum(unit_src_dst, dim=0)  # [Eu], e.g. [0,1,2,0,1,2,3,4,0,1], corresponding to edge id [0,0,0,1,1,1,1,1,2,2]
    unit_dst_n = edge_n_units[:, 1][edge_id]  # [Eu], each block edge has m*n unit pairs, here n refers to the number of units in the dst block
    # turn 1D indexes to 2D indexes (TODO: this block is memory-intensive)
    unit_src = torch.div(unit_src_dst, unit_dst_n, rounding_mode='floor') + block_offsets[block_src_dst[:, 0][edge_id]] # [Eu]
    unit_dst = torch.remainder(unit_src_dst, unit_dst_n)  # [Eu], e.g. [0,1,2,0,0,0,0,0,0,1] for block-pair shape 1*3, 5*1, 1*2
    unit_dist_local = unit_dst
    # release some memory
    del unit_dst_n, unit_src_dst  # release memory
    unit_edge_src_start = (unit_dst == 0)
    unit_dst = unit_dst + block_offsets[block_src_dst[:, 1][edge_id]]  # [Eu]
    del block_offsets, block_src_dst # release memory
    unit_edge_src_id = unit_edge_src_start.long()
    if len(unit_edge_src_id) > 1:
        unit_edge_src_id[0] = 0
    unit_edge_src_id = torch.cumsum(unit_edge_src_id, dim=0)  # [Eu], e.g. [0,0,0,1,2,3,4,5,6,6] for the above example
    
    if k is None:
        return (unit_src, unit_dst), (edge_id, unit_edge_src_start, unit_edge_src_id)

    # sparsify, each atom is connected to the nearest k atoms in the other block in the same block edge
    max_n = torch.max(scatter_sum(torch.ones_like(unit_edge_src_id), unit_edge_src_id))
    k = min(k, max_n)

    BIGINT = 1e10  # assign a large distance to invalid edges
    N = unit_edge_src_id.max() + 1
    # src_dst = src_dst.transpose(0, 1)  # [2, Ef]
    dist = torch.norm(Z[unit_src] - Z[unit_dst], dim=-1).sum(-1) # [Eu]

    dist_mat = torch.ones(N, max_n, device=dist.device, dtype=dist.dtype) * BIGINT  # [N, max_n]
    dist_mat[(unit_edge_src_id, unit_dist_local)] = dist
    del dist
    dist_neighbors, dst = torch.topk(dist_mat, k, dim=-1, largest=False)  # [N, topk]
    del dist_mat

    src = torch.arange(0, N, device=dst.device).unsqueeze(-1).repeat(1, k)
    unit_edge_src_start = torch.zeros_like(src).bool() # [N, k]
    unit_edge_src_start[:, 0] = True
    src, dst = src.flatten(), dst.flatten()
    unit_edge_src_start = unit_edge_src_start.flatten()
    dist_neighbors = dist_neighbors.flatten()
    is_valid = dist_neighbors < BIGINT
    src = src.masked_select(is_valid)
    dst = dst.masked_select(is_valid)
    unit_edge_src_start = unit_edge_src_start.masked_select(is_valid)

    # extract row, col and edge id
    mat = torch.ones(N, max_n, device=unit_src.device, dtype=unit_src.dtype) * -1
    mat[(unit_edge_src_id, unit_dist_local)] = unit_src
    unit_src = mat[(src, dst)]
    mat[(unit_edge_src_id, unit_dist_local)] = unit_dst
    unit_dst = mat[(src, dst)]
    mat[(unit_edge_src_id, unit_dist_local)] = edge_id
    edge_id = mat[(src, dst)]

    unit_edge_src_id = src
    
    return (unit_src, unit_dst), (edge_id, unit_edge_src_start, unit_edge_src_id)


def _block_edge_dist(X, block_id, src_dst):
    '''
    Several units constitute a block.
    This function calculates the distance of edges between blocks
    The distance between two blocks are defined as the minimum distance of unit-pairs between them.
    The distance between two units are defined as the minimum distance across different channels.
        e.g. For number of channels c = 2, suppose their distance is c1 and c2, then the distance between the two units is min(c1, c2)

    :param X: [N, c, 3], coordinates, each unit has c channels. Assume the units in the same block are aranged sequentially
    :param block_id [N], id of block of each unit. Assume X is sorted so that block_id starts from 0 to Nb - 1
    :param src_dst: [Eb, 2], all edges (block level) that needs distance calculation, represented in (src, dst)
    '''
    (unit_src, unit_dst), (edge_id, _, _) = _unit_edges_from_block_edges(block_id, src_dst)
    # calculate unit-pair distances
    src_x, dst_x = X[unit_src], X[unit_dst]  # [Eu, c, 3]
    dist = torch.norm(src_x - dst_x, dim=-1)  # [Eu, c]
    dist = torch.min(dist, dim=-1).values  # [Eu]
    dist = scatter_min(dist, edge_id)[0]  # [Eb]

    return dist


def variadic_arange(size):
    """
    from https://torchdrug.ai/docs/_modules/torchdrug/layers/functional/functional.html#variadic_arange

    Return a 1-D tensor that contains integer intervals of variadic sizes.
    This is a variadic variant of ``torch.arange(stop).expand(batch_size, -1)``.

    Suppose there are :math:`N` intervals.

    Parameters:
        size (LongTensor): size of intervals of shape :math:`(N,)`
    """
    starts = size.cumsum(0) - size

    range = torch.arange(size.sum(), device=size.device)
    range = range - starts.repeat_interleave(size)
    return range



def variadic_meshgrid(input1, size1, input2, size2):
    """
    from https://torchdrug.ai/docs/_modules/torchdrug/layers/functional/functional.html#variadic_meshgrid
    Compute the Cartesian product for two batches of sets with variadic sizes.

    Suppose there are :math:`N` sets in each input,
    and the sizes of all sets are summed to :math:`B_1` and :math:`B_2` respectively.

    Parameters:
        input1 (Tensor): input of shape :math:`(B_1, ...)`
        size1 (LongTensor): size of :attr:`input1` of shape :math:`(N,)`
        input2 (Tensor): input of shape :math:`(B_2, ...)`
        size2 (LongTensor): size of :attr:`input2` of shape :math:`(N,)`

    Returns
        (Tensor, Tensor): the first and the second elements in the Cartesian product
    """
    grid_size = size1 * size2
    local_index = variadic_arange(grid_size)
    local_inner_size = size2.repeat_interleave(grid_size)
    offset1 = (size1.cumsum(0) - size1).repeat_interleave(grid_size)
    offset2 = (size2.cumsum(0) - size2).repeat_interleave(grid_size)
    index1 = torch.div(local_index, local_inner_size, rounding_mode="floor") + offset1
    index2 = local_index % local_inner_size + offset2
    return input1[index1], input2[index2]


def scatter_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
    '''
    from https://github.com/rusty1s/pytorch_scatter/issues/48
    WARN: the range between src.max() and src.min() should not be too wide for numerical stability

    reproducible
    '''
    # f_src = src.float()
    # f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
    # norm = (f_src - f_min)/(f_max - f_min + eps) + index.float()*(-1)**int(descending)
    # perm = norm.argsort(dim=dim, descending=descending)

    # return src[perm], perm
    src, src_perm = torch.sort(src, dim=dim, descending=descending)
    index = index.take_along_dim(src_perm, dim=dim)
    index, index_perm = torch.sort(index, dim=dim, stable=True)
    src = src.take_along_dim(index_perm, dim=dim)
    perm = src_perm.take_along_dim(index_perm, dim=0)
    return src, perm


def scatter_topk(src: torch.Tensor, index: torch.Tensor, k: int, dim=0, largest=True):
    indices = torch.arange(src.shape[dim], device=src.device)
    src, perm = scatter_sort(src, index, dim, descending=largest)
    index, indices = index[perm], indices[perm]
    mask = torch.ones_like(index).bool()
    mask[k:] = index[k:] != index[:-k]
    return src[mask], indices[mask]


def fully_connect_edges(batch_ids):
    lengths = scatter_sum(torch.ones_like(batch_ids), batch_ids, dim=0)
    row, col = variadic_meshgrid(
        input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
        size1=lengths,
        input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
        size2=lengths,
    )
    return torch.stack([row, col], dim=0)


def knn_edges(block_ids, batch_ids, X, k_neighbors, specify_all_edges=None):
    '''
    :param X
    :param k_neighbors: int
    :param specify_all_edges: [2, E]
    '''
    if specify_all_edges is None:
        all_edges = fully_connect_edges(batch_ids)
    else:
        all_edges = specify_all_edges
    dist = _block_edge_dist(X, block_ids, all_edges.T)
    row, col = all_edges

    # get topk for each node
    _, indices = scatter_topk(dist, row, k=k_neighbors, largest=False)
    edges = torch.stack([all_edges[0][indices], all_edges[1][indices]], dim=0) # [2, k*N]
    return edges  # [2, E]