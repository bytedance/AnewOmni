# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
'''
beta version
'''
from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum


@torch.no_grad()
def graph_to_batch(tensor, batch_id, padding_value=0, mask_is_pad=True, max_len=-1):
    '''
    :param tensor: [N, D1, D2, ...]
    :param batch_id: [N]
    :param mask_is_pad: 1 in the mask indicates padding if set to True
    :param max_len: manually specify maximum length
    '''
    lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
    bs, max_n = lengths.shape[0], torch.max(lengths)
    if max_len > 0: max_n = max_len
    batch = torch.ones((bs, max_n, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device) * padding_value
    # generate pad mask: 1 for pad and 0 for data
    pad_mask = torch.zeros((bs, max_n + 1), dtype=torch.long, device=tensor.device)
    pad_mask[(torch.arange(bs, device=tensor.device), lengths)] = 1
    pad_mask = (torch.cumsum(pad_mask, dim=-1)[:, :-1]).bool()
    data_mask = torch.logical_not(pad_mask)
    # fill data
    batch[data_mask] = tensor
    mask = pad_mask if mask_is_pad else data_mask
    return batch, mask


@torch.no_grad()
def length_to_batch_id(lengths):
    # generate batch id
    batch_id = torch.zeros(lengths.sum(), dtype=torch.long, device=lengths.device) # [N]
    batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
    batch_id.cumsum_(dim=0)  # [N], item idx in the batch
    return batch_id


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


def std_conserve_scatter_mean(src, index, dim):
    '''
    std conserving scatter mean
    suppose n variables from N(0, sigma^2)
    the sum of the n variables results in N(0, n*sigma^2)
    if directly use mean pooling, the result will be N(0, (1/n)sigma^2)
    this function divides the results by sqrt(n), which leads to N(0, sigma^2) again
    '''
    ones = torch.ones_like(index)
    n = scatter_sum(ones, index, dim=0) # [N]
    value = scatter_sum(src, index, dim=dim) # [N, ...]
    value = value / torch.sqrt(n).unsqueeze(-1)
    return value


def scatter_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
    '''
    from https://github.com/rusty1s/pytorch_scatter/issues/48
    WARN: the range between src.max() and src.min() should not be too wide for numerical stability

    reproducible
    '''
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


def _edge_dist(X, atom_pad_mask, src_dst):
    '''
    :param X: [N, n_channel, 3], coordinates
    :param atom_pad_mask: [N, n_channel], mark the padding positions as 1
    :param src_dst: [Ef, 2], all edges that needs distance calculation, represented in (src, dst)
    '''
    BIGINT = 1e10  # assign a large distance to invalid edges
    dist = X[src_dst]  # [Ef, 2, n_channel, 3]
    dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
    dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
    atom_pad = atom_pad_mask[src_dst]  # [Ef, 2, n_channel]
    atom_pad = torch.logical_or(atom_pad[:, 0].unsqueeze(2), atom_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
    dist = dist + atom_pad * BIGINT  # [Ef, n_channel, n_channel]
    dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
    return dist


def fully_connect_edges(batch_ids):
    lengths = scatter_sum(torch.ones_like(batch_ids), batch_ids, dim=0)
    row, col = variadic_meshgrid(
        input1=torch.arange(batch_ids.shape[0], device=batch_ids.device),
        size1=lengths,
        input2=torch.arange(batch_ids.shape[0], device=batch_ids.device),
        size2=lengths,
    )
    return torch.stack([row, col], dim=0)


def knn_edges(k_neighbors, all_edges, dist):
    '''
    :param k_neighbors: int
    :param all_edges: [2, E], src and tgt of all edges
    :param dist: [E], distances of each edge
    '''
    row, col = all_edges

    # get topk for each node
    _, indices = scatter_topk(dist, row, k=k_neighbors, largest=False)
    edges = torch.stack([all_edges[0][indices], all_edges[1][indices]], dim=0) # [2, k*N]
    return edges  # [2, E]