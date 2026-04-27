# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


def expand_like(src, tgt):
    src = src.reshape(*src.shape, *[1 for _ in tgt.shape[len(src.shape):]]) # [..., 1, 1, ...]
    return src.expand_as(tgt)


def stable_norm(input, *args, **kwargs):
    '''
        For L2: norm = sqrt(\sum x^2) = (\sum x^2)^{1/2}
        the gradient will have zero in divider if \sum x^2 = 0
        it is not ok to direct add eps to all x, since x might be a small but negative value
        this function deals with this problem
    '''
    input = input.clone()
    with torch.no_grad():
        sign = torch.sign(input)
        input = torch.abs(input)
        input.clamp_(min=1e-10)
        input = sign * input
    return torch.norm(input, *args, **kwargs)


def std_conserve_scatter_sum(src, index, dim):
    ones = torch.ones_like(index)
    n = scatter_sum(ones, index, dim=0) # [N]
    value = scatter_sum(src, index, dim=dim) # [N, ...]
    value = value / torch.sqrt(n).unsqueeze(-1)
    return value


def graph_to_batch_nx(tensor, batch_id, padding_value=0, mask_is_pad=True, factor_req=8):
    '''
    :param tensor: [N, D1, D2, ...]
    :param batch_id: [N]
    :param mask_is_pad: 1 in the mask indicates padding if set to True
    '''
    lengths = scatter_sum(torch.ones_like(batch_id), batch_id)  # [bs]
    bs, max_n = lengths.shape[0], torch.max(lengths)
    max_n = max_n if (max_n % 8 == 0) else (max_n // 8 * 8 + 8)
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


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sin-Cos Positional Embedding
    """
    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim

    def forward(self, position_ids):
        device = position_ids.device
        position_ids = position_ids[None] # [1, N]
        indices = torch.arange(self.output_dim // 2, device=device, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(-1, self.output_dim)
        return embeddings


class SinusoidalTimeEmbeddings(nn.Module):
    '''
        sin(1*t*2pi), sin(2*t*2pi), ...,
        cos(1*t*2pi), cos(2*t*2pi)
    '''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        if self.dim == 1: # no projection
            return time.reshape(-1,1)
        device = time.device
        half_dim = self.dim // 2
        freq = 2 * torch.arange(half_dim, device=device) * math.pi
        t = freq * time[..., None]
        embeddings = torch.cat((t.sin(), t.cos()), dim=-1)
        return embeddings