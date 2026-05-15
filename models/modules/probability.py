# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
import torch

GAUSSIAN_COEF = 1.0 / math.sqrt(2 * math.pi)


def gaussian_probability(mu, sigma, query):
    '''
    Get probability of query in the gaussian N(mu, sigma^2)
    Args:
        mu: [N, 3], coordinate
        sigma: [N, 3], invariant
        query: [N, 3]
    '''
    errors = query - mu
    sigma = sigma + 1e-16
    p = GAUSSIAN_COEF * torch.exp(-0.5 * (errors / sigma) ** 2) / sigma
    p = torch.prod(p, dim=-1) # [N]
    return p


def continuous_nll(mu_pred, sigma_pred, y_true, reduction='mean'):
    prob = gaussian_probability(mu_pred, sigma_pred, y_true)
    nll = -torch.log(prob + 1e-16) #
    if reduction == 'mean':
        return nll.mean()
    elif reduction == 'none':
        return nll
    else:
        raise ValueError(f'Reduction {reduction} not implemented')