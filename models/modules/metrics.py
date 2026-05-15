# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch

from torch_scatter import scatter_mean


def batch_rmsd(y_pred, y_true, batch_ids, reduction='none'):
    '''
    Args:
        y_pred: [N, 3]
        y_true: [N, 3]
        batch_ids: [N]
    Returns:
        rmsd: [1] (w/ reduction) or [batch_size] (w/o reduction)
    '''
    square_error = ((y_pred - y_true) ** 2).sum(dim=-1) # [N]
    rmsd = torch.sqrt(
        scatter_mean(square_error, batch_ids, dim=0, dim_size=batch_ids.max() + 1
    )) # [batch_size]
    if reduction == 'mean':
        return rmsd.mean()
    elif reduction == 'none':
        return rmsd
    else:
        raise NotImplementedError(f'reduction {reduction} not implemented')
    

def batch_accu(y_pred, y_true, batch_ids, reduction='none'):
    '''
    Args:
        y_pred: [N, n_class] or [N] (binary)
        y_true: [N]
        batch_ids: [N]
    Returns:
        accu: [1] (w/ reduction) or [batch_size] (w/o reduction)
    '''
    if len(y_pred.shape) == 1: # binary
        y_pred_class = (y_pred > 0.5).long()
    else:
        y_pred_class = torch.argmax(y_pred, dim=-1)
    hit = (y_pred_class == y_true).float()  # [N]
    accu = scatter_mean(hit, batch_ids, dim=0) # [batch_size]
    if reduction == 'mean':
        return accu.mean()
    elif reduction == 'none':
        return accu
    else:
        raise NotImplementedError(f'reduction {reduction} not implemented')
    