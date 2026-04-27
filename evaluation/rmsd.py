# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from utils.chem_utils import get_rdkit_rmsd, get_rmsd_from_coord

# from https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py
def kabsch_rotation(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


# have been validated with kabsch from RefineGNN
def kabsch(a, b):
    # find optimal rotation matrix to transform a into b
    # a, b are both [N, 3]
    # a_aligned = aR + t
    a, b = np.array(a), np.array(b)
    a_mean = np.mean(a, axis=0)
    b_mean = np.mean(b, axis=0)
    a_c = a - a_mean
    b_c = b - b_mean

    rotation = kabsch_rotation(a_c, b_c)
    # a_aligned = np.dot(a_c, rotation)
    # t = b_mean - np.mean(a_aligned, axis=0)
    # a_aligned += t
    t = b_mean - np.dot(a_mean, rotation)
    a_aligned = np.dot(a, rotation) + t

    return a_aligned, rotation, t


# a: [N, 3], b: [N, 3]
def compute_rmsd(a, b, need_align=False):  # rmsd on single complex
    if need_align:
        a, _, _ = kabsch(a, b)
    dist = np.sum((a - b) ** 2, axis=-1)
    rmsd = np.sqrt(dist.sum() / a.shape[0])
    return float(rmsd)


def compute_ligand_rmsd(ligand_sdf:str, reference_sdf:str, compute_directly=False) -> float:
    """
    Two ways for computing RMSD:
    1. get RMSD from coordinates directly (match atoms by finding MCS before that) [compute_directly:On]
    2. Align and compute (follow Posebusters' code) [compute_directly:Off]
    """
    if not compute_directly:
        rmsd = get_rdkit_rmsd(ligand_sdf, reference_sdf)
    
    else:
        rmsd = get_rmsd_from_coord(ligand_sdf, reference_sdf)
    
    return rmsd


# def _np_kabsch(a, b, return_v=False):
#     '''get alignment matrix for two sets of coodinates'''
#     # _np = jnp if use_jax else np
#     _np = np
#     ab = a.swapaxes(-1,-2) @ b
#     u, s, vh = _np.linalg.svd(ab, full_matrices=False)
#     flip = _np.linalg.det(u @ vh) < 0
#     u_ = _np.where(flip, -u[...,-1].T, u[...,-1].T).T
#     # if use_jax: u = u.at[...,-1].set(u_)
#     # else: u[...,-1] = u_
#     u[...,-1] = u_
#     return u if return_v else (u @ vh)
# 
# 
# def complex_rmsd_after_align(true, pred, weights=None, L=None, include_L=True, copies=1):
#     # adapted from https://github.com/sokrypton/ColabDesign/blob/d024c4e846fea83c090afcbe89a313eeee8ec01e/colabdesign/af/loss.py
#     '''
#     get rmsd + alignment function
#     align based on the first L positions, computed weighted rmsd using all 
#     positions (if include_L=True) or remaining positions (if include_L=False).
#     '''
#     # normalize weights
#     length = true.shape[-2]
#     if weights is None:
#       weights = (np.ones(length) / length)[...,None]
#     else:
#       weights = (weights / (weights.sum(-1,keepdims=True) + 1e-8))[...,None]
# 
#     # determine alignment [L]ength and remaining [l]ength
#     if copies > 1:
#       if L is None:
#         L = iL = length // copies; C = copies - 1
#       else:
#         (iL, C) = ((length - L) // copies, copies)
#     else:
#       (L, iL, C) = (length, 0, 0) if L is None else (L, length - L, 1)
# 
#     # slice inputs
#     if iL == 0:
#       (T,P,W) = (true,pred,weights)
#     else:
#       (T,P,W) = (x[...,:L,:] for x in (true,pred,weights))
#       (iT,iP,iW) = (x[...,L:,:] for x in (true,pred,weights))
# 
#     # get alignment and rmsd functions
#     (T_mu,P_mu) = ((x*W).sum(-2,keepdims=True)/W.sum((-1,-2)) for x in (T,P))
#     aln = _np_kabsch((P-P_mu)*W, T-T_mu)   
#     align_fn = lambda x: (x - P_mu) @ aln + T_mu
#     msd_fn = lambda t,p,w: (w*np.square(align_fn(p)-t)).sum((-1,-2))
# 
#     # compute rmsd
#     if iL == 0:
#       msd = msd_fn(true,pred,weights)
#     elif C > 1:
#       # all vs all alignment of remaining, get min RMSD
#       iT = iT.reshape(-1, C, 1, iL, 3).swapaxes(0,-3)
#       iP = iP.reshape(-1, 1, C, iL, 3).swapaxes(0,-3)
#       imsd = msd_fn(iT, iP, iW.reshape(-1,C,1,iL,1).swapaxes(0,-3))
#       imsd = (imsd.min(0).sum(0) + imsd.min(1).sum(0)) / 2 
#       imsd = imsd.reshape(np.broadcast_shapes(true.shape[:-2],pred.shape[:-2]))
#       msd = (imsd + msd_fn(T,P,W)) if include_L else (imsd/iW.sum((-1,-2)))
#     else:
#       msd = msd_fn(true,pred,weights) if include_L else (msd_fn(iT,iP,iW)/iW.sum((-1,-2)))
#     rmsd = np.sqrt(msd + 1e-8)
# 
#     return {"rmsd":rmsd, "align":align_fn}