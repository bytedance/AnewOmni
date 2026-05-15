# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
from typing import List
from dataclasses import dataclass
from enum import Enum

from utils.logger import print_log


class FilterResult(Enum):
    PASSED = 1
    FAILED = 2
    ERROR = 3


def add_tag_to_path(path, tag):
    name, ext = os.path.splitext(path)
    new_path = f'{name}_{tag}{ext}'
    return new_path


@dataclass
class FilterInput:
    path_prefix: str
    tgt_chains: List[str]  # target protein
    lig_chains: List[str]  # ligands, should only have one chain except antibody
    smiles: str
    seq: str               # for peptides/antibodies
    confidence: float      # pairwise distance error
    likelihood: float      # likelihood from diffusion model


class BaseFilter:
    def __init__(self, *args, **kwargs):
        pass

    @property
    def name(self):
        return self.__class__.__name__
    
    def run(self, input: FilterInput):
        # Result (enum), Details
        return FilterResult.PASSED, {}
    
    def print_error(self, input, err_msg):
        print_log(f'{self.name}, error occurred for input {input}: {err_msg}', level='ERROR')