# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
from typing import Dict
from copy import deepcopy

_NAMESPACE = {}

def register(name):
    def decorator(cls):
        assert name not in _NAMESPACE, f'Class {name} already registered'
        _NAMESPACE[name] = cls
        return cls
    return decorator


def get(name):
    if name not in _NAMESPACE:
        raise ValueError(f'Class {name} not registered')
    return _NAMESPACE[name]


def recur_construct(val: any):
    if isinstance(val, dict):
        if 'class' in val: return construct(val) # leaf node
        for key in val:
            val[key] = recur_construct(val[key])
    elif isinstance(val, list):
        val = [recur_construct(v) for v in val]
    return val


def construct(config: Dict, **kwargs):
    config = deepcopy(config)
    cls_name = config.pop('class')
    cls = get(cls_name)
    config.update(kwargs)
    return cls(**config)