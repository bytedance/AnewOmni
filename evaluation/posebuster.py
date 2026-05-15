# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import posebusters
from posebusters import PoseBusters
from pathlib import Path
from yaml import safe_load

def create_fast_config(mode='dock', loose_th=False):
    # Create a "fast" configuration file by removing the `energy_ratio` module.
    assert mode in {'dock', 'redock', 'gen', 'mol'}, f"unknown mode: `{mode}`"
    posebusters_path = Path(posebusters.__file__).parent
    config_path = posebusters_path / "config" / f"{mode}.yml"
    with open(config_path, encoding="utf-8") as f:
        config = safe_load(f)

    config["modules"] = [
        module for module in config["modules"] 
        if module["function"] != "energy_ratio"
    ]

    # widen internal clash threshold
    if loose_th:
        for module in config['modules']:
            if module['function'] == 'distance_geometry': module['parameters']['threshold_clash'] = 0.4
            elif module['function'] == 'intermolecular_distance': module['parameters']['radius_scale'] = 0.8

    return config

def denovo_validity(pocket_file, mol_sdf, remove_energy_term=False, loose_th=False):
    pred_file = Path(mol_sdf)
    true_file = Path(mol_sdf)
    cond_file = Path(pocket_file)

    if remove_energy_term:
        fast_config = create_fast_config(mode='dock', loose_th=loose_th)
        buster = PoseBusters(config=fast_config)
    else:
        buster = PoseBusters(config='dock')

    try:
        df = buster.bust([pred_file], true_file, cond_file)
        data = df.to_dict(orient='index') # {row1: {col1: val1, col2: val2, ...}, row2: {...}}
        is_valid = []
        for key in data:
            valid = True
            for check_type in data[key]: valid = valid and data[key][check_type]
            is_valid.append(valid)
        return is_valid, data
    except:
        return [False], None