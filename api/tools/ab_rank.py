# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import shutil

import yaml


RANKING_WEIGHTS = { # default weights
    'metrics.bs_overlap': 1.0, # this is the factor of the first priority
    # the remainings sum up to 1.0
    'metrics.contact_cdr_ratio': 0.4,
    'cplx_confidences.cofold_iptm': 0.2,
    'cplx_confidences.cofold_binder_plddt': 0.1,
    'cplx_confidences.cofold_normalized_ipae': 0.05,
    'generative_confidences.normalized_cdr_design_confidence': 0.1,
    'metrics.normalized_scRMSD': 0.15,
}


FRAME_LIB = json.load(open(os.path.join(
    os.path.dirname(__file__), '..', 'templates', 'framework_lib.json'
), 'r'))


def get_score(summary, weights):
    val = 0
    print(weights)
    print(summary)
    for key in weights:
        depths = key.split('.')
        v = summary
        for d in depths: v = v[d]
        # v = dicts[key]
        if v is None: continue
        val += v * weights[key]

    return val


def satisfy(item, ths):
    summary = item[1]
    flag = True
    for key in ths:
        depths = key.split('.')
        v = summary
        for d in depths: v = v[d]
        if v is None: continue
        flag = flag and (ths[key](v))
    return flag


def assign_frame(summary):
    seq = summary['heavychain']['sequence']
    mark = summary['heavychain']['mark']
    s = ''
    for aa, m in zip(seq, mark):
        if m != '3': s += aa
        else: break
    for _id in FRAME_LIB:
        hseq = FRAME_LIB[_id]['heavychain']
        if s in hseq: return _id


def main(root_dir, output_dir, topk=20):
    # root_dir: xxx/results/
    # folders under root_dir: e.g. 000000/000001/0000002/...
    # rank_weights = { # default weights
    #     'af3_iptm': 0.1,
    #     'af3_binder_plddt': 0.05,
    #     'af3_normalized_ipae': 0.2,
    #     'normalized_cdr_design_confidence': 0.1,
    #     'normalized_scRMSD': 0.15,
    #     'bs_overlap': 0.3,
    #     'contact_cdr_ratio': 0.2
    # }
    ori_config = os.path.join(root_dir, '..', '..', 'config_v3.yaml')
    config = yaml.safe_load(open(ori_config, 'r'))

    if 'ranking_weights' in config:
        rank_weights = config['ranking_weights']
        print(f'using ranking weights from config: {rank_weights}')
    else:
        rank_weights = RANKING_WEIGHTS

    items = []

    for d in os.listdir(root_dir):
        path = os.path.join(root_dir, d)
        summary_path = os.path.join(path, 'summary.json')
        if not os.path.exists(summary_path): continue
        summary = json.load(open(summary_path, 'r'))
        if 'metrics:' in summary:
            print(summary)
            summary['metrics'] = summary['metrics:']
            del summary['metrics:']
            json.dump(summary, open(summary_path, 'w'), indent=2)
        summary['original_id'] = assign_frame(summary)
        items.append((path, summary, get_score(summary, rank_weights)))

    print(f'loaded {len(items)} items')

    # # statistics
    # vals, ab_cnt, types = {}, 0, []
    # for _, summary, _ in items:
    #     if summary['lightchain'] is not None:
    #         ab_cnt += 1
    #         types.append('antibody')
    #     else: types.append('nanobody')
    #     for category in ['metrics', 'confidences']:
    #         for key in summary[category]:
    #             if key not in vals: vals[key] = []
    #             vals[key].append(summary[category][key])
    # for key in vals:
    #     val_list = [v for v in vals[key] if v is not None]
    #     print(f'{key}, max {max(val_list)}, mean {sum(val_list) / len(val_list)}, min {min(val_list)}')

    # # antibody/nanobody ratio
    # print()
    # print(f'antibody {ab_cnt}, nanobody {len(items) - ab_cnt}')

    # # stats for different types
    # print()
    # print('stats for antibody')
    # for key in vals:
    #     val_list = [v for v, t in zip(vals[key], types) if t == 'antibody' and v is not None]
    #     if len(val_list) == 0:
    #         print(f'no values for {key}')
    #         continue
    #     else:
    #         print(f'{key}, max {max(val_list)}, mean {sum(val_list) / len(val_list)}, min {min(val_list)}')
    
    # print()
    # print('stats for nanobody')
    # for key in vals:
    #     val_list = [v for v, t in zip(vals[key], types) if t == 'nanobody' and v is not None]
    #     print(f'{key}, max {max(val_list)}, mean {sum(val_list) / len(val_list)}, min {min(val_list)}')

    # stats of original frame
    cnts = {}
    for _, summary, _ in items:
        pdb_id = summary['original_id']
        if pdb_id not in cnts: cnts[pdb_id] = 0
        cnts[pdb_id] += 1
    total = sum(cnts.values())
    print()
    for pdb_id in sorted(cnts, key=lambda k: cnts[k], reverse=True):
        print(f'{pdb_id}: {cnts[pdb_id] / total * 100}%')

    # rank
    items = sorted(items, key=lambda tup: tup[-1], reverse=True)

    # topk
    os.makedirs(output_dir, exist_ok=True)
    print()
    for i, item in enumerate(items[:topk]):
        print(i, item[0])
        print(item[1])
        shutil.copyfile(os.path.join(item[0], 'model.cif'), os.path.join(output_dir, f'{i}.cif'))
        shutil.copyfile(os.path.join(item[0], 'model.pdb'), os.path.join(output_dir, f'{i}.pdb'))
        gen_file = os.path.join(item[0], 'generated', 'model.cif')
        if os.path.exists(gen_file):
            shutil.copyfile(gen_file, os.path.join(output_dir, f'{i}_gen.cif'))
            shutil.copyfile(gen_file.replace('.cif', '.pdb'), os.path.join(output_dir, f'{i}_gen.pdb'))

    # use some thresholds for filtering
    ths = {
        'metrics.contact_cdr_ratio': lambda v: v > 0.4,
        'metrics.lig_temp_scRMSD': lambda v: v < 5.0,
        # 'metrics.custom_metrics.n_term_contact_dist': lambda v: v > 0.0,
    }

    print()
    cnt = 0
    output_dir = os.path.join(output_dir, 'thresholds')
    def ignore(src, names):
        if os.path.basename(src) == 'design':
            return names
        if os.path.basename(src) == 'generated':
            return ['model.sdf', 'model_confidence.json']
        return ['design', 'model.sdf', 'model_confidence.json', 'full_data.json', 'model.cif', 'model.pdb']
    os.makedirs(output_dir, exist_ok=True)
    for item in items:
        if satisfy(item, ths):
            print(item[0])
            print(item[1])
            cnt += 1
            _id = os.path.basename(item[0])
            shutil.copytree(item[0], os.path.join(output_dir, _id), ignore=ignore)
    print(f'A total of {cnt} candidates satisfy the thresholds')


if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
