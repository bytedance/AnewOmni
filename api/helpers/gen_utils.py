# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import random
import tempfile
from copy import deepcopy

import torch
from rdkit import Chem

from data.bioparse.hierarchy import Block, Atom
from data.bioparse.numbering import assign_pos_ids, get_nsys
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from data.bioparse.writer.complex_to_mmcif import complex_to_mmcif
from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.utils import extract_sub_complex
from data.file_loader import MolType
from models.LDM.data_utils import OverwriteTask, Recorder, _get_item_multitype
from utils.logger import print_log

from ..pdb_dataset import PDBDataset, ComplexDesc
from ..templates import BaseTemplate, Antibody


def load_model(ckpt, cfd_ckpt, device):
    # load model
    assert not ((ckpt is None) and (cfd_ckpt is None)), f'Either LDM ckpt or confidence ckpt should be provided'
    # load confidence model
    if cfd_ckpt is not None:
        print_log(f'Using confidence model from {cfd_ckpt}')
        conf_model = torch.load(cfd_ckpt, map_location='cpu', weights_only=False)
        conf_model.to(device)
        conf_model.eval()
        print_log(f'Using the base model saved with the confidence model')
        model = conf_model.base_model
    else:
        conf_model = None
        # load model
        print_log(f'Using checkpoint {ckpt}')
        model = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.to(device)
        model.eval()
    return model, conf_model

# utils
def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
    elif isinstance(data, list) or isinstance(data, tuple):
        res = [to_device(item, device) for item in data]
        data = type(data)(res)
    elif hasattr(data, 'to'):
        data = data.to(device)
    return data


def data_to_cplx(
        cplx_desc: ComplexDesc, template: BaseTemplate,
        S: list, X: list, A: list, ll: list, inter_bonds: tuple, intra_bonds: tuple,
        out_path: str, check_validity: bool=False, check_filters: bool=False,
        ):
    '''
        Args:
            bonds: [row, col, prob, type], row and col are atom index, prob has confidence and distance
    '''

    covalent_modifications = None
    if cplx_desc.additional_props is not None:
        covalent_modifications = cplx_desc.additional_props.get('covalent_modifications', None)

    task = OverwriteTask(
        cplx = cplx_desc.cplx,
        select_indexes = cplx_desc.pocket_block_ids + cplx_desc.lig_block_ids,
        generate_mask = cplx_desc.generate_mask,
        target_chain_ids = cplx_desc.tgt_chains,
        ligand_chain_ids = cplx_desc.lig_chains,
        S = S,
        X = X,
        A = A,
        ll = ll['fm_ll'],
        inter_bonds = inter_bonds,
        intra_bonds = intra_bonds,
        confidence = ll.get('confidence', None),
        likelihood = ll.get('likelihood', None),
        out_path = out_path,
        save_cif = True,
        covalent_modifications = covalent_modifications
    )

    def template_filter(cplx):
        cplx_desc.cplx = cplx
        return template.validate(cplx_desc)

    cplx, gen_mol, overwrite_indexes = task.get_overwritten_results(
        check_validity = check_validity,
        filters = [template_filter] if check_filters else None
    )

    if cplx is None or gen_mol is None:
        return None, None

    cplx_desc.cplx = cplx

    details = {
        'pmetric': task.get_total_likelihood(),
        'smiles': Chem.MolToSmiles(gen_mol),
        'gen_seq': task.get_generated_seq(),
        'confidence': task.confidence.confidence if task.confidence is not None else None,
        'likelihood': task.likelihood,
        'normalized_likelihood': task.get_normalized_likelihood()
    }

    return cplx_desc, details


def format_log(cplx_desc: ComplexDesc, details: dict, n: int):
    basics = {
        'id': cplx_desc.id,
        'n': n,
        'pmetric': details['pmetric'],
        'confidence': details['confidence'],
        'likelihood': details['likelihood'],
        'normalized_likelihood': details['normalized_likelihood'],
        'smiles': details['smiles'],
        'gen_seq': details['gen_seq'],
        'tgt_chains': cplx_desc.tgt_chains,
        'lig_chains': cplx_desc.lig_chains,
        'gen_block_idx': cplx_desc.lig_block_ids,
    }
    if cplx_desc.additional_props is not None: basics.update(cplx_desc.additional_props)
    return basics


def modify_additional_props(cplx, props):
    if props is None: return props
    props = deepcopy(props)
    if 'covalent_modifications' in props:   # the covalent bond might have changed
        bonds = []
        for bond in cplx.bonds:
            if bond.index1[0] != bond.index2[0]: bonds.append((bond.index1, bond.index2, bond.bond_type.value))
        props['covalent_modifications'] = bonds
    return props


def generate_for_one_template(
        model, dataset, n_samples, batch_size, save_dir, device, sample_opt,
        n_cycles=0, conf_model=None, max_retry=None, verbose=True,
        check_validity=True, resample_mode=False):
    recorder = Recorder(dataset, n_samples, save_dir, max_retry=max_retry, verbose=verbose)
    tmp_save_dir = os.path.join(save_dir, 'tmp')
    if n_cycles == 0 and dataset.config.name == 'Molecule':
        n_cycles = 1    # at least reconstruct once for small molecules
    
    model_autoencoder = model.autoencoder

    while not recorder.is_finished():
        batch_list = recorder.get_next_batch_list(batch_size)
        batch = dataset.collate_fn([dataset[i] for i, _ in batch_list])
        
        with torch.no_grad():
            cplx_descs = batch.pop('cplx_desc')
            # to GPU
            batch = to_device(batch, device)
            # inference
            if resample_mode:
                batch_S, batch_X, batch_A, batch_ll, batch_inter_bonds, batch_intra_bonds = model_autoencoder.generate(
                    **batch, confidence_model=conf_model if n_cycles == 0 else None, resample_mode=True,
                    disable_avoid_clash=sample_opt.get('vae_disable_avoid_clash', False))
            else:   # denovo generation
                for try_i in range(5):  # try for maximal of five times:
                    try:
                        batch_S, batch_X, batch_A, batch_ll, batch_inter_bonds, batch_intra_bonds = model.sample(
                            **batch, sample_opt=deepcopy(sample_opt), confidence_model=conf_model if n_cycles == 0 else None)
                        break
                    except AssertionError:
                        print_log(f'Generation failed, maybe due to large value of w in CFG, will try for the {try_i + 1}-th time', level='WARN')
            likelihoods = [ll.get('likelihood', None) for ll in batch_ll]   # saved here as it is derived from the LDM

        vae_batch_list = []
        for i in range(len(cplx_descs)):
            S, X, A, ll = batch_S[i], batch_X[i], batch_A[i], batch_ll[i]
            inter_bonds, intra_bonds = batch_inter_bonds[i], batch_intra_bonds[i]
            item_idx, n = batch_list[i]
            cplx_desc = cplx_descs[i]

            if n == 0:
                os.makedirs(os.path.join(tmp_save_dir, cplx_desc.id), exist_ok=True)
                os.makedirs(os.path.join(save_dir, cplx_desc.id), exist_ok=True)
                pocket_cplx = extract_sub_complex(cplx_desc.cplx, cplx_desc.pocket_block_ids)
                complex_to_pdb(pocket_cplx, os.path.join(tmp_save_dir, cplx_desc.id, 'pocket.pdb'), cplx_desc.tgt_chains)
                complex_to_pdb(pocket_cplx, os.path.join(save_dir, cplx_desc.id, 'pocket.pdb'), cplx_desc.tgt_chains)

            if n_cycles == 0: out_path = os.path.join(save_dir, cplx_desc.id, str(n) + '.pdb')
            else: out_path = os.path.join(tmp_save_dir, cplx_desc.id, str(n) + '.pdb')
            cplx_desc, details = data_to_cplx(
                cplx_desc, dataset.config, S, X, A, ll, inter_bonds, intra_bonds, out_path,
                check_validity = check_validity and (dataset.config.moltype == MolType.MOLECULE) and (n_cycles == 0),
                check_filters = (n_cycles == 0)
            )
            if cplx_desc is None:
                log = None
                assert n_cycles == 0  # only the last cycle has such possibility
            else: log = format_log(cplx_desc, details, n) 
            
            if n_cycles == 0: recorder.check_and_save(log, item_idx, n)
            else:
                data, cplx, pocket_block_ids, ligand_block_ids = _get_item_multitype(
                    os.path.join(tmp_save_dir, cplx_desc.id, f'pocket.pdb'),
                    out_path.rstrip('.pdb') + '.sdf',
                    # out_path,
                    out_path.rstrip('.pdb') + '.cif',
                    cplx_desc.tgt_chains,
                    cplx_desc.lig_chains,
                    getattr(dataset.config, 'cdr_type', None),
                    getattr(dataset.config, 'fr_len', None),
                    dataset.config.moltype
                )
                data['cplx_desc'] = ComplexDesc(
                    id=cplx_desc.id,
                    cplx=cplx,
                    tgt_chains=cplx_desc.tgt_chains,
                    lig_chains=list({bid[0]: 1 for bid in ligand_block_ids}.keys()), # keep the original order (for antibody)
                    pocket_block_ids=pocket_block_ids,
                    lig_block_ids=ligand_block_ids,
                    center_mask=data['center_mask'].tolist(),
                    generate_mask=data['generate_mask'].tolist(),
                    additional_props=modify_additional_props(cplx, cplx_desc.additional_props)
                )
                vae_batch_list.append(data)

        for cyc_i in range(n_cycles):
            print_log(f'Cycle: {cyc_i}', level='DEBUG')
            final_cycle = cyc_i == n_cycles - 1
            batch = dataset.collate_fn(vae_batch_list)
            vae_batch_list = []
            with torch.no_grad():
                if final_cycle: batch['topo_generate_mask'] = torch.zeros_like(batch['generate_mask'])
                cplx_descs = batch.pop('cplx_desc')
                batch = to_device(batch, device)
                batch_S, batch_X, batch_A, batch_ll, batch_inter_bonds, batch_intra_bonds = model_autoencoder.generate(
                    **batch, confidence_model=conf_model, disable_avoid_clash=sample_opt.get('vae_disable_avoid_clash', False))
            for i in range(len(cplx_descs)):
                S, X, A, ll = batch_S[i], batch_X[i], batch_A[i], batch_ll[i]
                # set likelihood
                ll['likelihood'] = likelihoods[i]
                inter_bonds, intra_bonds = batch_inter_bonds[i], batch_intra_bonds[i]
                item_idx, n = batch_list[i]
                cplx_desc = cplx_descs[i]
                # cplx_desc = ori_vae_batch_list[i]['cplx_desc']

                if final_cycle: out_path = os.path.join(save_dir, cplx_desc.id, str(n) + '.pdb')
                else: out_path = os.path.join(tmp_save_dir, cplx_desc.id, f'{n}_cyc{cyc_i}.pdb')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cplx_desc, details = data_to_cplx(
                    cplx_desc, dataset.config, S, X, A, ll, inter_bonds, intra_bonds, out_path,
                    check_validity = check_validity and (dataset.config.moltype == MolType.MOLECULE) and final_cycle,
                    check_filters = final_cycle
                )
                if cplx_desc is None:
                    log = None
                    assert final_cycle  # only the last cycle has such possibility
                else: log = format_log(cplx_desc, details, n) 

                if final_cycle: recorder.check_and_save(log, item_idx, n)
                else:
                    data, cplx, pocket_block_ids, ligand_block_ids = _get_item_multitype(
                        os.path.join(tmp_save_dir, cplx_desc.id, f'pocket.pdb'),
                        out_path.rstrip('.pdb') + '.sdf',
                        # out_path,
                        out_path.rstrip('.pdb') + '.cif',
                        cplx_desc.tgt_chains,
                        cplx_desc.lig_chains,
                        getattr(dataset.config, 'cdr_type', None),
                        getattr(dataset.config, 'fr_len', None),
                        dataset.config.moltype
                    )
                    data['cplx_desc'] = ComplexDesc(
                        id=cplx_desc.id,
                        cplx=cplx,
                        tgt_chains=cplx_desc.tgt_chains,
                        lig_chains=list({bid[0]: 1 for bid in ligand_block_ids}.keys()), # keep the original order (for antibody)
                        pocket_block_ids=pocket_block_ids,
                        lig_block_ids=ligand_block_ids,
                        center_mask=data['center_mask'].tolist(),
                        generate_mask=data['generate_mask'].tolist(),
                        additional_props=modify_additional_props(cplx, cplx_desc.additional_props)
                    )
                    vae_batch_list.append(data)

    print_log(f'Failed rate: {recorder.num_failed / recorder.num_generated}', level='DEBUG')


def change_cdr_length(in_cif, out_cif, cdr_type, mark_dict, chain_id, length):
    mark_dict = { c: mark_dict[c] for c in mark_dict }
    if in_cif.endswith('.pdb'): cplx = pdb_to_complex(in_cif)
    else: cplx = mmcif_to_complex(in_cif)
    all_chain_ids = [chain.id for chain in cplx]
    chain_type, cdr_digit = cdr_type[0], cdr_type[-1]

    if chain_id in mark_dict: mark = mark_dict[chain_id]
    else: # first get the cdr mark
        Nsys = get_nsys()
        if cdr_type.startswith('H'): mark = Nsys.mark_heavy_seq([block.id[0] for block in cplx[chain_id]])
        else: mark = Nsys.mark_light_seq([block.id[0] for block in cplx[chain_id]])
    start_index, end_index = mark.index(cdr_digit), mark.rindex(cdr_digit)

    # create blocks
    chain_blocks = cplx[chain_id].blocks
    num_add = length - (end_index - start_index + 1)
    if num_add == 0: pass   # nothing changed
    if num_add < 0:    # delete
        chain_blocks = chain_blocks[:start_index + length] + chain_blocks[end_index + 1:]
    else: # addition
        coord = chain_blocks[end_index][0].get_coord()  # dummy coordinates
        blocks = [Block(
            name='GLY', atoms=[Atom(name='CA', coordinate=coord, element='C', id=-1)], id=[None, '']
        ) for _ in range(num_add)]
        chain_blocks = chain_blocks[:end_index + 1] + blocks + chain_blocks[end_index + 1:]
    mark = mark[:start_index] + cdr_digit * length + mark[end_index + 1:]
    pos_ids = assign_pos_ids(mark, chain_type)
    start_index, end_index = mark.index(cdr_digit), mark.rindex(cdr_digit)
    assert len(mark) == len(chain_blocks), f'{len(mark)} vs {len(chain_blocks)}'
    for block, pos_id in zip(chain_blocks[start_index:end_index + 1], pos_ids[start_index:end_index + 1]):
        block.id = pos_id

    cplx[chain_id].blocks = chain_blocks

    # delete bonds
    mol_idx = cplx.id2idx[chain_id]
    new_bonds = []
    for bond in cplx.bonds:
        if bond.index1[0] == mol_idx or bond.index2[0] == mol_idx: continue
        new_bonds.append(bond)
    cplx.bonds = new_bonds

    # save
    complex_to_mmcif(cplx, out_cif, all_chain_ids)

    mark_dict[chain_id] = mark
    return mark_dict


def merge_sdfs(sdfs, out_sdf):
    all_mols = []
    for sdf in sdfs:
        supplier = Chem.SDMolSupplier(sdf)
        all_mols.extend([mol for mol in supplier if mol is not None])
    writer = Chem.SDWriter(out_sdf)
    for mol in all_mols: writer.write(mol)
    writer.close()


def merge_jsons(new_keys, jsons, out_json):
    item = {}
    for key, f in zip(new_keys, jsons):
        item[key] = json.load(open(f, 'r'))
    json.dump(item, open(out_json, 'w'))


def generate_multiple_cdrs(
        model, dataset, n_samples, batch_size, save_dir, device, sample_opt,
        n_cycles=0, conf_model=None, max_retry=None, verbose=True,
        check_validity=True, resample_mode=False
    ):

    assert len(dataset.pdb_paths) == 1, f'Only single target file supported for multiple CDR design'
    pdb_path = dataset.pdb_paths[0]
    id = os.path.basename(os.path.splitext(pdb_path)[0])

    sample_trajs = [[] for _ in range(n_samples)]   # recording paths and logs of each cdr for each sample

    dataset = deepcopy(dataset)
    dummy_template = dataset.config

    pdb_paths = []
    tgt_chains = dataset.tgt_chains * n_samples
    lig_chains = dataset.lig_chains * n_samples
    marks = [dummy_template.specify_regions] * n_samples

    tmp_files = []
    for _ in range(n_samples):
        tmp_file = tempfile.NamedTemporaryFile(suffix='.' + dataset.pdb_paths[0].split('.')[-1])
        with open(tmp_file.name, 'w') as fin: fin.write(open(dataset.pdb_paths[0], 'r').read())
        tmp_files.append(tmp_file)
        pdb_paths.append(tmp_file.name)

    for cdr_type in dummy_template.cdr_types:

        # change CDR length
        if cdr_type in dummy_template.length_ranges:
            l, r = dummy_template.length_ranges[cdr_type]
            new_paths, new_marks = [], []
            for path, chain_ids, mark in zip(pdb_paths, lig_chains, marks):
                length = random.randint(l, r + 1)
                tmp_file = tempfile.NamedTemporaryFile(suffix='.cif')
                cur_chain_id = chain_ids[0] if cdr_type.startswith('H') else chain_ids[1]
                new_mark = change_cdr_length(
                    path, tmp_file.name, cdr_type, mark, cur_chain_id, length)
                new_paths.append(tmp_file.name)
                tmp_files.append(tmp_file)
                new_marks.append(new_mark)
            pdb_paths, marks = new_paths, new_marks
        
        # generate new dataset
        dataset = PDBDataset(
            pdb_paths=pdb_paths,
            tgt_chains=tgt_chains,
            template_config=None,
            lig_chains=lig_chains,
        )

        print_log(f'processing {cdr_type}...')
        cur_save_dir = os.path.join(save_dir, 'intermediate', cdr_type)
        raw_save_dir = os.path.join(cur_save_dir, 'raw')
        os.makedirs(raw_save_dir, exist_ok=True)
        template = Antibody(cdr_type=cdr_type, fr_len=dummy_template.fr_len)
        dataset.config = template
        generate_for_one_template(
            model, dataset, 1, batch_size, raw_save_dir, device, sample_opt,
            n_cycles, conf_model, max_retry, verbose, check_validity, resample_mode
        )

        with open(os.path.join(raw_save_dir, 'results.jsonl'), 'r') as fin:
            items = [json.loads(l) for l in fin.readlines()]
            assert len(items) == len(sample_trajs)
        pdb_paths, tgt_chains, lig_chains = [], [], []  # update
        tgt_dir = os.path.join(cur_save_dir, id)
        os.makedirs(tgt_dir, exist_ok=True)
        for i, item in enumerate(items):
            # change CDR length
            path_prefix = os.path.join(raw_save_dir, item['id'], str(item['n']))
            tgt_path = os.path.join(tgt_dir, str(i) + '.cif')
            os.system(f'mv {path_prefix}.cif {tgt_path}')
            pdb_paths.append(tgt_path)
            tgt_chains.append(item['tgt_chains'])
            lig_chains.append(item['lig_chains'])
            sample_trajs[i].append((item, path_prefix))

        # # generate new dataset
        # dataset = PDBDataset(
        #     pdb_paths=pdb_paths,
        #     tgt_chains=tgt_chains,
        #     template_config=None,
        #     lig_chains=lig_chains,
        # )
        # n_samples = 1   # overwrite n_samples
        for tmp_file in tmp_files: tmp_file.close()
        tmp_files = []    


    # aggregate results
    # cif files
    os.system(f'cp -r {tgt_dir} {save_dir}')
    # copy pocket file
    os.system(f'cp {os.path.join(os.path.dirname(sample_trajs[0][0][1]), "pocket.pdb")} {os.path.join(save_dir, id, "pocket.pdb")}')
    res_file = open(os.path.join(save_dir, 'results.jsonl'), 'w')
    for i, trajs in enumerate(sample_trajs):
        tgt_chains, lig_chains = trajs[0][0]['tgt_chains'], trajs[0][0]['lig_chains']
        # 1. write pdb
        cplx = mmcif_to_complex(os.path.join(save_dir, id, str(i) + '.cif'), selected_chains=tgt_chains + lig_chains)
        complex_to_pdb(cplx, os.path.join(save_dir, id, str(i) + '.pdb'), selected_chains=tgt_chains + lig_chains)
        # 2. write sdf
        merge_sdfs([(tup[1] + '.sdf') for tup in trajs], os.path.join(save_dir, id, str(i) + '.sdf'))
        # 3. write confidence
        merge_jsons(dummy_template.cdr_types, [(tup[1] + '_confidence.json') for tup in trajs], os.path.join(save_dir, id, str(i) + '_confidence.json'))
        # 4. write log
        item = {
            'id': id,
            'n': i,
        }
        for key in ['pmetric', 'confidence', 'likelihood', 'normalized_likelihood']:
            if key not in trajs[0][0]: continue
            vals = [tup[0][key] for tup in trajs]
            item[key] = sum(vals) / len(vals)
            item[key + '_details'] = vals
        item['smiles'] = '.'.join([tup[0]['smiles'] for tup in trajs])
        item['gen_seq'] = '|'.join([tup[0]['gen_seq'] for tup in trajs])
        item['tgt_chains'] = tgt_chains
        item['lig_chains'] = lig_chains
        gen_block_idx = []
        for tup in trajs: gen_block_idx.extend(tup[0]['gen_block_idx'])
        item['gen_block_idx'] = gen_block_idx
        item['struct_only'] = trajs[0][0]['struct_only']
        item['mark'] = marks[i]
        res_file.write(json.dumps(item) + '\n')
    res_file.close()