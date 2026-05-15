# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import json
import shutil
import random
from copy import deepcopy
from typing import List, Optional, Union
from dataclasses import dataclass, asdict

import numpy as np

from api.tools.cofold.backends import get_backend, ChainData, CofoldConfidences, CofoldTask
from api.tools.cofold import utils as cofold_utils
from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from data.bioparse import tools
from utils.logger import print_log
from utils.singleton import singleton

from .funcs import get_binding_site, get_contact_is_cdr_ratio, renumber_ab


@singleton
class MetaData:

    def __init__(self):
        self.global_count = 0
        self.finish_init = False

    def to_dict(self): return self.__dict__

    def update_dict(self, data: dict):
        for key in data: setattr(self, key, data[key])


RANKING_WEIGHTS = { # default weights
    'metrics.bs_overlap': 10.0, # this is the factor of the first priority
    'metrics.contact_cdr_ratio': 5.0,   # second priority
    # the remainings sum up to 1.0
    'cplx_confidences.cofold_iptm': 0.3,
    'cplx_confidences.cofold_binder_plddt': 0.2,
    'cplx_confidences.cofold_normalized_ipae': 0.1,
    'generative_confidences.normalized_cdr_design_likelihood': 0.2,
    'metrics.normalized_scRMSD_cdr': 0.2,
}

PRIORITY_FILTERS = {
    'metrics.bs_overlap': lambda x: x > 0.4,
    'metrics.contact_cdr_ratio': lambda x: x > 0.4,
    'metrics.scRMSD': lambda x: x < 5.0,
    'metrics.lig_only_scRMSD': lambda x: x < 3.0
}


METADATA = MetaData()


@dataclass
class Metrics:
    bs_overlap: float
    contact_cdr_ratio: float

    # from complex prediction
    scRMSD: float
    scRMSD_cdr: float
    normalized_scRMSD: Optional[float] = None
    normalized_scRMSD_cdr: Optional[float] = None

    # from ligand-only prediction
    lig_only_scRMSD: Optional[float] = None
    normalized_lig_only_scRMSD: Optional[float] = None
    lig_only_scRMSD_cdr: Optional[float] = None
    normalized_lig_only_scRMSD_cdr: Optional[float] = None

    # from ligand templated complex prediction
    lig_temp_scRMSD: Optional[float] = None
    normalized_lig_temp_scRMSD: Optional[float] = None
    lig_temp_scRMSD_cdr: Optional[float] = None
    normalized_lig_temp_scRMSD_cdr: Optional[float] = None

    custom_metrics: Optional[dict] = None

    def process_metrics(self):
        def _normalize(rmsd):
            if rmsd is None: return None
            return 1.0 - rmsd / 40.0
        self.normalized_scRMSD = _normalize(self.scRMSD)
        self.normalized_scRMSD_cdr = _normalize(self.scRMSD_cdr)
        self.normalized_lig_only_scRMSD = _normalize(self.lig_only_scRMSD)
        self.normalized_lig_only_scRMSD_cdr = _normalize(self.lig_only_scRMSD_cdr)
        self.normalized_lig_temp_scRMSD = _normalize(self.lig_temp_scRMSD)
        self.normalized_lig_temp_scRMSD_cdr = _normalize(self.lig_temp_scRMSD_cdr)

    def to_str(self):
        self_dict, s = self.__dict__, []
        for key in self_dict:
            s.append(f'{key} ({self_dict[key]})')
        return ','.join(s)


@dataclass
class GenerativeConfidences:

    # for self
    confidence: Optional[float] = None
    likelihood: Optional[float] = None
    # for downstream generations based on this start point
    cdr_design_confidence: Optional[float] = None
    cdr_design_likelihood: Optional[float] = None

    # normalized
    normalized_confidence: Optional[float] = None
    normalized_likelihood: Optional[float] = None
    normalized_cdr_design_confidence: Optional[float] = None
    normalized_cdr_design_likelihood: Optional[float] = None
    
    def load_confidence(self, item):
        self.confidence = item['confidence']
        self.likelihood = item['likelihood']
        self.normalized_confidence = 1.0 - (self.confidence / 3.0)
        # TODO: normalized likelihood?
        # self.normalized_likelihood = 1.0 - (self.likelihood / len(item['gen_seq'])) / 400.0
        self.normalized_likelihood = item['normalized_likelihood']
        return True

    def load_cdr_design_confidence(self, jsonl_path):
        if not os.path.exists(jsonl_path): return False
        with open(jsonl_path, 'r') as fin:
            lines = fin.readlines()
        confs, lls, normalized_lls = [], [], []
        for line in lines:
            item = json.loads(line)
            confs.append(item['confidence'])
            lls.append(item['likelihood'])
            normalized_lls.append(item['likelihood'] / len(item['gen_seq']))
        if len(confs) > 0:
            self.cdr_design_confidence = sum(confs) / len(confs)
            self.normalized_cdr_design_confidence = 1.0 - (self.cdr_design_confidence / 3.0)
        if len(lls) > 0:
            self.cdr_design_likelihood = sum(lls) / len(lls)
            self.normalized_cdr_design_likelihood = 1.0 - (sum(normalized_lls) / len(normalized_lls)) / 400.0
        return True

    def to_str(self):
        self_dict, s = self.__dict__, []
        for key in self_dict:
            s.append(f'{key} ({self_dict[key]})')
        return ','.join(s)

            

@dataclass
class Candidate:
    id: str
    tgt_data: List[ChainData]

    # heavy chain
    hdata: Optional[ChainData] = None
    hmark: Optional[str] = None

    # light chain
    ldata: Optional[ChainData] = None
    lmark: Optional[str] = None

    # confidences (from cofold and Generation)
    cplx_confidences: Optional[CofoldConfidences] = None   # cplx prediction
    lig_only_confidences: Optional[CofoldConfidences] = None   # ligand only prediction
    lig_temp_confidences: Optional[CofoldConfidences] = None   # ligand templated complex prediction
    generative_confidences: Optional[GenerativeConfidences] = None

    # cofold results
    cofold_model: Optional[str] = 'boltz2'
    cofold_status: Optional[int] = None
    cofold_finish_flag: Optional[dict] = None
    cofold_struct_path: Optional[dict] = None
    metrics: Optional[Metrics] = None

    # CDR generation results
    candidate_dir: Optional[str] = None

    # Original generation path
    generated_path: Optional[str] = None

    # parent
    parent: Optional[str] = None

    # ood flag for subsequent generation
    ood_flag: bool = False


    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Union[str, dict]):
        if isinstance(data, str):   # a path is given
            with open(data, 'r') as fin: data = json.load(fin)
        data['tgt_data'] = [ChainData(**chain) for chain in data['tgt_data']]
        key2cls = {
            'hdata': ChainData,
            'ldata': ChainData,
            'cplx_confidences': CofoldConfidences,
            'lig_only_confidences': CofoldConfidences,
            'lig_temp_confidences': CofoldConfidences,
            'generative_confidences': GenerativeConfidences,
            'metrics': Metrics
        }
        for key in key2cls:
            if data[key] is None: continue
            data[key] = key2cls[key](**data[key])
        return cls(**data)

    def get_val_for_ranking(self):
        global RANKING_WEIGHTS, PRIORITY_FILTERS

        # If cofold failed (or not finished), metrics/confidences may be missing.
        # Return a very small value so these candidates sink to the bottom and
        # the pipeline can keep running.
        if self.metrics is None or self.cplx_confidences is None:
            return -1e9

        priority_flag = True
        for key in PRIORITY_FILTERS:
            depths = key.split('.')
            v = self
            for d in depths:
                if isinstance(v, dict): v = v[d]
                else: v = getattr(v, d)
            if v is None or (not PRIORITY_FILTERS[key](v)):
                priority_flag = False
        if priority_flag: # use cofold iptm only
            return 1e2 + self.cplx_confidences.cofold_iptm

        val = 0
        for key in RANKING_WEIGHTS:
            depths = key.split('.')
            v = self
            for d in depths:
                if isinstance(v, dict): v = v[d]
                else: v = getattr(v, d)
            # v = dicts[key]
            if v is None: continue
            val += v * RANKING_WEIGHTS[key]

        return val

    def _cofold_input_suffix(self) -> str:
        return get_backend(self.cofold_model).preferred_input_suffix()

    def _build_cofold_task(self, include_target: bool, ligand_template: bool, name_suffix: str) -> CofoldTask:
        """
        Build a backend-agnostic cofold task. Backend-specific formatting happens in CofoldTask.write_input().
        """
        backend = get_backend(self.cofold_model)
        chains = []
        if include_target:
            chains.extend([chain for chain in self.tgt_data])

        if self.hdata is not None:
            if ligand_template and self.generated_path is not None:
                chains.append(backend.apply_manual_template(self.hdata, self.generated_path))
            else:
                chains.append(self.hdata)
        if self.ldata is not None:
            if ligand_template and self.generated_path is not None:
                chains.append(backend.apply_manual_template(self.ldata, self.generated_path))
            else:
                chains.append(self.ldata)

        return CofoldTask(
            name=self.id + name_suffix,
            chains=chains,
            props={}
        )
    
    def launch_cofold(self, proj_dir, n_seeds):

        # complex prediction
        task = self._build_cofold_task(include_target=True, ligand_template=False, name_suffix='')
        input_path = os.path.join(proj_dir, task.name + self._cofold_input_suffix())
        task.write_input(self.cofold_model, input_path, n_seeds=n_seeds)

        if self.generated_path is not None:
            # ligand prediction (for scRMSD on the ligand)
            task = self._build_cofold_task(include_target=False, ligand_template=False, name_suffix='_ligand_only')
            input_path = os.path.join(proj_dir, task.name + self._cofold_input_suffix())
            task.write_input(self.cofold_model, input_path, n_seeds=n_seeds)

            # complex prediction with ligand-side templates
            task = self._build_cofold_task(include_target=True, ligand_template=True, name_suffix='_ligand_template')
            input_path = os.path.join(proj_dir, task.name + self._cofold_input_suffix())
            task.write_input(self.cofold_model, input_path, n_seeds=n_seeds)

    def check_cofold_finished(self, proj_dir):
        if (self.cofold_finish_flag is not None) and all(flag > 0 for flag in self.cofold_finish_flag.values()): # all finished
            return max(self.cofold_finish_flag.values())
        
        # update self.cofold_finish_flag, self.cofold_struct_path, and self.confidences
        def _check_name(name):
            status_file = os.path.join(proj_dir, 'logs', name, '_STATUS')
            if not os.path.exists(status_file): return  0   # still waiting
            mark = open(status_file, 'r').read().strip('\n')
            if mark == 'SUCCEEDED':
                print_log(f'cofold prediction succeeded for {name}.')
                return 1
            elif mark == 'FAILED':
                print_log(f'cofold prediction failed for {name}. Please check its logs.', level='WARN')
                return 2
            return 0
        
        # check finish
        if self.cofold_finish_flag is None: self.cofold_finish_flag = {}

        if self.cofold_finish_flag.get('cplx', 0) == 0:    # check
            self.cofold_finish_flag['cplx'] = _check_name(self.id)
        if self.generated_path is not None:
            if self.cofold_finish_flag.get('ligand_only', 0) == 0:
                self.cofold_finish_flag['ligand_only'] = _check_name(self.id + '_ligand_only')
            if self.cofold_finish_flag.get('ligand_template', 0) == 0:
                self.cofold_finish_flag['ligand_template'] = _check_name(self.id + '_ligand_template')
        
        if all(flag > 0 for flag in self.cofold_finish_flag.values()): # all finished
            # setup structure path and confidences
            backend = get_backend(self.cofold_model)
            
            if self.cofold_struct_path is None: self.cofold_struct_path = {}
            if self.cofold_finish_flag.get('cplx', 0) == 1:
                name = self.id.lower()
                if self.cplx_confidences is None: self.cplx_confidences = CofoldConfidences()
                self.cofold_struct_path['cplx'] = os.path.join(proj_dir, 'output', name, f'{name}_model.cif')
                self.cplx_confidences.apply_standardized(backend.load_confidences(
                    os.path.join(proj_dir, 'output', name, f'{name}_summary_confidences.json'),
                    self.get_tgt_chains(), self.get_lig_chains()
                ))
            if self.cofold_finish_flag.get('ligand_only', 0) == 1:
                name = self.id.lower() + '_ligand_only'
                if self.lig_only_confidences is None: self.lig_only_confidences = CofoldConfidences()
                self.cofold_struct_path['ligand_only'] = os.path.join(proj_dir, 'output', name, f'{name}_model.cif')
                self.lig_only_confidences.apply_standardized(backend.load_confidences(
                    os.path.join(proj_dir, 'output', name, f'{name}_summary_confidences.json'),
                    [], self.get_lig_chains()
                ))
            if self.cofold_finish_flag.get('ligand_template', 0) == 1:
                name = self.id.lower() + '_ligand_template'
                if self.lig_temp_confidences is None: self.lig_temp_confidences = CofoldConfidences()
                self.cofold_struct_path['ligand_template'] = os.path.join(proj_dir, 'output', name, f'{name}_model.cif')
                self.lig_temp_confidences.apply_standardized(backend.load_confidences(
                    os.path.join(proj_dir, 'output', name, f'{name}_summary_confidences.json'),
                    self.get_tgt_chains(), self.get_lig_chains()
                ))
            self.cofold_status = max(self.cofold_finish_flag.values())
            return self.cofold_status
        return 0

    def update_metrics(self, ref_bs_residues, specify_cdrs=None, cofold_mode='ligand_template', custom_metrics=None):
        tgt_chains = [item.id for item in self.tgt_data]
        lig_chains = []
        if self.hdata is not None: lig_chains.append(self.hdata.id)
        if self.ldata is not None: lig_chains.append(self.ldata.id)
        bs_residues = get_binding_site(self.cofold_struct_path[cofold_mode], tgt_chains, lig_chains)
        contact_cdr_ratio = get_contact_is_cdr_ratio(
            self.cofold_struct_path[cofold_mode], tgt_chains,
            self.hdata.id if self.hdata is not None else None, self.hmark,
            self.ldata.id if self.ldata is not None else None, self.lmark,
            specify_cdrs
        )
        bs_overlap = len(bs_residues.intersection(ref_bs_residues)) / (min(len(ref_bs_residues), len(bs_residues)) + 1e-5)
        if self.generated_path is not None:
            # get CDR mark
            is_cdr = []
            if self.hmark is not None:
                for m in self.hmark: is_cdr.append(int(m != '0'))
            if self.lmark is not None:
                for m in self.lmark: is_cdr.append(int(m != '0'))
            is_cdr = np.array(is_cdr, dtype=bool)
            scRMSD, scRMSD_cdr = cofold_utils.get_scRMSD(self.cofold_struct_path['cplx'], self.generated_path, self.get_tgt_chains(), self.get_lig_chains(), is_cdr)
            lig_only_scRMSD, lig_only_scRMSD_cdr = cofold_utils.get_scRMSD(self.cofold_struct_path['ligand_only'], self.generated_path, [], self.get_lig_chains(), is_cdr, align_by_target=False)
            lig_temp_scRMSD, lig_temp_scRMSD_cdr = cofold_utils.get_scRMSD(self.cofold_struct_path['ligand_template'], self.generated_path, self.get_tgt_chains(), self.get_lig_chains(), is_cdr)
        else: scRMSD, scRMSD_cdr, lig_only_scRMSD, lig_only_scRMSD_cdr, lig_temp_scRMSD, lig_temp_scRMSD_cdr = None, None, None, None, None, None
        self.metrics = Metrics(
            bs_overlap = bs_overlap,
            contact_cdr_ratio = contact_cdr_ratio,
            scRMSD = scRMSD,
            scRMSD_cdr = scRMSD_cdr,
            lig_only_scRMSD = lig_only_scRMSD,
            lig_only_scRMSD_cdr = lig_only_scRMSD_cdr,
            lig_temp_scRMSD = lig_temp_scRMSD,
            lig_temp_scRMSD_cdr = lig_temp_scRMSD_cdr
        )
        self.metrics.process_metrics()
        if custom_metrics is not None:
            self.metrics.custom_metrics = {}
            for name in custom_metrics: self.metrics.custom_metrics[name] = custom_metrics[name](self, cofold_mode)
        return self.metrics
    
    def set_msa_from_cofold_results(self, proj_dir):
        data = json.load(open(os.path.join(proj_dir, 'output', self.id.lower(), self.id.lower() + '_data.json'), 'r'))
        id2msa = {}
        for item in data['sequences']:
            for key in item: id2msa[item[key]['id']] = item[key]
        for chain in self.tgt_data:
            chain.set_msa(id2msa[chain.id])

        if self.hdata is not None: self.hdata.set_msa(id2msa[self.hdata.id])
        if self.ldata is not None: self.ldata.set_msa(id2msa[self.ldata.id])

    def get_summary(self):
        global RANKING_WEIGHTS

        s = self.id + ','
        s += '\n\tmetrics: ' + self.metrics.to_str()
        if self.cplx_confidences is not None: s += '\n\tcplx: ' + self.cplx_confidences.to_str()
        if self.lig_only_confidences is not None: s += '\n\tligand only: ' + self.lig_only_confidences.to_str()
        if self.lig_temp_confidences is not None: s += '\n\tligand template: ' + self.lig_temp_confidences.to_str()
        s += f'\n\tOverall score: {self.get_val_for_ranking()}, \n\tconfig {RANKING_WEIGHTS}'
        return s
    
    def get_briefs(self):
        return {
            'heavychain': None if self.hdata is None else { 'sequence': self.hdata.sequence, 'mark': self.hmark },
            'lightchain': None if self.ldata is None else { 'sequence': self.ldata.sequence, 'mark': self.lmark },
            'metrics': self.metrics.__dict__,
            'cplx_confidences': self.cplx_confidences.__dict__,
            'lig_only_confidences': self.lig_only_confidences.__dict__,
            'lig_temp_confidences': self.lig_temp_confidences.__dict__,
            'generative_confidences': self.generative_confidences.__dict__
        }

    def save_results(self, save_dir, renumber=True, cofold_mode='ligand_template', update_only=False):
        out_dir = os.path.join(save_dir, self.id)
        if not update_only: # need to copy cofold results
            os.makedirs(out_dir, exist_ok=True)
            # copy structure file and renumber the antibody
            path = self.cofold_struct_path[cofold_mode] if cofold_mode in self.cofold_struct_path else self.cofold_struct_path['cplx']
            if not os.path.exists(os.path.join(out_dir, 'model.cif')):
                if renumber:
                    renumber_ab(
                        path, os.path.join(out_dir, 'model.cif'),
                        [item.id for item in self.tgt_data], None if self.hdata is None else self.hdata.id,
                        self.hmark, None if self.ldata is None else self.ldata.id, self.lmark
                    )
                else:
                    shutil.copyfile(path, os.path.join(out_dir, 'model.cif'))
            # add a conversion from cif to pdb in case needed
            cplx = mmcif_to_complex(os.path.join(out_dir, 'model.cif'))
            complex_to_pdb(cplx, os.path.join(out_dir, 'model.pdb'), [mol.id for mol in cplx])
            # copy other cofold results
            for mode in self.cofold_struct_path:
                shutil.copyfile(self.cofold_struct_path[mode], os.path.join(out_dir, f'{mode}_model.cif'))
        # write summary
        summary = open(os.path.join(out_dir, 'summary.json'), 'w')
        json.dump(self.get_briefs(), summary, indent=2)
        summary.close()
    
    def save_full_data(self, save_dir):
        path = os.path.join(save_dir, self.id, 'full_data.json')
        if os.path.exists(path): return # already saved
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as fout: json.dump(self.to_dict(), fout, indent=2)

    def get_tgt_chains(self):
        return [item.id for item in self.tgt_data]
    
    def get_lig_chains(self):
        lig_chains = []
        if self.hdata is not None: lig_chains.append(self.hdata.id)
        if self.ldata is not None: lig_chains.append(self.ldata.id)
        return lig_chains

    # regarding designing
    def set_candidate_dir(self, candidate_dir):
        self.candidate_dir = candidate_dir

    def update_design_confidence(self):
        if self.generative_confidences is None: self.generative_confidences = GenerativeConfidences()
        return self.generative_confidences.load_cdr_design_confidence(os.path.join(self.candidate_dir, 'results.jsonl'))
    
    def get_new_candidates(self, save_dir, n=5, use_global_id=True): 

        with open(os.path.join(self.candidate_dir, 'results.jsonl'), 'r') as fin:
            lines = fin.readlines()
        items = [json.loads(line) for line in (lines if n is None else lines[:n])]
        candidates = []
        for item in items:
            cif_path = os.path.join(self.candidate_dir, item['id'], str(item['n']) + '.cif')
            seqs = tools.extract_seqs(cif_path, item['lig_chains'])
            marks = item.get('marks', None)
            if self.hdata is None: hdata, hmark = None, None
            else:
                hdata = deepcopy(self.hdata)
                hdata.update_sequence(seqs.pop(0))
                hmark = self.hmark if marks is None else marks.pop(0)
            if self.ldata is None: ldata, lmark = None, None
            else:
                ldata = deepcopy(self.ldata)
                ldata.update_sequence(seqs.pop(0))
                lmark = self.lmark if marks is None else marks.pop(0)
            
            # copy generated structure to new folder
            if use_global_id: cand_id = str(MetaData().global_count).zfill(6)
            else: cand_id = self.id + '_' + item['id'] + '_' + str(item['n'])
            cur_save_dir = os.path.join(save_dir, cand_id, 'generated')
            os.makedirs(cur_save_dir, exist_ok=True)
            for suffix in ['.cif', '.pdb', '.sdf', '_confidence.json']:
                shutil.copyfile(
                    os.path.join(self.candidate_dir, item['id'], str(item['n']) + suffix),
                    os.path.join(cur_save_dir, 'model' + suffix)
                )

            candidate = Candidate(
                id=cand_id,
                tgt_data=self.tgt_data,
                hdata=hdata,
                hmark=hmark,
                ldata=ldata,
                lmark=lmark,
                cplx_confidences=CofoldConfidences(),
                lig_only_confidences=CofoldConfidences(),
                lig_temp_confidences=CofoldConfidences(),
                generative_confidences=GenerativeConfidences(),
                cofold_model=self.cofold_model,
                generated_path=os.path.join(cur_save_dir, 'model.cif'),
                parent=self.id
            )
            candidate.generative_confidences.load_confidence(item)
            candidates.append(candidate)
            MetaData().global_count += 1
        return candidates
