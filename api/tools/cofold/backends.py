# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
#
# Backend adapters for different cofolding models.
#
# Design goal: each model-specific logic lives in a single class so it is easy
# to extend/maintain.

from __future__ import annotations

from copy import deepcopy
import json
import os
import random
import tempfile
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from . import utils as cofold_utils
from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
from data.bioparse.writer.complex_to_mmcif import complex_to_mmcif


def _quote(s: str) -> str:
    import shlex
    return shlex.quote(s)


def _rewrite_mmcif_chain_ids(src_path: str, out_path: str, chain_ids: List[str]) -> None:
    """
    Rewrite model mmCIF chain ids to match the original input chain ids by order.

    Both boltz2 and protenix may emit consecutive asym ids (A/B/C/...) regardless
    of the user-provided chain labels (e.g. A/H/L). Downstream code in this repo
    expects the original chain ids.
    """
    cplx = mmcif_to_complex(src_path)
    if len(cplx) != len(chain_ids):
        raise ValueError(
            f"Number of chains in model ({len(cplx)}) does not match input chains ({len(chain_ids)}) for {src_path}"
        )
    for mol, cid in zip(cplx, chain_ids): mol.id = cid
    complex_to_mmcif(cplx, out_path, selected_chains=chain_ids)


@dataclass
class ChainData:
    id: str  # chain id
    sequence: str
    modifications: list
    # AF3-style MSA/template payloads (strings/objects embedded in input json).
    # If MSA is disabled, set to ''.
    unpairedMsa: Optional[str] = None
    pairedMsa: Optional[str] = None
    templates: Optional[list] = None
    type: str = "protein"

    def set_null_msa(self):
        self.unpairedMsa = ""
        self.pairedMsa = ""
        self.templates = []

    def set_msa(self, data_dict):
        # Some backends (e.g. boltz2) write *_data.json with `templates: []`
        # even when the original task used templates (e.g. target chain template).
        # Preserve existing templates unless the new payload provides a non-empty
        # replacement.
        self.unpairedMsa = data_dict["unpairedMsa"]
        self.pairedMsa = data_dict["pairedMsa"]
        new_templates = data_dict.get("templates")
        if new_templates:
            self.templates = new_templates
        elif self.templates is None:
            self.templates = new_templates

    def update_sequence(self, seq):
        # Update MSA query information (assume MSA remains the same for CDR redesign).
        if self.unpairedMsa == "":
            pass
        elif self.unpairedMsa is not None:
            self.unpairedMsa = self.unpairedMsa.replace(f"query\n{self.sequence}", f"query\n{seq}")
        if self.pairedMsa == "":
            pass
        elif self.pairedMsa is not None:
            self.pairedMsa = self.pairedMsa.replace(f"query\n{self.sequence}", f"query\n{seq}")
        self.sequence = seq

    def get_data_with_manual_template(self, cif_path):
        # Legacy helper: embed template mmCIF as AF3-style per-chain template.
        cplx = mmcif_to_complex(cif_path, selected_chains=[self.id])
        cif_str = cofold_utils._get_chain_str(cplx, self.id)
        idx = list(range(len(self.sequence)))
        template = {"mmcif": cif_str, "queryIndices": idx, "templateIndices": idx}
        data = deepcopy(self)
        data.unpairedMsa = ""
        data.pairedMsa = ""
        data.templates = [template]
        return data

    def get_data_with_boltz2_template(self, cif_path):
        # Legacy helper: stash boltz template metadata on the chain.
        data = deepcopy(self)
        data.unpairedMsa = ""
        data.pairedMsa = ""
        data.templates = [{"cif": os.path.abspath(cif_path), "chain_id": self.id}]
        return data


@dataclass
class CofoldConfidences:
    # unified, model-agnostic fields used by ranking/filters in this repo.
    cofold_iptm: Optional[float] = None
    cofold_ptm: Optional[float] = None
    cofold_ranking_score: Optional[float] = None
    cofold_binder_plddt: Optional[float] = None  # normalized to [0, 1] if available
    cofold_ipae: Optional[float] = None
    cofold_normalized_ipae: Optional[float] = None

    def apply_standardized(self, item: dict):
        self.cofold_iptm = item.get("cofold_iptm")
        self.cofold_ptm = item.get("cofold_ptm")
        self.cofold_ranking_score = item.get("cofold_ranking_score")
        self.cofold_binder_plddt = item.get("cofold_binder_plddt")
        self.cofold_ipae = item.get("cofold_ipae")
        if self.cofold_ipae is not None:
            self.cofold_normalized_ipae = 1.0 - (self.cofold_ipae / 31.0)
        else:
            self.cofold_normalized_ipae = None

    def to_str(self):
        self_dict, s = self.__dict__, []
        for key in self_dict:
            s.append(f"{key} ({self_dict[key]})")
        return ",".join(s)


@dataclass
class CofoldTask:
    name: str
    chains: List[ChainData]
    props: Optional[dict] = None

    def write_input(self, model: str, out_path: str, n_seeds: int = 1, include_chains: Optional[List[str]] = None):
        backend = get_backend(model)
        return backend.write_input(
            name=self.name,
            chains=self.chains,
            out_path=out_path,
            n_seeds=n_seeds,
            include_chains=include_chains,
        )


@dataclass(frozen=True)
class BackendTaskConfig:
    # Paths/config passed from the top-level pipeline.
    repo_dir: str = ""
    env: str = ""
    db: str = ""
    param: str = ""


class CofoldBackend(ABC):
    name: str

    @abstractmethod
    def input_suffixes(self) -> List[str]:
        raise NotImplementedError

    def preferred_input_suffix(self) -> str:
        return self.input_suffixes()[0]

    @abstractmethod
    def get_task_name(self, input_path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def write_input(
        self,
        name: str,
        chains: List[ChainData],
        out_path: str,
        n_seeds: int = 1,
        include_chains: Optional[List[str]] = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def load_confidences(
        self,
        summary_json_path: str,
        tgt_chains: List[str],
        lig_chains: List[str],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def build_command(self, cfg: BackendTaskConfig, gpu_ids: List[str], input_path: str, out_dir: str) -> str:
        raise NotImplementedError

    def preprocess_input(self, cfg: BackendTaskConfig, input_path: str) -> None:
        """
        Optional hook executed in the cofold worker process before launching the backend.
        Use it to do model-specific preparation steps (e.g. convert template CIFs).
        """
        _ = cfg, input_path
        return

    def postprocess_outputs(self, out_dir: str, input_name: str, input_path: str) -> None:
        # Optional hook. By default, assume outputs are already in the expected layout.
        return

    def apply_manual_template(self, chain: Any, cif_path: str) -> Any:
        """
        Apply a model-specific "manual template" to a single chain and return a
        modified copy of the chain object.
        """
        _ = cif_path
        return chain


'''
    Implementation of different cofolding backends
'''

class AlphaFold3Backend(CofoldBackend):
    name = "alphafold3"

    def input_suffixes(self) -> List[str]:
        return [".json"]

    def get_task_name(self, input_path: str) -> str:
        return json.load(open(input_path, "r"))["name"]

    def _chain_to_input_entry(self, chain: ChainData) -> Dict[str, Any]:
        item = {
            "id": chain.id,
            "sequence": chain.sequence,
            "modifications": chain.modifications,
        }
        if chain.unpairedMsa is not None:
            item["unpairedMsa"] = chain.unpairedMsa
        if chain.pairedMsa is not None:
            item["pairedMsa"] = chain.pairedMsa
        if chain.templates is not None:
            item["templates"] = chain.templates
        return {chain.type: item}

    def write_input(
        self,
        name: str,
        chains: List[ChainData],
        out_path: str,
        n_seeds: int = 1,
        include_chains: Optional[List[str]] = None,
    ) -> str:
        if include_chains is None:
            include_chains = [c.id for c in chains]
        data = [self._chain_to_input_entry(chain) for chain in chains if chain.id in include_chains]
        input_json = {
            "name": name,
            "dialect": "alphafold3",
            "version": 2,
            "modelSeeds": [random.randint(0, 4294967295) for _ in range(n_seeds)],
            "bondedAtomPairs": None,
            "userCCD": None,
            "sequences": data,
        }
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as fout:
            json.dump(input_json, fout, indent=2)
        return out_path

    def load_confidences(
        self,
        summary_json_path: str,
        tgt_chains: List[str],
        lig_chains: List[str],
    ) -> Dict[str, Any]:
        c2i = {c: i for i, c in enumerate(sorted(tgt_chains + lig_chains))}
        if len(tgt_chains) == 0 or len(lig_chains) == 0:
            iptm_row_cols = None
        else:
            iptm_row_cols = [(c2i[c1], c2i[c2]) for c1 in tgt_chains for c2 in lig_chains]
        item = cofold_utils.load_confidences(summary_json_path, tgt_chains, lig_chains, iptm_row_cols)
        binder_plddt = item.get("binder_plddt")
        return {
            "cofold_iptm": item.get("iptm"),
            "cofold_ptm": item.get("ptm"),
            "cofold_ranking_score": item.get("ranking_score"),
            # AF3 parser returns binder_plddt in [0, 100]
            "cofold_binder_plddt": (binder_plddt * 0.01) if binder_plddt is not None else None,
            "cofold_ipae": item.get("ipae"),
        }

    def build_command(self, cfg: BackendTaskConfig, gpu_ids: List[str], input_path: str, out_dir: str) -> str:
        script_path = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "af3_scripts", "alphafold3_predict.sh")
        )
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Cannot find AF3 script: {script_path}")

        cmd = f"""
            CUDA_VISIBLE_DEVICES={','.join(gpu_ids)} \\
            AF3_REPO_DIR={cfg.repo_dir} \\
            AF3_ENV={cfg.env} \\
            AF3_DB={cfg.db} \\
            AF3_PARAM={cfg.param} \\
            bash {script_path} \\
            --json_path {input_path} \\
            --output_dir {out_dir}
        """
        return cmd

    def apply_manual_template(self, chain: Any, cif_path: str) -> Any:
        # AF3 templates are embedded per-chain with explicit indices.
        data = chain.__class__(**chain.__dict__)
        cplx = mmcif_to_complex(cif_path, selected_chains=[chain.id])
        cif_str = cofold_utils._get_chain_str(cplx, chain.id)
        idx = list(range(len(chain.sequence)))
        template = {
            "mmcif": cif_str,
            "queryIndices": idx,
            "templateIndices": idx,
        }
        data.unpairedMsa = ""
        data.pairedMsa = ""
        data.templates = [template]
        return data


class Boltz2Backend(CofoldBackend):
    name = "boltz2"

    def input_suffixes(self) -> List[str]:
        return [".yaml", ".yml"]

    def get_task_name(self, input_path: str) -> str:
        base = os.path.basename(input_path)
        for ext in (".yaml", ".yml"):
            if base.endswith(ext):
                return base[: -len(ext)]
        return os.path.splitext(base)[0]

    def _chain_to_sequence_entry(self, chain: ChainData, force_single_seq: bool = True) -> Dict[str, Any]:
        if chain.type != "protein":
            raise ValueError(f"boltz2 backend currently supports only protein chains, got type={chain.type}")
        item: Dict[str, Any] = {
            "id": chain.id,
            "sequence": chain.sequence,
        }
        if chain.modifications:
            item["modifications"] = chain.modifications
        if force_single_seq:
            item["msa"] = "empty"
        return {"protein": item}

    def _extract_templates(self, chains: List[ChainData], include_chains: List[str]) -> List[Dict[str, Any]]:
        templates: List[Dict[str, Any]] = []
        for chain in chains:
            if chain.id not in include_chains:
                continue
            if not chain.templates:
                continue
            for template in chain.templates:
                if not isinstance(template, dict):
                    continue
                # Native Boltz-style template definition already present.
                if "cif" in template or "pdb" in template:
                    templates.append(dict(template))
                    continue
                # Convert AF3-style per-chain template metadata into a Boltz
                # top-level template entry. Boltz will auto-match the provided
                # template structure to the specified chain.
                mmcif_path = template.get("mmcifPath")
                if mmcif_path:
                    # Use the per-chain mmcif emitted by this repo (e.g. chain_A.cif).
                    # It is what the target/template pipeline intends to use; if it
                    # lacks entity metadata, we enrich it in the boltz2 preprocess step.
                    cif_path = mmcif_path
                    boltz_tpl = {
                        "cif": os.path.abspath(cif_path),
                        "chain_id": chain.id,
                    }
                    # Explicitly preserve mapping when the template chain id
                    # differs from the input chain id (e.g., template R -> input A).
                    template_chain_id = template.get("boltzTemplateChainId") or template.get("templateChainId")
                    if template_chain_id:
                        boltz_tpl["template_id"] = template_chain_id
                    templates.append(boltz_tpl)
        return templates

    def write_input(
        self,
        name: str,
        chains: List[ChainData],
        out_path: str,
        n_seeds: int = 1,
        include_chains: Optional[List[str]] = None,
    ) -> str:
        if include_chains is None:
            include_chains = [c.id for c in chains]
        seqs = [self._chain_to_sequence_entry(chain, force_single_seq=True) for chain in chains if chain.id in include_chains]
        input_yaml: Dict[str, Any] = {
            "version": 1,
            "sequences": seqs,
        }
        templates = self._extract_templates(chains, include_chains)
        if templates:
            input_yaml["templates"] = templates
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as fout:
            yaml.safe_dump(input_yaml, fout, default_flow_style=False, sort_keys=False)
        return out_path

    def load_confidences(
        self,
        summary_json_path: str,
        tgt_chains: List[str],
        lig_chains: List[str],
    ) -> Dict[str, Any]:
        """
        Boltz confidence JSON is indexed by numeric chain ids (strings "0", "1"...).
        This method computes interface aggregates based on the caller's
        target/ligand chain ids:
          iptm := mean of pair_chains_iptm over (T,L) and (L,T) for all T in targets and L in ligands
          ipae := mean of PAE submatrices between target and ligand tokens, in both directions
        """
        def _find_dir_case_insensitive(parent: str, base: str) -> str:
            cand = os.path.join(parent, base)
            if os.path.isdir(cand):
                return cand
            try:
                want = base.lower()
                for fn in os.listdir(parent):
                    p = os.path.join(parent, fn)
                    if os.path.isdir(p) and fn.lower() == want:
                        return p
            except Exception:
                pass
            return cand

        # Derive paths to full boltz outputs from the standardized summary location:
        #   <out_dir>/<lower_name>/<lower_name>_summary_confidences.json
        #
        # NOTE: boltz output directories preserve original case from the input yaml
        # basename (e.g. chain ids like "A_B_"). Our standardized outputs use
        # `lower_name`, so we resolve run roots and prediction subdirs
        # case-insensitively.
        lower_name = os.path.basename(os.path.dirname(summary_json_path))
        out_dir = os.path.dirname(os.path.dirname(summary_json_path))

        run_root = _find_dir_case_insensitive(out_dir, f"boltz_results_{lower_name}")
        if not os.path.isdir(run_root):
            raise FileNotFoundError(f"Boltz run root not found under {out_dir} for {lower_name}")

        pred_parent = os.path.join(run_root, "predictions")
        pred_dir = _find_dir_case_insensitive(pred_parent, lower_name)
        if not os.path.isdir(pred_dir):
            raise FileNotFoundError(f"Boltz predictions directory not found under {pred_parent} for {lower_name}")

        # Prefer "confidence_*_model_0.json" if filenames differ in case.
        conf_src = os.path.join(pred_dir, f"confidence_{lower_name}_model_0.json")
        if not os.path.exists(conf_src):
            cands = [
                os.path.join(pred_dir, fn)
                for fn in os.listdir(pred_dir)
                if fn.startswith("confidence_") and fn.endswith("_model_0.json")
            ]
            if len(cands) != 1:
                raise FileNotFoundError(f"Boltz confidence json not found (or ambiguous) in {pred_dir}")
            conf_src = cands[0]
        conf = json.load(open(conf_src, "r"))

        # Map chain_name -> numeric chain_id and derive token ranges from records.
        chain_map: Dict[str, int] = {}
        chain_ranges: Dict[str, slice] = {}
        # rec_dir = os.path.join(run_root, "processed", "records")
        # rec_path = os.path.join(rec_dir, f"{lower_name}.json")
        rec_path = os.path.join(run_root, "processed", "manifest.json")
        # if not os.path.exists(rec_path):
        #     # Case-insensitive fallback.
        #     try:
        #         for fn in os.listdir(rec_dir):
        #             if fn.lower() == f"{lower_name}.json".lower():
        #                 rec_path = os.path.join(rec_dir, fn)
        #                 break
        #     except Exception:
        #         pass
        if os.path.exists(rec_path):
            rec = json.load(open(rec_path, "r"))['records'][0]
            chains = sorted(rec.get("chains", []), key=lambda x: int(x.get("chain_id", 0)))
            start = 0
            for ch in chains:
                cid = ch.get("chain_name")
                idx = ch.get("chain_id")
                nres = ch.get("num_residues")
                if cid is None or idx is None or nres is None:
                    continue
                chain_map[cid] = int(idx)
                chain_ranges[cid] = slice(start, start + int(nres))
                start += int(nres)

        # Compute interface ipTM from pair_chains_iptm.
        pair = conf.get("pair_chains_iptm") or {}
        iptm_vals: List[float] = []
        for t in tgt_chains:
            for l in lig_chains:
                if t not in chain_map or l not in chain_map:
                    continue
                ti = str(chain_map[t])
                li = str(chain_map[l])
                v_tl = (pair.get(ti) or {}).get(li)
                v_lt = (pair.get(li) or {}).get(ti)
                if v_tl is not None:
                    iptm_vals.append(float(v_tl))
                if v_lt is not None:
                    iptm_vals.append(float(v_lt))
        interface_iptm = (sum(iptm_vals) / len(iptm_vals)) if iptm_vals else conf.get("iptm")

        # Compute interface ipAE from PAE matrix using token ranges.
        pae_val = None
        pae_path = os.path.join(pred_dir, f"pae_{lower_name}_model_0.npz")
        if not os.path.exists(pae_path):
            # Case-insensitive / wildcard fallback.
            try:
                cands = [fn for fn in os.listdir(pred_dir) if fn.startswith("pae_") and fn.endswith("_model_0.npz")]
                if len(cands) == 1:
                    pae_path = os.path.join(pred_dir, cands[0])
            except Exception:
                pass
        if os.path.exists(pae_path) and chain_ranges:
            pae = np.load(pae_path)["pae"]
            blocks: List[np.ndarray] = []
            for t in tgt_chains:
                for l in lig_chains:
                    if t not in chain_ranges or l not in chain_ranges:
                        continue
                    ts = chain_ranges[t]
                    ls = chain_ranges[l]
                    blocks.append(pae[ts, ls])
                    blocks.append(pae[ls, ts])
            if blocks:
                pae_val = float(np.concatenate([b.reshape(-1) for b in blocks]).mean())

        return {
            "cofold_iptm": interface_iptm,
            "cofold_ptm": conf.get("ptm"),
            "cofold_ranking_score": conf.get("confidence_score"),
            "cofold_binder_plddt": conf.get("complex_plddt"),
            "cofold_ipae": pae_val,
        }

    def _boltz_bin(self, env_prefix_or_name: str) -> str:
        cand = os.path.join(env_prefix_or_name, "bin", "boltz")
        if os.path.exists(cand):
            return cand
        return "boltz"

    def preprocess_input(self, cfg: BackendTaskConfig, input_path: str) -> None:
        """
        Convert template mmCIFs referenced by the input YAML into a boltz-friendly
        mmCIF in-place (adds entity/polymer metadata).

        This is executed in the worker process where we know the boltz env path.
        """
        py = os.path.join(cfg.env, "bin", "python")

        script = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "boltz2", "boltz2_template_convert.py")
        )
        # Keep this as a subprocess call (no heredoc) so the command stays clean.
        subprocess.check_call([py, script, os.path.abspath(input_path)])

    def build_command(self, cfg: BackendTaskConfig, gpu_ids: List[str], input_path: str, out_dir: str) -> str:
        boltz = self._boltz_bin(cfg.env)

        extra_flags: List[str] = []
        if cfg.param:
            extra_flags += [f"--cache {_quote(cfg.param)}"]

        # Runtime tuning knobs (useful for smoke tests or mixed clusters):
        # - ANEW_BOLTZ2_ACCELERATOR: gpu|cpu (default: gpu)
        # - ANEW_BOLTZ2_DEVICES: int (default: 1)
        # - ANEW_BOLTZ2_SAMPLING_STEPS / ANEW_BOLTZ2_RECYCLING_STEPS: ints (optional)
        # accelerator = os.environ.get("ANEW_BOLTZ2_ACCELERATOR", "gpu")
        # devices = int(os.environ.get("ANEW_BOLTZ2_DEVICES", "1"))
        sampling_steps = os.environ.get("ANEW_BOLTZ2_SAMPLING_STEPS")
        recycling_steps = os.environ.get("ANEW_BOLTZ2_RECYCLING_STEPS")

        if sampling_steps:
            extra_flags += [f"--sampling_steps {int(sampling_steps)}"]
        if recycling_steps:
            extra_flags += [f"--recycling_steps {int(recycling_steps)}"]

        cmd = f"""
            CUDA_VISIBLE_DEVICES={','.join(gpu_ids)} \\
            {boltz} predict {input_path} \\
            --out_dir {out_dir} \\
            --output_format mmcif \\
            --diffusion_samples 1 \\
            --write_full_pae \\
            {' '.join(extra_flags)}
        """
        return cmd

    def postprocess_outputs(self, out_dir: str, input_name: str, input_path: str) -> None:
        """
        Convert Boltz outputs into unified layout so downstream code
        can remain unchanged:
          out_dir/<lower_name>/<lower_name>_model.cif
          out_dir/<lower_name>/<lower_name>_summary_confidences.json
          out_dir/<lower_name>/<lower_name>_data.json
        """
        lower = input_name.lower()
        # Boltz writes into:
        #   <out_dir>/boltz_results_<input_name>/predictions/<input_name>/
        # rather than directly under <out_dir>/predictions/.
        run_root = os.path.join(out_dir, f"boltz_results_{input_name}")
        if not os.path.isdir(run_root):
            run_root = os.path.join(out_dir, f"boltz_results_{lower}")
        if not os.path.isdir(run_root):
            raise FileNotFoundError(
                f"Boltz run root not found under {out_dir} for {input_name}"
            )

        pred_dir = os.path.join(run_root, "predictions", input_name)
        if not os.path.isdir(pred_dir):
            pred_dir = os.path.join(run_root, "predictions", lower)
        if not os.path.isdir(pred_dir):
            raise FileNotFoundError(
                f"Boltz predictions directory not found under {run_root}/predictions for {input_name}"
            )

        model_src = os.path.join(pred_dir, f"{input_name}_model_0.cif")
        if not os.path.exists(model_src):
            model_src = os.path.join(pred_dir, f"{lower}_model_0.cif")
        if not os.path.exists(model_src):
            raise FileNotFoundError(f"Boltz model file not found in {pred_dir}")

        conf_src = os.path.join(pred_dir, f"confidence_{input_name}_model_0.json")
        if not os.path.exists(conf_src):
            conf_src = os.path.join(pred_dir, f"confidence_{lower}_model_0.json")
        if not os.path.exists(conf_src):
            raise FileNotFoundError(f"Boltz confidence json not found in {pred_dir}")

        out_std = os.path.join(out_dir, lower)
        os.makedirs(out_std, exist_ok=True)

        # Create an *_data.json so the pipeline can keep "MSA disabled" behavior.
        input_yaml = yaml.safe_load(open(input_path, "r"))
        seq_items, chain_ids = [], []
        for entry in input_yaml.get("sequences", []):
            if "protein" not in entry:
                continue
            p = entry["protein"]
            seq_items.append(
                {
                    "protein": {
                        "id": p["id"],
                        "sequence": p.get("sequence", ""),
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": [],
                    }
                }
            )
            chain_ids.append(p["id"])
        with open(os.path.join(out_std, f"{lower}_data.json"), "w") as fout:
            json.dump({"sequences": seq_items}, fout, indent=2)

        # Rewrite boltz's consecutive asym ids (A/B/C/...) back to the original input ids.
        _rewrite_mmcif_chain_ids(model_src, os.path.join(out_std, f"{lower}_model.cif"), chain_ids)

        conf = json.load(open(conf_src, "r"))
        summary = {
            "iptm": conf.get("iptm"),
            "ptm": conf.get("ptm"),
            "ranking_score": conf.get("confidence_score"),
            # Boltz complex_plddt is in [0, 1]. We do not have chain-local pLDDT here.
            "binder_plddt": conf.get("complex_plddt"),
            # Optional (not computed for now): interface PAE between target and binder.
            "ipae": None,
        }
        with open(os.path.join(out_std, f"{lower}_summary_confidences.json"), "w") as fout:
            json.dump(summary, fout, indent=2)

    def apply_manual_template(self, chain: ChainData, cif_path: str) -> ChainData:
        # Boltz templates are declared at YAML top-level `templates:`. We attach
        # a boltz-style template record to the chain and let write_input() lift it.
        data = chain.__class__(**chain.__dict__)
        data.unpairedMsa = ""
        data.pairedMsa = ""
        data.templates = [{
            "cif": os.path.abspath(cif_path),
            "chain_id": chain.id,
        }]
        return data


class ProtenixBackend(CofoldBackend):
    name = "protenix"

    def input_suffixes(self) -> List[str]:
        return [".json"]

    def get_task_name(self, input_path: str) -> str:
        data = json.load(open(input_path, "r"))
        if isinstance(data, list):
            if len(data) != 1:
                raise ValueError(f"Expected a single-sample protenix input in {input_path}, got {len(data)} entries")
            data = data[0]
        return data["name"]

    def _materialize_msa_path(self, content: Optional[str], out_dir: str, basename: str) -> str:
        if content is None or content == "": return ""
        path = os.path.join(out_dir, basename)
        with open(path, "w") as fout: fout.write(content)
        return path

    def _normalize_template_items(self, chain: ChainData) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        if not chain.templates:
            return items
        for template in chain.templates:
            if not isinstance(template, dict):
                continue
            mmcif = template.get("mmcif")
            if mmcif is None:
                mmcif_path = template.get("mmcifPath") or template.get("cif") or template.get("pdb")
                if mmcif_path:
                    with open(mmcif_path, "r") as fin:
                        mmcif = fin.read()
            if mmcif is None:
                continue
            query_idx = template.get("queryIndices")
            template_idx = template.get("templateIndices")
            if query_idx is None:
                query_idx = list(range(len(chain.sequence)))
            if template_idx is None:
                template_idx = list(range(len(query_idx)))
            items.append(
                {
                    "mmcif": mmcif,
                    "queryIndices": query_idx,
                    "templateIndices": template_idx,
                }
            )
        return items

    def _materialize_template_path(self, chain: ChainData, out_dir: str) -> str:
        items = self._normalize_template_items(chain)
        if not items:
            return ""
        path = os.path.join(out_dir, f"{chain.id}_templates.json")
        with open(path, "w") as fout:
            json.dump(items, fout, indent=2)
        return path

    def write_input(
        self,
        name: str,
        chains: List[ChainData],
        out_path: str,
        n_seeds: int = 1,
        include_chains: Optional[List[str]] = None,
    ) -> str:
        if include_chains is None:
            include_chains = [c.id for c in chains]

        sidecar_dir = os.path.join(os.path.dirname(os.path.abspath(out_path)), f"{name}_protenix_inputs")
        os.makedirs(sidecar_dir, exist_ok=True)

        seqs: List[Dict[str, Any]] = []
        original_chain_ids: List[str] = []
        for chain in chains:
            if chain.id not in include_chains:
                continue
            original_chain_ids.append(chain.id)
            protein = {
                "count": 1,
                "sequence": chain.sequence,
                "unpairedMsaPath": self._materialize_msa_path(chain.unpairedMsa, sidecar_dir, f"{chain.id}_unpaired.a3m"),
                "pairedMsaPath": self._materialize_msa_path(chain.pairedMsa, sidecar_dir, f"{chain.id}_paired.a3m"),
            }
            ############# Future function ###########
            # TODO: waiting for protenix to support custom template
            # tpl_path = self._materialize_template_path(chain, sidecar_dir)
            # if tpl_path:
            #     protein["templatesPath"] = tpl_path
            ############# END ###########
            if chain.modifications:
                protein["modifications"] = chain.modifications
            seqs.append({"proteinChain": protein})

        input_json = [
            {
                "name": name,
                "sequences": seqs,
                "modelSeeds": [random.randint(0, 4294967295) for _ in range(n_seeds)],
                "assembly_id": "1",
            }
        ]
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as fout:
            json.dump(input_json, fout, indent=2)
        with open(os.path.join(sidecar_dir, "chain_ids.json"), "w") as fout:
            json.dump(original_chain_ids, fout, indent=2)
        return out_path

    def load_confidences(
        self,
        summary_json_path: str,
        tgt_chains: List[str],
        lig_chains: List[str],
    ) -> Dict[str, Any]:
        conf = json.load(open(summary_json_path, "r"))
        data_json_path = summary_json_path.replace("_summary_confidences.json", "_data.json")
        chain_ids: List[str] = []
        if os.path.exists(data_json_path):
            data = json.load(open(data_json_path, "r"))
            for entry in data.get("sequences", []):
                p = entry.get("protein")
                if p and p.get("id") is not None:
                    chain_ids.append(p["id"])
        idx_map = {cid: i for i, cid in enumerate(chain_ids)}

        pair_iptm = conf.get("chain_pair_iptm") or conf.get("chain_pair_iptm_global") or []
        iptm_vals: List[float] = []
        for t in tgt_chains:
            for l in lig_chains:
                if t not in idx_map or l not in idx_map:
                    continue
                ti, li = idx_map[t], idx_map[l]
                try:
                    iptm_vals.append(float(pair_iptm[ti][li]))
                    iptm_vals.append(float(pair_iptm[li][ti]))
                except Exception:
                    pass
        interface_iptm = (sum(iptm_vals) / len(iptm_vals)) if iptm_vals else conf.get("iptm")

        pair_pae = conf.get("chain_pair_pae_mean") or []
        pae_vals: List[float] = []
        for t in tgt_chains:
            for l in lig_chains:
                if t not in idx_map or l not in idx_map:
                    continue
                ti, li = idx_map[t], idx_map[l]
                try:
                    pae_vals.append(float(pair_pae[ti][li]))
                    pae_vals.append(float(pair_pae[li][ti]))
                except Exception:
                    pass
        interface_ipae = (sum(pae_vals) / len(pae_vals)) if pae_vals else None

        chain_plddt = conf.get("chain_plddt") or []
        binder_plddt_vals: List[float] = []
        for l in lig_chains:
            if l not in idx_map:
                continue
            try:
                binder_plddt_vals.append(float(chain_plddt[idx_map[l]]))
            except Exception:
                pass
        binder_plddt = (sum(binder_plddt_vals) / len(binder_plddt_vals)) if binder_plddt_vals else conf.get("plddt")
        if binder_plddt is not None and binder_plddt > 1.0:
            binder_plddt *= 0.01

        return {
            "cofold_iptm": interface_iptm,
            "cofold_ptm": conf.get("ptm"),
            "cofold_ranking_score": conf.get("ranking_score"),
            "cofold_binder_plddt": binder_plddt,
            "cofold_ipae": interface_ipae,
        }

    def _protenix_bin(self, env_prefix_or_name: str) -> str:
        cand = os.path.join(env_prefix_or_name, "bin", "protenix")
        if os.path.exists(cand):
            return cand
        return "protenix"

    def build_command(self, cfg: BackendTaskConfig, gpu_ids: List[str], input_path: str, out_dir: str) -> str:
        protenix = self._protenix_bin(cfg.env)
        # model_name = os.environ.get("ANEW_PROTENIX_MODEL_NAME", "protenix_base_default_v1.0.0")
        model_name = os.environ.get("ANEW_PROTENIX_MODEL_NAME", "protenix_mini_esm_v0.5.0")
        # change to the running folder based output directory as the working directory to prevent shared cache of esm embeddings
        tmpdir = tempfile.mkdtemp()
        cmd = f"""
            cd {tmpdir}; \\   
            PROTENIX_ROOT_DIR={cfg.param} \\
            CUDA_VISIBLE_DEVICES={','.join(gpu_ids)} \\
            {protenix} pred -i {input_path} -o {out_dir} -n {model_name} \\
            --use_msa false --use_seeds_in_json true; \\
            cd {out_dir}; rm -r {tmpdir}
        """
        return cmd

    def postprocess_outputs(self, out_dir: str, input_name: str, input_path: str) -> None:
        lower = input_name.lower()
        input_json = json.load(open(input_path, "r"))
        if isinstance(input_json, list):
            input_json = input_json[0]

        seq_items: List[Dict[str, Any]] = []
        sidecar_dir = os.path.join(os.path.dirname(os.path.abspath(input_path)), f"{input_name}_protenix_inputs")
        chain_map_path = os.path.join(sidecar_dir, "chain_ids.json")
        if os.path.exists(chain_map_path):
            orig_chain_ids = json.load(open(chain_map_path, "r"))
        else:
            orig_chain_ids = []
        for entry in input_json.get("sequences", []):
            p = entry.get("proteinChain")
            if not p:
                continue
            cid = orig_chain_ids[len(seq_items)] if len(orig_chain_ids) > len(seq_items) else chr(ord("A") + len(seq_items))
            seq_items.append(
                {
                    "protein": {
                        "id": cid,
                        "sequence": p.get("sequence", ""),
                        "unpairedMsa": "",
                        "pairedMsa": "",
                        "templates": [],
                    }
                }
            )

        sample_root = os.path.join(out_dir, input_name)
        if not os.path.isdir(sample_root):
            sample_root = os.path.join(out_dir, lower)
        if not os.path.isdir(sample_root):
            raise FileNotFoundError(f"Protenix output root not found under {out_dir} for {input_name}")

        best_rank = None
        best_summary = None
        best_cif = None
        for seed_dir_name in os.listdir(sample_root):
            if not seed_dir_name.startswith("seed_"):
                continue
            pred_dir = os.path.join(sample_root, seed_dir_name, "predictions")
            if not os.path.isdir(pred_dir):
                continue
            for fn in os.listdir(pred_dir):
                if not fn.endswith("_summary_confidence_sample_0.json"):
                    continue
                summary_path = os.path.join(pred_dir, fn)
                summary = json.load(open(summary_path, "r"))
                rank = summary.get("ranking_score")
                if (best_rank is None) or (rank is not None and rank > best_rank):
                    best_rank = rank
                    best_summary = summary_path
                    best_cif = summary_path.replace("_summary_confidence_sample_0.json", "_sample_0.cif")
        if best_summary is None or best_cif is None or not os.path.exists(best_cif):
            raise FileNotFoundError(f"Protenix prediction artifacts not found under {sample_root}")

        out_std = os.path.join(out_dir, lower)
        os.makedirs(out_std, exist_ok=True)

        with open(os.path.join(out_std, f"{lower}_data.json"), "w") as fout:
            json.dump({"sequences": seq_items}, fout, indent=2)

        chain_ids = [entry["protein"]["id"] for entry in seq_items]
        _rewrite_mmcif_chain_ids(best_cif, os.path.join(out_std, f"{lower}_model.cif"), chain_ids)
        with open(best_summary, "r") as fin:
            conf = json.load(fin)
        with open(os.path.join(out_std, f"{lower}_summary_confidences.json"), "w") as fout:
            json.dump(conf, fout, indent=2)

    def apply_manual_template(self, chain: Any, cif_path: str) -> Any:
        # Protenix template JSON stored in a sidecar json file.
        data = chain.__class__(**chain.__dict__)
        cplx = mmcif_to_complex(cif_path, selected_chains=[chain.id])
        cif_str = cofold_utils._get_chain_str(cplx, chain.id)
        idx = list(range(len(chain.sequence)))
        template = {
            "mmcif": cif_str,
            "queryIndices": idx,
            "templateIndices": idx,
        }
        data.unpairedMsa = ""
        data.pairedMsa = ""
        data.templates = [template]
        return data


def get_backend(model: str) -> CofoldBackend:
    if model == AlphaFold3Backend.name:
        return AlphaFold3Backend()
    if model == Boltz2Backend.name:
        return Boltz2Backend()
    if model == ProtenixBackend.name:
        return ProtenixBackend()
    raise ValueError(f"Unsupported cofold backend: {model}")
