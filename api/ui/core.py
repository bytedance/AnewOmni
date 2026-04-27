#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import annotations

import json
import os
import random
import re
import shutil
import tempfile
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ray
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from api.filters.base import FilterInput, FilterResult
from api.filters.chem import ChiralCentersFilter, RotatableBondsFilter
from api.filters.geom import AbnormalConfidenceFilter, PhysicalValidityFilter
from api.filters.L_type_AA import LTypeAAFilter
from api.filters.mol_beauty import MolBeautyFilter
from api.helpers.gen_utils import generate_for_one_template, generate_multiple_cdrs, load_model, merge_jsons, merge_sdfs
from api.helpers.template_creator import Creator
from api.pdb_dataset import PDBDataset
from api.templates import AntibodyMultipleCDR
from api.templates.antibody import Antibody
from api.templates.base import BaseTemplate, ComplexDesc
from data.bioparse import VOCAB
from data.bioparse import const
from data.bioparse.hierarchy import Atom, Block, Bond, BondType, Complex, Molecule
from data.bioparse.numbering import get_nsys
from data.bioparse.parser.mmcif_to_complex import mmcif_to_complex
from data.bioparse.parser.pdb_to_complex import pdb_to_complex
from data.bioparse.writer.complex_to_mmcif import complex_to_mmcif
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from data.bioparse.utils import bond_type_to_rdkit
from data.bioparse.utils import recur_index
from data.file_loader import MolLoader, MolType, PeptideLoader, _extract_antibody_masks
from data.bioparse.tokenizer.tokenize_3d import tokenize_3d
from models.modules.adapter.model import ConditionConfig
from utils.chem_utils import find_term_and_delete_dummy


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_project_path(path: str) -> str:
    if not isinstance(path, str) or path == "":
        return path
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def _default_checkpoint_path() -> str:
    return resolve_project_path("checkpoints/model.ckpt")


def _stringify_iter(values: Iterable[Any]) -> str:
    return ", ".join(str(value) for value in values)


def _coerce_single_pdb_path(kwargs: Dict[str, Any]) -> List[str]:
    if "pdb_path" in kwargs:
        value = kwargs["pdb_path"]
        if not isinstance(value, str) or not value:
            raise ValueError("pdb_path must be a non-empty string")
        return [resolve_project_path(value)]
    if "pdb_paths" in kwargs:
        value = kwargs["pdb_paths"]
        if isinstance(value, str):
            return [resolve_project_path(value)]
        value = list(value)
        if len(value) != 1:
            raise ValueError("UI demo only supports a single target: please pass exactly one pdb_path")
        return [resolve_project_path(value[0])]
    raise ValueError("pdb_path is required for generation")


def _coerce_single_chain_list(value: Any, field_name: str) -> List[List[str]]:
    # Preferred: ["R"] or ["H", "L"]
    if isinstance(value, (list, tuple)) and (len(value) == 0 or isinstance(value[0], str)):
        return [list(value)]
    # Backward-compat: [["R"]] or [["H","L"]]
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        return [list(value[0])]
    # Allow passing "HL" as shorthand for ["H","L"].
    if isinstance(value, str):
        return [list(value)]
    raise ValueError(f"{field_name} must be a list of chain IDs, e.g. ['R'] or ['H','L']")


def _coerce_single_hotspots(value: Any) -> List[List[tuple]] | None:
    if value is None:
        return None
    # Preferred: [("R", 30, "")]
    if isinstance(value, (list, tuple)) and (len(value) == 0 or (isinstance(value[0], (list, tuple)) and len(value[0]) == 3)):
        # If it's already a list of 3-tuples, wrap it.
        if len(value) == 0 or (isinstance(value[0], (list, tuple)) and isinstance(value[0][0], str)):
            return [list(value)]
    # Backward-compat: [[("R", 30, "")]]
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        return [list(value[0])]
    raise ValueError("hotspots must be a list of 3-tuples (chain_id, seq_number, insertion_code)")

def _bond_type_from_value(value: Any) -> BondType:
    if isinstance(value, BondType):
        return value
    if isinstance(value, str):
        mapping = {
            "single": BondType.SINGLE,
            "double": BondType.DOUBLE,
            "triple": BondType.TRIPLE,
            "aromatic": BondType.AROMATIC,
        }
        key = value.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unsupported bond type: {value}")
        return mapping[key]
    if isinstance(value, int):
        return BondType(value)
    raise TypeError(f"Unsupported bond type value: {value}")


def _find_atom_index(block: Block, atom_name: str) -> int:
    for index, atom in enumerate(block):
        if atom.name == atom_name:
            return index
    raise ValueError(f"Atom {atom_name} not found in block {block.name}")


def _atom_label_sort_key(label: str) -> Tuple[str, int]:
    match = re.fullmatch(r"([a-z]+)(\d+)", label)
    if match is None:
        return label, -1
    return match.group(1), int(match.group(2))


def _make_condition_config(mask_2d: List[int], mask_3d: List[int], w: float | None) -> ConditionConfig:
    tensor_2d = torch.tensor(mask_2d, dtype=torch.bool)
    tensor_3d = torch.tensor(mask_3d, dtype=torch.bool)
    return ConditionConfig(
        mask_2d=tensor_2d,
        mask_3d=tensor_3d,
        mask_incomplete_2d=torch.zeros_like(tensor_2d),
        w=w,
    )


# #region debug-point A:report-helper
def _debug_report(hypothesis_id: str, location: str, msg: str, data: Optional[Dict[str, Any]] = None, run_id: str = "pre-fix") -> None:
    import json, urllib.request
    _p = ".dbg/antibody-ncaa-bonds.env"
    _u, _s = "http://127.0.0.1:7777/event", "antibody-ncaa-bonds"
    try:
        with open(_p, "r", encoding="utf-8") as _f:
            _c = _f.read()
        for _line in _c.splitlines():
            if _line.startswith("DEBUG_SERVER_URL="):
                _u = _line.split("=", 1)[1]
            elif _line.startswith("DEBUG_SESSION_ID="):
                _s = _line.split("=", 1)[1]
        urllib.request.urlopen(
            urllib.request.Request(
                _u,
                data=json.dumps(
                    {
                        "sessionId": _s,
                        "runId": run_id,
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "msg": msg,
                        "data": data or {},
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=2,
        ).read()
    except Exception:
        pass
# #endregion


def _mol_to_svg(mol: Chem.Mol, atom_labels: Optional[Dict[int, str]] = None, width: int = 520, height: int = 360) -> str:
    mol = Chem.Mol(mol)
    if atom_labels is not None:
        for atom_index, label in atom_labels.items():
            mol.GetAtomWithIdx(atom_index).SetProp("atomNote", label)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().annotationFontScale = 0.85
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def smiles_to_svg(smiles: str, width: int = 520, height: int = 360) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for SVG rendering: {smiles}")
    return _mol_to_svg(mol, width=width, height=height)


def sdf_to_svg(sdf_path: str, width: int = 520, height: int = 360) -> str:
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mol = supplier[0] if len(supplier) > 0 else None
    if mol is None:
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
        mol = supplier[0] if len(supplier) > 0 else None
    if mol is None:
        raise ValueError(f"Failed to load molecule from SDF: {sdf_path}")
    return _mol_to_svg(mol, width=width, height=height)


def sdf_to_svg_pages(sdf_path: str, width: int = 520, height: int = 360) -> List[str]:
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=False)
    mols = [mol for mol in supplier if mol is not None]
    if not mols:
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=False, removeHs=False)
        mols = [mol for mol in supplier if mol is not None]
    if not mols:
        raise ValueError(f"Failed to load molecules from SDF: {sdf_path}")
    return [_mol_to_svg(mol, width=width, height=height) for mol in mols]


def nsaa_smiles_to_svg(smiles: str, width: int = 360, height: int = 220) -> str:
    # For UI preview, keep dummy atoms visible so users can see the backbone
    # connection sites directly in the 2D depiction.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid nsAA SMILES for SVG rendering: {smiles}")
    return _mol_to_svg(mol, atom_labels=None, width=width, height=height)


def _load_complex_for_ui(path: str):
    if path.endswith(".cif"):
        return mmcif_to_complex(path)
    return pdb_to_complex(path)


def _detect_antibody_chains(cplx, preferred: Optional[List[str]] = None) -> Dict[str, str]:
    preferred = preferred or []
    chain_map: Dict[str, str] = {}
    nsys = get_nsys()

    candidates = []
    for chain in cplx:
        chain_id = chain.id
        seq_positions = [block.id[0] for block in chain.blocks if isinstance(block.id, tuple)]
        if not seq_positions:
            continue
        hmark = nsys.mark_heavy_seq(seq_positions)
        lmark = nsys.mark_light_seq(seq_positions)
        hscore = sum(1 for ch in hmark if ch in "123")
        lscore = sum(1 for ch in lmark if ch in "123")
        candidates.append((chain_id, hscore, lscore))

    # Honor preferred chain ordering from context if present.
    for chain_id in preferred:
        match = next((item for item in candidates if item[0] == chain_id), None)
        if match is None:
            continue
        _, hscore, lscore = match
        if hscore >= lscore and "H" not in chain_map:
            chain_map["H"] = chain_id
        elif lscore > hscore and "L" not in chain_map:
            chain_map["L"] = chain_id

    if "H" not in chain_map:
        heavy = sorted(candidates, key=lambda item: item[1], reverse=True)
        if heavy and heavy[0][1] > 0:
            chain_map["H"] = heavy[0][0]
    if "L" not in chain_map:
        light = [item for item in candidates if item[0] != chain_map.get("H")]
        light = sorted(light, key=lambda item: item[2], reverse=True)
        if light and light[0][2] > 0:
            chain_map["L"] = light[0][0]
    return chain_map


def _extract_antibody_visual_info(path: str, preferred_chain_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    cplx = _load_complex_for_ui(path)
    chain_map = _detect_antibody_chains(cplx, preferred_chain_ids)
    nsys = get_nsys()
    out: Dict[str, Any] = {"chains": {}}
    for chain_type, chain_id in chain_map.items():
        chain = cplx[chain_id]
        blocks = [block for block in chain.blocks]
        seq = "".join([(VOCAB.abrv_to_symbol(block.name) or "X")[0] for block in blocks])
        positions = [block.id[0] for block in blocks]
        mark = nsys.mark_heavy_seq(positions) if chain_type == "H" else nsys.mark_light_seq(positions)
        out["chains"][chain_type] = {
            "chain_id": chain_id,
            "sequence": seq,
            "positions": positions,
            "mark": mark,
        }
    return out


def _get_target_antibody_chain_id(lig_chains: Sequence[str], cdr_type: str) -> Optional[str]:
    if not lig_chains:
        return None
    if len(lig_chains) == 1:
        return lig_chains[0]
    return lig_chains[0] if cdr_type.startswith("H") else lig_chains[1]


def _normalize_motif_sequence(seq: str) -> str:
    seq = str(seq).strip().upper()
    if not seq:
        raise ValueError("motif sequence must be non-empty")
    invalid = [aa for aa in seq if VOCAB.symbol_to_abrv(aa) is None]
    if invalid:
        raise ValueError(f"Unsupported motif residue(s): {invalid}")
    return seq


def _normalize_relative_positions(positions: Optional[Sequence[int]], field_name: str) -> Optional[List[int]]:
    if positions is None:
        return None
    values = [int(pos) for pos in positions]
    if not values:
        raise ValueError(f"{field_name} must be non-empty when provided")
    if any(pos <= 0 for pos in values):
        raise ValueError(f"{field_name} must be positive 1-based positions")
    if len(set(values)) != len(values):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


def _get_antibody_cdr_block_ids(cplx, lig_chains: Sequence[str], cdr_type: str) -> List[Tuple[str, Tuple[int, str]]]:
    chain_id = _get_target_antibody_chain_id(lig_chains, cdr_type)
    if chain_id is None:
        return []
    positions = [block.id[0] for block in cplx[chain_id]]
    nsys = get_nsys()
    mark = nsys.mark_heavy_seq(positions) if cdr_type.startswith("H") else nsys.mark_light_seq(positions)
    cdr_digit = cdr_type[-1]
    out: List[Tuple[str, Tuple[int, str]]] = []
    for block, digit in zip(cplx[chain_id], mark):
        if str(digit) == cdr_digit:
            out.append((chain_id, block.id))
    return out


def _build_standard_aa_block(old_block: Block, new_abrv: str, atom_id_prefix: str) -> Block:
    atom_names = VOCAB.abrv_to_atoms(new_abrv)
    elements = VOCAB.abrv_to_elements(new_abrv)
    if not atom_names or not elements:
        raise ValueError(f"Unsupported amino acid for motif replacement: {new_abrv}")

    old_atoms_by_name = {atom.name: atom for atom in old_block}
    if "CA" in old_atoms_by_name:
        fallback_coord = list(old_atoms_by_name["CA"].coordinate)
    elif len(old_block.atoms) > 0:
        fallback_coord = [
            sum(atom.coordinate[i] for atom in old_block.atoms) / len(old_block.atoms)
            for i in range(3)
        ]
    else:
        fallback_coord = [0.0, 0.0, 0.0]

    atoms: List[Atom] = []
    for atom_name, element in zip(atom_names, elements):
        old_atom = old_atoms_by_name.get(atom_name)
        atoms.append(
            Atom(
                name=atom_name,
                coordinate=list(old_atom.coordinate) if old_atom is not None else list(fallback_coord),
                element=element,
                id=old_atom.id if old_atom is not None else f"{atom_id_prefix}:{atom_name}",
            )
        )
    return Block(name=new_abrv, atoms=atoms, id=old_block.id, properties=deepcopy(old_block.properties))


@dataclass
class _ResidueReplacementSpec:
    blocks: List[Block]
    bonds: List[Tuple[int, int, int, int, BondType]]
    atom_name_map: Dict[str, Tuple[int, int]]


def _build_standard_aa_replacement(old_block: Block, new_abrv: str, atom_id_prefix: str) -> _ResidueReplacementSpec:
    new_block = _build_standard_aa_block(old_block, new_abrv, atom_id_prefix)
    return _ResidueReplacementSpec(
        blocks=[new_block],
        bonds=[(0, atom_i, 0, atom_j, bond_type) for atom_i, atom_j, bond_type in VOCAB.abrv_to_bonds(new_block.name)],
        atom_name_map={atom.name: (0, idx) for idx, atom in enumerate(new_block.atoms)},
    )


def _build_ncaa_fragment_replacement(old_block: Block, nsaa_smiles: str) -> _ResidueReplacementSpec:
    nsaa_mol, n_term_idx, c_term_idx = find_term_and_delete_dummy(nsaa_smiles)
    if nsaa_mol is None:
        raise ValueError(f"Invalid nsAA SMILES: {nsaa_smiles}")
    canonical = Chem.MolToSmiles(nsaa_mol, canonical=True)

    creator = Creator(0)
    added_blocks, added_bonds = creator.decompose_and_add_fragment(nsaa_mol, old_block.id[0])
    fragment_blocks = deepcopy(added_blocks)
    atom_name_map: Dict[str, Tuple[int, int]] = {}
    for block_idx, block in enumerate(fragment_blocks):
        if block.properties is None:
            block.properties = {}
        block.properties["original_name"] = canonical
        for atom_idx, atom in enumerate(block):
            source_atom_idx = int(atom.id) - 1
            if source_atom_idx == int(n_term_idx):
                atom_name_map["N"] = (block_idx, atom_idx)
            elif source_atom_idx == int(c_term_idx):
                atom_name_map["C"] = (block_idx, atom_idx)

    if "N" not in atom_name_map or "C" not in atom_name_map:
        raise ValueError("Could not identify nsAA N/C termini after fragment decomposition")

    # #region debug-point A:ncaa-build
    _debug_report(
        "A",
        "api/ui/core.py:_build_ncaa_fragment_replacement",
        "[DEBUG] built ncAA fragment replacement before insertion",
        {
            "old_block_name": old_block.name,
            "old_block_id": list(old_block.id),
            "canonical_smiles": canonical,
            "fragment_count": len(fragment_blocks),
            "fragment_names": [block.name for block in fragment_blocks],
            "fragment_sizes": [len(block.atoms) for block in fragment_blocks],
            "bond_count": len(added_bonds),
            "n_term_anchor": list(atom_name_map["N"]),
            "c_term_anchor": list(atom_name_map["C"]),
        },
    )
    # #endregion

    return _ResidueReplacementSpec(
        blocks=fragment_blocks,
        bonds=[
            (bond.index1[1], bond.index1[2], bond.index2[1], bond.index2[2], bond.bond_type)
            for bond in added_bonds
        ],
        atom_name_map=atom_name_map,
    )


def _replace_blocks_and_rebuild_bonds_general(
    cplx: Complex,
    replacements: Dict[Tuple[str, Tuple[int, str]], _ResidueReplacementSpec],
) -> Complex:
    if not replacements:
        return deepcopy(cplx)

    replaced_molecules = []
    old_blocks: Dict[Tuple[str, Tuple[int, str]], Block] = {}
    block_index_map: Dict[Tuple[str, Tuple[int, str]], List[int]] = {}
    realized_specs: Dict[Tuple[str, Tuple[int, str]], _ResidueReplacementSpec] = {}

    for mol in cplx:
        new_blocks: List[Block] = []
        for old_block in mol:
            key = (mol.id, old_block.id)
            old_blocks[key] = deepcopy(old_block)
            if key not in replacements:
                block_index_map[key] = [len(new_blocks)]
                new_blocks.append(deepcopy(old_block))
                continue
            spec = replacements[key]
            start = len(new_blocks)
            copied_blocks = [deepcopy(block) for block in spec.blocks]
            new_blocks.extend(copied_blocks)
            block_index_map[key] = list(range(start, start + len(copied_blocks)))
            realized_specs[key] = _ResidueReplacementSpec(
                blocks=copied_blocks,
                bonds=list(spec.bonds),
                atom_name_map=dict(spec.atom_name_map),
            )
        replaced_molecules.append(Molecule(mol.name, new_blocks, mol.id, deepcopy(mol.properties)))

    replaced = Complex(cplx.name, replaced_molecules, [], deepcopy(cplx.properties))

    def _remap_endpoint(endpoint: Tuple[int, int, int]) -> Optional[Tuple[int, int, int]]:
        mol_idx, block_idx, atom_idx = endpoint
        mol = cplx[mol_idx]
        block = mol.blocks[block_idx]
        block_key = (mol.id, block.id)
        if block_key not in replacements:
            new_mol_idx = replaced.id2idx[mol.id]
            new_block_idx = block_index_map[block_key][0]
            return (new_mol_idx, new_block_idx, atom_idx)
        atom_name = old_blocks[block_key].atoms[atom_idx].name
        new_local_endpoint = realized_specs[block_key].atom_name_map.get(atom_name)
        if new_local_endpoint is None:
            return None
        new_mol_idx = replaced.id2idx[mol.id]
        new_block_idx = block_index_map[block_key][new_local_endpoint[0]]
        return (new_mol_idx, new_block_idx, new_local_endpoint[1])

    rebuilt_bonds: List[Bond] = []
    for bond in cplx.bonds:
        mol1 = cplx[bond.index1[0]]
        mol2 = cplx[bond.index2[0]]
        key1 = (mol1.id, mol1.blocks[bond.index1[1]].id)
        key2 = (mol2.id, mol2.blocks[bond.index2[1]].id)
        # Always drop intra-block bonds for replaced residues and rebuild from the new block.
        if key1 == key2 and key1 in replacements:
            continue
        idx1 = _remap_endpoint(bond.index1)
        idx2 = _remap_endpoint(bond.index2)
        if idx1 is None or idx2 is None:
            continue
        rebuilt_bonds.append(Bond(idx1, idx2, bond.bond_type))

    # Rebuild intra-block bonds for each replaced block.
    for (chain_id, block_id), spec in realized_specs.items():
        mol_idx = replaced.id2idx[chain_id]
        mapped_block_indices = block_index_map[(chain_id, block_id)]
        for block_i, atom_i, block_j, atom_j, bond_type in spec.bonds:
            rebuilt_bonds.append(
                Bond(
                    (mol_idx, mapped_block_indices[block_i], atom_i),
                    (mol_idx, mapped_block_indices[block_j], atom_j),
                    bond_type,
                )
            )

    # #region debug-point B:replacement-summary
    _debug_report(
        "B",
        "api/ui/core.py:_replace_blocks_and_rebuild_bonds_general",
        "[DEBUG] rebuilt complex after antibody residue replacement",
        {
            "replacement_count": len(replacements),
            "replacement_keys": [f"{chain_id}:{block_id[0]}:{block_id[1]}" for chain_id, block_id in replacements.keys()],
            "replacement_blocks": {
                f"{chain_id}:{block_id[0]}:{block_id[1]}": {
                    "block_names": [block.name for block in realized_specs[(chain_id, block_id)].blocks],
                    "block_sizes": [len(block.atoms) for block in realized_specs[(chain_id, block_id)].blocks],
                    "bond_count": len(realized_specs[(chain_id, block_id)].bonds),
                }
                for chain_id, block_id in replacements.keys()
            },
            "total_bond_count": len(rebuilt_bonds),
        },
    )
    # #endregion

    return Complex(replaced.name, list(replaced.molecules), rebuilt_bonds, deepcopy(replaced.properties))


def _make_mmcif_safe_complex(cplx: Complex) -> Complex:
    safe_bonds: List[Bond] = []
    for bond in cplx.bonds:
        bond_type = bond.bond_type
        if bond_type == BondType.AROMATIC:
            bond_type = BondType.SINGLE
        safe_bonds.append(Bond(bond.index1, bond.index2, bond_type))
    return Complex(cplx.name, list(cplx.molecules), safe_bonds, deepcopy(cplx.properties))


def _write_antibody_preprocessed_input(
    source_path: str,
    lig_chains: Sequence[str],
    cdr_type: str,
    length_range: Optional[Tuple[int, int]],
    resolved_constraint: Dict[str, Any],
) -> List[str]:
    tmp_paths: List[str] = []
    current_path = source_path

    if length_range is not None:
        l, r = length_range
        if int(l) != int(r):
            raise ValueError("Single-CDR UI currently only supports a fixed antibody CDR length")
        tmp_file = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
        tmp_file.close()
        chain_id = _get_target_antibody_chain_id(lig_chains, cdr_type)
        if chain_id is None:
            raise ValueError(f"Could not determine target antibody chain for {cdr_type}")
        from api.helpers.gen_utils import change_cdr_length

        change_cdr_length(current_path, tmp_file.name, cdr_type, {}, chain_id, int(l))
        current_path = tmp_file.name
        tmp_paths.append(current_path)

    motif_spec = resolved_constraint.get("motif")
    motif_positions = list(resolved_constraint.get("motif_positions", []))
    nsaa_spec = resolved_constraint.get("nsaa")
    nsaa_positions = list(resolved_constraint.get("nsaa_positions", []))
    if (motif_spec is None or not motif_positions) and (nsaa_spec is None or not nsaa_positions):
        return tmp_paths

    cplx = _load_complex_for_ui(current_path)
    cdr_block_ids = _get_antibody_cdr_block_ids(cplx, lig_chains, cdr_type)
    replacements: Dict[Tuple[str, Tuple[int, str]], _ResidueReplacementSpec] = {}

    if motif_spec is not None and motif_positions:
        for aa, rel_pos in zip(motif_spec["seq"], motif_positions):
            rel_pos = int(rel_pos)
            if rel_pos < 1 or rel_pos > len(cdr_block_ids):
                raise ValueError(f"motif position {rel_pos} is outside {cdr_type} with length {len(cdr_block_ids)}")
            block_key = cdr_block_ids[rel_pos - 1]
            chain_id, block_id = block_key
            old_block = cplx[chain_id][block_id]
            atom_id_prefix = f"{chain_id}:{block_id[0]}:{block_id[1] or '_'}"
            new_abrv = VOCAB.symbol_to_abrv(aa)
            replacements[block_key] = _build_standard_aa_replacement(old_block, new_abrv, atom_id_prefix)

    if nsaa_spec is not None and nsaa_positions:
        nsaa_smiles = str(nsaa_spec.get("smiles", "")).strip()
        if not nsaa_smiles:
            raise ValueError("nsAA smiles must be non-empty")
        for rel_pos in nsaa_positions:
            rel_pos = int(rel_pos)
            if rel_pos < 1 or rel_pos > len(cdr_block_ids):
                raise ValueError(f"nsAA position {rel_pos} is outside {cdr_type} with length {len(cdr_block_ids)}")
            block_key = cdr_block_ids[rel_pos - 1]
            chain_id, block_id = block_key
            old_block = cplx[chain_id][block_id]
            atom_id_prefix = f"{chain_id}:{block_id[0]}:{block_id[1] or '_'}:nsaa"
            replacements[block_key] = _build_ncaa_fragment_replacement(old_block, nsaa_smiles)

    rewritten_cplx = _replace_blocks_and_rebuild_bonds_general(cplx, replacements)
    tmp_file = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
    tmp_file.close()
    mmcif_cplx = _make_mmcif_safe_complex(rewritten_cplx)
    # #region debug-point C:mmcif-export
    _debug_report(
        "C",
        "api/ui/core.py:_write_antibody_preprocessed_input",
        "[DEBUG] exporting antibody preprocessed complex to mmCIF",
        {
            "cdr_type": cdr_type,
            "tmp_file": tmp_file.name,
            "replacement_count": len(replacements),
            "safe_bond_count": len(mmcif_cplx.bonds),
            "aromatic_bonds_after_safe": sum(1 for bond in mmcif_cplx.bonds if bond.bond_type == BondType.AROMATIC),
        },
    )
    # #endregion
    complex_to_mmcif(mmcif_cplx, tmp_file.name, selected_chains=[mol.id for mol in mmcif_cplx])
    tmp_paths.append(tmp_file.name)
    return tmp_paths


class ConstrainedAntibodyTemplate(Antibody):
    def __init__(
        self,
        cdr_type: str,
        fr_len: int,
        motif_seq: Optional[str],
        motif_positions: Sequence[int],
        nsaa_smiles: Optional[str],
        nsaa_positions: Sequence[int],
        guidance: Optional[float] = None,
    ):
        super().__init__(cdr_type=cdr_type, fr_len=fr_len)
        self.motif_seq = motif_seq
        self.motif_positions = [int(pos) for pos in motif_positions]
        self.nsaa_smiles = nsaa_smiles
        self.nsaa_positions = [int(pos) for pos in nsaa_positions]
        self.guidance = guidance

    def add_dummy_lig(self, cplx_desc):
        cplx = cplx_desc.cplx
        cdr_block_ids = _get_antibody_cdr_block_ids(cplx, cplx_desc.lig_chains, self.cdr_type)
        replacements: Dict[Tuple[str, Tuple[int, str]], _ResidueReplacementSpec] = {}

        if self.motif_seq is not None and self.motif_positions:
            for aa, rel_pos in zip(self.motif_seq, self.motif_positions):
                if 1 <= int(rel_pos) <= len(cdr_block_ids):
                    chain_id, block_id = cdr_block_ids[int(rel_pos) - 1]
                    old_block = cplx[chain_id][block_id]
                    atom_id_prefix = f"{chain_id}:{block_id[0]}:{block_id[1] or '_'}"
                    new_abrv = VOCAB.symbol_to_abrv(aa)
                    replacements[(chain_id, block_id)] = _build_standard_aa_replacement(old_block, new_abrv, atom_id_prefix)

        if self.nsaa_smiles is not None and self.nsaa_positions:
            for rel_pos in self.nsaa_positions:
                if 1 <= int(rel_pos) <= len(cdr_block_ids):
                    chain_id, block_id = cdr_block_ids[int(rel_pos) - 1]
                    old_block = cplx[chain_id][block_id]
                    replacements[(chain_id, block_id)] = _build_ncaa_fragment_replacement(old_block, self.nsaa_smiles)

        if replacements:
            cplx = _replace_blocks_and_rebuild_bonds_general(cplx, replacements)
            cplx_desc.cplx = cplx

        generate_mask, center_mask, lig_block_ids = _extract_antibody_masks(
            cplx, cplx_desc.lig_chains, cplx_desc.pocket_block_ids, self.cdr_type, self.fr_len
        )
        all_block_ids = cplx_desc.pocket_block_ids + lig_block_ids
        fixed_block_ids = {
            (chain_id, block.id)
            for (chain_id, _), spec in replacements.items()
            for block in spec.blocks
        }
        condition_mask_2d = [0 for _ in generate_mask]
        condition_mask_3d = [0 for _ in generate_mask]

        for idx, (m, block_id) in enumerate(zip(generate_mask, all_block_ids)):
            if m == 0:
                continue
            if block_id in fixed_block_ids:
                condition_mask_2d[idx] = 1
                continue
            block: Block = recur_index(cplx, block_id)
            block.name = "GLY"
            block.atoms = [
                Atom(
                    name="C",
                    coordinate=[0, 0, 0],
                    element="C",
                    id=atom.id,
                )
                for atom in block.atoms
            ]

        cplx_desc.generate_mask = generate_mask
        cplx_desc.center_mask = center_mask
        cplx_desc.lig_block_ids = lig_block_ids
        cplx_desc.condition_config = _make_condition_config(condition_mask_2d, condition_mask_3d, self.guidance)
        return cplx

    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        data = super().to_data(cplx_desc)
        data["condition_config"] = cplx_desc.condition_config
        return data


def _load_result_items(result_path: str) -> List[dict]:
    if not os.path.exists(result_path):
        return []
    with open(result_path, "r") as fin:
        return [json.loads(line) for line in fin.readlines()]


def _normalize_chain_group(value: Any) -> List[str]:
    if isinstance(value, str):
        return list(value)
    return [str(item) for item in value]


def _normalize_chain_groups(values: Sequence[Any], n_expected: int) -> List[List[str]]:
    groups = [_normalize_chain_group(value) for value in values]
    if len(groups) != n_expected:
        raise ValueError(f"Expected {n_expected} chain groups, got {len(groups)}")
    return groups


def generate_multiple_cdrs_constrained(
    model,
    dataset,
    n_samples,
    batch_size,
    save_dir,
    device,
    sample_opt,
    n_cycles=0,
    conf_model=None,
    max_retry=None,
    verbose=True,
    check_validity=False,
    resample_mode=False,
):
    assert len(dataset.pdb_paths) == 1, "Only single target file supported for multiple CDR design"
    pdb_path = dataset.pdb_paths[0]
    target_id = os.path.basename(os.path.splitext(pdb_path)[0])

    sample_trajs = [[] for _ in range(n_samples)]
    dataset = deepcopy(dataset)
    dummy_template = dataset.config
    resolved_constraints = getattr(dummy_template, "ui_resolved_constraints", {}) or {}
    guidance = getattr(dummy_template, "ui_guidance", None)

    pdb_paths = []
    tgt_chains = _normalize_chain_groups(dataset.tgt_chains * n_samples, n_samples)
    lig_chains = _normalize_chain_groups(dataset.lig_chains * n_samples, n_samples)
    marks = [getattr(dummy_template, "specify_regions", {}) for _ in range(n_samples)]

    tmp_files: List[str] = []
    src_ext = "." + dataset.pdb_paths[0].split(".")[-1]
    for _ in range(n_samples):
        tmp_file = tempfile.NamedTemporaryFile(suffix=src_ext, delete=False)
        tmp_file.close()
        shutil.copyfile(dataset.pdb_paths[0], tmp_file.name)
        tmp_files.append(tmp_file.name)
        pdb_paths.append(tmp_file.name)

    for cdr_type in dummy_template.cdr_types:
        if cdr_type in dummy_template.length_ranges:
            l, r = dummy_template.length_ranges[cdr_type]
            new_paths, new_marks = [], []
            for path, chain_ids, mark in zip(pdb_paths, lig_chains, marks):
                length = random.randint(int(l), int(r))
                tmp_file = tempfile.NamedTemporaryFile(suffix=".cif", delete=False)
                tmp_file.close()
                cur_chain_id = _get_target_antibody_chain_id(chain_ids, cdr_type)
                if cur_chain_id is None:
                    raise ValueError(f"Could not determine target antibody chain for {cdr_type}")
                from api.helpers.gen_utils import change_cdr_length

                new_mark = change_cdr_length(path, tmp_file.name, cdr_type, mark, cur_chain_id, length)
                new_paths.append(tmp_file.name)
                tmp_files.append(tmp_file.name)
                new_marks.append(new_mark)
            pdb_paths, marks = new_paths, new_marks

        dataset = PDBDataset(
            pdb_paths=pdb_paths,
            tgt_chains=tgt_chains,
            template_config=None,
            lig_chains=lig_chains,
        )

        cur_save_dir = os.path.join(save_dir, "intermediate", cdr_type)
        raw_save_dir = os.path.join(cur_save_dir, "raw")
        os.makedirs(raw_save_dir, exist_ok=True)
        resolved_constraint = resolved_constraints.get(cdr_type, {})
        has_constraints = resolved_constraint.get("motif") is not None or resolved_constraint.get("nsaa") is not None
        if has_constraints:
            template = ConstrainedAntibodyTemplate(
                cdr_type=cdr_type,
                fr_len=dummy_template.fr_len,
                motif_seq=resolved_constraint.get("motif", {}).get("seq") if resolved_constraint.get("motif") is not None else None,
                motif_positions=resolved_constraint.get("motif_positions", []),
                nsaa_smiles=resolved_constraint.get("nsaa", {}).get("smiles") if resolved_constraint.get("nsaa") is not None else None,
                nsaa_positions=resolved_constraint.get("nsaa_positions", []),
                guidance=guidance,
            )
        else:
            template = Antibody(cdr_type=cdr_type, fr_len=dummy_template.fr_len)
        dataset.config = template
        generate_for_one_template(
            model,
            dataset,
            1,
            batch_size,
            raw_save_dir,
            device,
            sample_opt,
            n_cycles,
            conf_model,
            max_retry,
            verbose,
            check_validity,
            resample_mode,
        )

        items = _load_result_items(os.path.join(raw_save_dir, "results.jsonl"))
        assert len(items) == len(sample_trajs)
        pdb_paths, tgt_chains, lig_chains = [], [], []
        tgt_dir = os.path.join(cur_save_dir, target_id)
        os.makedirs(tgt_dir, exist_ok=True)
        for i, item in enumerate(items):
            path_prefix = os.path.join(raw_save_dir, item["id"], str(item["n"]))
            tgt_path = os.path.join(tgt_dir, f"{i}.cif")
            shutil.move(path_prefix + ".cif", tgt_path)
            pdb_paths.append(tgt_path)
            tgt_chains.append(_normalize_chain_group(item["tgt_chains"]))
            lig_chains.append(_normalize_chain_group(item["lig_chains"]))
            sample_trajs[i].append((item, path_prefix))

    shutil.copytree(tgt_dir, os.path.join(save_dir, target_id), dirs_exist_ok=True)
    shutil.copyfile(
        os.path.join(os.path.dirname(sample_trajs[0][0][1]), "pocket.pdb"),
        os.path.join(save_dir, target_id, "pocket.pdb"),
    )
    with open(os.path.join(save_dir, "results.jsonl"), "w") as res_file:
        for i, trajs in enumerate(sample_trajs):
            sample_tgt_chains, sample_lig_chains = trajs[0][0]["tgt_chains"], trajs[0][0]["lig_chains"]
            cplx = mmcif_to_complex(
                os.path.join(save_dir, target_id, f"{i}.cif"),
                selected_chains=sample_tgt_chains + sample_lig_chains,
            )
            complex_to_pdb(
                cplx,
                os.path.join(save_dir, target_id, f"{i}.pdb"),
                selected_chains=sample_tgt_chains + sample_lig_chains,
            )
            merge_sdfs([tup[1] + ".sdf" for tup in trajs], os.path.join(save_dir, target_id, f"{i}.sdf"))
            merge_jsons(
                dummy_template.cdr_types,
                [tup[1] + "_confidence.json" for tup in trajs],
                os.path.join(save_dir, target_id, f"{i}_confidence.json"),
            )
            item = {"id": target_id, "n": i}
            for key in ["pmetric", "confidence", "likelihood", "normalized_likelihood"]:
                if key not in trajs[0][0]:
                    continue
                vals = [tup[0][key] for tup in trajs]
                valid_vals = [val for val in vals if val is not None]
                item[key] = (sum(valid_vals) / len(valid_vals)) if valid_vals else None
                item[key + "_details"] = vals
            item["smiles"] = ".".join([tup[0]["smiles"] for tup in trajs])
            item["gen_seq"] = "|".join([tup[0]["gen_seq"] for tup in trajs])
            item["tgt_chains"] = sample_tgt_chains
            item["lig_chains"] = sample_lig_chains
            gen_block_idx = []
            for tup in trajs:
                gen_block_idx.extend(tup[0]["gen_block_idx"])
            item["gen_block_idx"] = gen_block_idx
            item["struct_only"] = trajs[0][0]["struct_only"]
            item["mark"] = marks[i]
            res_file.write(json.dumps(item) + "\n")

    for path in tmp_files:
        try:
            os.remove(path)
        except Exception:
            pass


def _filter_display_name(filter_obj: Any) -> str:
    return getattr(filter_obj, "name", filter_obj.__class__.__name__)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe_value(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _summarize_filter_outputs(items: Sequence[dict]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for item in items:
        for output in item.get("filter_outputs", []) or []:
            name = str(output.get("name", "unknown"))
            status = str(output.get("status", "UNKNOWN"))
            if name not in summary:
                summary[name] = {"PASSED": 0, "FAILED": 0, "ERROR": 0}
            if status not in summary[name]:
                summary[name][status] = 0
            summary[name][status] += 1
    return summary


def _apply_filters_to_item(save_dir: str, item: dict, filters: Sequence[Any]) -> dict:
    item = dict(item)
    if not filters:
        item["filter_outputs"] = []
        item["filter_passed"] = True
        return item

    inputs = FilterInput(
        path_prefix=os.path.join(save_dir, item["id"], str(item["n"])),
        tgt_chains=item["tgt_chains"],
        lig_chains=item["lig_chains"],
        smiles=item.get("smiles", ""),
        seq=item.get("gen_seq", ""),
        confidence=item.get("confidence", None),
        likelihood=item.get("likelihood", None),
    )
    passed = True
    filter_outputs: List[dict] = []
    for filter_obj in filters:
        result, metrics = ray.get(filter_obj.run.remote(filter_obj, inputs))
        filter_outputs.append(
            {
                "name": _filter_display_name(filter_obj),
                "status": result.name,
                "detail": _json_safe_value(metrics),
            }
        )
        if result != FilterResult.PASSED:
            passed = False
            break
    item["filter_outputs"] = filter_outputs
    item["filter_passed"] = passed
    return item


def _write_result_items(file_path: str, items: Sequence[dict]) -> None:
    with open(file_path, "w") as fout:
        for item in items:
            fout.write(json.dumps(item) + "\n")


def _load_incremental_result_items(results_path: str, processed_line_count: int) -> Tuple[List[dict], int]:
    if not os.path.exists(results_path):
        return [], processed_line_count
    with open(results_path, "r") as fin:
        lines = fin.readlines()
    if lines and not lines[-1].endswith("\n"):
        lines = lines[:-1]
    if processed_line_count >= len(lines):
        return [], len(lines)
    return [json.loads(line) for line in lines[processed_line_count:]], len(lines)


def _generate_and_filter_attempt(
    attempt_dir: str,
    filters: Sequence[Any],
    gen_runner,
    log_cb=None,
) -> Tuple[List[dict], List[dict]]:
    def _log(message: str) -> None:
        if log_cb is None:
            return
        log_cb(message)

    results_path = os.path.join(attempt_dir, "results.jsonl")
    filtered_results_path = os.path.join(attempt_dir, "filtered_results.jsonl")
    raw_items: List[dict] = []
    passed_items: List[dict] = []
    processed_line_count = 0
    generation_error: Dict[str, BaseException] = {}

    def _run_generation() -> None:
        try:
            gen_runner()
        except BaseException as exc:  # propagate worker exceptions after streaming loop drains
            generation_error["exc"] = exc

    gen_thread = threading.Thread(target=_run_generation, daemon=True)
    gen_thread.start()

    while gen_thread.is_alive() or generation_error or os.path.exists(results_path):
        new_items, processed_line_count = _load_incremental_result_items(results_path, processed_line_count)
        if new_items:
            for item in new_items:
                enriched_item = _apply_filters_to_item(attempt_dir, item, filters)
                raw_items.append(enriched_item)
                if enriched_item.get("filter_passed", False):
                    passed_items.append(dict(enriched_item))
                filter_outputs = enriched_item.get("filter_outputs", [])
                final_status = filter_outputs[-1]["status"] if filter_outputs else ("PASSED" if enriched_item.get("filter_passed", False) else "UNKNOWN")
                _log(
                    f"[ui] filter item {item.get('id')}:{item.get('n')} -> {final_status.lower()}"
                )
            continue
        if not gen_thread.is_alive():
            break
        time.sleep(0.2)

    gen_thread.join()
    new_items, processed_line_count = _load_incremental_result_items(results_path, processed_line_count)
    for item in new_items:
        enriched_item = _apply_filters_to_item(attempt_dir, item, filters)
        raw_items.append(enriched_item)
        if enriched_item.get("filter_passed", False):
            passed_items.append(dict(enriched_item))
        filter_outputs = enriched_item.get("filter_outputs", [])
        final_status = filter_outputs[-1]["status"] if filter_outputs else ("PASSED" if enriched_item.get("filter_passed", False) else "UNKNOWN")
        _log(f"[ui] filter item {item.get('id')}:{item.get('n')} -> {final_status.lower()}")

    if "exc" in generation_error:
        raise generation_error["exc"]

    _write_result_items(results_path, raw_items)
    _write_result_items(filtered_results_path, passed_items)
    return raw_items, passed_items


def _apply_filters(save_dir: str, filters: Sequence[Any]) -> List[dict]:
    if not filters:
        return _load_result_items(os.path.join(save_dir, "results.jsonl"))

    items = _load_result_items(os.path.join(save_dir, "results.jsonl"))
    passed_items: List[dict] = []
    for item in items:
        enriched_item = _apply_filters_to_item(save_dir, item, filters)
        item.update(enriched_item)
        if enriched_item["filter_passed"]:
            passed_items.append(item)

    _write_result_items(os.path.join(save_dir, "results.jsonl"), items)
    _write_result_items(os.path.join(save_dir, "filtered_results.jsonl"), passed_items)
    return passed_items


@dataclass
class _AttemptItem:
    attempt_dir: str
    item: dict


def _confidence_sort_key(item: dict) -> float:
    confidence = item.get("confidence", None)
    if confidence is None:
        return float("inf")
    try:
        return float(confidence)
    except Exception:
        return float("inf")


def _copy_result_artifacts(src_dir: str, item: dict, dst_dir: str, dst_item: dict) -> None:
    src_prefix = os.path.join(src_dir, item["id"], str(item["n"]))
    dst_prefix = os.path.join(dst_dir, dst_item["id"], str(dst_item["n"]))
    os.makedirs(os.path.dirname(dst_prefix), exist_ok=True)
    for suffix in [".cif", ".pdb", ".sdf", "_confidence.json"]:
        src_path = src_prefix + suffix
        dst_path = dst_prefix + suffix
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)

    src_pocket = os.path.join(src_dir, item["id"], "pocket.pdb")
    dst_pocket = os.path.join(dst_dir, dst_item["id"], "pocket.pdb")
    if os.path.exists(src_pocket) and not os.path.exists(dst_pocket):
        os.makedirs(os.path.dirname(dst_pocket), exist_ok=True)
        shutil.copyfile(src_pocket, dst_pocket)


def _materialize_attempt_outputs(
    save_dir: str,
    raw_attempt_items: Sequence[_AttemptItem],
    selected_source_keys: set[Tuple[str, str, int]],
) -> Tuple[List[dict], List[dict]]:
    raw_items: List[dict] = []
    selected_items: List[dict] = []
    source_to_new: Dict[Tuple[str, str, int], dict] = {}
    index_by_id: Dict[str, int] = {}

    for entry in raw_attempt_items:
        item = dict(entry.item)
        item_id = str(item["id"])
        new_index = index_by_id.get(item_id, 0)
        index_by_id[item_id] = new_index + 1
        new_item = dict(item)
        new_item["n"] = new_index
        new_item["attempt_dir"] = entry.attempt_dir
        _copy_result_artifacts(entry.attempt_dir, item, save_dir, new_item)
        raw_items.append(new_item)
        source_to_new[(entry.attempt_dir, item_id, int(item["n"]))] = new_item

    for key in selected_source_keys:
        mapped = source_to_new.get(key)
        if mapped is not None:
            selected_items.append(dict(mapped))

    with open(os.path.join(save_dir, "results.jsonl"), "w") as fout:
        for item in raw_items:
            fout.write(json.dumps(item) + "\n")

    with open(os.path.join(save_dir, "filtered_results.jsonl"), "w") as fout:
        for item in selected_items:
            fout.write(json.dumps(item) + "\n")

    return raw_items, selected_items


def _build_fragment_blueprint(smiles: str) -> Tuple[List[Block], List[Tuple[int, int, int, int, BondType]], Dict[str, Tuple[int, int]]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    creator = Creator(0)
    added_blocks, added_bonds = creator.decompose_and_add_fragment(mol, 1)
    fragment_blocks = deepcopy(added_blocks)
    fragment_bonds = [
        (bond.index1[1], bond.index1[2], bond.index2[1], bond.index2[2], bond.bond_type)
        for bond in added_bonds
    ]

    _, atom_groups = tokenize_3d(None, None, rdkit_mol=mol)
    atom_labels = {
        atom.GetIdx(): f"{atom.GetSymbol().lower()}{atom.GetIdx()}"
        for atom in mol.GetAtoms()
    }
    atom_map: Dict[str, Tuple[int, int]] = {}
    for block_index, atom_group in enumerate(atom_groups):
        for local_atom_index, atom_index in enumerate(atom_group):
            atom_map[atom_labels[atom_index]] = (block_index, local_atom_index)

    return fragment_blocks, fragment_bonds, atom_map


@dataclass
class GenerationContext:
    pdb_paths: List[str]
    tgt_chains: List[List[str]]
    lig_chains: Optional[List[List[str]]] = None
    hotspots: Optional[List[List[tuple]]] = None
    checkpoint: str = field(default_factory=_default_checkpoint_path)
    batch_size: int = 8
    n_samples: int = 1
    filter_batch_quota: int = 6
    gpu: int = 0
    sample_opt: Dict[str, Any] = field(default_factory=dict)
    max_retry: Optional[int] = None

    @classmethod
    def from_kwargs(cls, **kwargs) -> "GenerationContext":
        # UI demo: single-target only. Internally keep list-of-targets to reuse PDBDataset as-is.
        pdb_paths = _coerce_single_pdb_path(kwargs)
        tgt_chains = _coerce_single_chain_list(kwargs["tgt_chains"], "tgt_chains")
        lig_chains = _coerce_single_chain_list(kwargs.get("lig_chains", None), "lig_chains") if kwargs.get("lig_chains", None) is not None else None
        hotspots = _coerce_single_hotspots(kwargs.get("hotspots", None))
        if lig_chains is None and hotspots is None:
            raise ValueError("Either lig_chains or hotspots must be provided")
        return cls(
            pdb_paths=pdb_paths,
            tgt_chains=tgt_chains,
            lig_chains=lig_chains,
            hotspots=hotspots,
            checkpoint=resolve_project_path(
                kwargs.get("checkpoint", kwargs.get("confidence_ckpt", kwargs.get("ckpt", _default_checkpoint_path())))
            ),
            batch_size=kwargs.get("batch_size", 8),
            n_samples=kwargs.get("n_samples", 1),
            filter_batch_quota=kwargs.get("filter_batch_quota", 6),
            gpu=kwargs.get("gpu", 0),
            sample_opt=deepcopy(kwargs.get("sample_opt", {})),
            max_retry=kwargs.get("max_retry", None),
        )

    def merged(self, **overrides) -> "GenerationContext":
        data = {
            "pdb_paths": self.pdb_paths,
            "tgt_chains": self.tgt_chains,
            "lig_chains": self.lig_chains,
            "hotspots": self.hotspots,
            "checkpoint": self.checkpoint,
            "batch_size": self.batch_size,
            "n_samples": self.n_samples,
            "filter_batch_quota": self.filter_batch_quota,
            "gpu": self.gpu,
            "sample_opt": deepcopy(self.sample_opt),
            "max_retry": self.max_retry,
        }
        data.update(overrides)
        return GenerationContext.from_kwargs(**data)


@dataclass
class GenerationResult:
    save_dir: str
    results_path: str
    filtered_results_path: Optional[str]
    num_generated: int
    num_passed: int
    attempted_batches: int = 1
    filter_batch_quota: int = 1
    used_fallback: bool = False
    fallback_reason: Optional[str] = None

    def summary(self) -> str:
        lines = [
            f"Save directory: {self.save_dir}",
            f"Generated samples: {self.num_generated}",
            f"Passed filters: {self.num_passed}",
            f"Results file: {self.results_path}",
        ]
        if self.filtered_results_path is not None:
            lines.append(f"Filtered results file: {self.filtered_results_path}")
            lines.append(f"Filter batches tried: {self.attempted_batches}/{self.filter_batch_quota}")
        if self.used_fallback:
            lines.append(f"Fallback: {self.fallback_reason or 'best confidence'}")
        return "\n".join(lines)


@dataclass
class CompiledPrompt:
    template: BaseTemplate
    filters: List[Any]
    sample_opt: Dict[str, Any]
    metadata: Dict[str, Any]

    def summary(self) -> str:
        lines = [
            f"Template: {self.template.name}",
            f"Modality: {self.metadata.get('modality', 'unknown')}",
            f"Guidance: {self.metadata.get('guidance', None)}",
        ]
        if self.filters:
            lines.append(f"Filters: {_stringify_iter(_filter_display_name(f) for f in self.filters)}")
        else:
            lines.append("Filters: none")
        for key, value in self.metadata.items():
            if key in {"modality", "guidance"}:
                continue
            lines.append(f"{key}: {value}")
        return "\n".join(lines)


@dataclass
class FragmentHandle:
    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class _FragmentEntry:
    name: str
    smiles: str
    blocks: List[Block]
    bonds: List[Tuple[int, int, int, int, BondType]]
    atom_map: Dict[str, Tuple[int, int]]


class ProgrammedMoleculeTemplate(BaseTemplate):
    def __init__(
        self,
        blocks: List[Block],
        bonds: List[Tuple[int, int, int, int, BondType]],
        mask_2d: List[int],
        mask_3d: List[int],
        growth_min: int,
        growth_max: int,
        guidance: Optional[float],
    ):
        super().__init__(size_min=None, size_max=None)
        self._blocks = deepcopy(blocks)
        self._bonds = list(bonds)
        self._mask_2d = list(mask_2d)
        self._mask_3d = list(mask_3d)
        self.growth_min = growth_min
        self.growth_max = growth_max
        self.guidance = guidance

    @property
    def name(self) -> str:
        return "ProgrammedMoleculeTemplate"

    @property
    def moltype(self) -> MolType:
        return MolType.MOLECULE

    def sample_size(self, _: ComplexDesc):
        if self.growth_max < self.growth_min:
            raise ValueError("growth_max must be >= growth_min")
        return np.random.randint(self.growth_min, self.growth_max + 1)

    def dummy_lig_block_bonds(self, cplx_desc: ComplexDesc, size: int) -> List[Block]:
        blocks = deepcopy(self._blocks)
        mol_index = len(cplx_desc.cplx)
        bonds = [
            Bond((mol_index, b1, a1), (mol_index, b2, a2), bond_type)
            for b1, a1, b2, a2, bond_type in self._bonds
        ]
        for _ in range(size):
            blocks.append(
                Block(
                    name="UNK",
                    atoms=[Atom(name="C", coordinate=[0, 0, 0], element="C", id=-1)],
                    id=(1, str(len(blocks))),
                )
            )
        return blocks, bonds

    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        loader = MolLoader(tgt_chains=cplx_desc.tgt_chains, lig_chains=cplx_desc.lig_chains)
        data = loader.cplx_to_data(cplx_desc.cplx, cplx_desc.pocket_block_ids, cplx_desc.lig_block_ids)
        growth_blocks = len(cplx_desc.lig_block_ids) - len(self._blocks)
        mask_2d = [0 for _ in cplx_desc.pocket_block_ids] + self._mask_2d + [0 for _ in range(growth_blocks)]
        mask_3d = [0 for _ in cplx_desc.pocket_block_ids] + self._mask_3d + [0 for _ in range(growth_blocks)]
        data["condition_config"] = _make_condition_config(mask_2d, mask_3d, self.guidance)
        return data


class ProgrammedPeptideTemplate(BaseTemplate):
    def __init__(
        self,
        length_min: int,
        length_max: int,
        fixed_residues: Dict[int, str],
        noncanonicals: Dict[int, str],
        guidance: Optional[float],
        cyclize_mode: Optional[str] = None,
    ):
        super().__init__(size_min=length_min, size_max=length_max + 1)
        self.length_min = length_min
        self.length_max = length_max
        self.fixed_residues = dict(fixed_residues)
        self.noncanonicals = dict(noncanonicals)
        self.guidance = guidance
        self.cyclize_mode = cyclize_mode
        self._last_fixed_mask: List[int] = []

    @property
    def name(self) -> str:
        return "ProgrammedPeptideTemplate"

    @property
    def moltype(self) -> MolType:
        return MolType.PEPTIDE

    def sample_size(self, _: ComplexDesc):
        return np.random.randint(self.length_min, self.length_max + 1)

    def dummy_lig_block_bonds(self, cplx_desc: ComplexDesc, size: int):
        invalid_positions = [pos for pos in list(self.fixed_residues) + list(self.noncanonicals) if pos > size or pos < 1]
        if invalid_positions:
            raise ValueError(f"Prompt positions {invalid_positions} are outside sampled peptide length {size}")

        # Explicit bond indices must point to the ligand molecule that will be appended
        # after the existing complex molecules.
        creator = Creator(len(cplx_desc.cplx))
        fixed_mask: List[int] = []
        prev_c_term: Tuple[int, int] | None = None

        for position in range(1, size + 1):
            if position in self.noncanonicals:
                nsaa_smiles = self.noncanonicals[position]
                nsaa_mol, n_term_idx, c_term_idx = find_term_and_delete_dummy(nsaa_smiles)
                offset = len(creator.blocks)
                added_blocks, _ = creator.decompose_and_add_fragment(nsaa_mol, position)
                cur_n_term = None
                cur_c_term = None
                for block_offset, block in enumerate(added_blocks):
                    for atom_index, atom in enumerate(block):
                        atom_id = int(atom.id) - 1
                        if atom_id == n_term_idx:
                            cur_n_term = (offset + block_offset, atom_index)
                        if atom_id == c_term_idx:
                            cur_c_term = (offset + block_offset, atom_index)
                if prev_c_term is not None and cur_n_term is not None:
                    creator.add_bond(prev_c_term[0], cur_n_term[0], prev_c_term[1], cur_n_term[1], BondType.SINGLE)
                prev_c_term = cur_c_term
                fixed_mask.extend([1 for _ in added_blocks])
                continue

            if position in self.fixed_residues:
                residue_name = self.fixed_residues[position]
                creator.add_block(residue_name, (position, ""))
                block_index = len(creator.blocks) - 1
                cur_n_term = (block_index, _find_atom_index(creator.blocks[block_index], "N"))
                cur_c_term = (block_index, _find_atom_index(creator.blocks[block_index], "C"))
                if prev_c_term is not None:
                    creator.add_bond(prev_c_term[0], cur_n_term[0], prev_c_term[1], cur_n_term[1], BondType.SINGLE)
                prev_c_term = cur_c_term
                fixed_mask.append(1)
                continue

            creator.add_unk((position, ""), is_aa=True)
            prev_c_term = None
            fixed_mask.append(0)

        blocks, bonds = creator.get_results()
        if self.cyclize_mode is not None:
            if len(blocks) < 2:
                raise ValueError("Cyclization requires at least two residues")
            if self.cyclize_mode == "head_tail":
                head_n = _find_atom_index(blocks[0], "N")
                tail_c = _find_atom_index(blocks[-1], "C")
                creator.add_bond(0, len(blocks) - 1, head_n, tail_c, BondType.SINGLE)
            elif self.cyclize_mode == "disulfide":
                head_sg = _find_atom_index(blocks[0], "SG")
                tail_sg = _find_atom_index(blocks[-1], "SG")
                creator.add_bond(0, len(blocks) - 1, head_sg, tail_sg, BondType.SINGLE)
            else:
                raise ValueError(f"Unsupported cyclize mode: {self.cyclize_mode}")
            blocks, bonds = creator.get_results()
        self._last_fixed_mask = fixed_mask
        return blocks, bonds

    def to_data(self, cplx_desc: ComplexDesc) -> dict:
        loader = PeptideLoader(tgt_chains=cplx_desc.tgt_chains, lig_chains=cplx_desc.lig_chains)
        data = loader.cplx_to_data(cplx_desc.cplx, cplx_desc.pocket_block_ids, cplx_desc.lig_block_ids)
        mask_2d = [0 for _ in cplx_desc.pocket_block_ids] + self._last_fixed_mask
        mask_3d = [0 for _ in cplx_desc.pocket_block_ids] + [0 for _ in self._last_fixed_mask]
        data["condition_config"] = _make_condition_config(mask_2d, mask_3d, self.guidance)
        return data


class PromptProgram:
    def __init__(self):
        self._filters: List[Any] = []
        self._guidance: Optional[float] = None
        self._context: Optional[GenerationContext] = None

    def add_filter(self, filter_obj: Any) -> "PromptProgram":
        self._filters.append(filter_obj)
        return self

    def _ensure_default_filters(self, defaults: Sequence[Any]) -> None:
        existing = {filter_obj.__class__.__name__ for filter_obj in self._filters}
        for filter_obj in defaults:
            if filter_obj.__class__.__name__ not in existing:
                self._filters.append(filter_obj)
                existing.add(filter_obj.__class__.__name__)

    def set_guidance(self, value: float) -> "PromptProgram":
        self._guidance = float(value)
        return self

    def set_context(self, **kwargs) -> "PromptProgram":
        self._context = GenerationContext.from_kwargs(**kwargs)
        return self

    def get_context(self) -> Optional[GenerationContext]:
        return self._context

    def summary(self) -> str:
        return self.inspect()

    def inspect(self) -> str:
        raise NotImplementedError()

    def compile(self) -> CompiledPrompt:
        raise NotImplementedError()

    def to_visual_payload(self) -> Dict[str, Any]:
        return {"kind": "text", "text": self.inspect(), "svg": None}

    def run_generation(self, save_dir: str, **overrides) -> GenerationResult:
        ui_log = overrides.pop("_ui_log", None)

        def _log(msg: str) -> None:
            if ui_log is None:
                return
            try:
                ui_log(f"{msg}\n")
            except Exception:
                pass

        context = self._context
        if context is None:
            context = GenerationContext.from_kwargs(**overrides)
        elif overrides:
            context = context.merged(**overrides)
        compiled = self.compile()
        os.makedirs(save_dir, exist_ok=True)
        device = torch.device("cpu" if context.gpu == -1 else f"cuda:{context.gpu}")
        model, conf_model = load_model(None, context.checkpoint, device)
        effective_batch_size = int(context.batch_size)
        effective_n_samples = int(context.n_samples)
        filter_batch_quota = max(int(context.filter_batch_quota), 1)
        if compiled.filters:
            effective_batch_size = max(effective_batch_size, 8)
            effective_n_samples = effective_batch_size
            _log(
                f"[ui] filters enabled: {len(compiled.filters)} filters, "
                f"batch_size={effective_batch_size}, filter_batch_quota={filter_batch_quota}"
            )
        tmp_input_paths: List[str] = []
        if compiled.metadata.get("modality") == "antibody" and context.lig_chains:
            cdr_type = compiled.metadata.get("cdrs", [None])[0]
            preprocess_length_ranges = compiled.metadata.get("preprocess_length_ranges", {})
            if cdr_type is not None and cdr_type in preprocess_length_ranges:
                tmp_input_paths = _write_antibody_preprocessed_input(
                    source_path=context.pdb_paths[0],
                    lig_chains=context.lig_chains[0],
                    cdr_type=cdr_type,
                    length_range=preprocess_length_ranges.get(cdr_type),
                    resolved_constraint={},
                )
                if tmp_input_paths:
                    context = GenerationContext(
                        pdb_paths=[tmp_input_paths[-1]],
                        tgt_chains=context.tgt_chains,
                        lig_chains=context.lig_chains,
                        hotspots=context.hotspots,
                        checkpoint=context.checkpoint,
                        batch_size=effective_batch_size,
                        n_samples=effective_n_samples,
                        filter_batch_quota=filter_batch_quota,
                        gpu=context.gpu,
                        sample_opt=deepcopy(context.sample_opt),
                        max_retry=context.max_retry,
                    )
        dataset = PDBDataset(
            pdb_paths=context.pdb_paths,
            tgt_chains=context.tgt_chains,
            lig_chains=context.lig_chains,
            hotspots=context.hotspots,
            template_config=compiled.template,
        )
        sample_opt = deepcopy(context.sample_opt)
        sample_opt.update(compiled.sample_opt)
        effective_conf_model = conf_model
        run_n_cycles = 1
        if isinstance(compiled.template, AntibodyMultipleCDR):
            if getattr(compiled.template, "ui_use_constrained_generation", False):
                gen_func = generate_multiple_cdrs_constrained
            else:
                gen_func = generate_multiple_cdrs
        else:
            gen_func = generate_for_one_template
        try:
            if compiled.filters:
                attempts_root = os.path.join(save_dir, "_filter_attempts")
                os.makedirs(attempts_root, exist_ok=True)
                all_raw_attempt_items: List[_AttemptItem] = []
                all_passed_attempt_items: List[_AttemptItem] = []
                actual_num_passed = 0
                attempts_used = 0
                used_fallback = False
                fallback_reason = None
                for attempt_idx in range(filter_batch_quota):
                    attempts_used = attempt_idx + 1
                    attempt_dir = os.path.join(attempts_root, f"attempt_{attempt_idx:02d}")
                    os.makedirs(attempt_dir, exist_ok=True)
                    _log(f"[ui] filter batch {attempts_used}/{filter_batch_quota} started")
                    raw_items, passed_items = _generate_and_filter_attempt(
                        attempt_dir=attempt_dir,
                        filters=compiled.filters,
                        gen_runner=lambda: gen_func(
                            model,
                            dataset,
                            effective_n_samples,
                            effective_batch_size,
                            attempt_dir,
                            device,
                            sample_opt,
                            n_cycles=run_n_cycles,
                            conf_model=effective_conf_model,
                            max_retry=context.max_retry,
                            check_validity=False
                        ),
                        log_cb=lambda message: _log(message.rstrip("\n")),
                    )
                    _log(
                        f"[ui] filter batch {attempts_used}/{filter_batch_quota} finished: "
                        f"generated={len(raw_items)}, passed={len(passed_items)}"
                    )
                    filter_summary = _summarize_filter_outputs(raw_items)
                    if filter_summary:
                        summary_parts = []
                        for filter_name, counts in filter_summary.items():
                            summary_parts.append(
                                f"{filter_name}: passed={counts.get('PASSED', 0)}, "
                                f"failed={counts.get('FAILED', 0)}, error={counts.get('ERROR', 0)}"
                            )
                        _log(
                            f"[ui] filter batch {attempts_used}/{filter_batch_quota} details: "
                            + " | ".join(summary_parts)
                        )
                    all_raw_attempt_items.extend(_AttemptItem(attempt_dir, item) for item in raw_items)
                    all_passed_attempt_items.extend(_AttemptItem(attempt_dir, item) for item in passed_items)
                    actual_num_passed += len(passed_items)
                    if passed_items:
                        _log(f"[ui] stopping early after batch {attempts_used}: found filtered results")
                        break

                if all_passed_attempt_items:
                    selected_attempt_items = sorted(
                        all_passed_attempt_items,
                        key=lambda entry: _confidence_sort_key(entry.item),
                    )
                else:
                    if all_raw_attempt_items:
                        best_raw = min(all_raw_attempt_items, key=lambda entry: _confidence_sort_key(entry.item))
                        best_item = dict(best_raw.item)
                        best_item["selection_reason"] = "best_confidence_fallback"
                        used_fallback = True
                        fallback_reason = "no sample passed filters; selected lowest PDE/confidence"
                        _log("[ui] no sample passed filters; using best-confidence fallback")
                        all_raw_attempt_items = [
                            _AttemptItem(
                                entry.attempt_dir,
                                dict(entry.item, selection_reason="best_confidence_fallback") if entry is best_raw else dict(entry.item),
                            )
                            for entry in all_raw_attempt_items
                        ]
                        selected_attempt_items = [_AttemptItem(best_raw.attempt_dir, best_item)]
                    else:
                        selected_attempt_items = []

                selected_keys = {
                    (entry.attempt_dir, str(entry.item["id"]), int(entry.item["n"]))
                    for entry in selected_attempt_items
                }
                raw_items, selected_items = _materialize_attempt_outputs(save_dir, all_raw_attempt_items, selected_keys)
                passed_items = selected_items
                filtered_path = os.path.join(save_dir, "filtered_results.jsonl")
            else:
                gen_func(
                    model,
                    dataset,
                    effective_n_samples,
                    effective_batch_size,
                    save_dir,
                    device,
                    sample_opt,
                    n_cycles=run_n_cycles,
                    conf_model=effective_conf_model,
                    max_retry=context.max_retry,
                )
                raw_items = _load_result_items(os.path.join(save_dir, "results.jsonl"))
                passed_items = raw_items
                actual_num_passed = len(passed_items)
                filtered_path = None
            return GenerationResult(
                save_dir=save_dir,
                results_path=os.path.join(save_dir, "results.jsonl"),
                filtered_results_path=filtered_path,
                num_generated=len(raw_items),
                num_passed=actual_num_passed,
                attempted_batches=attempts_used if compiled.filters else 1,
                filter_batch_quota=filter_batch_quota if compiled.filters else 1,
                used_fallback=used_fallback if compiled.filters else False,
                fallback_reason=fallback_reason if compiled.filters else None,
            )
        finally:
            for path in tmp_input_paths:
                try:
                    os.remove(path)
                except Exception:
                    pass


class MoleculePrompt(PromptProgram):
    def __init__(self):
        super().__init__()
        self._ensure_default_filters([
            AbnormalConfidenceFilter(),
            ChiralCentersFilter(center_max=8, ring_mode=True),
            RotatableBondsFilter(max_num_rot_bonds=7),
            PhysicalValidityFilter(),
            MolBeautyFilter(th=1),
        ])
        self._fragments: Dict[str, _FragmentEntry] = {}
        self._explicit_bonds: List[Tuple[str, str, BondType]] = []
        self._pins: List[str] = []
        self._placements: Dict[str, Tuple[List[float], float]] = {}
        self._attachment_points: Dict[str, str] = {}
        self._growth_min = 3
        self._growth_max = 6

    def add_fragment(self, smiles: str, name: Optional[str] = None, pin: bool = True) -> FragmentHandle:
        if name is None:
            name = self._make_default_fragment_name()
        if name in self._fragments:
            raise ValueError(f"Fragment name {name} already exists")
        blocks, bonds, atom_map = _build_fragment_blueprint(smiles)
        self._fragments[name] = _FragmentEntry(name=name, smiles=smiles, blocks=blocks, bonds=bonds, atom_map=atom_map)
        if pin:
            self.pin(name)
        return FragmentHandle(name)

    def add_attachment_point(self, target: str | FragmentHandle, atom: str, label: str) -> "MoleculePrompt":
        fragment_name = self._coerce_fragment_name(target)
        atom_label = f"{fragment_name}:{atom}"
        self._resolve_atom_label(atom_label)
        self._attachment_points[label] = atom_label
        return self

    def add_bond(self, atom1: str, atom2: str, bond_type: Any = "single") -> "MoleculePrompt":
        resolved_atom1 = self._resolve_reference(atom1)
        resolved_atom2 = self._resolve_reference(atom2)
        self._resolve_atom_label(resolved_atom1)
        self._resolve_atom_label(resolved_atom2)
        self._explicit_bonds.append((resolved_atom1, resolved_atom2, _bond_type_from_value(bond_type)))
        return self

    def pin(self, target: str | FragmentHandle) -> "MoleculePrompt":
        resolved = self._resolve_reference(target)
        if resolved not in self._pins:
            self._pins.append(resolved)
        return self

    def place(self, target: str | FragmentHandle, center: Sequence[float], std: float = 1.0) -> "MoleculePrompt":
        resolved = self._resolve_reference(target)
        if len(center) != 3:
            raise ValueError("center must have exactly three values")
        self._placements[resolved] = ([float(v) for v in center], float(std))
        return self

    def allow_growth(self, min_blocks: int, max_blocks: int) -> "MoleculePrompt":
        if min_blocks < 0 or max_blocks < min_blocks:
            raise ValueError("allow_growth expects 0 <= min_blocks <= max_blocks")
        self._growth_min = int(min_blocks)
        self._growth_max = int(max_blocks)
        return self

    def fragments(self) -> List[str]:
        return list(self._fragments.keys())

    def atoms(self, fragment_name: str) -> List[str]:
        fragment = self._get_fragment(fragment_name)
        return [f"{fragment_name}:{label}" for label in sorted(fragment.atom_map.keys(), key=_atom_label_sort_key)]

    def labels(self) -> List[str]:
        labels = []
        for fragment_name in self.fragments():
            labels.extend(self.atoms(fragment_name))
        labels.extend(sorted(self._attachment_points.keys()))
        return labels

    def fragment_mappings(self) -> List[Dict[str, str]]:
        return [
            {"name": fragment_name, "smiles": fragment.smiles}
            for fragment_name, fragment in self._fragments.items()
        ]

    def suggest_open_bonds(self) -> List[str]:
        return [label for label in self.labels() if ":" in label]

    def inspect_atoms(self) -> str:
        lines = []
        for fragment_name in self.fragments():
            lines.append(f"{fragment_name}: {_stringify_iter(self.atoms(fragment_name))}")
        return "\n".join(lines) if lines else "No fragments"

    def highlight(self, target: str | FragmentHandle) -> str:
        resolved = self._resolve_reference(target)
        if ":" in resolved:
            return f"Selected atom: {resolved}"
        return f"Selected fragment: {resolved}\nAtoms: {_stringify_iter(self.atoms(resolved))}"

    def inspect(self) -> str:
        lines = ["Prompt Type: molecule", "Fragments:"]
        if not self._fragments:
            lines.append("  - none")
        for fragment_name, fragment in self._fragments.items():
            atoms = " ".join(sorted(fragment.atom_map.keys(), key=_atom_label_sort_key))
            lines.append(f"  - {fragment_name} -> {fragment.smiles}   atoms: {atoms}")
        lines.append("Pinned:")
        if self._pins:
            for item in self._pins:
                lines.append(f"  - {item}")
        else:
            lines.append("  - none")
        lines.append("Placements:")
        if self._placements:
            for label, (center, std) in self._placements.items():
                lines.append(f"  - {label} -> center={center}, std={std}")
        else:
            lines.append("  - none")
        lines.append("Bonds:")
        if self._explicit_bonds:
            for atom1, atom2, bond_type in self._explicit_bonds:
                lines.append(f"  - {atom1} --{bond_type.name.lower()}--> {atom2}")
        else:
            lines.append("  - none")
        lines.append(f"Growth Budget: [{self._growth_min}, {self._growth_max}]")
        lines.append("Filters:")
        if self._filters:
            for filter_obj in self._filters:
                lines.append(f"  - {_filter_display_name(filter_obj)}")
        else:
            lines.append("  - none")
        lines.append("Compiled Template:")
        lines.append("  - ProgrammedMoleculeTemplate")
        return "\n".join(lines)

    def to_svg(self, width: int = 520, height: int = 360) -> str:
        mols: List[Chem.Mol] = []
        atom_label_map: Dict[int, str] = {}
        atom_offset = 0
        global_index_map: Dict[str, int] = {}

        for fragment_name, fragment in self._fragments.items():
            mol = Chem.MolFromSmiles(fragment.smiles)
            if mol is None:
                continue
            mols.append(mol)
            for atom in mol.GetAtoms():
                atom_label = f"{fragment_name}:{atom.GetSymbol().lower()}{atom.GetIdx()}"
                atom_label_map[atom_offset + atom.GetIdx()] = atom_label
                global_index_map[atom_label] = atom_offset + atom.GetIdx()
            atom_offset += mol.GetNumAtoms()

        if not mols:
            raise ValueError("No fragments available for SVG rendering")

        combined = mols[0]
        for mol in mols[1:]:
            combined = Chem.CombineMols(combined, mol)

        editable = Chem.RWMol(combined)
        for atom1, atom2, bond_type in self._explicit_bonds:
            resolved_atom1 = self._resolve_reference(atom1)
            resolved_atom2 = self._resolve_reference(atom2)
            idx1 = global_index_map.get(resolved_atom1, None)
            idx2 = global_index_map.get(resolved_atom2, None)
            if idx1 is None or idx2 is None:
                continue
            if editable.GetBondBetweenAtoms(idx1, idx2) is None:
                editable.AddBond(idx1, idx2, bond_type_to_rdkit(bond_type))

        return _mol_to_svg(editable.GetMol(), atom_labels=atom_label_map, width=width, height=height)

    def to_visual_payload(self) -> Dict[str, Any]:
        try:
            svg = self.to_svg()
        except Exception:
            svg = None
        return {
            "kind": "molecule",
            "text": self.inspect(),
            "svg": svg,
            "fragment_mappings": self.fragment_mappings(),
        }

    def compile(self) -> CompiledPrompt:
        if not self._fragments:
            raise ValueError("MoleculePrompt requires at least one fragment")

        global_blocks: List[Block] = []
        global_bonds: List[Tuple[int, int, int, int, BondType]] = []
        block_owner: List[str] = []
        global_atom_map: Dict[str, Tuple[int, int]] = {}

        for fragment_name, fragment in self._fragments.items():
            block_offset = len(global_blocks)
            placement = self._placements.get(fragment_name, None)
            for block in deepcopy(fragment.blocks):
                if placement is not None:
                    center, _ = placement
                    for atom in block:
                        atom.coordinate = list(center)
                block.id = (1, str(len(global_blocks)))
                global_blocks.append(block)
                block_owner.append(fragment_name)
            for block_i, atom_i, block_j, atom_j, bond_type in fragment.bonds:
                global_bonds.append((block_offset + block_i, atom_i, block_offset + block_j, atom_j, bond_type))
            for atom_label, (block_i, atom_i) in fragment.atom_map.items():
                global_atom_map[f"{fragment_name}:{atom_label}"] = (block_offset + block_i, atom_i)

        for atom1, atom2, bond_type in self._explicit_bonds:
            block_i, atom_i = global_atom_map[self._resolve_reference(atom1)]
            block_j, atom_j = global_atom_map[self._resolve_reference(atom2)]
            global_bonds.append((block_i, atom_i, block_j, atom_j, bond_type))

        pin_blocks = set()
        for item in self._pins:
            pin_blocks.update(self._resolve_target_blocks(item, global_atom_map, block_owner))

        place_blocks = set()
        for item in self._placements:
            place_blocks.update(self._resolve_target_blocks(item, global_atom_map, block_owner))

        template = ProgrammedMoleculeTemplate(
            blocks=global_blocks,
            bonds=global_bonds,
            mask_2d=[1 if index in pin_blocks else 0 for index in range(len(global_blocks))],
            mask_3d=[1 if index in place_blocks else 0 for index in range(len(global_blocks))],
            growth_min=self._growth_min,
            growth_max=self._growth_max,
            guidance=self._guidance,
        )
        return CompiledPrompt(
            template=template,
            filters=list(self._filters),
            sample_opt={},
            metadata={
                "modality": "molecule",
                "guidance": self._guidance,
                "fragments": self.fragments(),
                "explicit_bonds": len(self._explicit_bonds),
                "growth_budget": [self._growth_min, self._growth_max],
            },
        )

    def _coerce_fragment_name(self, target: str | FragmentHandle) -> str:
        if isinstance(target, FragmentHandle):
            return target.name
        if isinstance(target, str) and target in self._fragments:
            return target
        raise ValueError(f"Unknown fragment reference: {target}")

    def _resolve_reference(self, value: str | FragmentHandle) -> str:
        if isinstance(value, FragmentHandle):
            return value.name
        if value in self._attachment_points:
            return self._attachment_points[value]
        return value

    def _resolve_atom_label(self, value: str) -> Tuple[int, int]:
        if ":" not in value:
            raise ValueError(f"Atom label must be in the form fragment:aN, got {value}")
        fragment_name, atom_label = value.rsplit(":", 1)
        fragment = self._get_fragment(fragment_name)
        if atom_label not in fragment.atom_map:
            raise ValueError(f"Unknown atom label {value}")
        return fragment.atom_map[atom_label]

    def _resolve_target_blocks(
        self,
        target: str,
        global_atom_map: Dict[str, Tuple[int, int]],
        block_owner: List[str],
    ) -> set[int]:
        resolved = self._resolve_reference(target)
        if resolved in self._fragments:
            return {index for index, owner in enumerate(block_owner) if owner == resolved}
        if ":" in resolved:
            block_index, _ = global_atom_map[resolved]
            return {block_index}
        return {index for index, owner in enumerate(block_owner) if owner == resolved}

    def _get_fragment(self, fragment_name: str) -> _FragmentEntry:
        if fragment_name not in self._fragments:
            raise ValueError(f"Unknown fragment: {fragment_name}")
        return self._fragments[fragment_name]

    def _make_default_fragment_name(self) -> str:
        prefix = "F"
        next_index = len(self._fragments) + 1
        while f"{prefix}{next_index}" in self._fragments:
            next_index += 1
        return f"{prefix}{next_index}"


class PeptidePrompt(PromptProgram):
    def __init__(self, length: int = 12):
        super().__init__()
        if isinstance(length, (list, tuple)):
            # UI: single length only.
            if len(length) != 2 or int(length[0]) != int(length[1]):
                raise ValueError("PeptidePrompt(length=...) expects a single integer length in the UI demo")
            length = int(length[0])
        self.length = int(length)
        self._fixed_positions: Dict[int, str] = {}
        self._noncanonical_positions: Dict[int, str] = {}
        self._random_motif: Optional[str] = None
        self._random_nsaa: List[str] = []
        self._random_nsaa_count: int = 0
        self._cyclize_mode: Optional[str] = None
        self._resolved_fixed_positions: Optional[Dict[int, str]] = None
        self._resolved_noncanonical_positions: Optional[Dict[int, str]] = None
        self._resolved_random_motif_positions: List[int] = []
        self._resolved_random_nsaa_positions: List[int] = []
        self._ensure_default_filters([
            AbnormalConfidenceFilter(),
            PhysicalValidityFilter(),
            LTypeAAFilter(),
        ])

    def _invalidate_resolution(self):
        self._resolved_fixed_positions = None
        self._resolved_noncanonical_positions = None
        self._resolved_random_motif_positions = []
        self._resolved_random_nsaa_positions = []

    def set_length(self, length: int) -> "PeptidePrompt":
        if int(length) <= 0:
            raise ValueError("set_length expects length > 0")
        self.length = int(length)
        self._invalidate_resolution()
        return self

    def add_motif(self, seq: str, positions: Optional[Sequence[int]] = None, fixed: bool = True) -> "PeptidePrompt":
        if not fixed:
            raise ValueError("The demo only supports fixed motifs")
        if positions is None:
            # Random insertion per generation.
            self._random_motif = str(seq)
            self._invalidate_resolution()
            return self
        if len(seq) != len(positions):
            raise ValueError("positions must have the same length as seq")
        for aa, position in zip(seq, positions):
            self._fixed_positions[int(position)] = VOCAB.symbol_to_abrv(aa)
        self._invalidate_resolution()
        return self

    def add_noncanonical(self, smiles: str, positions: Optional[Sequence[int]] = None, count: Optional[int] = None) -> "PeptidePrompt":
        if positions is None:
            # Random insertion per generation. If count is omitted, default to 1.
            self._random_nsaa.append(smiles)
            self._random_nsaa_count = int(count) if count is not None else 1
            self._invalidate_resolution()
            return self
        if count is not None and count != len(positions):
            raise ValueError("count must match the number of positions when both are provided")
        for position in positions:
            self._noncanonical_positions[int(position)] = smiles
        self._invalidate_resolution()
        return self

    def cyclize(self, mode: str = "head_tail") -> "PeptidePrompt":
        if mode not in {"head_tail", "disulfide"}:
            raise ValueError("Supported cyclize modes: 'head_tail', 'disulfide'")
        self._cyclize_mode = mode
        self._invalidate_resolution()
        return self

    def _resolve_random_features(self):
        if self._resolved_fixed_positions is not None and self._resolved_noncanonical_positions is not None:
            return

        fixed = dict(self._fixed_positions)
        noncanonicals = dict(self._noncanonical_positions)
        occupied = set(fixed.keys()) | set(noncanonicals.keys())
        reserved = set()

        if self._cyclize_mode == "head_tail":
            if 1 in noncanonicals or self.length in noncanonicals:
                raise ValueError("head_tail cyclization does not support nsAA at the N/C termini in the demo")
            # Match the original template logic: head excludes PRO/CYS, tail excludes CYS.
            head_symbols = [symbol for symbol, abrv in const.aas if abrv not in {"PRO", "CYS"}]
            tail_symbols = [symbol for symbol, abrv in const.aas if abrv not in {"CYS"}]
            if 1 not in fixed:
                fixed[1] = VOCAB.symbol_to_abrv(str(np.random.choice(head_symbols)))
            if self.length not in fixed:
                fixed[self.length] = VOCAB.symbol_to_abrv(str(np.random.choice(tail_symbols)))
            reserved.update({1, self.length})
        elif self._cyclize_mode == "disulfide":
            if 1 in noncanonicals or self.length in noncanonicals:
                raise ValueError("disulfide cyclization does not support nsAA at the S-S termini in the demo")
            fixed[1] = VOCAB.symbol_to_abrv("C")
            fixed[self.length] = VOCAB.symbol_to_abrv("C")
            reserved.update({1, self.length})

        occupied = set(fixed.keys()) | set(noncanonicals.keys())

        random_motif_positions: List[int] = []
        if self._random_motif is not None:
            motif_len = len(self._random_motif)
            if motif_len > self.length:
                raise ValueError("Motif length exceeds peptide length")
            candidates = []
            for start in range(1, self.length - motif_len + 2):
                positions = list(range(start, start + motif_len))
                # avoid overwriting reserved/occupied positions if possible
                if any(pos in occupied or pos in reserved for pos in positions):
                    continue
                candidates.append(positions)
            if not candidates:
                # fallback: allow overlap with editable positions only, but never overwrite explicit fixed/nsAA
                for start in range(1, self.length - motif_len + 2):
                    positions = list(range(start, start + motif_len))
                    if any(pos in self._fixed_positions or pos in self._noncanonical_positions for pos in positions):
                        continue
                    candidates.append(positions)
            if not candidates:
                raise ValueError("Could not place random motif without conflicting with fixed constraints")
            random_motif_positions = list(candidates[np.random.randint(0, len(candidates))])
            for aa, pos in zip(self._random_motif, random_motif_positions):
                fixed[pos] = VOCAB.symbol_to_abrv(aa)
            occupied.update(random_motif_positions)

        random_nsaa_positions: List[int] = []
        if self._random_nsaa:
            count = max(int(self._random_nsaa_count), 1)
            available = [pos for pos in range(1, self.length + 1) if pos not in occupied and pos not in reserved]
            if len(available) < count:
                raise ValueError("Not enough free positions to place random nsAA")
            chosen = np.random.choice(available, count, replace=False).tolist()
            for index, pos in enumerate(chosen):
                smi = self._random_nsaa[index % len(self._random_nsaa)]
                noncanonicals[int(pos)] = smi
            random_nsaa_positions = [int(pos) for pos in chosen]

        self._resolved_fixed_positions = fixed
        self._resolved_noncanonical_positions = noncanonicals
        self._resolved_random_motif_positions = random_motif_positions
        self._resolved_random_nsaa_positions = random_nsaa_positions

    def inspect(self) -> str:
        self._resolve_random_features()
        fixed_positions = self._resolved_fixed_positions or {}
        noncanonical_positions = self._resolved_noncanonical_positions or {}
        lines = [
            "Prompt Type: peptide",
            f"Length: {self.length}",
            "Fixed Residues:",
        ]
        if fixed_positions:
            for position in sorted(fixed_positions):
                lines.append(f"  - {position}: {fixed_positions[position]}")
        else:
            lines.append("  - none")
        lines.append("Non-Canonical Residues:")
        if noncanonical_positions:
            for position in sorted(noncanonical_positions):
                lines.append(f"  - {position}: {noncanonical_positions[position]}")
        else:
            lines.append("  - none")

        lines.append("Random Motif:")
        lines.append(f"  - {self._random_motif if self._random_motif is not None else 'none'}")
        if self._resolved_random_motif_positions:
            lines.append(f"  - resolved_positions: {self._resolved_random_motif_positions}")

        lines.append("Random nsAA:")
        if self._random_nsaa:
            lines.append(f"  - count: {self._random_nsaa_count}")
            lines.append(f"  - library_size: {len(self._random_nsaa)}")
            lines.append(f"  - resolved_positions: {self._resolved_random_nsaa_positions}")
        else:
            lines.append("  - none")

        lines.append("Cyclization:")
        lines.append(f"  - {self._cyclize_mode if self._cyclize_mode is not None else 'none'}")
        lines.append("Filters:")
        if self._filters:
            for filter_obj in self._filters:
                lines.append(f"  - {_filter_display_name(filter_obj)}")
        else:
            lines.append("  - none")
        lines.append("Compiled Template:")
        lines.append("  - (dynamic; see compile())")
        return "\n".join(lines)

    def compile(self) -> CompiledPrompt:
        self._resolve_random_features()
        fixed_positions = self._resolved_fixed_positions or {}
        noncanonical_positions = self._resolved_noncanonical_positions or {}
        if self.length <= 0:
            raise ValueError("Invalid peptide length")
        template = ProgrammedPeptideTemplate(
            length_min=self.length,
            length_max=self.length,
            fixed_residues=fixed_positions,
            noncanonicals=noncanonical_positions,
            guidance=self._guidance,
            cyclize_mode=self._cyclize_mode,
        )
        template_name = "ProgrammedPeptideTemplate"
        return CompiledPrompt(
            template=template,
            filters=list(self._filters),
            sample_opt={},
            metadata={
                "modality": "peptide",
                "guidance": self._guidance,
                "fixed_positions": sorted(fixed_positions.keys()),
                "noncanonical_positions": sorted(noncanonical_positions.keys()),
                "template_name": template_name,
            },
        )

    def to_visual_payload(self) -> Dict[str, Any]:
        self._resolve_random_features()
        try:
            svg = self.to_svg()
        except Exception:
            svg = None
        nsaa_previews: List[Dict[str, Any]] = []
        for smi in self._random_nsaa[:3]:
            try:
                nsaa_previews.append({"smiles": smi, "svg": nsaa_smiles_to_svg(smi)})
            except Exception:
                nsaa_previews.append({"smiles": smi, "svg": None})
        return {
            "kind": "peptide",
            "text": self.inspect(),
            "svg": svg,
            "random_motif": self._random_motif,
            "random_motif_positions": list(self._resolved_random_motif_positions),
            "random_nsaa_count": self._random_nsaa_count if self._random_nsaa else 0,
            "random_nsaa_positions": list(self._resolved_random_nsaa_positions),
            "random_nsaa_previews": nsaa_previews,
            "cyclize_mode": self._cyclize_mode,
        }

    def to_svg(self, width: int = 520, height: int = 260) -> str:
        self._resolve_random_features()
        fixed_positions = self._resolved_fixed_positions or {}
        noncanonical_positions = self._resolved_noncanonical_positions or {}
        motif_positions = set(self._resolved_random_motif_positions)
        nsaa_positions = set(self._resolved_random_nsaa_positions)

        # Render a compact sequence bar:
        # - fixed residues: show one-letter symbol in a colored box
        # - editable residues: show "."
        # - nsAA: show "X"
        display_len = max(self.length, 1)
        cell_w = max(int((width - 20) / max(display_len, 1)), 10)
        cell_h = 18
        left = 10
        top = 30
        font_size = 12
        annotation_lines = 1
        if self._random_motif is not None:
            annotation_lines += 1
        if self._random_nsaa:
            annotation_lines += 1
        annotation_height = 16 * annotation_lines
        seq_top = top + annotation_height + 10

        def esc(text: str) -> str:
            return (
                text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )

        def one_letter_from_abrv(abrv: str) -> str:
            sym = VOCAB.abrv_to_symbol(abrv)
            if sym is None:
                return "?"
            # For standard amino acids, symbol is typically one letter.
            return str(sym)[0]

        parts: List[str] = []
        parts.append("<?xml version='1.0' encoding='utf-8'?>")
        parts.append(
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}px' height='{height}px' viewBox='0 0 {width} {height}'>"
        )
        parts.append(f"<rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff' />")
        title = f"Peptide Prompt  (len: {self.length})"
        parts.append(f"<text x='{left}' y='18' font-family='monospace' font-size='{font_size}' fill='#0f172a'>{esc(title)}</text>")
        if self._random_motif is not None:
            parts.append(
                f"<text x='{left}' y='34' font-family='monospace' font-size='11' fill='#334155'>{esc('motif (random): ' + self._random_motif)}</text>"
            )
        if self._random_nsaa:
            parts.append(
                f"<text x='{left}' y='48' font-family='monospace' font-size='11' fill='#334155'>{esc('nsAA (random): count=' + str(self._random_nsaa_count) + ', lib=' + str(len(self._random_nsaa)))}</text>"
            )

        for i in range(1, display_len + 1):
            x = left + (i - 1) * cell_w
            y = seq_top
            fixed = i in fixed_positions
            nsaa = i in noncanonical_positions
            motif = i in motif_positions
            random_nsaa = i in nsaa_positions

            if nsaa:
                fill = "#fee2e2"
                stroke = "#ef4444" if random_nsaa else "#f87171"
                label = "X"
            elif fixed:
                fill = "#dbeafe" if motif else "#e2e8f0"
                stroke = "#2563eb" if motif else "#94a3b8"
                label = one_letter_from_abrv(fixed_positions[i])
            else:
                fill = "#f1f5f9"
                stroke = "#cbd5e1"
                label = "."
            parts.append(f"<rect x='{x}' y='{y}' width='{cell_w-1}' height='{cell_h}' rx='3' ry='3' fill='{fill}' stroke='{stroke}' />")
            parts.append(
                f"<text x='{x + (cell_w/2)}' y='{y + 13}' text-anchor='middle' font-family='monospace' font-size='{font_size}' fill='#0f172a'>{esc(label)}</text>"
            )
            if i % 5 == 0 or i == 1 or i == display_len:
                parts.append(
                    f"<text x='{x + (cell_w/2)}' y='{y + 35}' text-anchor='middle' font-family='monospace' font-size='10' fill='#475569'>{i}</text>"
                )
            if motif or random_nsaa:
                marker = "motif" if motif and not random_nsaa else "nsAA" if random_nsaa and not motif else "motif+nsAA"
                marker_color = "#1d4ed8" if motif and not random_nsaa else "#dc2626" if random_nsaa and not motif else "#7c3aed"
                parts.append(
                    f"<text x='{x + (cell_w/2)}' y='{y - 4}' text-anchor='middle' font-family='monospace' font-size='9' fill='{marker_color}'>{marker}</text>"
                )

        # Cyclization overlay (visual hint)
        if self._cyclize_mode in {"head_tail", "disulfide"}:
            x1 = left + 2
            x2 = left + (display_len - 1) * cell_w + (cell_w - 3)
            y = seq_top - 12
            color = "#eab308" if self._cyclize_mode == "disulfide" else "#0ea5e9"
            label = "S-S" if self._cyclize_mode == "disulfide" else "N-C"
            parts.append(f"<path d='M {x1} {y} C {x1} {y-18}, {x2} {y-18}, {x2} {y}' stroke='{color}' stroke-width='2' fill='none'/>")
            parts.append(f"<text x='{(x1+x2)/2}' y='{y-22}' text-anchor='middle' font-family='monospace' font-size='11' fill='{color}'>{label}</text>")

        # Legend
        legend_y = seq_top + 60
        parts.append(f"<text x='{left}' y='{legend_y}' font-family='monospace' font-size='11' fill='#334155'>Legend:</text>")
        parts.append(f"<text x='{left}' y='{legend_y + 16}' font-family='monospace' font-size='11' fill='#334155'>motif=blue, nsAA=red, editable='.'</text>")
        parts.append("</svg>")
        return "\n".join(parts)


class AntibodyPrompt(PromptProgram):
    def __init__(self, framework_path: Optional[str] = None, cdrs: Optional[Sequence[str]] = None, fr_len: int = 3):
        super().__init__()
        self.framework_path = resolve_project_path(framework_path) if framework_path else framework_path
        self.cdrs: List[str] = []
        self.fr_len = int(fr_len)
        self.length_ranges: Dict[str, Tuple[int, int]] = {}
        self._freeze_framework = True
        self._motifs: Dict[str, Dict[str, Any]] = {}
        self._noncanonicals: Dict[str, Dict[str, Any]] = {}
        self._resolved_antibody_constraints: Optional[Dict[str, Dict[str, Any]]] = None
        self._ensure_default_antibody_filters()
        if cdrs is not None:
            self.design_cdrs(cdrs)

    def _ensure_default_antibody_filters(self) -> None:
        self._ensure_default_filters([
            AbnormalConfidenceFilter(),
            PhysicalValidityFilter(),
            LTypeAAFilter(),
        ])

    def _ensure_cdr_registered(self, cdr_type: str) -> None:
        cdr_type = str(cdr_type)
        if cdr_type not in self.cdrs:
            self.cdrs.append(cdr_type)
            self._resolved_antibody_constraints = None

    def _reference_antibody_path(self) -> Optional[str]:
        if self.framework_path is not None and os.path.exists(self.framework_path):
            return self.framework_path
        context = self.get_context()
        if context is not None and context.pdb_paths:
            return context.pdb_paths[0]
        return None

    def _preferred_antibody_chain_ids(self) -> Optional[List[str]]:
        context = self.get_context()
        if context is not None and context.lig_chains:
            return list(context.lig_chains[0])
        return None

    def _native_cdr_length(self, cdr_type: str) -> Optional[int]:
        ref_path = self._reference_antibody_path()
        if ref_path is None or not os.path.exists(ref_path):
            return None
        try:
            cplx = _load_complex_for_ui(ref_path)
            preferred_chain_ids = self._preferred_antibody_chain_ids() or []
            chain_ids = preferred_chain_ids if preferred_chain_ids else list(_detect_antibody_chains(cplx).values())
            return len(_get_antibody_cdr_block_ids(cplx, chain_ids, cdr_type))
        except Exception:
            return None

    def design_cdr(self, cdr_type: str) -> "AntibodyPrompt":
        self._ensure_cdr_registered(cdr_type)
        return self

    def design_cdrs(self, cdr_types: Sequence[str]) -> "AntibodyPrompt":
        unique = list(dict.fromkeys(str(cdr_type) for cdr_type in cdr_types))
        self.cdrs = unique
        self._resolved_antibody_constraints = None
        return self

    def set_length(self, cdr_type: str, min_len: int, max_len: Optional[int] = None) -> "AntibodyPrompt":
        self._ensure_cdr_registered(cdr_type)
        if max_len is None:
            max_len = min_len
        if int(min_len) != int(max_len):
            raise ValueError("UI antibody prompt currently only supports a fixed antibody CDR length")
        self.length_ranges[cdr_type] = (int(min_len), int(max_len))
        self._resolved_antibody_constraints = None
        return self

    def freeze_framework(self) -> "AntibodyPrompt":
        self._freeze_framework = True
        return self

    def add_motif(self, cdr_type: str, seq: str, positions: Optional[Sequence[int]] = None) -> "AntibodyPrompt":
        self._ensure_cdr_registered(cdr_type)
        self._motifs[cdr_type] = {
            "seq": _normalize_motif_sequence(seq),
            "positions": _normalize_relative_positions(positions, "motif positions"),
        }
        self._resolved_antibody_constraints = None
        return self

    def add_noncanonical(self, cdr_type: str, smiles: str, positions: Optional[Sequence[int]] = None, count: Optional[int] = 1) -> "AntibodyPrompt":
        self._ensure_cdr_registered(cdr_type)
        self._noncanonicals[cdr_type] = {
            "smiles": str(smiles),
            "positions": _normalize_relative_positions(positions, "nsAA positions"),
            "count": int(count) if count is not None else 1,
        }
        self._resolved_antibody_constraints = None
        return self

    def set_region(self, cdr_type: str, mode: str = "generate") -> "AntibodyPrompt":
        if mode == "generate":
            return self.design_cdr(cdr_type)
        raise ValueError("The demo currently only supports mode='generate' for antibodies")

    def _resolve_antibody_constraints(self) -> Dict[str, Dict[str, Any]]:
        if self._resolved_antibody_constraints is not None:
            return self._resolved_antibody_constraints
        resolved: Dict[str, Dict[str, Any]] = {}
        for cdr_type in self.cdrs:
            native_cdr_len = self._native_cdr_length(cdr_type)
            cdr_len = native_cdr_len
            if cdr_type in self.length_ranges:
                cdr_len = int(self.length_ranges[cdr_type][0])
            motif_spec = self._motifs.get(cdr_type)
            nsaa_spec = self._noncanonicals.get(cdr_type)
            motif_positions: List[int] = []
            nsaa_positions: List[int] = []
            if motif_spec is not None and cdr_len is not None:
                seq = motif_spec["seq"]
                if motif_spec["positions"] is not None:
                    if len(seq) != len(motif_spec["positions"]):
                        raise ValueError("motif positions must have the same length as motif sequence")
                    if max(motif_spec["positions"]) > cdr_len:
                        raise ValueError(f"motif positions exceed {cdr_type} length {cdr_len}")
                    motif_positions = list(motif_spec["positions"])
                else:
                    if len(seq) <= cdr_len:
                        start = int(np.random.randint(1, cdr_len - len(seq) + 2))
                        motif_positions = list(range(start, start + len(seq)))
                    else:
                        raise ValueError(f"motif length {len(seq)} exceeds {cdr_type} length {cdr_len}")
            elif motif_spec is not None:
                raise ValueError(f"Could not determine length for {cdr_type}; cannot resolve motif positions")
            if nsaa_spec is not None and cdr_len is not None:
                if nsaa_spec["positions"] is not None:
                    if max(nsaa_spec["positions"]) > cdr_len:
                        raise ValueError(f"nsAA positions exceed {cdr_type} length {cdr_len}")
                    nsaa_positions = list(nsaa_spec["positions"])
                else:
                    available = [i for i in range(1, cdr_len + 1) if i not in motif_positions]
                    count = min(int(nsaa_spec["count"]), len(available))
                    if count > 0:
                        nsaa_positions = sorted(np.random.choice(available, count, replace=False).tolist())
            overlap = sorted(set(motif_positions).intersection(nsaa_positions))
            if overlap:
                raise ValueError(f"motif positions and nsAA positions overlap on {cdr_type}: {overlap}")
            resolved[cdr_type] = {
                "motif": None if motif_spec is None else dict(motif_spec),
                "motif_positions": motif_positions,
                "nsaa": None if nsaa_spec is None else dict(nsaa_spec),
                "nsaa_positions": nsaa_positions,
                "native_length": native_cdr_len,
                "target_length": cdr_len,
                "backend_enforced": motif_spec is not None or nsaa_spec is not None,
            }
        self._resolved_antibody_constraints = resolved
        return resolved

    def inspect(self) -> str:
        resolved = self._resolve_antibody_constraints()
        lines = [
            "Prompt Type: antibody",
            f"Framework Path: {self.framework_path}",
            f"CDRs: {_stringify_iter(self.cdrs) if self.cdrs else 'none'}",
            f"Freeze Framework: {self._freeze_framework}",
            "CDR Lengths:",
        ]
        if self.length_ranges:
            for cdr_type, (min_len, max_len) in self.length_ranges.items():
                if min_len == max_len:
                    lines.append(f"  - {cdr_type}: {min_len}")
                else:
                    lines.append(f"  - {cdr_type}: [{min_len}, {max_len}]")
        else:
            for cdr_type in self.cdrs:
                native_len = self._native_cdr_length(cdr_type)
                if native_len is None:
                    lines.append(f"  - {cdr_type}: native")
                else:
                    lines.append(f"  - {cdr_type}: native ({native_len})")
            if not self.cdrs:
                lines.append("  - none")
        lines.append("Motifs:")
        if self._motifs:
            for cdr_type, item in self._motifs.items():
                lines.append(f"  - {cdr_type}: {item['seq']} @ {resolved.get(cdr_type, {}).get('motif_positions', item.get('positions'))}")
        else:
            lines.append("  - none")
        lines.append("Non-Canonical Residues:")
        if self._noncanonicals:
            for cdr_type, item in self._noncanonicals.items():
                lines.append(f"  - {cdr_type}: count={item['count']} @ {resolved.get(cdr_type, {}).get('nsaa_positions', item.get('positions'))}")
        else:
            lines.append("  - none")
        lines.append("Backend Generation:")
        if len(self.cdrs) > 1:
            lines.append("  - multiple CDRs are generated sequentially")
        if self._motifs or self._noncanonicals:
            lines.append("  - constraints are applied in-memory before antibody loader/tokenization")
            if self.length_ranges:
                lines.append("  - fixed CDR length still uses the canonical temporary-cif length rewrite")
            lines.append("  - constrained motif/nsAA positions are fixed through condition_config.mask_2d")
        else:
            lines.append("  - ordinary antibody generation")
        if not self._noncanonicals:
            lines.append("  - no nsAA constraints")
        lines.append("Compiled Template:")
        lines.append("  - AntibodyMultipleCDR" if len(self.cdrs) > 1 else "  - Antibody")
        return "\n".join(lines)

    def compile(self) -> CompiledPrompt:
        if not self.cdrs:
            raise ValueError("AntibodyPrompt requires at least one CDR")
        resolved = self._resolve_antibody_constraints()
        if len(self.cdrs) > 1:
            template = AntibodyMultipleCDR(
                cdr_types=list(self.cdrs),
                length_ranges={cdr_type: tuple(length_range) for cdr_type, length_range in self.length_ranges.items()},
                fr_len=self.fr_len,
            )
            template.ui_resolved_constraints = deepcopy(resolved)
            template.ui_guidance = self._guidance
            template.ui_use_constrained_generation = True
            template_name = "AntibodyMultipleCDR(constrained)"
            enforced = sorted(
                {
                    name
                    for item in resolved.values()
                    for name, enabled in [
                        ("motif", item.get("motif") is not None),
                        ("nsaa", item.get("nsaa") is not None),
                    ]
                    if enabled
                }
            )
            return CompiledPrompt(
                template=template,
                filters=list(self._filters),
                sample_opt={},
                metadata={
                    "modality": "antibody",
                    "guidance": self._guidance,
                    "cdrs": list(self.cdrs),
                    "length_ranges": dict(self.length_ranges),
                    "preprocess_length_ranges": {},
                    "motifs": deepcopy(self._motifs),
                    "noncanonicals": deepcopy(self._noncanonicals),
                    "resolved_constraints": deepcopy(resolved),
                    "backend_enforced_constraints": enforced,
                    "template_name": template_name,
                },
            )

        cdr_type = self.cdrs[0]
        resolved_constraint = resolved.get(cdr_type, {})
        has_motif = resolved_constraint.get("motif") is not None
        has_nsaa = resolved_constraint.get("nsaa") is not None
        has_constraints = has_motif or has_nsaa
        # For absolute correctness: always compile to the existing templates without
        # any UI-side antibody-specific generation wrappers.
        #
        # - If a fixed length is requested, route through AntibodyMultipleCDR with a
        #   single cdr_type so the length change happens in the canonical pipeline.
        # - Otherwise, use the canonical single-CDR Antibody template directly.
        if has_constraints:
            template = ConstrainedAntibodyTemplate(
                cdr_type=cdr_type,
                fr_len=self.fr_len,
                motif_seq=resolved_constraint.get("motif", {}).get("seq") if resolved_constraint.get("motif") is not None else None,
                motif_positions=resolved_constraint.get("motif_positions", []),
                nsaa_smiles=resolved_constraint.get("nsaa", {}).get("smiles") if resolved_constraint.get("nsaa") is not None else None,
                nsaa_positions=resolved_constraint.get("nsaa_positions", []),
                guidance=self._guidance,
            )
            template_name = "ConstrainedAntibodyTemplate"
        elif cdr_type in self.length_ranges:
            l, r = self.length_ranges[cdr_type]
            if int(l) != int(r):
                raise ValueError("Single-CDR UI currently only supports a fixed antibody CDR length")
            template = AntibodyMultipleCDR(cdr_types=[cdr_type], length_ranges={cdr_type: (int(l), int(r))}, fr_len=self.fr_len)
            template_name = "AntibodyMultipleCDR(single)"
        else:
            template = Antibody(cdr_type=cdr_type, fr_len=self.fr_len)
            template_name = "Antibody"
        return CompiledPrompt(
            template=template,
            filters=list(self._filters),
            sample_opt={},
            metadata={
                "modality": "antibody",
                "guidance": self._guidance,
                "cdrs": [cdr_type],
                "length_ranges": dict(self.length_ranges),
                "preprocess_length_ranges": {cdr_type: self.length_ranges[cdr_type]} if has_constraints and cdr_type in self.length_ranges else {},
                "motifs": deepcopy(self._motifs),
                "noncanonicals": deepcopy(self._noncanonicals),
                "resolved_constraints": deepcopy(resolved),
                "backend_enforced_constraints": [name for name, enabled in [("motif", has_motif), ("nsaa", has_nsaa)] if enabled],
                "template_name": template_name,
            },
        )

    def to_visual_payload(self) -> Dict[str, Any]:
        resolved = self._resolve_antibody_constraints()
        visual = None
        preferred_chain_ids = None
        if self.get_context() is not None and self.get_context().lig_chains:
            preferred_chain_ids = list(self.get_context().lig_chains[0])
        if self.framework_path is not None and os.path.exists(self.framework_path):
            try:
                visual = _extract_antibody_visual_info(self.framework_path, preferred_chain_ids=preferred_chain_ids)
            except Exception:
                visual = None

        svg = None
        if visual is not None:
            try:
                svg = self.to_svg(visual)
            except Exception:
                svg = None

        nsaa_previews: List[Dict[str, Any]] = []
        for cdr_type, item in self._noncanonicals.items():
            try:
                nsaa_previews.append(
                    {
                        "cdr_type": cdr_type,
                        "smiles": item["smiles"],
                        "positions": resolved.get(cdr_type, {}).get("nsaa_positions", []),
                        "svg": nsaa_smiles_to_svg(item["smiles"]),
                    }
                )
            except Exception:
                nsaa_previews.append(
                    {
                        "cdr_type": cdr_type,
                        "smiles": item["smiles"],
                        "positions": resolved.get(cdr_type, {}).get("nsaa_positions", []),
                        "svg": None,
                    }
                )

        return {
            "kind": "antibody",
            "text": self.inspect(),
            "svg": svg,
            "cdr_control_pages": self._build_cdr_control_pages(visual, resolved),
            "resolved_constraints": resolved,
            "nsaa_previews": nsaa_previews,
            "framework_visual": visual,
            "enforcement_note": "motif/nsAA are applied in-memory before antibody loader/tokenization and fixed through condition_config.mask_2d",
        }

    def _build_cdr_control_pages(self, visual: Optional[Dict[str, Any]], resolved: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
        if visual is None:
            return []
        pages: List[Dict[str, str]] = []
        for cdr_type in self.cdrs:
            try:
                pages.append({"cdr_type": cdr_type, "svg": self._cdr_control_svg(cdr_type, visual, resolved)})
            except Exception:
                continue
        return pages

    def _cdr_control_svg(
        self,
        cdr_type: str,
        visual: Dict[str, Any],
        resolved: Dict[str, Dict[str, Any]],
        width: int = 720,
        height: int = 180,
    ) -> str:
        chain_type = cdr_type[0]
        cdr_digit = cdr_type[-1]
        info = visual["chains"].get(chain_type, None)
        if info is None:
            raise ValueError(f"Missing chain info for {cdr_type}")
        mark = info["mark"]
        seq = info["sequence"]
        indices = [i for i, digit in enumerate(mark) if digit == cdr_digit]
        if not indices:
            raise ValueError(f"No indices found for {cdr_type}")

        item = resolved.get(cdr_type, {})
        motif = item.get("motif")
        motif_positions = set(item.get("motif_positions", []))
        nsaa_positions = set(item.get("nsaa_positions", []))
        native_cdr_len = len(indices)
        cdr_len = int(item.get("target_length") or native_cdr_len)
        cell_w = max(min((width - 40) / max(cdr_len, 1), 42), 18)
        left = 20
        top = 72

        def esc(text: str) -> str:
            return (
                str(text)
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )

        parts = [
            "<?xml version='1.0' encoding='utf-8'?>",
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}px' height='{height}px' viewBox='0 0 {width} {height}'>",
            f"<rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff' />",
            f"<text x='{left}' y='18' font-family='monospace' font-size='13' fill='#0f172a'>{esc(cdr_type)} Control Prompt</text>",
            f"<text x='{left}' y='34' font-family='monospace' font-size='11' fill='#475569'>{esc('length=' + str(cdr_len) + '   native=' + str(native_cdr_len))}</text>",
        ]
        if motif:
            motif_pos_text = item.get("motif_positions", [])
            parts.append(
                f"<text x='{left}' y='50' font-family='monospace' font-size='11' fill='#1d4ed8'>{esc('motif: ' + motif['seq'] + ' @ ' + str(motif_pos_text))}</text>"
            )
        elif item.get("nsaa"):
            parts.append(
                f"<text x='{left}' y='50' font-family='monospace' font-size='11' fill='#dc2626'>{esc('nsAA positions: ' + str(item.get('nsaa_positions', [])))}</text>"
            )
        parts.append(
            f"<text x='{left}' y='{height - 14}' font-family='monospace' font-size='11' fill='#475569'>Legend: editable='.'  motif=blue  nsAA=red</text>"
        )

        motif_seq = motif["seq"] if motif else ""
        motif_pos_to_aa: Dict[int, str] = {}
        if motif and item.get("motif_positions"):
            for aa, pos in zip(motif_seq, item["motif_positions"]):
                motif_pos_to_aa[int(pos)] = aa

        for rel_idx in range(1, cdr_len + 1):
            x = left + (rel_idx - 1) * cell_w
            if rel_idx <= native_cdr_len:
                seq_idx = indices[rel_idx - 1]
                aa = seq[seq_idx] if seq_idx < len(seq) else "X"
            else:
                aa = "."
            fill = "#f1f5f9"
            stroke = "#cbd5e1"
            label = "."
            if rel_idx in motif_positions:
                fill = "#dbeafe"
                stroke = "#2563eb"
                label = motif_pos_to_aa.get(rel_idx, aa)
            elif rel_idx in nsaa_positions:
                fill = "#fee2e2"
                stroke = "#dc2626"
                label = "X"
            parts.append(f"<rect x='{x:.2f}' y='{top}' width='{cell_w - 2:.2f}' height='26' rx='4' ry='4' fill='{fill}' stroke='{stroke}' />")
            parts.append(f"<text x='{x + (cell_w - 2)/2:.2f}' y='{top + 17}' text-anchor='middle' font-family='monospace' font-size='12' fill='#0f172a'>{esc(label)}</text>")
            parts.append(f"<text x='{x + (cell_w - 2)/2:.2f}' y='{top + 42}' text-anchor='middle' font-family='monospace' font-size='10' fill='#475569'>{rel_idx}</text>")

        parts.append(f"<text x='{left}' y='{top + 70}' font-family='monospace' font-size='11' fill='#475569'>Blue = motif-fixed residue, red = nsAA placeholder.</text>")
        parts.append("</svg>")
        return "\n".join(parts)

    def to_svg(self, visual: Dict[str, Any], width: int = 720, height: int = 240) -> str:
        def esc(text: str) -> str:
            return (
                text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
            )

        rows: List[Tuple[str, Dict[str, Any]]] = []
        for chain_type in ["H", "L"]:
            if chain_type in visual["chains"]:
                rows.append((chain_type, visual["chains"][chain_type]))
        if not rows:
            raise ValueError("No antibody chains available for visualization")

        parts = [
            "<?xml version='1.0' encoding='utf-8'?>",
            f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}px' height='{height}px' viewBox='0 0 {width} {height}'>",
            f"<rect x='0' y='0' width='{width}' height='{height}' fill='#ffffff' />",
            "<text x='12' y='18' font-family='monospace' font-size='13' fill='#0f172a'>Antibody Prompt</text>",
        ]

        row_y = 44
        bar_x = 120
        bar_w = width - 150
        row_h = 46
        for chain_type, info in rows:
            mark = info["mark"]
            seq = info["sequence"]
            n = max(len(mark), 1)
            cell_w = max(bar_w / n, 4)
            parts.append(f"<text x='12' y='{row_y + 14}' font-family='monospace' font-size='12' fill='#334155'>{chain_type}-chain ({info['chain_id']})</text>")
            for i, digit in enumerate(mark):
                x = bar_x + i * cell_w
                fill = "#e2e8f0"
                if digit in "123":
                    cdr_type = f"{chain_type}CDR{digit}"
                    fill = "#dbeafe" if cdr_type in self.cdrs else "#fef3c7"
                parts.append(f"<rect x='{x:.2f}' y='{row_y}' width='{max(cell_w-1,1):.2f}' height='18' fill='{fill}' stroke='#cbd5e1' />")
                if cell_w >= 10:
                    aa = esc(seq[i]) if i < len(seq) else "X"
                    parts.append(f"<text x='{x + cell_w/2:.2f}' y='{row_y + 13}' text-anchor='middle' font-family='monospace' font-size='10' fill='#0f172a'>{aa}</text>")

            for cdr_idx in ["1", "2", "3"]:
                cdr_type = f"{chain_type}CDR{cdr_idx}"
                if cdr_type not in self.cdrs:
                    continue
                indices = [i for i, digit in enumerate(mark) if digit == cdr_idx]
                if not indices:
                    continue
                start_i, end_i = indices[0], indices[-1]
                x1 = bar_x + start_i * cell_w
                x2 = bar_x + (end_i + 1) * cell_w
                parts.append(f"<rect x='{x1:.2f}' y='{row_y - 4}' width='{x2 - x1:.2f}' height='26' fill='none' stroke='#2563eb' stroke-width='1.5' rx='3' />")
                parts.append(f"<text x='{x1:.2f}' y='{row_y - 8}' font-family='monospace' font-size='10' fill='#2563eb'>{cdr_type}</text>")

            row_y += row_h

        parts.append("<text x='12' y='214' font-family='monospace' font-size='11' fill='#475569'>Designed CDRs are boxed in blue. Use the CDR prompt pager below for detailed motif/nsAA controls.</text>")
        parts.append("</svg>")
        return "\n".join(parts)
