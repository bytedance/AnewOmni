#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Convert template CIFs referenced by a boltz2 input YAML into a boltz-friendly mmCIF.

We do this as a separate script so the worker code can simply `subprocess` call it
under the boltz env python, avoiding inline heredoc Python in shell commands.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import Any, Dict, List

import yaml


def _looks_converted(path: str) -> bool:
    # Heuristic to skip re-writing already enriched mmCIFs.
    try:
        with open(path, "r") as f:
            head = f.read(200_000)
        return ("_entity_poly_seq" in head) or ("_entity_poly.pdbx_seq_one_letter_code" in head)
    except Exception:
        return False


def _convert_in_place(cif_path: str, conv_script_path: str) -> None:
    cif_path = os.path.abspath(cif_path)
    if not os.path.exists(cif_path):
        return
    if _looks_converted(cif_path):
        return

    # Lock per template file to avoid concurrent conversion races across Ray workers.
    lock_path = cif_path + ".boltz_convert.lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    # fcntl is available on linux (our runtime).
    import fcntl  # noqa: PLC0415

    with open(lock_path, "w") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        # Re-check under lock.
        if _looks_converted(cif_path):
            return

        tmp_dir = os.path.dirname(cif_path)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tmp.cif", prefix=os.path.basename(cif_path) + ".", dir=tmp_dir, delete=False
        ) as tf:
            tmp_path = tf.name

        try:
            # Import converter from the colocated script (gemmi dependency must exist).
            # We avoid spawning another python process here to keep it fast.
            conv_dir = os.path.dirname(os.path.abspath(conv_script_path))
            if conv_dir not in sys.path:
                sys.path.insert(0, conv_dir)
            from gemmi_convert_template import convert_one  # type: ignore

            convert_one(cif_path, tmp_path)
            os.replace(tmp_path, cif_path)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass


def convert_yaml_templates(yaml_path: str) -> List[str]:
    data: Dict[str, Any] = yaml.safe_load(open(yaml_path, "r"))
    tpls = data.get("templates") or []
    if not isinstance(tpls, list) or not tpls:
        return []

    conv_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemmi_convert_template.py")
    changed: List[str] = []
    for t in tpls:
        if not isinstance(t, dict):
            continue
        cif = t.get("cif")
        pdb = t.get("pdb")
        if cif:
            before = os.path.getmtime(cif) if os.path.exists(cif) else None
            _convert_in_place(cif, conv_script_path)
            after = os.path.getmtime(cif) if os.path.exists(cif) else None
            if before is None or (after is not None and after != before):
                changed.append(os.path.abspath(cif))
        elif pdb:
            # Not supported without rewriting YAML (boltz accepts pdb, but the
            # metadata fix requires mmCIF). We keep this explicit.
            continue
    return changed


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <input.yaml>", file=sys.stderr)
        return 2
    yaml_path = os.path.abspath(argv[1])
    if not os.path.exists(yaml_path):
        print(f"YAML not found: {yaml_path}", file=sys.stderr)
        return 2
    convert_yaml_templates(yaml_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
