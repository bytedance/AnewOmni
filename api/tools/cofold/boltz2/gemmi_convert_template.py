#!/usr/bin/python
# -*- coding:utf-8 -*-

"""
Convert a structure file into a Boltz-friendly mmCIF using Gemmi.

This is intended for template CIF/PDBs that may miss entity/polymer metadata
required by boltz's mmCIF parser.

Based on a Boltz community snippet (Slack).
"""

import re
import sys
from pathlib import Path

import gemmi


def _sanitize_chain_names(st: gemmi.Structure):
    for model in st:
        for ch in model:
            name = (ch.name or "").strip()
            if not name or name in {".", "?"}:
                ch.name = f"X{model.index}-{ch.index}"
            else:
                ch.name = re.sub(r"[^A-Za-z0-9_-]", "_", name)


def _populate_entity_sequences_from_model(st: gemmi.Structure):
    """
    For each polymer entity, build a 3-letter-code sequence from the longest
    polymer subchain in model 0 and store it in Entity.full_sequence.
    Gemmi will then write _entity_poly/_entity_poly_seq to mmCIF.

    This implementation is defensive about Gemmi versions where Chain.subchains()
    may yield either Subchain objects (with .get_polymer()) or ResidueSpan directly.
    """
    if len(st) == 0:
        return
    model0: gemmi.Model = st[0]

    # Build an index: subchain_id -> polymer ResidueSpan
    subchain_by_id: dict[str, gemmi.ResidueSpan] = {}
    for ch in model0:
        for sc in ch.subchains():
            # Normalize: obtain a ResidueSpan for the polymer portion
            if hasattr(sc, "get_polymer"):
                poly = sc.get_polymer()
                sid = sc.subchain_id() if hasattr(sc, "subchain_id") else None
            else:
                poly = sc
                sid = sc.subchain_id() if hasattr(sc, "subchain_id") else None

            if sid is None:
                chain_name = getattr(ch, "name", "?")
                sid = f"{chain_name}:{poly.first().seqid}" if len(poly) else f"{chain_name}:empty"

            if len(poly) == 0:
                continue
            subchain_by_id[str(sid)] = poly

    for ent in st.entities:
        if ent.entity_type != gemmi.EntityType.Polymer:
            continue

        candidate_seqs: list[list[str]] = []
        for sid in ent.subchains:
            poly = subchain_by_id.get(str(sid))
            if poly is None or len(poly) == 0:
                continue
            seq = [res.name for res in poly]
            candidate_seqs.append(seq)

        if not candidate_seqs:
            continue

        best = max(candidate_seqs, key=len)
        ent.full_sequence = best


def _chimera_like_setup(st: gemmi.Structure):
    """
    Make mmCIF-compatible like Chimera exports:
      - non-empty chain/subchain ids
      - entities rebuilt and deduplicated
      - label sequence numbers present
      - entity polymer sequences populated so _entity_poly_seq is written
    """
    _sanitize_chain_names(st)
    st.assign_subchains(force=True, fail_if_unknown=False)
    st.setup_entities()
    st.ensure_entities()
    st.add_entity_types(overwrite=True)
    st.add_entity_ids(overwrite=True)
    st.deduplicate_entities()

    _populate_entity_sequences_from_model(st)
    st.assign_label_seq_id(force=True)


def convert_one(inp: str | Path, out: str | Path):
    inp = Path(inp)
    out = Path(out)
    st = gemmi.read_structure(str(inp))
    _chimera_like_setup(st)
    doc = st.make_mmcif_document()
    out.write_text(doc.as_string())


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} <in.(cif|pdb)> <out.cif>", file=sys.stderr)
        return 2
    convert_one(argv[1], argv[2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

