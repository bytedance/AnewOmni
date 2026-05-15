# Prompt Language Design

## Goal

This document describes the original design direction for a lightweight programming-language-style interface for controllable generation.

The purpose is not to replace the existing backend templates, but to provide a unified, composable, and interactive abstraction on top of them so users can control generation by writing prompt programs.

Target modalities:

- small molecules
- peptides
- antibodies

Core principles:

- users interact with a `PromptProgram`, not directly with backend template internals
- the model consumes a unified intermediate representation
- the surface API should feel like Python so it works in both REPL and browser demos

## Architecture

The design uses three layers.

### User Layer

Users construct a prompt through a Python-like API:

```python
graph = MoleculePrompt()
graph.add_fragment("c1ccccc1")
graph.add_filter(MolBeautyFilter(th=1))
graph.run_generation(save_dir="./outputs/mol_case")
```

### Compiler Layer

Prompt objects are compiled into an intermediate representation containing:

- nodes
- edges
- filters
- generation metadata

### Execution Layer

The UI compiler adapts the IR to the existing backend stack:

- repository templates
- masks
- filters
- `sample_opt`
- generation context

## Object Model

### Shared Base Class

```python
class PromptProgram:
    def summary(self) -> str: ...
    def inspect(self) -> str: ...
    def compile(self) -> CompiledPrompt: ...
    def run_generation(self, save_dir: str, **kwargs) -> GenerationResult: ...
```

### Molecules

Design goals:

- fragment composition
- explicit bond control
- scaffold pinning
- growth budget
- chemistry filters

Example:

```python
graph = MoleculePrompt()
graph.add_fragment("c1ccccc1", name="ring")
graph.add_fragment("C", name="methyl")
graph.add_bond("ring:c0", "methyl:c0")
graph.allow_growth(3, 6)
```

### Peptides

Design goals:

- motif control
- length control
- nsAA insertion
- cyclization

Example:

```python
pep = PeptidePrompt(length=12)
pep.add_motif("RGD")
pep.add_noncanonical("SMILES_STRING", count=1)
pep.cyclize(mode="disulfide")
```

### Antibodies

Design goals:

- CDR selection
- CDR length control
- framework preservation
- motif control

Example:

```python
ab = AntibodyPrompt(
    framework_path="demo/data/8u4r_chothia.pdb",
    cdrs=["HCDR3", "LCDR3"],
)
ab.set_length("HCDR3", 12)
ab.set_length("LCDR3", 9)
ab.add_motif("HCDR3", "YYG", positions=[2, 3, 4])
```

## Referencing Rules

For molecule prompts, atom labels are generated in a stable way:

- `ring:c0`
- `ring:c1`
- `methyl:c0`

This makes `add_bond(...)` and prompt inspection deterministic.

## Intermediate Representation

Recommended fields:

- modality
- nodes
- edges
- filters
- guidance
- generation budget
- metadata

This IR is the boundary between the language surface and execution.

## Why This Matters

The repository already contains multiple hidden DSL-like entry points:

- fragment growth templates
- peptide motif templates
- nsAA peptide templates
- multi-CDR antibody design templates

The UI prompt language makes them one coherent system by providing:

- one user-facing object model
- one inspectable prompt state
- one compilation flow
- one runtime bridge

## Status

The current implementation already realizes a large part of this design in `api/ui/core.py`, `api/ui/repl.py`, and `api/ui/web.py`, even though some long-term ideas in this design note remain broader than the current shipped behavior.
