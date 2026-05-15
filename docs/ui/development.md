# UI Development Guide

## Overview

`api/ui` is the interactive layer for prompt-driven controllable generation. It translates user-authored prompt programs into backend templates and generation contexts that the existing generation pipeline can execute.

User-facing entry points:

- REPL: `api.ui.repl`
- Web UI: `api.ui.web`

Design goals:

- reuse the existing model, dataset, template, filter, and file-output pipeline as much as possible
- keep UI-specific behavior inside `api/ui`
- preserve inspectable prompt state before execution
- support lightweight browser-based demos without adding a heavy service stack

## Directory Layout

- `api/ui/core.py`
  - prompt classes
  - prompt compilation
  - runtime bridge logic
  - UI-side filtering and result materialization
  - SVG helpers
- `api/ui/repl.py`
  - restricted Python REPL
  - safe prompt execution and session state
- `api/ui/web.py`
  - HTTP server
  - demo code
  - task submission, task switching, and result payload building
- `api/ui/web_static/`
  - frontend page, styling, and browser logic
- `api/ui/examples/run_with_env.sh`
  - environment-aware launcher

## Core Data Objects

### `GenerationContext`

Normalizes UI-friendly inputs into the format expected by `PDBDataset`.

Important fields:

- `pdb_paths`
- `tgt_chains`
- `lig_chains`
- `hotspots`
- `checkpoint`
- `batch_size`
- `n_samples`
- `filter_batch_quota`
- `gpu`
- `sample_opt`

Notes:

- the UI demo currently supports one target complex at a time
- relative paths are resolved against the project root in the UI path

### `CompiledPrompt`

Represents the result of prompt compilation.

Fields:

- `template`
- `filters`
- `sample_opt`
- `metadata`

### `GenerationResult`

Returned by `PromptProgram.run_generation()`.

Current summary fields include:

- output directory
- `results.jsonl`
- optional `filtered_results.jsonl`
- generated count
- passed count
- attempted batch count
- whether fallback was used

## Prompt Classes

### `PromptProgram`

Shared base class for all prompt types.

Common methods:

- `add_filter()`
- `set_guidance()`
- `set_context()`
- `run_generation()`

Subclass responsibilities:

- `inspect()`
- `compile()`
- `to_visual_payload()`

### `MoleculePrompt`

Responsibilities:

- fragment management
- explicit inter-fragment bonds
- pinning, placement, and growth budget
- compilation into a UI-side programmed molecule template

Current default growth budget:

- `_growth_min = 3`
- `_growth_max = 6`

Current default filter order:

1. `AbnormalConfidenceFilter`
2. `ChiralCentersFilter(center_max=8, ring_mode=True)`
3. `RotatableBondsFilter(max_num_rot_bonds=7)`
4. `PhysicalValidityFilter`
5. `MolBeautyFilter(th=1)`

### `PeptidePrompt`

Responsibilities:

- peptide length
- motifs
- non-canonical amino acids
- cyclization mode
- compilation into a programmed peptide template

Default filters:

- `AbnormalConfidenceFilter`
- `PhysicalValidityFilter`
- `LTypeAAFilter`

### `AntibodyPrompt`

Responsibilities:

- framework structure
- selected CDRs
- per-CDR lengths
- motif constraints
- compilation into antibody runtime templates

Important current behavior:

- framework freezing defaults to enabled
- the demo currently designs `HCDR3` and `LCDR3`
- motif placement uses relative positions inside each CDR

Default filters:

- `AbnormalConfidenceFilter`
- `PhysicalValidityFilter`
- `LTypeAAFilter`

## Runtime Flow

High-level `run_generation()` flow:

1. build or merge `GenerationContext`
2. compile the prompt into `CompiledPrompt`
3. load model and confidence model
4. build `PDBDataset`
5. select the proper generation function
6. execute generation
7. apply UI-side filters
8. materialize raw and selected outputs
9. return `GenerationResult`

## Task Execution Model

The web UI submits tasks through Ray.

Current model:

- CPU task: `num_gpus=0`
- GPU task: currently configured in `web.py` with fractional GPU allocation

The web task wrapper:

- evaluates prompt code
- snapshots prompt state
- runs generation
- writes `run.log`
- returns a serialized result payload

## UI Filtering Model

### Batch Quota

When filters are enabled, the UI does not stop after a single generation attempt.

It uses:

- `filter_batch_quota`
- one attempt directory per batch under `_filter_attempts/`

The default quota is currently `6`.

### Result Selection

Per attempt:

- raw results are generated
- filters run in UI space
- filter outputs are attached to each item

Across attempts:

- if any item passes, the passed set is used
- otherwise the UI selects the item with the lowest `confidence` value

### Short-Circuit Filtering

For each item, filters now stop at the first non-`PASSED` result. This reduces unnecessary work for expensive trailing filters such as `MolBeautyFilter`.

### Batch Logging

`run.log` records:

- per-item streaming progress while a batch is still generating
- batch start
- batch finish
- generated count
- passed count
- per-filter `passed / failed / error` summary
- fallback selection message

## Streaming Filter Path

The current UI implementation supports batch-internal overlap between generation and filtering.

How it works:

- the generator writes `results.jsonl` incrementally
- a UI-side watcher polls for newly appended rows
- each new item is filtered immediately
- enriched results are written back once the batch completes

This is implemented only in `api/ui`, without changing the shared generation helpers.

Current limitation:

- filtering overlaps with generation inside a batch
- but the running generation batch is not terminated early when one sample passes

## Result Files

Primary files:

- `results.jsonl`
- `filtered_results.jsonl`
- `<id>/<n>.cif`
- `<id>/<n>.pdb`
- `<id>/<n>.sdf`
- `<id>/<n>_confidence.json`
- `run.log`

The web payload builder prefers `filtered_results.jsonl` when it contains selected records.

## Web Frontend

### `app.js`

Responsibilities:

- poll UI state
- render prompt visualization
- append the current task log into the `Program` output box
- render result SVG and paging
- render top-record summary
- switch tasks and demos

### Result Rendering

The top-record view currently shows:

- PDE (`confidence`)
- likelihood
- smiles
- generated sequence
- target chains
- ligand chains
- filter outputs


Current layout note:

- there is no standalone run-log panel anymore
- the `Program` panel output area is used for both execution status messages and the current task log
- the `Prompt State` area is configured to expand with the center panel height

## Extension Points

Recommended process for a new prompt type:

1. add a new `PromptProgram` subclass in `core.py`
2. implement `inspect()`, `compile()`, and `to_visual_payload()`
3. export the type in `api/ui/__init__.py`
4. expose it in the REPL globals
5. add a demo snippet in `web.py`
6. add browser rendering if needed

## Maintenance Notes

- keep UI-only generation bridges in `api/ui` unless shared backend changes are clearly justified
- update `inspect()` and `to_visual_payload()` together when prompt semantics change
- verify both raw and filtered output behavior after changing the filter pipeline
- for antibody changes, verify both single-CDR and multi-CDR paths
- for filter changes, inspect both `run.log` and `filtered_results.jsonl`
