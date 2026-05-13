# Antibody Pipeline

## Quick Links

- [Overview](#overview)
- [Set Up Cofolding Models](#set-up-cofolding-models)
- [Entry Point](#entry-point)
  - [Supported CLI Arguments](#supported-cli-arguments)
  - [Example Commands](#example-commands)
- [YAML Configuration](#yaml-configuration)
- [Full Example](#full-example)

## Overview

This document describes the de novo antibody/nanobody pipeline implemented in `api/tools/ab_design.py`, including:

1. cofolding model setup and configuration
2. the pipeline entry point and supported CLI arguments
3. the supported YAML configuration schema

The antibody pipeline is different from the general generation entry point `api.generate_with_rank`.
It performs iterative generation and cofold-based rectification:

1. initialize antibody or nanobody frameworks
2. run cofolding against the target
3. rank candidates
4. launch antibody CDR design jobs from top-ranked docking poses
5. re-run cofolding for newly generated binders
6. repeat until the iteration limit is reached

The main entry point is (default using protenix as the backend):

```bash
python -m api.tools.ab_design \
  --config /path/to/pipeline.yaml \
  --save_dir /path/to/output
```

## Set Up Cofolding Models

The pipeline currently supports:

- `protenix`
- `boltz2`
- `alphafold3`

The default backend is `protenix` and can be changed with the `--cofold_model` argument. You can choose the most suitable one for you to install.

### Protenix

`Protenix` is the default `--cofold_model`.

Recommended setup:

```bash
bash ./api/tools/cofold/protenix/setup.sh
```

Default layout after setup:

```bash
./api/tools/cofold/protenix/env
./api/tools/cofold/protenix/params
```

Backend notes:

- `--cofold_env` should point to the path of its conda environment.
- `--cofold_param` should point to the Protenix checkpoint directory.
- If `--cofold_env` and `--cofold_param` are omitted, `ab_design.py` automatically uses `./api/tools/cofold/protenix/env` and `./api/tools/cofold/protenix/params`.
- This repository currently uses `protenix_mini_esm_v0.5.0` in the backend implementation because custom templates are not officially supported yet by Protenix. Protenix-mini with ESM embedding performs better without MSAs and templates.

### Boltz2

Recommended setup:

```bash
bash ./api/tools/cofold/boltz2/setup.sh
```

Default layout after setup:

```bash
./api/tools/cofold/boltz2/env
./api/tools/cofold/boltz2/params
```

Backend notes:

- `--cofold_env` should point to the path of its conda environment.
- `--cofold_param` is used as the Boltz2 cache / weights directory.
- If `--cofold_env` and `--cofold_param` are omitted, `ab_design.py` automatically uses `./api/tools/cofold/boltz2/env` and `./api/tools/cofold/boltz2/params`.

### AlphaFold3

Recommended setup:

```bash
bash ./api/tools/af3_scripts/setup.sh
```

The setup script prepares the following local layout:

```bash
./api/tools/af3_scripts/alphafold3
./api/tools/af3_scripts/alphafold3/env
./api/tools/af3_scripts/alphafold3/dummy_databases
```

Required paths:

- `--cofold_repo_dir`: AlphaFold3 repository root
- `--cofold_env`: AlphaFold3 environment
- `--cofold_db`: AlphaFold3 databases
- `--cofold_param`: AlphaFold3 model parameters

Backend notes:

- Unlike `protenix` and `boltz2`, AlphaFold3 does not auto-fill backend paths from `api/tools/cofold/<model>/...`.
- A typical local configuration after running `setup.sh` is:

```bash
--cofold_repo_dir ./api/tools/af3_scripts/alphafold3
--cofold_env ./api/tools/af3_scripts/alphafold3/env
--cofold_db ./api/tools/af3_scripts/alphafold3/dummy_databases
--cofold_param /path/to/alphafold3/params
```

- The `dummy_databases` folder is only used to bypass AlphaFold3 code checks. This antibody pipeline does not rely on online MSA or template search inside AlphaFold3.
- You still need to obtain the official AlphaFold3 model parameters separately.
- A typical parameter directory should look like:

```bash
/path/to/alphafold3/params/
`-- af3.bin.zst
```

- Point `--cofold_param` to that parameter directory.
- If you are using V100 GPUs, add the following line to `api/tools/af3_scripts/alphafold3_predict.sh`:

```bash
export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
```

## Entry Point

The pipeline entry point is:

```bash
python -m api.tools.ab_design \
  --config /path/to/pipeline.yaml \
  --save_dir /path/to/output
```

### Supported CLI Arguments

#### Required

- `--config`: path to the pipeline YAML file
- `--save_dir`: output directory

#### Common

- `--name`: overrides `config["name"]`
- `--ckpt`: confidence model checkpoint, default `checkpoints/model.ckpt`
- `--max_num_iterations`: maximum number of pipeline iterations, default `60`. Each iteration will generate 50 candidates.

#### Cofolding Backend

- `--cofold_model`: one of `alphafold3`, `boltz2`, `protenix`; default `protenix`
- `--cofold_env`: backend environment path
- `--cofold_param`: backend parameter / cache directory
- `--cofold_db`: backend database path, used by `alphafold3`
- `--cofold_repo_dir`: backend repo root, used by `alphafold3`

Auto-filled defaults:

- for `protenix`, omitted `--cofold_env` and `--cofold_param` are resolved to `./api/tools/cofold/protenix/{env,params}`
- for `boltz2`, omitted `--cofold_env` and `--cofold_param` are resolved to `./api/tools/cofold/boltz2/{env,params}`

#### Execution / Search Behavior

- `--gpu_ids`: list of GPU ids to use; default is all visible GPUs
- `--n_beam`: number of candidates selected each iteration for CDR design; default `10`
- `--n_loser_up`: number of additional candidates sampled from the remaining pool; default `2`. For example, under default settings, each iteration will choose top 8 candidates plus 2 randomly chosen candidates for CDR design.
- `--disable_cleanup`: keep all intermediate outputs
- `--retain_topk`: when cleanup is enabled, keep only top-ranked candidates; default `100`
- `--allowed_type`: allowed framework types loaded from `framework_lib`; choices are `antibody` and `nanobody`; default is both

### Example Commands

#### Default Run

```bash
python -m api.tools.ab_design \
  --config demo/antibody_pipeline.yaml \
  --save_dir ./output/ab_pipeline_px
```

#### Protenix with Explicit Backend Paths

```bash
python -m api.tools.ab_design \
  --config demo/antibody_pipeline.yaml \
  --save_dir ./output/ab_pipeline_px \
  --cofold_model protenix \
  --cofold_env /path/to/protenix/env \
  --cofold_param /path/to/protenix/params
```

#### AlphaFold3 with Explicit Backend Paths

```bash
python -m api.tools.ab_design \
  --config demo/antibody_pipeline.yaml \
  --save_dir ./output/ab_pipeline_af3 \
  --cofold_model alphafold3 \
  --cofold_repo_dir /path/to/alphafold3 \
  --cofold_env /path/to/alphafold3/env \
  --cofold_db /path/to/alphafold3/databases \
  --cofold_param /path/to/alphafold3/params
```

## YAML Configuration

The antibody pipeline YAML is different from the general generation YAML.
At a high level, it contains:

- target protein definition
- epitope definition through `epitope` or `reference`
- initial antibody / nanobody sources
- `template_options` for the generation stage

### Minimal Structure

```yaml
name: cxcr4

target:
  - type: protein
    chain_id: A
    sequence: ...
    template:
      cif: ...
      template_chain_id: ...

epitope: [[A, 2], [A, 3]]

framework_lib:
  path: ./api/templates/framework_lib.json
  template: ./api/templates/framework_lib_templates

template_options:
  class: AntibodyMultipleCDR
  cdr_types: [HCDR3]
```

### Top-Level Fields

#### Required or Practically Required

- `name`: experiment name; also used to build the default output root folder
- `target`: list of target chains, with given sequences and structure templates
- `template_options`: generation template configuration passed into the design stage
- one of `epitope` or `reference`: defines the target binding site residues
- `framework_lib`: defines the initial antibody / nanobody pool. The repo has prepared a libraries of frameworks with interactions most attributed to CDRs, which are better for design.

#### Optional

- `batch_size`: generation batch size for each design subprocess; default `8`
- `batch_n_samples`: number of generated samples per design job; default `10`
- `cofold_n_seeds`: number of cofold seeds used for iterative cofold jobs; default `1`
- `ranking_weights`: overrides default ranking weights
- `interaction_specify_cdrs`: restricts contact-based metrics to selected CDRs

### target

`target` is a list of chain descriptors.
Each item is converted to backend-agnostic `ChainData`.

Supported fields per target chain:

- `type`: chain type, typically `protein`
- `chain_id`: chain id used in cofold input
- `sequence`: target amino-acid sequence
- `modifications`: optional residue modification records
- `unpairedMsa`: optional raw A3M content, or a path ending in `.a3m`
- `pairedMsa`: optional raw A3M content, or a path ending in `.a3m`
- `template`: optional template description

`template` supports:

- `cif`: path to the template mmCIF / CIF file
- `template_chain_id`: chain id inside the template structure

Example:

```yaml
target:
  - type: protein
    chain_id: A
    sequence: MKEPCFREENANFNKIFLPTIY...
    template:
      cif: ./demo/data/8u4r_target.cif
      template_chain_id: R
```

Notes:

- If `unpairedMsa` or `pairedMsa` is omitted, the pipeline uses an empty string and does not trigger slow online MSA search.
- If `template` is omitted, the target chain is passed without template features.

### Binding Site Definition

The pipeline requires one of the following:

- `epitope`
- `reference`

#### epitope

`epitope` is a list of residue identifiers.
Each residue is noted as `[chain_id, residue_number]`.

Example:

```yaml
epitope: [[A, 2], [A, 3], [A, 155]]
```

> The residue number should be aligned with the sequence (starting from 1), instead of the template structure!

#### reference

`reference` defines the binding site from a reference complex.

Supported fields:

- `path`: path to a reference complex file
- `tgt_chains`: target chain ids in that complex
- `lig_chains`: ligand chain ids in that complex

Example:

```yaml
reference:
  path: ./demo/data/8u4r_chothia.pdb
  tgt_chains: R
  lig_chains: HL
```

### Initial Library 

The pipeline can initialize from `framework_lib`
This loads antibody / nanobody frameworks from a JSON library.

Supported fields:

- `path`: framework JSON file
- `template`: directory containing framework template CIFs
- `heavy_chain_id`: output heavy-chain id; default `H`
- `light_chain_id`: output light-chain id; default `L`

Example:

```yaml
framework_lib:
  heavy_chain_id: H
  light_chain_id: L
  path: ./api/templates/framework_lib.json
  template: ./api/templates/framework_lib_templates
```

Notes:

- The CLI flag `--allowed_type` filters which entries are loaded from the framework library.
- Valid values are `antibody`, `nanobody`, or both.

### template_options

`template_options` is the generation template passed into the design stage for each selected candidate.
It uses the same antibody template classes as the general generation entry point.


### ranking_weights

`ranking_weights` overrides the default weighted ranking used by the pipeline.
The default weights live in `api/tools/data_defs.py`.

Default keys include:

- `metrics.bs_overlap`
- `metrics.contact_cdr_ratio`
- `cplx_confidences.cofold_iptm`
- `cplx_confidences.cofold_binder_plddt`
- `cplx_confidences.cofold_normalized_ipae`
- `generative_confidences.normalized_cdr_design_likelihood`
- `metrics.normalized_scRMSD_cdr`

Example:

```yaml
ranking_weights:
  metrics.bs_overlap: 10.0
  metrics.contact_cdr_ratio: 5.0
  cplx_confidences.cofold_iptm: 0.3
```

### interaction_specify_cdrs

Restricts interaction-related metric calculation to specific CDRs.

Example:

```yaml
interaction_specify_cdrs:
  - HCDR3
  - LCDR3
```

### Example: Single-CDR Design

```yaml
template_options:
  class: AntibodyMultipleCDR
  cdr_type: [HCDR3]
```

### Example: Multiple-CDR Design

```yaml
template_options:
  class: AntibodyMultipleCDR
  cdr_types: [HCDR1, HCDR2, HCDR3, LCDR3]
  length_ranges:
    HCDR1: [6, 8]
    HCDR2: [5, 7]
    HCDR3: [15, 18]
    LCDR3: [8, 10]
```

Supported fields:

- `cdr_types`: list of CDRs to design
- `length_ranges`: optional map from CDR name to `[min_len, max_len]`

## Full Example

```yaml
name: cxcr4

target:
  - type: protein
    chain_id: A
    sequence: MKEPCFREENANFNKIFLPTIYSIIFLTGIVGNGLVILVMGYQKKLRSMTDKYRLHLSVADLLFVITLPFWAVDAVANWYFGNFLCKAVHVIYTVSLYSSVLILAFISLDRYLAIVHATNSQRPRKLLAEKVVYVGVWIPALLLTIPDFIFANVSEADDRYICDRFYPNDLWVVVFQFQHIMVGLILPGIVILSCYCIIISKLSHSKGHQKRKALKTTVILILAFFACWLPYYIGISIDSFILLEIIKQGCEFENTVHKWISITEALAFFHCCLNPILYAFLGA
    template:
      cif: ./demo/data/8u4r_target.cif
      template_chain_id: R

framework_lib:
  heavy_chain_id: H
  light_chain_id: L
  path: ./api/templates/framework_lib.json
  template: ./api/templates/framework_lib_templates

epitope: [[A, 2], [A, 3], [A, 4], [A, 5], [A, 6], [A, 7], [A, 9], [A, 10], [A, 11], [A, 155], [A, 150], [A, 181], [A, 159], [A, 162], [A, 164], [A, 165], [A, 166], [A, 167], [A, 168], [A, 169], [A, 170], [A, 232], [A, 265]]

template_options:
  class: AntibodyMultipleCDR
  cdr_types:
    - HCDR3

interaction_specify_cdrs:
  - HCDR3

batch_size: 8
batch_n_samples: 10
```