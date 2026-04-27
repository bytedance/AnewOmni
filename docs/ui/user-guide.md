# UI User Guide

## What This Is

`api/ui` provides a local prompt-programming interface for controllable generation. It supports:

- prompt editing in a restricted REPL
- a lightweight browser UI with built-in demos
- prompt-state inspection and visualization
- 2D and 3D result inspection
- task submission and task switching

Supported prompt types:

- `MoleculePrompt`
- `PeptidePrompt`
- `AntibodyPrompt`

## Start the UI

### Web UI

From the repository root:

```bash
ANEW_UI_PORT=8766 python -m api.ui.web
```

Then open:

```text
http://127.0.0.1:8766
```

If `ANEW_UI_PORT` is not set, the default port is `8765`.

### REPL

To start the command-line interface:

```bash
python -m api.ui.repl
```

## Web Workflow

The web app includes four built-in demos:

- `demo:molecule`
- `demo:peptide`
- `demo:cyclic_peptide`
- `demo:antibody`

Recommended first run:

1. Start the web UI.
2. Select a demo.
3. Inspect the generated code in the editor.
4. Click `Execute`.
5. Click `Run Generation`.
6. See the records and visualizations.
7. Open the output directory if needed and inspect `run.log`, `results.jsonl`, and `filtered_results.jsonl`.

## Minimal REPL Example

```python
from api.ui import *

graph = MoleculePrompt()
graph.add_fragment("c1ccccc1", name="ring")
graph.set_context(
    pdb_path="demo/data/8u4r_chothia.pdb",
    tgt_chains="R",
    lig_chains="HL",
    checkpoint="checkpoints/model.ckpt",
    gpu=0,
)
```

Then run:

```text
:state
:compile
:run ./outputs/demo_run
```

Relative paths are resolved against the project root in the UI path, so the examples above do not need machine-specific absolute paths.

## REPL Commands

Available meta-commands:

- `:help`
- `:state`
- `:labels`
- `:compile`
- `:run <save_dir>`
- `:reset`
- `:quit`

## General Prompt Workflow

1. Create a prompt object.
2. Add prompt constraints.
3. Attach generation context with `set_context(...)`.
4. Inspect the prompt with `:state` or the web preview.
5. Run generation.
6. Inspect raw and filtered outputs.

Typical context fields:

- `pdb_path`
- `tgt_chains`
- `lig_chains`
- `checkpoint`
- `batch_size`
- `n_samples`
- `gpu`

Notes:

- `gpu=-1` means CPU.
- The UI demo currently supports only a single target complex.

## Molecule Prompt

### Example

```python
graph = MoleculePrompt()
graph.add_fragment("c1ccccc1", name="ring")
graph.add_fragment("C", name="methyl")
graph.add_bond("ring:c0", "methyl:c0")
graph.allow_growth(3, 6)
```

### Common Methods

- `add_fragment(smiles, name=None, pin=True)`
- `add_bond(atom1, atom2, bond_type="single")`
- `allow_growth(min_blocks, max_blocks)`
- `pin(name)`
- `add_filter(filter_obj)`

### Default Molecule Filters

The current default molecule filters are:

- `AbnormalConfidenceFilter`
- `ChiralCentersFilter(center_max=8, ring_mode=True)`
- `RotatableBondsFilter(max_num_rot_bonds=7)`
- `PhysicalValidityFilter`
- `MolBeautyFilter(th=1)`

The filter loop stops on the first non-passing filter for each sample.

## Peptide Prompt

### Example

```python
pep = PeptidePrompt(length=12)
pep.add_motif("RGD")
pep.add_noncanonical(
    "NC1=CC=C(C=C1)C=C(C#N)C2=CC=CC(C(C(=O)*)N*)=C2",
    count=1,
)
pep.cyclize(mode="disulfide")
```

### Common Methods

- `add_motif(seq, positions=None)`
- `add_noncanonical(smiles, positions=None, count=None)`
- `cyclize(mode="head_tail" | "disulfide")`

### Default Peptide Filters

- `AbnormalConfidenceFilter`
- `PhysicalValidityFilter`
- `LTypeAAFilter`

## Antibody Prompt

### Example

```python
ab = AntibodyPrompt(
    framework_path="demo/data/8u4r_chothia.pdb",
    cdrs=["HCDR3", "LCDR3"],
)
ab.set_length("HCDR3", 12)
ab.set_length("LCDR3", 9)
ab.add_motif("HCDR3", "YYG", positions=[2, 3, 4])
```

### Current Antibody Behavior

- Framework freezing is enabled by default.
- Relative CDR positions are used for motif placement.
- The web demo uses two designed CDRs:
  - `HCDR3 = 12`
  - `LCDR3 = 9`

### Default Antibody Filters

- `AbnormalConfidenceFilter`
- `PhysicalValidityFilter`
- `LTypeAAFilter`

## Filtering and Result Selection

When filters are enabled in the UI path:

- generation is attempted in batches
- the default filter batch quota is `6`
- generation and filtering overlap within each batch
- if any sample passes, the UI keeps the passed set
- if no sample passes after all attempts, the UI selects the sample with the lowest `confidence` value as fallback

Here, `confidence` is interpreted as PDE, so lower is better.

### Files

Relevant result files:

- `results.jsonl`
- `filtered_results.jsonl`
- `<id>/<n>.cif`
- `<id>/<n>.pdb`
- `<id>/<n>.sdf`
- `run.log`

### Logs

`run.log` includes:

- batch start and finish lines
- per-item streaming filter progress during batch execution
- per-batch generated and passed counts
- per-filter `passed / failed / error` counts
- fallback messages when no sample passes

## Current Result View

The web UI result panel shows:

- prompt summary
- display source (`filtered` vs `raw`)
- PDE
- likelihood
- smiles
- generated sequence
- target chains
- ligand chains
- filter outputs

The `AbnormalConfidenceFilter` details are intentionally hidden from the top-record view, even though the filter still runs.

## Current Layout Notes

The browser layout currently behaves as follows:

- the `Program` output box shows execution status messages and the current task log
- there is no separate dedicated run-log panel
- the `Latest Result` panel remains focused on visual and textual result inspection
- the `Prompt State` area in the middle panel expands to use the remaining vertical space

## Limitations

- Antibody nsAA support is still weaker than motif and length control.
- The UI streaming filter path currently overlaps generation and filtering within a batch, but it does not terminate a running generation batch early once one item passes.
