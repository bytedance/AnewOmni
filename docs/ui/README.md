# UI Documentation

This folder contains the consolidated documentation for the interactive prompt-programming UI.

The UI layer lives in `api/ui`, but the documentation now lives under the repository root `docs/` directory so it is easier to discover and maintain.

## Documents

- `README.md`
  - landing page for the UI docs set
- `user-guide.md`
  - how to start the UI
  - how to use the built-in demos
  - prompt syntax for molecule, peptide, cyclic peptide, and antibody workflows
- `development.md`
  - architecture and runtime flow
  - task execution model
  - filter pipeline behavior
  - extension and maintenance notes
- `prompt-language-design.md`
  - original prompt-language design note
  - high-level object model and long-term direction

## Main Code Entry Points

- `api/ui/core.py`
- `api/ui/repl.py`
- `api/ui/web.py`

## Quick Start

Start the web UI (if `ANEW_UI_PORT` is not provided, port 8765 will be used):

```bash
ANEW_UI_PORT=8766 python -m api.ui.web
```

Start the REPL (command-line user interface):

```bash
python -m api.ui.repl
```

## Notes

- The UI demo currently supports a single target complex.
- Built-in demos are provided for molecule, peptide, cyclic peptide, and antibody workflows.
- The web UI supports task submission, task switching, prompt inspection, 2D SVG results, and 3D Mol* viewing.
- The generation pipeline now supports batch-level filtering (predefined physiochemical filters) with quota retries and a best-confidence fallback.
- The UI demo was completed mostly by vibe coding (thank GPT-5.4), so there might be a lot of hidden bugs. The UI demo only serves as a proof of concept.