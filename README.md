# SASRec: Background Preparation, Training and Evaluation

Anonymous source supplement for reviewing the study's SASRec baseline. The
implementation exposes the background construction, sequence model, training
objective, checkpoint selection and fixed-candidate evaluation in separate
modules.

**This repository contains source code and configuration only.** The experiment
datasets, split files, item mappings, trained weights and experimental outputs
are not distributed. Training and evaluation cannot run from this checkout
alone. The entry points report the required missing files rather than downloading
data or substituting demonstration data.

## Repository structure

```text
.
├── configs/
│   └── sasrec.json          # Architecture and per-domain training settings
├── data/
│   └── README.md           # External input requirements; no data files
├── docs/
│   ├── data_contract.md    # File schemas, catalog mapping and split invariants
│   └── implementation.md   # Background recovery, numerical fix and protocol
├── sasrec/
│   ├── __main__.py         # python -m sasrec
│   ├── cli.py              # Prepare, train and evaluate entry points
│   ├── config.py           # Configuration, preflight and input fingerprints
│   ├── data.py             # Query validation and sequence tensor construction
│   ├── background.py       # Background filtering and prefix restoration
│   ├── model.py            # Causal sequence encoder and finite-value guards
│   ├── objectives.py       # Next-item sampled-softmax objective
│   ├── checkpoint.py       # Validation selection and checkpoint persistence
│   ├── trainer.py          # Epoch loop, early stopping and final evaluation
│   └── evaluation.py       # Fixed-candidate ranking
├── tests/
│   └── test_invariants.py  # In-memory implementation checks
└── pyproject.toml
```

## Implementation focus

- **Training background:** retain eligible background users within the existing
  catalog; remove all selected evaluation users from the raw stream and restore
  only their verified history prefixes.
- **Numerical scoring:** disable the optimized multi-head attention evaluation
  path for left-padded causal inputs and reject non-finite scores before ranking.
- **Model selection:** select the earliest checkpoint attaining the best
  validation accuracy. Evaluate the held-out test splits after selection.

The network uses item and positional embeddings followed by a causal Transformer
encoder. Training predicts the next item from the preceding sequence. At
evaluation, the final sequence representation scores each supplied candidate by
its dot product with the candidate embedding.

## Configuration

The default architecture uses sequence length 20, embedding dimension 64, two
layers, two heads, feedforward dimension 256 and dropout 0.2. Training uses Adam
with learning rate 0.001 and 128 uniform negative draws per valid position.

The configuration covers `auto`, `music`, `kindle`, `health`, `sports` and `books`.
The first five use expanded background inputs; `music` denotes CDs & Vinyl.
Books retains its original background and uses a longer training schedule.
Domain-specific settings are explicit in [configs/sasrec.json](configs/sasrec.json).

## Entry points

From the repository root, install the source package in a Python environment
with an appropriate PyTorch build:

```bash
python -m pip install -e .
```

After separately obtaining and normalizing the required experiment inputs:

```bash
python -m sasrec prepare --domain auto --data-root data --output-root artifacts
python -m sasrec train --domain auto --data-root data --output-root artifacts --device cuda
python -m sasrec evaluate --domain auto --data-root data --output-root artifacts --device cuda
```

Use `--device cpu` for CPU execution. All three commands require the external
files described in [docs/data_contract.md](docs/data_contract.md). Running them
without those inputs exits with a missing-input error before creating artifacts.
Existing training directories are not overwritten.

Implementation checks use in-memory tensors and frames and do not run a
recommendation experiment:

```bash
python -m unittest discover -s tests
```

## Scope and provenance

The code is a modular refactoring of the study's training and audit scripts.
Machine-specific paths, private dataset readers, and experiment scheduling are
replaced by explicit input contracts and a command-line interface. The model,
objective, padding correction and validation-selection logic are retained.
This source release is not a self-contained reproduction package.

The holdout protocol excludes each selected user's target event; it does not
impose a global chronological cutoff. Details and boundaries of the comparison
are documented in [docs/implementation.md](docs/implementation.md).
