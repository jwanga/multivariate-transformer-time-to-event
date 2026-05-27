# Project Instructions

## General

- This is a single-notebook learning project. The notebook
  (`notebook.ipynb`) is the source of truth for all code and teaching.
- The reader writes the implementation inside scaffolded code cells.
  Markdown cells are pre-written pedagogy.
- Built from scratch — avoid PyTorch high-level shortcuts like
  `nn.TransformerEncoder`, `nn.MultiheadAttention`, or
  `F.scaled_dot_product_attention`. The point is to implement them.

## Build

There is no build step. Install dependencies and the notebook is ready.

```bash
pip install -r requirements.txt
python -m ipykernel install --user --name=tte-transformer
```

## Run

```bash
jupyter lab notebook.ipynb
```

Work through the notebook top-to-bottom. Cells are ordered so each one
depends only on those above it.

## Test

There is no separate test suite. Each Part of the notebook ends with a
sanity-check cell (e.g., synthetic-batch forward pass shape check,
loss-curves render, calibration plot). Running the cells in order is
the test.

## Deploy

Not applicable — this is a learning artifact, not a deployable service.

## Overrides

- Do not introduce a `src/` Python package. Code lives in the notebook.
- Do not add automated unit tests. Sanity-check cells in the notebook
  serve that role.
- Do not commit the CMAPSS data files. They are large and licensed
  separately by NASA; users download them on their own.
- Do not commit `~/.kaggle/kaggle.json` or any other API credentials.
- Voice in all written content is universal — no first-person references
  to any specific person, no author attribution, no anecdotes.
