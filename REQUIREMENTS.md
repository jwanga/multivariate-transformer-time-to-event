# Multivariate Time-to-Event Transformer — Requirements

A learning project: a from-scratch PyTorch causal transformer with a
Weibull time-to-event head, trained on the NASA CMAPSS jet engine
degradation dataset.

## Table of Contents

- [Milestones](#milestones)
- [Requirements](#requirements)
  - [Notebook structure](#notebook-structure)
  - [Data pipeline](#data-pipeline)
  - [Model](#model)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Pedagogy](#pedagogy)
  - [Project hygiene](#project-hygiene)
- [Architecture](#architecture)

## Milestones

- [ ] **M0 — Project scaffold**: notebook structure built, all markdown
      pedagogy written, all code cells stubbed with signatures + docstrings
      + TODO hints, project files in place
- [ ] **M1 — Data pipeline**: CMAPSS FD002 loads, regime-aware
      normalization works, causal training tuples (sequence, target_time,
      censored) are constructed, `DataLoader` yields padded batches
- [ ] **M2 — Model components**: continuous embedding, positional encoding,
      scaled dot-product attention with masks, multi-head attention,
      transformer block, and full `TTETransformer` all implemented and
      pass the synthetic-batch sanity check
- [ ] **M3 — Training**: Weibull NLL with censoring implemented, training
      loop runs end-to-end, loss decreases over epochs
- [ ] **M4 — Evaluation**: C-index and Brier score computed on test set;
      Weibull (k, λ) converted to point estimates with 90% confidence
      intervals; per-engine PDF plot, forecast-trajectory plot, and
      calibration plot all render
- [ ] **M5 — Extensions**: add a second head following the registry
      pattern, swap causal mask for bidirectional, or add attention
      visualization

## Requirements

### Notebook structure

- Single `notebook.ipynb` at the repo root contains all teaching content
- Organized in 7 parts: Overview, Setup, Data, Model components,
  Training, Evaluation, Extensions
- Every code cell paired with a preceding markdown cell explaining the
  concept
- Mermaid diagrams used wherever they clarify architecture, data flow,
  or math

### Data pipeline

- Auto-download CMAPSS from Kaggle (`behrad3d/nasa-cmaps`)
- Document NASA `data.nasa.gov` manual fallback
- Load FD002 by default; data loader contract supports FD001/FD003/FD004
  with no architecture change
- Regime-aware normalization: cluster the 3 ops settings into 6
  operating regimes (KMeans), z-score each sensor per regime
- Drop near-constant sensors automatically; report which were dropped
- Construct causal training tuples: each engine contributes one supervised
  sample per timestep, with the target being remaining cycles from that
  timestep
- Handle variable sequence lengths via padding + attention mask
- Train engines are uncensored (run to failure); test engines are
  right-censored at the last observed cycle

### Model

- Built from scratch in PyTorch — no `nn.TransformerEncoder` or
  `F.scaled_dot_product_attention` shortcuts
- Continuous-input embedding (no NLP-style vocabulary)
- Sinusoidal positional encoding (fixed, precomputed)
- Scaled dot-product attention supporting combined causal + padding masks
- Multi-head attention as a tensor reshape (no per-head loop)
- Pre-norm transformer blocks: `x = x + MHA(LN(x)); x = x + FFN(LN(x))`
- Causal masking — each timestep can only attend to past and present
- Head registry pattern: each head is a separately registered module
  with its own forward + loss; encoder is shared
- Weibull head outputs `(k, λ)` via softplus for strict positivity
- `YourHeadHere` stub demonstrating the extension contract

### Training

- Weibull NLL with right-censoring term derived from the survival function
- Multi-head weighted loss aggregation
- AdamW optimizer with warmup + cosine schedule
- Gradient clipping
- Per-epoch logging of train/eval losses and metrics
- Per-epoch loss curves plot at end of training

### Evaluation

- Concordance index (C-index) — implemented from scratch with numpy
- Integrated Brier Score — implemented from scratch with numpy
- Weibull `(k, λ)` → point estimates (median, mean) and quantiles
- Batch conversion of test predictions to
  `(point_estimate, lower_90, upper_90)` tuples
- Per-engine predicted-PDF plot with median, 90% CI shading, true RUL marker
- **Forecast-trajectory plot** for sample engines showing how the
  predicted time-to-event and its error band evolve as cycles accumulate
- Calibration plot (empirical vs predicted survival probabilities)
- Headline results table: RMSE, NASA score (derived from median),
  C-index, Brier

### Pedagogy

- Intuition-first explanations; rigorous math in clearly-marked sections
  the reader can skip on first pass
- Mermaid diagrams for architecture, data flow, and conceptual breakdowns
- Code cells are scaffolded — signatures, docstrings, TODO hints,
  `raise NotImplementedError` placeholder; the reader writes the logic
- Markdown cells are fully written

### Project hygiene

- Public repository, MIT licensed
- No personally identifying information anywhere
- No Kaggle credentials, API keys, or secrets committed
- `.gitignore` excludes data files, virtual environments, notebook
  checkpoints, and Kaggle credential paths
- Universal voice — no first-person references to any specific maintainer

## Architecture

See `docs/superpowers/specs/2026-05-26-tte-transformer-design.md` for the
full design spec and architecture diagram.

High-level data flow:

```
Raw sensors → regime normalization → continuous embedding
            → + positional encoding → N × causal transformer blocks
            → per-timestep hidden states → head registry
            → Weibull head (k, λ) → NLL with censoring
                                  → quantile conversion → (point, low, high)
```

The head registry is the extensibility seam: additional heads attach to
the same encoder output and contribute to a weighted total loss.
