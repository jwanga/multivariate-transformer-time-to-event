# Multivariate Time-to-Event Transformer — Design Spec

**Date:** 2026-05-26
**Status:** Approved

## Goal

Build a single, comprehensive Jupyter notebook that scaffolds a from-scratch
PyTorch implementation of a **causal multivariate transformer with a Weibull
time-to-event head**, applied to NASA's CMAPSS jet engine degradation dataset.

The notebook is a **learning artifact**. The scaffolder (this spec) writes
all markdown pedagogy, function signatures, docstrings, and TODO hints. The
reader writes the implementation logic inside each code cell.

## Non-Goals

- Production deployment, model serving, or hyperparameter sweeps
- State-of-the-art accuracy on CMAPSS benchmarks
- Multi-notebook curriculum (deliberately single-notebook for end-to-end flow)
- A separate `src/` package — all code lives in the notebook so the reader
  writes it themselves rather than importing pre-built helpers

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Primary head framing | Causal transformer + Weibull TTE head | "Time to event" terminology maps to survival analysis; Weibull handles right-censored test engines naturally; causal masking gives streaming inference + dense per-timestep supervision |
| Notebook scope | One comprehensive notebook | End-to-end narrative beats chapter splits for a learning artifact |
| Primary dataset | CMAPSS FD002 (6 ops conditions, single fault) | Regime variability is pedagogically valuable without the noise floor of FD004 |
| Extension head | "YourHeadHere" documented stub | Minimum scaffold weight; demonstrates the extension contract without prescribing a second task |
| Data acquisition | Auto-download from Kaggle (`behrad3d/nasa-cmaps`) with NASA `data.nasa.gov` manual fallback documented | Convenience without broken-link traps |
| Math depth | Intuition-first with collapsible rigorous math | Reader controls depth on first pass |
| Code/markdown split | Markdown cells fully written; code cells = signatures + docstrings + TODO hints + `raise NotImplementedError` | Reader does the implementation work, scaffold does the teaching |
| Repo layout | Notebook at root + engineering-plugin context files; no `src/` package | All code lives in the notebook |

## Architecture

```
Raw sensors (3 ops settings + 21 sensors)
        │
        ▼
Regime-aware normalization  (cluster ops → per-cluster z-score)
        │
        ▼
Continuous embedding  (Linear F → d_model)
        │
        ▼
+ Sinusoidal positional encoding
        │
        ▼
N × Causal Transformer Block
   (Pre-norm: x = x + MHA(LN(x))
              x = x + FFN(LN(x))
    with causal + padding masks)
        │
        ▼
Per-timestep hidden states h_t
        │
        ├──► Weibull head ──► (k_t, λ_t) ──► Weibull NLL w/ censoring
        │
        └──► YourHeadHere stub ──► (your loss)
                                          │
                                          ▼
                                Weighted total loss
```

The same diagram is rendered as a mermaid diagram inside the notebook's
Part 0 overview.

### Loss

Weibull negative log-likelihood with right-censoring:

```
L(t, u; k, λ) = -u · log f(t; k, λ) - (1-u) · log S(t; k, λ)
```

where `u=1` for observed events (train engines run to failure), `u=0` for
right-censored observations (test engines stopped before failure — we only
know they survived ≥ t cycles).

### Distribution → number with error bars

Given predicted (k, λ), use the Weibull quantile function
`Q(p; k, λ) = λ · (−ln(1−p))^(1/k)` to derive:

- Point estimate (median): `λ · (ln 2)^(1/k)`
- Point estimate (mean): `λ · Γ(1 + 1/k)`
- 90% confidence interval: `[Q(0.05), Q(0.95)]`
- 50% confidence interval: `[Q(0.25), Q(0.75)]`

Reported as "Engine N will fail in X cycles (90% CI: low–high)."

A trajectory plot shows how the predicted (k, λ) — and thus the point
estimate and CI — **evolve as cycles accumulate**, which is the payoff of
causal training.

## Notebook Cell Outline

### Part 0 — Title & Architecture Overview (markdown only)

- 0.1  Title, learning objectives, what you'll build
- 0.2  📊 Top-level architecture mermaid diagram
- 0.3  Why a causal transformer — intuition + streaming inference story
- 0.4  Why a Weibull head — intuition + collapsible math (PDF, CDF, S(t), censoring)
- 0.5  Roadmap of the notebook

### Part 1 — Setup

- 1.1  md: Environment, dependencies, device-selection rationale
- 1.2  code: `pip install` cell
- 1.3  code: Imports, seed setting, device detection (`CUDA` / `MPS` / `CPU`)

### Part 2 — Data

- 2.1  md: CMAPSS overview + FD002 specifics + file format
- 2.2  code: Kaggle auto-download with NASA manual fallback documented
- 2.3  code: Load FD002 train / test / RUL into DataFrames
- 2.4  md: 📊 Raw files → tensors pipeline
- 2.5  md: Exploratory analysis — what we're looking for
- 2.6  code: EDA plots — sequence-length histogram, sample sensor traces, ops scatter
- 2.7  md: Why 6 operating conditions break naive normalization + 📊
- 2.8  code: KMeans cluster ops settings → 6 regimes; per-regime z-score sensors
- 2.9  md: Sensor selection — which sensors carry signal
- 2.10 code: Drop near-constant sensors; report which survived
- 2.11 md: Causal target construction — each engine → many supervised samples + 📊
- 2.12 code: Build (sequence, target_time, censored_flag) tuples for train + test
- 2.13 md: PyTorch `Dataset` + variable-length padding/masking strategy
- 2.14 code: `CMAPSSDataset` class + `collate_fn` with padding mask
- 2.15 code: `DataLoader` wiring

### Part 3 — Model components (from-scratch)

- 3.1  md: Continuous sensor embedding — contrast with NLP token embeddings
- 3.2  code: `ContinuousEmbedding(nn.Module)`
- 3.3  md: Sinusoidal positional encoding — intuition + math + 📊 encoding matrix
- 3.4  code: `PositionalEncoding`
- 3.5  md: Scaled dot-product attention — intuition + 📊 Q/K/V flow + math
- 3.6  code: `scaled_dot_product_attention(q, k, v, mask)`
- 3.7  md: Multi-head attention as parallel low-dim attentions + 📊
- 3.8  code: `MultiHeadAttention(nn.Module)`
- 3.9  md: Position-wise feed-forward
- 3.10 code: `FeedForward(nn.Module)`
- 3.11 md: One transformer block — pre-norm residual structure + 📊
- 3.12 code: `TransformerBlock(nn.Module)`
- 3.13 md: Stacking blocks + the head registry pattern + 📊
- 3.14 code: `WeibullHead(nn.Module)` — outputs `(k, λ)` via softplus
- 3.15 code: `YourHeadHere(nn.Module)` — documented stub
- 3.16 code: `TTETransformer(nn.Module)` — full model wiring
- 3.17 code: Sanity-check forward pass on synthetic batch — verify shapes

### Part 4 — Training

- 4.1  md: Weibull NLL with censoring — intuition + collapsible derivation
- 4.2  code: `weibull_nll_loss(k, lambda_, t, censored)`
- 4.3  md: Full training objective — multi-head weighted sum + 📊
- 4.4  code: `compute_total_loss(model_outputs, targets, head_weights)`
- 4.5  md: Optimizer (AdamW), warmup + cosine schedule, gradient clipping
- 4.6  code: Optimizer + scheduler setup
- 4.7  md: The training loop
- 4.8  code: `train_one_epoch()` and `evaluate_one_epoch()`
- 4.9  code: Main training loop with tqdm + per-epoch metric logging
- 4.10 code: Loss curves plot

### Part 5 — Evaluation & distribution → number conversion

- 5.1  md: What "evaluating a survival model" means — C-index, Brier, calibration
- 5.2  code: C-index on test set (via `lifelines`)
- 5.3  code: Integrated Brier Score
- 5.4  md: From (k, λ) to a number with error bars — quantile derivation + 📊
- 5.5  code: `weibull_median`, `weibull_mean`, `weibull_quantile`
- 5.6  code: Batch-convert test predictions → (point_estimate, lower_90, upper_90)
- 5.7  md: Predicted PDFs as a sanity check
- 5.8  code: For 6 sample test engines: plot Weibull PDF at last observed cycle,
       mark median, shade 90% CI, mark true RUL
- 5.9  md: Forecast trajectory — how predictions refresh as cycles accumulate
- 5.10 code: For 2-3 sample engines: plot predicted TTE with error band vs cycles observed
- 5.11 md: Calibration plot interpretation
- 5.12 code: Calibration plot — empirical vs predicted survival probabilities
- 5.13 md: Headline results table — RMSE, NASA score (from median), C-index, Brier

### Part 6 — Extensions

- 6.1  md: Adding another dataset — data loader contract
- 6.2  md: Adding another head — registry pattern walkthrough
- 6.3  md: Swapping causal for bidirectional (offline analysis)
- 6.4  md: Open exercises

## Scaffolding Contract

Every **code cell** follows this pattern:

```python
def function_name(args) -> ReturnType:
    """Full docstring with shapes, args, returns, pedagogical notes.

    Implementation hints:
        1. Step one
        2. Step two
        3. Step three
    """
    # TODO: hint for step 1
    # TODO: hint for step 2
    raise NotImplementedError("Implement function_name")
```

Every **markdown cell** is fully written — pedagogy is not a fill-in exercise.

## Repo Layout

```
multivariate-transformer-time-to-event/
├── notebook.ipynb            # the main learning notebook
├── README.md                 # project intro, setup, what you'll learn
├── REQUIREMENTS.md           # captured spec (engineering-plugin convention)
├── INSTRUCTIONS.md           # build/run conventions
├── MEMORY.md                 # progress log
├── requirements.txt          # pinned Python deps
├── .gitignore                # excludes data/, .venv/, .ipynb_checkpoints/
├── LICENSE                   # MIT
├── docs/
│   └── superpowers/specs/
│       └── 2026-05-26-tte-transformer-design.md  # this file
└── data/
    └── README.md             # where CMAPSS files go + links
```

## Dependencies

Pinned in `requirements.txt`:

- `torch>=2.5` (latest stable PyTorch)
- `numpy`, `pandas`, `scikit-learn`, `scipy`
- `matplotlib`, `tqdm`
- `lifelines` (for C-index and Brier score)
- `kaggle` (for dataset auto-download)
- `jupyterlab`, `ipykernel`

## Privacy & Public-Repo Hygiene

- No author names, emails, or other PII anywhere
- No API keys, Kaggle credentials, or secrets committed
- `.gitignore` excludes `data/`, virtual environments, notebook checkpoints,
  and `~/.kaggle/`
- Voice is universal ("you'll learn", "we'll build") — no first-person
  references to any specific maintainer or origin story
- MIT license so anyone can use the material

## Open Questions / Deferred

- Specific PyTorch minor version pin — leave at `>=2.5` until the notebook
  runs end-to-end and we can lock a tested version
- Whether to include attention visualization as a Part 6 exercise vs an
  in-line cell in Part 5 (currently deferred to Part 6 exercises)
