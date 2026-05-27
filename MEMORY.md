# Project Memory

## Current State

- **Active Milestone**: M0 — Project scaffold
- **Current Issue**: None (pre-issue scaffolding phase)
- **Current Branch**: main
- **Plugin Version**: engineering-plugin (current)

## Progress Log

<!-- Each entry MUST use the format: [YYYY-MM-DD HH:MM] @username: description -->

- [2026-05-26 14:40] @contributor: Project bootstrapped via `/next` Feature Classification workflow
  - Brainstormed model architecture and notebook structure
  - Decision: causal transformer + Weibull time-to-event head
  - Decision: single comprehensive notebook (no `src/` package)
  - Decision: CMAPSS FD002 as primary dataset (6 ops conditions, single fault)
  - Decision: "YourHeadHere" stub for multi-head extensibility demo
  - Decision: auto-download from Kaggle with NASA manual fallback documented
  - Decision: intuition-first pedagogy with collapsible rigorous math
  - Design spec written to `docs/superpowers/specs/2026-05-26-tte-transformer-design.md`
- [2026-05-26 14:45] @contributor: Bootstrap files created — README, REQUIREMENTS, INSTRUCTIONS, MEMORY, requirements.txt, .gitignore, LICENSE, data/README.md
- [2026-05-26 18:30] @contributor: Dropped `lifelines` dependency. C-index and Integrated Brier Score now scaffolded as from-scratch numpy implementations in cells 5.2 and 5.3 — better alignment with the "build from scratch" pedagogy, removes the pandas version conflict (lifelines pinned pandas<2.1, project uses pandas>=3.0.0), and reduces the dep footprint.

## Key Decisions

<!-- Each entry MUST use the format: [YYYY-MM-DD HH:MM] @username: description -->

- [2026-05-26 14:40] @contributor: Causal masking (not bidirectional) chosen because (a) "time to event" prediction must not peek at future degradation, (b) gives a clean streaming-inference story, (c) each engine becomes many supervised samples (one per timestep) instead of one
- [2026-05-26 14:40] @contributor: Weibull head chosen over scalar RUL regression because the term "time to event" maps to survival analysis, the Weibull NLL naturally handles right-censored test engines, and the predicted distribution yields calibrated error bars via the quantile function
- [2026-05-26 14:40] @contributor: Single notebook chosen over multi-notebook or notebook + `src/` because the reader writes the code themselves — splitting code across files would defeat that
- [2026-05-26 14:40] @contributor: Head registry pattern adopted to satisfy the multi-head extensibility requirement; each head registers its own forward + loss, all share the encoder

## Notes

<!-- Each entry MUST use the format: [YYYY-MM-DD HH:MM] @username: description -->

- [2026-05-26 14:40] @contributor: Privacy constraint — repo must contain no PII, no first-person references, no author attribution. Apply across all written content.
