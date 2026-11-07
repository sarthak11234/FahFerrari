# FahFerrari 2027 — Implementation Plan
**Status:** Stage plan reconstructed from the approved project context. Confirm task status against the repository before execution.

## Guiding principles
1. Ferrari-focused forecasting, with full-grid competitive benchmarking.
2. Hybrid ML + Bayesian statistics + Monte Carlo simulation.
3. Extend the existing FahFerrari repository; preserve 2026.
4. Freeze the first preseason forecast before Bahrain testing.
5. Audit data and existing implementation before training or frontend rebuild.

## Stage 0 — Architecture and design foundation
- [x] Document 2027 architecture.
- [x] Document staged implementation plan.
- [x] Document frontend design system.
- [x] Create development branch `docs/fahferrari-2027-stage-0`.
- [x] Keep `main` unchanged during planning.

## Stage 1 — Repository and data audit
### FF27-01: Repository and dependency audit
- Inspect repository structure, runtime versions, dependencies, scripts, and deployment configuration.
- Identify existing 2026 entry points and preserve them.
- Record reproducible setup and test commands.

### FF27-02: Trace the existing pipeline
- Trace raw data through cleaning, feature engineering, training, inference, and API output.
- Identify hardcoded assumptions, random components, and manual inputs.
- Map each output field to its source and transformation.

### FF27-03: Raw data audit
- Inventory all CSVs and external data.
- Check row counts, season coverage, unique events, rounds, drivers, constructors, sessions, duplicates, nulls, and timestamp quality.
- Verify whether apparent 10-row/year samples are incomplete.
- Check whether round zero is a placeholder or a data defect.

### FF27-04: Data provider evaluation
- Compare source coverage, reliability, licensing/terms, historical depth, update cadence, and API stability.
- Document source attribution and retrieval timestamps.
- Do not select a provider solely for convenience.

### FF27-05: Canonical entity model
- Establish stable IDs for seasons, events, sessions, circuits, drivers, and constructors.
- Create explicit mapping tables for aliases and historical team/driver changes.

### FF27-06: Data manifests and point-in-time integrity
- Store source manifests and dataset versions.
- Record event time and availability time where possible.
- Eliminate look-ahead leakage, including unsafe backward filling.
- Ensure historical features respect the original forecast cutoff.

### FF27-07: Validation framework
- Add schema validation, uniqueness checks, range checks, missingness reports, and season/event consistency checks.
- Make data quality failures visible and reproducible.

### FF27-08: Multi-team dataset sample
- Produce an auditable sample covering multiple teams and drivers.
- Preserve race/session/circuit granularity.
- Review feature definitions before modelling.

## Stage 2 — Modelling research and baselines
- Establish simple, transparent baselines.
- Define chronological backtesting splits.
- Prototype hierarchical pace estimation.
- Test nonlinear residual modelling.
- Prototype reliability hazard modelling.
- Define probabilistic race-order and points generation.
- Evaluate calibration, interval coverage, and forecast sharpness.
- Reject model complexity that does not improve out-of-sample performance.

## Stage 3 — Season simulation
- Integrate pace, reliability, race outcomes, and points.
- Validate scoring and calendar assumptions from authoritative season rules.
- Start with 10,000 simulations.
- Test convergence and sensitivity to random seed.
- Preserve paired simulation outcomes for driver/team comparisons.
- Store simulation metadata and immutable forecast snapshots.

## Stage 4 — Forecast API and provenance
- Verify actual endpoint contracts.
- Define typed response schemas.
- Expose forecast version, cutoff, seed, simulation count, data manifest, model version, and assumptions.
- Distinguish missing, partial, stale, and unavailable data.
- Add API tests and schema validation.

## Stage 5 — Frontend implementation
### UI-0: Audit
- Inspect existing React/Vite app and backend payloads.
- Verify endpoints and real response shapes.
- Identify legacy 2026 UI that must remain accessible.

### UI-1: Design foundation
- Implement tokens, typography, app shell, navigation, cards, buttons, tooltips, skeletons, and status states.

### UI-2: Cinematic hero
- Add approved Ferrari imagery, responsive crop, text overlay, and restrained motion.

### UI-3: API integration
- Replace hardcoded values with API data.
- Implement loading, error, partial, stale, and missing states.

### UI-4: Homepage analytics
- Build Ferrari outlook, driver comparison, constructors table, race watchlist, and provenance panel.

### UI-5: Responsive and accessibility QA
- Test 360px mobile through desktop.
- Check keyboard navigation, contrast, chart labels, reduced motion, and horizontal overflow.

## Stage 6 — Validation and release
- Run full historical backtests.
- Review calibration and uncertainty communication.
- Verify every displayed metric against backend output.
- Confirm forecast cutoff and immutable snapshot.
- Publish methodology and limitations.
- Preserve 2026 retrospective and results.

## Critical acceptance criteria
- No future-data leakage.
- No fabricated or hardcoded forecast values.
- Probabilities are calibrated and correctly labelled.
- Uncertainty intervals have documented definitions.
- Driver head-to-head metrics use paired simulation outcomes.
- Provenance is visible and reproducible.
- The 2026 implementation remains preserved.
- Frontend handles loading, errors, partial and stale data.
- Mobile layout and accessibility are tested.

## Important scheduling note
The planned preseason freeze date and 2027 calendar assumptions must be reconfirmed against official Formula 1 announcements before being treated as fixed.
