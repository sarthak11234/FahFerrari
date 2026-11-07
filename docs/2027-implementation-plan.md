# FahFerrari 2027 — Implementation Plan

**Parent architecture:** [2027-architecture.md](./2027-architecture.md)  
**Branch:** `docs/fahferrari-2027-stage-0`  
**Status:** Proposed execution plan

## Guiding rule

Do not start by training a more complicated model. First establish a trustworthy, point-in-time-correct dataset and a reproducible baseline. A complex model is promoted only if it improves out-of-sample performance and calibration.

## Stage 1 — Repository and data audit

### 1.1 Repository audit
- Inventory Python and frontend dependencies, runtime versions, scripts, API routes and model artifacts.
- Map existing 2026 data flow from source collection to dashboard output.
- Identify public interfaces that must remain backward-compatible.
- Record reproducibility gaps, hidden assumptions and unused modules.

### 1.2 Existing data audit
The previously inspected repository snapshot showed a committed `race_results_2019_2025.csv` with only a small sample (10 rows per year), all `round` values set to zero. The feature file consequently cannot be treated as a complete season-level training dataset.

Audit and resolve:
- Why the collector's round field was not populated in the committed export.
- Whether the CSV is a sample, a failed/partial collection, or an incomplete artifact.
- Duplicate and missing driver-event records.
- Whether points, sprint points, fastest-lap points (where historically applicable), penalties and disqualifications are represented correctly.
- Whether status strings are being incorrectly counted as DNFs.
- Whether all constructors and drivers are present, not just Ferrari.
- Whether 2026 is complete or provisional at each data snapshot.

### 1.3 Known feature-engineering defects to investigate
- Fixed divisors of 24 races and 48 driver-race entries distort seasons with different event counts and Sprint formats.
- Backward filling can leak future-season information into earlier rows.
- Season-level aggregation discards driver-, session-, circuit- and race-level variation.
- Manually assigned technical scores are subjective and lack a documented measurement source.
- Current prediction code uses hand-authored driver splits, rival points multipliers and track multipliers rather than learned, validated distributions.
- The existing confidence interval uses disagreement between four models as if it were a calibrated predictive interval; this is not statistically justified.
- Randomly generated race outcomes are not constrained by an empirically calibrated finishing-order model.

These are audit findings to verify against the current branch before changing implementation. Preserve the original files as a baseline.

### 1.4 Data acquisition spike
Implement small, isolated experiments for:
- FastF1: one complete recent event with practice, qualifying, race, lap timing, stints and weather availability.
- Jolpica: season-level results, qualifying, standings, constructors, drivers and circuits.
- FIA: current technical/sporting regulation documents and official classifications where accessible.

For each provider record rate limits, schema stability, coverage gaps, terms/usage constraints, caching behaviour and a reproducible retrieval timestamp. Never scrape or infer unavailable telemetry.

### Stage 1 deliverables
1. Repository audit report.
2. Data coverage and missingness report.
3. Source-provider comparison.
4. Canonical data dictionary and schema proposal.
5. Small validated end-to-end ingestion proof of concept.
6. Prioritised defect register.

**Exit criteria:** one reproducible multi-team historical dataset sample with correct event/driver identifiers, source provenance, schema validation and documented gaps. No model training is required to exit Stage 1.

## Stage 2 — Canonical data and feature contracts

Define stable IDs for season, event, session, driver, constructor and circuit. Retain provider-native IDs.

Every row should have:
- `source_provider`
- `source_record_id`
- `retrieved_at_utc`
- `event/session timestamp` where applicable
- `schema_version`
- units and null semantics

Build point-in-time feature definitions. Features must be computed as-of a declared cutoff and tested so that no future row contributes to an earlier forecast.

**Exit criteria:** documented schemas, data-quality tests, reproducible feature build and leakage tests.

## Stage 3 — Exploratory analysis and baselines

Produce:
- Data coverage by season, team, driver and session.
- Ferrari teammate-relative qualifying and race-pace analysis.
- Constructor-relative pace trends.
- Circuit and weather splits where sample sizes permit.
- Reliability and DNF analysis with failure categories.
- Regulation-era comparisons with uncertainty.

Build simple baselines before advanced ML. All metrics must be computed using rolling-origin backtests.

**Exit criteria:** baseline report and agreed model target definitions.

## Stage 4 — Hybrid model development

Develop separate components rather than one monolithic model:
1. Hierarchical latent pace model for constructor and driver effects.
2. Gradient-boosted model for nonlinear residual structure, if it improves on the statistical baseline.
3. Reliability hazard model for retirement risk.
4. Probabilistic race-order / points model.
5. Calibrated uncertainty layer.

Use nested temporal validation for model selection. Keep feature preprocessing inside each training fold. Do not tune on the final holdout.

**Exit criteria:** documented out-of-sample improvement over baselines, or a decision to retain the baseline.

## Stage 5 — Simulator verification

- Implement current season points and Sprint scoring as versioned rules.
- Unit-test tie-breaking, classification, non-classified drivers and penalties.
- Add seeded simulation for deterministic test runs.
- Check that every simulated race and season obeys scoring invariants.
- Check Monte Carlo convergence by comparing estimates at increasing simulation counts.
- Run sensitivity analysis for uncertain 2027 regulation and team-performance assumptions.

**Exit criteria:** deterministic tests pass and simulation outputs are statistically stable at the chosen run count.

## Stage 6 — Preseason forecast release

Target freeze: 23 February 2027, before Bahrain testing scheduled for 24–27 February. Reconfirm official schedule before the freeze.

Freeze:
- raw-data manifest and retrieval cutoffs,
- feature schema and code revision,
- model artifacts and hyperparameters,
- regulations/calendar/roster versions,
- random seeds and simulation configuration.

Publish:
- Ferrari Constructors' Championship points and rank distributions,
- Leclerc and Hamilton driver points/rank distributions if the lineup is officially confirmed,
- race-by-race probabilities,
- assumptions, limitations and calibration evidence.

Do not revise the archived preseason forecast after seeing testing data. Create a separate post-testing forecast version if desired.

## Suggested issue breakdown

| ID | Task | Priority | Dependency |
|---|---|---|---|
| FF27-01 | Audit repository and dependency graph | P0 | None |
| FF27-02 | Trace existing 2026 collection and prediction pipeline | P0 | FF27-01 |
| FF27-03 | Audit raw CSV completeness and round identifiers | P0 | FF27-02 |
| FF27-04 | Evaluate FastF1/Jolpica/FIA data coverage | P0 | FF27-01 |
| FF27-05 | Define canonical event/driver/constructor IDs | P0 | FF27-04 |
| FF27-06 | Implement ingestion manifests and raw snapshots | P0 | FF27-05 |
| FF27-07 | Implement data validation and quality report | P0 | FF27-05 |
| FF27-08 | Build first all-team historical dataset sample | P0 | FF27-06, FF27-07 |
| FF27-09 | Define point-in-time feature contracts | P1 | FF27-08 |
| FF27-10 | Build rolling-origin baseline evaluation | P1 | FF27-09 |

## Scope guardrails

- Keep the initial release Ferrari-focused. Collect competitor data only to estimate relative competitiveness and championship context.
- Do not add an elaborate frontend before the data and model outputs are stable.
- Do not add orchestration, distributed compute or a database without evidence they are needed.
- Do not create synthetic data to fill real-world data gaps.
- Keep every published forecast immutable and traceable to a data/model version.
