# FahFerrari 2027 — Stage 0 Architecture

**Status:** Proposed architecture  
**Project:** FahFerrari 2027  
**Repository strategy:** Extend the existing FahFerrari repository  
**Forecast scope:** Ferrari-focused, with all-team data retained for competitive benchmarking  
**Modelling strategy:** Hybrid machine learning + Bayesian estimation + Monte Carlo simulation  
**Initial forecast:** Preseason forecast, frozen before 2027 Bahrain testing  
**Planning baseline date:** 7 November 2026

## 1. Product objective

Build a reproducible, probabilistic forecasting system that estimates Ferrari's 2027 Formula 1 performance at race, driver and constructors' championship levels.

The system is Ferrari-focused, not Ferrari-only in its data. Competitor observations are required to estimate Ferrari's relative pace and championship position. The system must not assume Ferrari wins, assign subjective team ratings, or manufacture rival points totals.

### Primary outputs

- Ferrari race-result and points distributions for every Grand Prix and Sprint.
- Charles Leclerc and Lewis Hamilton driver championship distributions, conditional on the confirmed 2027 driver lineup.
- Ferrari Constructors' Championship points distribution and finishing-position probabilities.
- Probabilities for wins, podiums, top-five finishes, points finishes and DNFs.
- Prediction intervals, calibration diagnostics, feature explanations and model/data provenance.
- Immutable forecast snapshots so preseason predictions can be evaluated separately from in-season updates.

### Non-goals for the first release

- Predicting every team's championship in equal depth.
- Live race strategy optimisation or driver-in-the-loop telemetry control.
- Computer-vision analysis of onboard footage.
- Treating subjective technical ratings as ground-truth model inputs.
- Presenting a single deterministic championship order as if it were certain.

## 2. 2027 season and regulation context

The official 2027 calendar comprises 24 Grands Prix across 22 countries and 10 Sprint events. The published calendar lists Bahrain testing for 24–27 February 2027 and the Bahrain Grand Prix for 12–14 March 2027.

The 2027 power-unit package rebalances ICE and MGU-K contribution. Published figures include 420 kW maximum ICE power, 300 kW maximum MGU-K power, and a 58:42 ICE/MGU-K split. The 2027 package also includes fuel-flow and energy-management changes. Race distance is reduced to 290 km for standard-distance events, and gearbox homologation is removed. The exact operative regulations and event-specific sporting rules must be versioned from FIA publications before implementation.

**Preseason forecast freeze:** target 23 February 2027, before the first scheduled Bahrain test day. This forecast may use historical data, confirmed regulations, official calendar information, confirmed driver/team information, and any explicitly timestamped public information available before the freeze. It must not use 2027 testing observations or race-weekend data.

The calendar, regulations, driver lineup, Sprint venues and technical assumptions are versioned inputs, not hardcoded truths. Re-check official FIA/F1 sources during Stage 1 and before the forecast freeze.

## 3. System architecture

```text
Official / historical sources
  ├── FastF1 session, lap, timing, tyre, pit-stop and weather data
  ├── Jolpica historical results, standings, drivers, constructors, circuits
  ├── FIA regulations, classifications, penalties and technical documents
  └── Curated event metadata and documented public sources
             |
             v
     Ingestion + source adapters
             |
             v
     Immutable raw data / provenance
             |
             v
     Validation, normalization, canonical IDs
             |
             v
     Feature store (point-in-time safe)
       ├── Ferrari team form and pace
       ├── Driver-relative performance
       ├── Circuit suitability
       ├── Reliability / failure history
       ├── Tyre and stint performance
       └── Competitor-relative pace
             |
             v
     Hybrid prediction layer
       ├── Bayesian latent team/driver pace
       ├── Gradient-boosted residual / outcome models
       ├── Reliability and DNF hazard model
       └── Probabilistic qualifying and race-order model
             |
             v
     Monte Carlo race + season simulator
             |
             v
     Forecast snapshots + evaluation
             |
             v
     Existing application/API and Ferrari dashboard
```

## 4. Modelling design

### 4.1 Latent team and driver performance

Use a hierarchical model to estimate latent car/team pace and driver contribution separately. The model should partially pool estimates across seasons, drivers and circuits, with explicit regime indicators for regulation eras.

A conceptual decomposition is:

`observed pace = team/car pace + driver effect + circuit fit + session conditions + residual noise`

This is an estimation framework, not a claim that these effects are perfectly identifiable. Teammate comparisons and field-wide data help separate driver and car effects.

### 4.2 Gradient-boosted models

Evaluate LightGBM and XGBoost as nonlinear residual or outcome models against simple baselines. Train only on features available at the forecast timestamp. Do not select an ensemble solely because it fits historical data well.

### 4.3 Reliability model

Estimate DNF / mechanical-failure risk with a survival or discrete-time hazard approach. Separate mechanical failures from collisions, penalties, disqualifications and other non-performance retirements where source data permits.

### 4.4 Probabilistic race model

Estimate qualifying pace and race pace distributions, then generate plausible finishing orders. Race simulation must respect grid, points rules, Sprint format, retirements and classification rules. Do not infer finishing positions directly from a season-points regression.

### 4.5 Monte Carlo season simulation

Run an initial 10,000 season simulations, with a configurable higher-count mode after runtime and convergence tests. Each simulation samples race outcomes and reliability, applies the season's actual points rules, and aggregates Ferrari's driver and constructor totals. Report distributions and calibrated probabilities, not only means.

### 4.6 Baselines

Every complex model must be compared with:
- Historical Ferrari points-per-event baseline, adjusted for calendar length and points rules.
- Recent-season weighted baseline.
- Teammate-relative pace baseline.
- Simple probabilistic ranking / regression baseline.

If the hybrid system does not beat a baseline on rolling-origin evaluation, the simpler model remains eligible for production.

## 5. Data strategy

### Coverage

Use all-team historical data where possible, targeting 2014–2026. Give the 2022–2026 regulation era greater relevance, but do not discard earlier data without testing. The 2026 season is still in progress as of the planning baseline date; mark it provisional until officially complete.

### Canonical datasets

| Dataset | Grain | Key fields |
|---|---|---|
| events | event | season, round, circuit_id, dates, format |
| sessions | session | event_id, session_type, start_time |
| entries | driver-event | driver_id, constructor_id, car_number |
| race_results | driver-event | grid, classified_position, points, status |
| qualifying_results | driver-event | Q1/Q2/Q3 times, position |
| sprint_results | driver-event | grid, position, points, status |
| lap_times | driver-session-lap | lap, time, position, pit/in-lap flags |
| stints | driver-session-stint | compound, start/end lap, tyre age |
| pit_stops | driver-event-stop | lap, duration |
| weather | session-time | air/track temperature, rain, wind where available |
| circuits | circuit | layout and stable circuit attributes |
| driver_seasons | driver-season | team, teammate, experience |
| regulations | season-version | technical/sporting rule version and effective date |
| source_manifest | source-record | provider, retrieved_at, source URL, checksum/version |

Preserve provider-native identifiers alongside canonical IDs. Raw source payloads should be immutable. Normalized tables must record ingestion time and source version.

### Data quality requirements

- Correct season and round values; never silently substitute zero for missing round metadata.
- Explicitly distinguish missing, zero, unclassified, retired and disqualified values.
- No backward-fill of future season information into earlier rows.
- Validate event counts, driver counts, duplicate keys, impossible lap times, points totals and classifications.
- Store units and timezone conventions explicitly.
- Treat missing technical telemetry as missing; do not fabricate it.
- Track data completeness by season, event, session and provider.

## 6. Point-in-time integrity and leakage prevention

Every feature must have an availability timestamp. A historical backtest for a given forecast cutoff can use only records published or knowable by that cutoff.

Examples:
- Preseason 2027 cannot use 2027 testing or race observations.
- A post-race update can use that event's completed sessions but not future events.
- Full-season aggregates cannot be used to predict an earlier round.
- Feature scalers, imputers, encoders and model selection must be fitted inside each training fold.

Use rolling-origin / expanding-window validation, not random train-test splitting.

## 7. Evaluation framework

Evaluate at multiple levels:

**Continuous targets:** MAE and RMSE for pace and points.

**Classification / probability targets:** log loss, Brier score, precision/recall where useful, and reliability diagrams / calibration error.

**Ranking targets:** rank correlation and top-k accuracy for qualifying and race classifications.

**Championship targets:** points error, finishing-position error, and probability calibration for Ferrari's P1–P10 outcomes.

Backtests should simulate the information cutoff realistically. Report metrics by regulation era, circuit type, wet/dry conditions, Sprint/non-Sprint weekends and driver where sample sizes support it. Keep a final untouched holdout period where feasible.

## 8. Repository strategy

Extend the existing repository without deleting or silently rewriting the 2026 implementation. Preserve current code and artifacts as a historical baseline. New modules should be introduced incrementally, with compatibility boundaries rather than a premature full rewrite.

Proposed additions:

```text
docs/
  2026-retrospective.md
  2027-architecture.md
  2027-data-dictionary.md
  2027-validation-plan.md

src/
  ingestion/
    fastf1_collector.py
    jolpica_collector.py
    fia_documents.py
  preprocessing/
    validation.py
    normalization.py
  features/
    driver_features.py
    team_features.py
    circuit_features.py
    race_features.py
  models/
    latent_pace.py
    pace_residual.py
    reliability.py
    race_outcome.py
  simulation/
    race_simulator.py
    season_simulator.py
  evaluation/
    backtesting.py
    metrics.py
    calibration.py
  pipeline/
    build_dataset.py
    train.py
    forecast.py

data/
  raw/
  staging/
  processed/
  predictions/
  manifests/

tests/
  unit/
  integration/
  data_quality/
```

The exact layout may be adapted to the repository's existing conventions after a dependency and code audit. Avoid introducing a database or orchestration platform until dataset size and refresh requirements justify it. Versioned Parquet datasets plus manifests are sufficient for the initial research pipeline.

## 9. Implementation phases and exit criteria

### Stage 0 — Architecture and project controls
- Agree scope, modelling principles, forecast freeze and non-goals.
- Document data contracts, leakage policy, validation metrics and repository boundaries.
- Record unresolved risks and assumptions.

**Exit:** architecture reviewed; Stage 1 tasks are estimable.

### Stage 1 — Repository audit and data foundation
- Audit existing dependencies, scripts, data files and frontend/API contracts.
- Reconcile existing CSVs against provider data; investigate the current zero-round and fixed-calendar assumptions.
- Build source adapters, canonical identifiers, manifests and validation tests.

**Exit:** reproducible all-team historical dataset with quality report and provenance.

### Stage 2 — Exploratory analysis and feature definitions
- Quantify data coverage and missingness.
- Define driver-relative, team-relative, circuit and reliability features.
- Add point-in-time availability metadata.

**Exit:** versioned feature dictionary and leakage tests.

### Stage 3 — Baselines and hybrid models
- Implement baselines first.
- Fit hierarchical pace and reliability components.
- Evaluate boosted models as incremental improvements.

**Exit:** rolling-origin report with calibration and baseline comparisons.

### Stage 4 — Race and season simulation
- Implement race outcome sampling and correct points rules.
- Validate deterministic replay with fixed random seeds.
- Test Monte Carlo convergence and output invariants.

**Exit:** repeatable season distributions with tested scoring.

### Stage 5 — Preseason forecast
- Freeze eligible data and model artifacts before 2027 testing.
- Publish Ferrari driver and constructor distributions with assumptions and limitations.
- Store a signed/versioned forecast snapshot.

**Exit:** reproducible preseason report and archived prediction.

### Stage 6 — In-season updates and retrospective
- Update only after new event data is available.
- Preserve all prior forecasts.
- Compare predicted distributions with actual outcomes.

**Exit:** race-by-race evaluation history and end-of-season retrospective.

## 10. Engineering standards

- Python type hints and documented public interfaces.
- Deterministic random seeds for reproducible simulation.
- Unit tests for feature calculations, points rules and simulator invariants.
- Data contracts and schema versioning.
- Model artifacts include training cutoff, data manifest hash, feature schema, hyperparameters and metrics.
- No credentials or private API keys committed to Git.
- CI runs linting, tests and lightweight data-contract checks.
- Dashboard consumes prediction API contracts; it does not calculate model predictions in the browser.

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Small effective sample size | Hierarchical partial pooling; conservative model complexity |
| 2026–27 regulation transition | Regime-aware features and explicit uncertainty |
| Data provider gaps / schema changes | Adapter boundaries, manifests and validation |
| Driver/team effects confounded | Teammate comparisons and hierarchical estimates |
| Data leakage | Availability timestamps and rolling-origin folds |
| Overconfident championship probabilities | Calibration and proper scoring rules |
| Manual subjective assumptions | Documented priors with sensitivity analysis |
| Late driver/team changes | Versioned roster inputs and scenario reruns |

## 12. Stage 0 decisions recorded

- Scope: Ferrari-focused forecast; all-team data used for relative competitiveness.
- Modelling: hybrid Bayesian + machine learning + Monte Carlo.
- Repository: extend the existing FahFerrari repository; preserve 2026 baseline.
- First forecast: preseason, frozen before 2027 Bahrain testing.
- Forecast target: 2027 race-by-race Ferrari performance, Ferrari driver outcomes and Constructors' Championship distribution.
- Initial simulation target: 10,000 seasons, increased after convergence checks.

## 13. Official reference sources

- FIA 2027 calendar announcement: https://www.fia.com/news/2027-fia-formula-one-world-championship-calendar-announced
- Official 2027 F1 calendar: https://www.formula1.com/en/racing/2027
- FIA 2027 technical regulations index: https://www.fia.com/regulation/category/110/technical-regulations
- FIA / F1 2027–2028 regulatory changes: https://www.fia.com/news/agreement-reached-proposed-regulatory-changes-2027-and-2028
- FastF1 documentation: https://docs.fastf1.dev/
- Jolpica API: https://api.jolpi.ca/ergast/f1/

Regulatory and calendar facts must be re-checked against the latest official documents before the preseason forecast freeze.
