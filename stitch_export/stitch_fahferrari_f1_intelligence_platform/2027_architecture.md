# FahFerrari 2027 — Architecture
**Status:** Architecture brief reconstructed from the approved project context. Verify assumptions against the live repository and backend before implementation.

## 1. Product objective
Build a scientifically defensible, Ferrari-focused forecast for the 2027 Formula 1 season. Ferrari is the primary subject; all teams and drivers are modelled to provide competitive benchmarking. Preserve the 2026 implementation as a historical baseline and retrospective.

## 2. Forecast strategy
- Hybrid approach: machine learning + Bayesian statistics + Monte Carlo simulation.
- First public forecast: preseason forecast frozen before Bahrain testing.
- Preserve immutable forecast snapshots, including data cutoff and model provenance.
- Treat all preseason predictions as probabilistic, not deterministic claims.

## 3. Proposed modelling architecture
1. Data ingestion and validation.
2. Canonical entity mapping for teams, drivers, circuits, sessions, and seasons.
3. Feature engineering with point-in-time integrity.
4. Bayesian hierarchical latent team/driver pace estimation.
5. Gradient-boosting model for nonlinear residual effects.
6. Reliability / DNF hazard model.
7. Probabilistic race-order and points model.
8. Monte Carlo season simulator.
9. Forecast aggregation, calibration, and provenance output.

The exact model family, features, and training procedure remain subject to data audit and backtesting.

## 4. Data strategy
Retain race/session/circuit-level granularity wherever possible. Use stable canonical IDs and document provider/source attribution. Track:
- source and retrieval timestamp;
- event/session timestamp;
- publication/availability timestamp where relevant;
- schema and dataset version;
- missingness and validation status.

Never use information unavailable at the forecast cutoff. Avoid backward filling or any transformation that leaks future information into historical predictions.

## 5. Point-in-time integrity
Every training and evaluation feature must be computable using only information available at the prediction timestamp. Historical backtests must reproduce the information set that would actually have been available then. Forecast snapshots must record the cutoff timestamp and source manifest.

## 6. Simulation
Initial planning baseline: 10,000 season simulations, with larger runs only after convergence and stability checks. Record random seed, simulation count, calendar assumptions, scoring assumptions, model version, and regulation assumptions. Validate Monte Carlo convergence rather than treating a large simulation count as proof of accuracy.

## 7. Forecast outputs
Potential outputs include:
- constructors' and drivers' expected and median points;
- finishing-position distributions;
- title and top-three probabilities;
- race-level outcome distributions;
- reliability/DNF risk;
- uncertainty intervals;
- provenance and model metadata.

Do not display a statistic unless the backend supplies it or it can be correctly derived from the underlying paired simulation samples.

## 8. Evaluation
Use chronological / rolling-origin backtesting rather than random train-test splits. Evaluate:
- point error for points forecasts;
- probabilistic calibration;
- Brier score for binary events where applicable;
- log loss for probabilistic outcomes where applicable;
- interval coverage and sharpness;
- comparison against simple baselines.

Compare forecasts with actual outcomes only after the relevant events have occurred.

## 9. Frontend architecture
Existing frontend context: React 18, Vite, Axios, Recharts, and Lucide React. Retain the existing stack initially and evolve incrementally. Do not replace the application architecture merely to match a generated design.

Frontend pages: Overview, Championship, Ferrari, Race Calendar, Simulation Lab, Model Intelligence, AI Strategist, 2026 Retrospective.

The frontend must use backend data and provenance. Include loading, error, stale, partial, and missing-data states.

## 10. API integration assumptions
Previously discussed endpoints include:
- `GET /api/predictions/2027`
- `GET /api/predictions/2027/provenance`
- `GET /api/predictions/2026`
- `GET /api/historical`
- `POST /api/chat`

These are integration references, not verified contracts. Inspect the running backend, schemas, and actual payloads before implementing. Do not hardcode sample values.

## 11. Known risks to audit
Previously identified issues in the 2026 implementation:
- Historical CSV sample appeared incomplete (10 rows per year) and used `round=0`.
- Fixed 24-race and 48 driver-race assumptions.
- Backward filling could introduce future leakage.
- Season aggregation discarded race/session/circuit granularity.
- Manually assigned technical scores.
- Hardcoded driver split and rival-points multipliers.
- Disagreement among four tree models was treated as confidence intervals.
- Random race outcomes were not empirically calibrated.

Treat these as audit findings to verify, not automatically as current defects.

## 12. Release and governance
- Keep the 2026 implementation intact.
- Develop 2027 changes on a separate development branch until reviewed.
- Record model/data versions and immutable forecast snapshots.
- Clearly label forecast versus actual data.
- Publish methodology, limitations, and data-source attribution.
