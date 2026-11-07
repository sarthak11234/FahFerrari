# FahFerrari 2026 Retrospective & Stage 1 Audit Report

**Tasks covered:** `FF27-01`, `FF27-02`, `FF27-03`  
**Parent documents:** [2027-architecture.md](./2027-architecture.md), [2027-implementation-plan.md](./2027-implementation-plan.md)  
**Audit date:** 7 November 2026  
**Baseline commit audited:** `ae91b36` (`docs/fahferrari-2027-stage-0`, preserving 2026 implementation from `a978f59`)

---

## 1. Repository & Dependency Audit (`FF27-01`)

### 1.1 Directory & Module Inventory

| Path | Role | Status / Audit Notes |
|---|---|---|
| `README.md` | Root README | Humorous 2026 project overview; paths assume running inside `ferrari-f1-predictor/`. |
| `package.json`, `package-lock.json` | Root Node manifest | Accidental duplicate dependency manifest (`axios`, `lucide-react`, `recharts` with no `scripts` or `react`). Actual frontend lives in `ferrari-f1-predictor/frontend/`. |
| `mechatroner.rainbow-csv-3.3.0.vsix` | VS Code extension binary | 744 KB binary committed to repository root in `a978f59`; unrelated to runtime. |
| `ferrari-f1-predictor/requirements.txt` | Python dependencies | Targets Python 3.12+; uses `>=` floor pins rather than locked versions. Includes unused packages (`catboost`, `streamlit`, `beautifulsoup4`, `plotly`, `seaborn`). |
| `ferrari-f1-predictor/src/data/data_collection.py` | Raw data ingestion | `FerrariDataCollector` using `fastf1.ergast.Ergast`. Writes `data/raw/race_results_2019_2025.csv`. |
| `ferrari-f1-predictor/src/data/feature_engineering.py` | Feature pipeline | `FerrariFeatureEngineer`. Joins `race_results_2019_2025.csv` with `technical_scores.csv` and writes `data/processed/ferrari_features.csv`. |
| `ferrari-f1-predictor/src/models/train.py` | Model training | `FerrariMLTrainer`. Fits `StandardScaler` + 4 regressors (`random_forest`, `xgboost`, `lightgbm`, `gradient_boosting`) and `FerrariEnsemble`. Saves `.pkl` files to `models/saved/`. |
| `ferrari-f1-predictor/src/models/predict.py` | 2026 prediction engine | `FerrariEnsemble` and `Ferrari2026Predictor`. Loads `.pkl` models, mutates 2025 row for 2026, synthesizes race/driver/constructor outputs. |
| `ferrari-f1-predictor/src/chatbot/query_engine.py` | Rule-based chatbot | `FerrariChatbot`. Keyword/regex intent router over `ferrari_features.csv`, 2026 predictions, and hardcoded strings. |
| `ferrari-f1-predictor/api/app.py` | Flask REST API | Serves 8 endpoints on port `5000`; computes `predictions_2026` once at import/startup time. |
| `ferrari-f1-predictor/frontend/` | React + Vite SPA | React 18 + Vite 5 + Recharts dashboard (`Dashboard.jsx`, `PredictionView.jsx`, `ChatBot.jsx`). |
| `ferrari-f1-predictor/tests/` | Pytest suite | `test_api.py` (4 mocked route tests) and `test_models.py` (4 toy-dataframe tests). |
| `ferrari-f1-predictor/{debug_fastf1,reproduce_issue,verify_chat,verify_pipeline,check_api,run_tests}.py` | Ad-hoc root scripts | Diagnostic and verification scripts created during `a978f59` debugging. |
| `ferrari-f1-predictor/cache/fastf1_http_cache.sqlite` | Requests-cache SQLite DB | 4.17 MB committed HTTP cache containing 299 cached responses from `2026-01-08`. |

### 1.2 Dependency & Runtime Findings
1. **Unpinned Python dependencies:** `ferrari-f1-predictor/requirements.txt` uses minimum version specifiers (`>=`) without a lockfile. Because `models/saved/*.pkl` are serialized via `joblib`, version drift in `scikit-learn`, `xgboost`, or `lightgbm` can break deserialization or alter predictions.
2. **Missing Bayesian / statistical packages for 2027:** `scipy` is imported directly in `src/models/predict.py` (line 9: `from scipy import stats`) and `pyarrow` is required for Parquet storage in the 2027 architecture, but neither `scipy` nor `pyarrow` is declared as a direct dependency in `requirements.txt`.
3. **Working directory sensitivity:** Every module (`data_collection.py`, `feature_engineering.py`, `train.py`, `predict.py`, `api/app.py`) uses relative file paths (`data/raw/...`, `data/processed/...`, `models/saved/...`, `cache`). Running any command from the repository root instead of `ferrari-f1-predictor/` fails with `FileNotFoundError`.

---

## 2. 2026 Pipeline Trace & Public Interfaces (`FF27-02`)

### 2.1 End-to-End 2026 Data Flow

```text
[Jolpica Ergast API via FastF1]
       │  GET /ergast/f1/{year}/results.json?limit=1000
       ▼
src/data/data_collection.py (FerrariDataCollector)
       │  Filters constructorId == 'ferrari'
       │  Drops driver, circuit, date, sprint; sets round = 0
       ▼
data/raw/race_results_2019_2025.csv (70 rows: 10 rows/year × 7 years)
       │
       ├──► Joined on `year` with data/raw/technical_scores.csv (7 rows, manual scores)
       ▼
src/data/feature_engineering.py (FerrariFeatureEngineer)
       │  Aggregates to 1 row per year (7 rows × 34 columns)
       │  Applies fixed /24 and /48 divisors + .fillna(method='bfill')
       ▼
data/processed/ferrari_features.csv (7 rows: 2019–2025)
       │
       ▼
src/models/train.py (FerrariMLTrainer)
       │  train_test_split(test_size=0.2, shuffle=False) -> 5 train rows (2019–2023), 2 test rows (2024–2025)
       │  3-fold CV on 5 rows (1–2 samples per fold)
       │  Saves scaler.pkl + 4 regressors + ensemble_model.pkl
       ▼
src/models/predict.py (Ferrari2026Predictor)
       │  Takes 2025 row, overwrites 4 technical scores with constants (86, 75, 78, 95)
       │  Predicts scalar `points_pred`, derives CI from std across 4 models
       │  Synthesizes 24 races (unseeded np.random), 52/48 driver split, and hardcoded rival multipliers
       ▼
api/app.py (Flask) ──► frontend/src/App.jsx (React UI on http://localhost:5173)
```

### 2.2 Backward-Compatible Public Interfaces
To preserve the existing application while introducing 2027 modules:
- **Flask REST Routes (`api/app.py`):**
  - `GET /api/health` → `{status, message, models_loaded}`
  - `GET /api/historical` → array of yearly feature records (`year`, `total_points`, `wins`, `podiums`, ...)
  - `GET /api/historical/<int:year>` → single yearly record
  - `GET /api/predictions/2026` → `{year, predicted_points, confidence_interval, predicted_position, predicted_wins, predicted_podiums, individual_predictions, prediction_std, confidence_level, race_predictions, driver_standings, constructors_standings}`
  - `GET /api/predictions/scenarios` → `{expected, optimistic, pessimistic}` prediction objects
  - `POST /api/chat` → `{query, response, timestamp}`
  - `GET /api/drivers` → driver summary object (currently hardcoded 2025 stats in `api/app.py:152-171`)
  - `POST /api/compare` → array of yearly records for requested `years`
  - `GET /api/stats/summary` → `{total_seasons, years_covered, total_points, total_wins, total_podiums, best_season, worst_season}`
- **Frontend UI Expectations (`frontend/src/components/`):**
  - `Dashboard.jsx` reads `data[].year`, `data[].total_points`, `data[].wins`, `data[].podiums`.
  - `PredictionView.jsx` reads `predicted_points`, `confidence_interval[0..1]`, `predicted_position`, `predicted_wins`, `predicted_podiums`, `individual_predictions`, `driver_standings.{leclerc,hamilton}.{points,wins,podiums}`, `constructors_standings[].{position,team,points}`, and `race_predictions[].{round,circuit,predicted_points,race_position,performance_score}`.

---

## 3. Forensic Audit of Committed Data (`FF27-03`)

By inspecting the committed CSVs alongside the 299 cached HTTP responses inside `ferrari-f1-predictor/cache/fastf1_http_cache.sqlite` (captured on `2026-01-08T04:52:51Z` – `04:58:06Z`), we resolved every data anomaly listed in §1.2 of the implementation plan:

### 3.1 Why `race_results_2019_2025.csv` has only 10 rows per year (70 rows total)
- **Root cause (Server-side pagination cap):** `FerrariDataCollector.collect_historical_data()` (`src/data/data_collection.py:42`) calls:
  ```python
  response = self.ergast.get_race_results(season=year, limit=1000)
  ```
  Inspection of the cached HTTP payloads from `https://api.jolpi.ca/ergast/f1/{year}/results.json?limit=1000` inside `fastf1_http_cache.sqlite` proves that **Jolpica enforces a maximum page size of `limit=100` driver-race results per request** (`MRData.limit = "100"`, `MRData.offset = "0"`), silently ignoring `limit=1000` above `100`:
  - **2019:** `total=420`, returned 100 rows = Rounds 1–5 (10 Ferrari rows)
  - **2020:** `total=340`, returned 100 rows = Rounds 1–5 (10 Ferrari rows)
  - **2021:** `total=440`, returned 100 rows = Rounds 1–5 (10 Ferrari rows)
  - **2022:** `total=440`, returned 100 rows = Rounds 1–5 (10 Ferrari rows)
  - **2023:** `total=440`, returned 100 rows = Rounds 1–5 (10 Ferrari rows)
  - **2024:** `total=479`, returned 100 rows = Rounds 1–5 plus P1 of Round 6 (because Round 3 Australia had 19 entries after Sargeant withdrew: `20+20+19+20+20+1 = 100` rows; Ferrari finished P3/P4 in Round 6 so Round 6 Ferrari rows were at indices 102–103 and truncated, leaving 10 Ferrari rows)
  - **2025:** `total=479`, returned 100 rows = Rounds 1–5 (10 Ferrari rows)
- Because `data_collection.py` never checked `response.is_complete` or iterated through `response.get_next_result_page()`, **75%–80% of every season (Rounds 6–24) was silently dropped.**
- Consequently, `ferrari_features.csv` claims Ferrari scored only `121.0` points in 2019 (actual: `504`), `142.0` points in 2024 (actual: `652`), and `66.0` points in 2025 (actual: `398`).

### 3.2 Why `round` is `0` for all 70 rows in `race_results_2019_2025.csv`
- **Root cause (`ErgastMultiResponse` structure mismatch):** In FastF1's `Ergast.get_race_results()`, race-level metadata (`season`, `round`, `raceName`, `date`, `circuitId`) is stored in `response.description` (one row per race), whereas `response.content` is a list of per-race DataFrames containing **only** the inner `Results` fields (`number`, `position`, `positionText`, `points`, `grid`, `laps`, `status`, `driverId`, `constructorId`, etc.).
- In `src/data/data_collection.py:48-68`, the loop iterates over `for race_df in response.content:` without zipping `response.description`, and evaluates:
  ```python
  'round': int(row.get('round', 0))
  ```
  Because `'round'` is not a column in `race_df`, `row.get('round', 0)` falls back to `0` on every row.

### 3.3 Empty `standings_2019_2025.csv` (1 byte)
- In commit `5b86cba`, `F1DataCollector` attempted to query `http://ergast.com/api/f1` (deprecated/offline), writing empty 1-byte files for both `race_results_2019_2025.csv` and `standings_2019_2025.csv`.
- When `data_collection.py` was rewritten in `a978f59` to use `fastf1.ergast`, standings collection was omitted entirely. `standings_2019_2025.csv` remains an empty 1-byte file, and constructor championship `position` is completely absent from `ferrari_features.csv`.

### 3.4 Missing Entities, Sprint Points, Disqualifications, and Status Semantics
1. **Ferrari-only filtering & stripped identifiers:** `data_collection.py:63-73` filters `race_df[race_df['constructorId'] == 'ferrari']` and drops `driverId`, `constructorId`, `circuitId`, `raceName`, and `date`. Neither teammate comparisons (Leclerc vs. Vettel/Sainz/Hamilton, or Bearman's 2024 Jeddah appearance) nor competitor pace can be computed from the exported CSV.
2. **Sprint points omitted:** `ergast.get_race_results()` only returns Grand Prix results. Sprint results (2021–2025, e.g., Hamilton's 2025 Chinese GP Sprint win) require the separate `/sprint.json` endpoint (`ergast.get_sprint_results()`) and are completely missing.
3. **Fastest-lap points:** Included inside Grand Prix `points` returned by Jolpica for 2019–2024 (e.g., 26 points for Leclerc in 2022 Bahrain and Australia), and abolished by FIA sporting regulations from 2025 onwards, but not tracked separately from base finishing points.
4. **DNF miscounting (`status != 'Finished'`):** `feature_engineering.py:42` computes:
   ```python
   yearly['dnfs'] = df[df['status'] != 'Finished'].groupby('year').size()
   ```
   In `race_results_2019_2025.csv`, this falsely counts **lapped finishes** (`+1 Lap` for Vettel P6 and Leclerc P11 in 2020 Round 3 Hungary) and **post-race disqualifications** (`Disqualified` for Leclerc P18 and Hamilton P19 in 2025 Round 2 China) as mechanical/crash DNFs, inflating `dnfs` and `dnf_rate`.
5. **2026 season completeness:** The committed raw data stops at 2025 (first 5 rounds). No 2026 events are ingested yet; 2026 must be ingested with explicit provisional/completeness flags per snapshot.

---

## 4. Verification of Known Feature-Engineering & Modelling Defects (§1.3)

Every defect listed in §1.3 of [2027-implementation-plan.md](./2027-implementation-plan.md) was verified against the current codebase:

| # | Defect | Code Location | Verification & Impact |
|---|---|---|---|
| 1 | Fixed divisors of `24` races and `48` entries | `src/data/feature_engineering.py:99-102` | `points_per_race = total_points / 24`, `podium_rate = podiums / 48`, `dnf_rate = dnfs / 48`, `win_rate = wins / 24`. Actual season lengths were 21 (2019), 17 (2020), 22 (2021), 22 (2022), 22 (2023), 24 (2024), 24 (2025)—and the input CSV only contained 5 races per season. |
| 2 | Backward-fill temporal leakage | `src/data/feature_engineering.py:135` | `df = df.fillna(method='bfill').fillna(0)` backfills `2020`'s `points_diff = -66.0` and `points_pct_change = -0.545` into `2019` (see `ferrari_features.csv` row 2), and backfills `2019` values into `points_lag1` / `points_lag2` for `2019` and `2020`. |
| 3 | Season-level aggregation | `src/data/feature_engineering.py:26-46` | Collapses all rows into `N = 7` annual rows (`2019–2025`) with `32` features (`p >> N`). Discards driver, circuit, qualifying, weather, tyre stint, and session variation. |
| 4 | Subjective technical scores & hardcoded 2026 overrides | `data/raw/technical_scores.csv`, `src/models/predict.py:42-45` | `technical_scores.csv` contains hand-authored 0–100 numbers with no measurement source. `prepare_2026_features()` overwrites 2025 values with hardcoded constants (`reliability_score=86`, `development_pace=75`, `strategy_effectiveness=78`, `driver_skill=95`). |
| 5 | Hand-authored splits, rival multipliers & track multipliers | `src/models/predict.py:93-95, 140-145, 188-236` | Hardcodes `52% / 48%` Leclerc/Hamilton split, `1.15x` Red Bull / `1.05x` McLaren / `0.85x` Mercedes points multipliers, and `1.5x` (Monaco/Monza/Singapore) vs `0.8x` (Britain/Belgium) track multipliers. |
| 6 | Uncalibrated model-disagreement confidence interval | `src/models/predict.py:87-90` | Uses `stats.norm.interval(0.95, loc=points_pred, scale=np.std(individual_preds))` across 4 trees fit on 5 rows. Model disagreement is not predictive uncertainty. |
| 7 | Unseeded, unconstrained race outcome generation | `src/models/predict.py:124-186` | `simulate_race_by_race()` uses unseeded `np.random.normal(1, 0.2)` and `np.random.random()` (producing different predictions on every API restart), caps single-car points at `26` (even though `avg_points` is a *two-car constructor* average where max is 43–44 or 58–59 on Sprint weekends), and maps constructor points to a single-driver finishing position (`P1`–`>P10`). |

---

## 5. Prioritised Defect Register (Stage 1 Deliverable #6)

| Defect ID | Severity | Component | Summary | Target Resolution Task |
|---|---|---|---|---|
| **DEF-01** | **P0 (Blocker)** | Ingestion (`data_collection.py:42`) | Jolpica `limit=100` server-side cap without pagination truncates 2019–2025 data to the first 5 rounds per season (10 Ferrari rows/year). | `FF27-04`, `FF27-06`, `FF27-08` |
| **DEF-02** | **P0 (Blocker)** | Ingestion (`data_collection.py:48-68`) | Iterating `response.content` without `response.description` drops race metadata and sets `round = 0` on every row. | `FF27-05`, `FF27-06` |
| **DEF-03** | **P0 (Blocker)** | Ingestion (`data_collection.py:63-73`) | Filtering `constructorId == 'ferrari'` and dropping `driverId`, `circuitId`, `date`, qualifying, and Sprint results prevents driver-relative and competitor-relative modelling. | `FF27-05`, `FF27-06`, `FF27-08` |
| **DEF-04** | **P0 (Blocker)** | Preprocessing (`feature_engineering.py:42`) | `status != 'Finished'` misclassifies lapped finishes (`+1 Lap`) and disqualifications (`Disqualified`) as mechanical/crash DNFs. | `FF27-05`, `FF27-07` |
| **DEF-05** | **P0 (Blocker)** | Features (`feature_engineering.py:135`) | `.fillna(method='bfill')` leaks future-season diffs and pct changes backward into earlier seasons. | `FF27-07`, `FF27-09` |
| **DEF-06** | **P1 (High)** | Features (`feature_engineering.py:99-102`) | Hardcoded `/24` and `/48` divisors distort rates across seasons with different event/Sprint counts. | `FF27-09` |
| **DEF-07** | **P1 (High)** | Features (`technical_scores.csv`, `predict.py:42-45`) | Unverified subjective technical ratings and hardcoded 2026 overrides dominate composite indices. | `FF27-09`, `FF27-10` |
| **DEF-08** | **P1 (High)** | Training (`train.py:30-139`) | Training 4 tree ensembles with `32` features on `N = 5` annual rows (`cv=3`) is severely overparameterized and statistically invalid. | `FF27-10`, Stage 3/4 |
| **DEF-09** | **P1 (High)** | Prediction (`predict.py:87-90`) | Confidence interval derived from standard deviation across 4 overfit tree models rather than calibrated predictive distribution. | Stage 3/4 |
| **DEF-10** | **P1 (High)** | Simulation (`predict.py:124-236`) | Unseeded random race generator conflates 2-car constructor points with 1-car finishing positions (`cap=26`) and uses hardcoded driver (`52/48`) and constructor (`1.15x`, `1.05x`) multipliers. | Stage 4/5 |
| **DEF-11** | **P2 (Medium)** | Environment (`requirements.txt`, root `package.json`) | Unpinned dependencies, missing direct `scipy`/`pyarrow` declarations, CWD-coupled relative paths, and stray root `package.json` / `.vsix` artifact. | `FF27-01`, `FF27-06` |

