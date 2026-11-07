# FahFerrari 2027 — Validation Plan & Baseline Benchmarks

**Tasks covered:** `FF27-09` (Point-in-Time Feature Contracts), `FF27-10` (Rolling-Origin Baseline Evaluation)  
**Parent documents:** [2027-architecture.md](./2027-architecture.md), [2027-implementation-plan.md](./2027-implementation-plan.md), [2027-data-dictionary.md](./2027-data-dictionary.md)  
**Status:** Completed Baseline Evaluation & Decision Gate

---

## 1. Objectives & Non-Negotiable Validation Principles

In accordance with the guiding rule of the 2027 architecture:
> **Do not start by training a more complicated model. First establish a trustworthy, point-in-time-correct dataset and a reproducible baseline. A complex model is promoted only if it improves out-of-sample performance and calibration.**

The validation framework guarantees:
1. **Zero Temporal Leakage:** Every feature for an event $E$ at timestamp $t_E$ is derived strictly from historical events with $t < t_E$. Future-season or post-event data cannot alter earlier feature vectors.
2. **Expanding-Window (Rolling-Origin) Evaluation:** No random train/test splitting across time. Models are fitted on historical folds and evaluated strictly on out-of-sample forward horizons.
3. **Multi-Faceted Metrics:** Continuous error (MAE, RMSE), ranking ability (Spearman rank correlation, Top-3 overlap), and calibrated uncertainty (Brier score, Expected Calibration Error).

---

## 2. Rolling-Origin Backtesting Architecture

### 2.1 Fold Structure (Expanding Window)

| Fold | Training Seasons | Test Season | Test Events | Test Entries | Information Cutoff ($t_{cutoff}$) |
|---|---|---|---|---|---|
| **Fold 1** | 2019, 2020 (58 events) | **2021** | 22 events | 440 entries | Pre-2021 Season ($<$ 2021-03-26) |
| **Fold 2** | 2019–2021 (80 events) | **2022** | 12 events | 240 entries | Pre-2022 Season ($<$ 2022-03-18) |
| **Fold 3** | 2019–2022 (92 events) | **2023** | 5 events | 100 entries | Pre-2023 Season ($<$ 2023-03-03) |
| **Fold 4** | 2019–2023 (97 events) | **2024** | 5 events | 99 entries | Pre-2024 Season ($<$ 2024-02-29) |
| **Fold 5** | 2019–2024 (102 events) | **2025** | 5 events | 100 entries | Pre-2025 Season ($<$ 2025-03-14) |

Each test fold is evaluated event-by-event, updating rolling event-level EWMA features within the test season as each race finishes, simulating actual point-in-time forecasting.

---

## 3. Four Baseline Models Evaluated

Four benchmark baselines defined in Section 4.6 of `2027-architecture.md` were implemented in `src/evaluation/backtesting.py`:

1. **Baseline 1 — Historical Team Mean:**
   Estimates constructor points per entry using the historical unweighted mean across all prior seasons.
2. **Baseline 2 — Recent-Season Weighted:**
   Estimates points per entry using 3-season exponential decay weighting (`50%` $t-1$, `30%` $t-2$, `20%` $t-3$).
3. **Baseline 3 — Teammate-Relative Pace:**
   Combines recent constructor finish position EWMA with the driver's historical intra-team finish delta.
4. **Baseline 4 — Probabilistic Ranking:**
   Synthesizes a latent pace metric ($50\%$ team form, $30\%$ driver form, $20\%$ circuit history), determines expected finishing ranks, and derives win and podium probabilities via Gaussian CDF uncertainty integration.

---

## 4. Baseline Evaluation Benchmarks

The expanding-window evaluation across all 5 test folds produced the following out-of-sample results (persisted in `data/manifests/baseline_evaluation_report.json`):

| Evaluation Metric | Baseline 1 (Historical Mean) | Baseline 2 (Recent Weighted) | Baseline 3 (Teammate Relative) | Baseline 4 (Probabilistic Ranking) | Optimal Direction |
|---|---|---|---|---|---|
| **Points MAE** | 3.952 | 3.883 | 3.903 | **3.719** | Lower |
| **Points RMSE** | 5.774 | **5.736** | 6.532 | 6.297 | Lower |
| **Finishing Order MAE** | 3.816 | **3.720** | 4.037 | 3.935 | Lower |
| **Spearman Rank Corr ($\rho$)** | 0.596 | **0.598** | 0.554 | 0.565 | Higher |
| **Top-3 Overlap Accuracy** | 56.5% | **60.5%** | 48.3% | 54.4% | Higher |
| **Brier Score (Win)** | 0.0867 | 0.0844 | 0.0488 | **0.0435** | Lower |
| **Brier Score (Podium)** | 0.1429 | 0.1372 | 0.1242 | **0.1024** | Lower |
| **Brier Score (DNF)** | **0.1243** | **0.1243** | 0.1245 | 0.1292 | Lower |
| **ECE (Win Probability)** | 0.1537 | 0.1538 | **0.0260** | 0.0302 | Lower |
| **ECE (Podium Probability)** | 0.1771 | 0.1742 | **0.0681** | 0.0765 | Lower |

### Key Benchmark Takeaways
- **Baseline 2 (Recent Weighted)** achieved the strongest deterministic ranking accuracy ($\rho = 0.598$, Top-3 overlap $60.5\%$, finish order MAE $3.72$).
- **Baseline 4 (Probabilistic Ranking)** achieved the lowest overall points error (**MAE 3.719**) and superior probability scores (Win Brier **0.0435**, Podium Brier **0.1024**).
- **Baselines 3 & 4** drastically outperformed Baselines 1 & 2 in probability calibration (Win ECE reduced from $\sim 0.154$ down to $\sim 0.026 - 0.030$), eliminating the severe overconfidence typical of uncalibrated point estimates.

---

## 5. Model Promotion Decision Gates (Stage 4 Exit Criteria)

Any complex model developed in **Stage 4** (Hierarchical Bayesian Pace Model, Gradient-Boosted Residual Model, or Reliability Hazard Model) must satisfy the following criteria on the rolling-origin backtest before being promoted to production:

1. **Points Prediction Gate:** Points MAE $< 3.719$ (beating Baseline 4).
2. **Ranking Gate:** Spearman rank correlation $\rho > 0.598$ and Top-3 accuracy $> 60.5\%$ (beating Baseline 2).
3. **Probabilistic Calibration Gate:** Win Brier score $< 0.0435$ and Win Expected Calibration Error (ECE) $< 0.030$.
4. **Reliability Gate:** Out-of-sample DNF Brier score $< 0.1243$.

If a candidate ML model fails any of these criteria, the corresponding baseline model will remain the production forecasting engine.

