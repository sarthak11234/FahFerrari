# FahFerrari 2027 — Official Preseason Forecast Release

**Status:** Frozen Preseason Forecast  
**Forecast Freeze Timestamp:** `2026-11-07T15:33:40Z` (Prior to 2027 Bahrain Testing)  
**Parent Architecture:** [2027-architecture.md](./2027-architecture.md)  
**Validation Plan & Baselines:** [2027-validation-plan.md](./2027-validation-plan.md)  
**Data Dictionary:** [2027-data-dictionary.md](./2027-data-dictionary.md)  
**Audit Baseline & Retrospective:** [2026-retrospective.md](./2026-retrospective.md)  
**Simulation Mode:** 10,000 Full-Season Monte Carlo Runs (Random Seed `2027`)  
**Artifact Path:** `data/predictions/preseason_2027_forecast.json`

---

## 1. Executive Summary: Scuderia Ferrari 2027 Outlook

Under the 2027 technical regulations (rebalanced 420 kW ICE / 300 kW MGU-K power units, 290 km standard race distance, 24 Grands Prix, and 10 Sprint events), Scuderia Ferrari is projected to finish **P3 in the Constructors' Championship with an expected 488.8 points** ($90\%$ credible interval: $[403.0, 577.0]$).

| Output Dimension | Expected / Mean | Median ($p_{50}$) | 80% Credible Interval | 90% Credible Interval | Probability |
|---|---|---|---|---|---|
| **Constructors' Points** | **488.8 pts** | 488.0 pts | $[421.0, 558.0]$ | $[403.0, 577.0]$ | — |
| **Constructors' Championship (P1)** | — | — | — | — | **1.1%** |
| **Constructors' P2 Finish** | — | — | — | — | **17.2%** |
| **Constructors' P3 Finish** | — | — | — | — | **80.3%** |
| **Constructors' P4 Finish** | — | — | — | — | **1.5%** |
| **Top-3 Championship Finish** | — | — | — | — | **98.5%** |
| **Total Team Grand Prix Wins** | **4.4 wins** | 4.0 wins | $[2.0, 7.0]$ | $[1.0, 8.0]$ | — |
| **Total Team Podiums** | **13.7 podiums**| 14.0 podiums | $[9.0, 18.0]$ | $[8.0, 20.0]$ | — |

---

## 2. Driver Championship Projections

Simulated on-track against the full 20-driver grid (no arbitrary percentage splits):

```
                                  2027 Ferrari Drivers Championship Projection
                     P10                   P50 (Median)                  P90
                      |                         |                         |
Charles Leclerc       [====== 158.0 pts ======= | 219.0 pts | ======= 285.0 pts ======]
Lewis Hamilton        [====== 205.0 pts ======= | 269.0 pts | ======= 335.0 pts ======]
```

### 2.1 Detailed Driver Breakdown

| Metric | Charles Leclerc (`#16`) | Lewis Hamilton (`#44`) |
|---|---|---|
| **Expected Points** | **219.5 pts** | **269.3 pts** |
| **Median Points ($p_{50}$)** | 219.0 pts | 269.0 pts |
| **80% Credible Interval** | $[172.0, 269.0]$ | $[219.0, 320.0]$ |
| **90% Credible Interval** | $[158.0, 285.0]$ | $[205.0, 335.0]$ |
| **Expected Grand Prix Wins** | 1.8 wins | 2.6 wins |
| **Expected Podiums** | 5.9 podiums | 7.8 podiums |
| **Expected DNF Count** | 2.4 retirements | 2.4 retirements |
| **P1 Drivers' Title Probability** | **0.2%** | **3.2%** |
| **Championship Rank Probabilities** | P3: $1.8\%$, P4: $19.4\%$, P5: $51.7\%$, P6: $24.8\%$, P7: $2.3\%$ | P2: $10.8\%$, P3: $36.1\%$, P4: $40.2\%$, P5: $11.8\%$, P6: $1.1\%$ |

---

## 3. Full Constructors' Championship Standing Expectations

Unlike the 2026 system—which applied fixed ad-hoc multipliers ($1.15\times$ Red Bull, $1.05\times$ McLaren) to Ferrari's total—the 2027 forecast simulates **all 10 constructors and 20 drivers concurrently across all 34 race sessions** (24 GPs + 10 Sprints), summing actual driver points:

| Expected Rank | Constructor | Mean Points | Median ($p_{50}$) | 90% Credible Interval | P1 Title Prob | Top-3 Prob |
|---|---|---|---|---|---|---|
| **P1** | **Mercedes** | 669.8 pts | 669.0 pts | $[579.0, 762.0]$ | **87.3%** | 100.0% |
| **P2** | **Red Bull Racing** | 567.6 pts | 566.0 pts | $[479.0, 661.0]$ | **11.6%** | 99.9% |
| **P3** | **Scuderia Ferrari** | 488.8 pts | 488.0 pts | $[403.0, 577.0]$ | **1.1%** | 98.5% |
| **P4** | **McLaren** | 325.0 pts | 323.0 pts | $[247.0, 407.0]$ | 0.0% | 1.6% |
| **P5** | **Alpine** | 216.0 pts | 214.0 pts | $[147.0, 290.0]$ | 0.0% | 0.0% |
| **P6** | **Aston Martin** | 183.4 pts | 181.0 pts | $[120.0, 252.0]$ | 0.0% | 0.0% |
| **P7** | **Racing Bulls (RB)** | 135.6 pts | 133.0 pts | $[79.0, 199.0]$ | 0.0% | 0.0% |
| **P8** | **Audi (Sauber)** | 91.2 pts | 89.0 pts | $[45.0, 147.0]$ | 0.0% | 0.0% |
| **P9** | **Williams** | 55.0 pts | 52.0 pts | $[20.0, 102.0]$ | 0.0% | 0.0% |
| **P10** | **Haas** | 51.6 pts | 49.0 pts | $[18.0, 97.0]$ | 0.0% | 0.0% |

---

## 4. Key Race-by-Race Forecast Highlights for Ferrari

| Round | Grand Prix | Format | Expected Ferrari Pts | 90% Credible Interval | Win Prob | Team Podium Prob | DNF Risk / Car |
|---|---|---|---|---|---|---|---|
| R01 | Bahrain | Conventional | 20.3 pts | $[6.0, 37.0]$ | 9.0% | 28.4% | 10.6% |
| R05 | China (Shanghai) | **Sprint** | 26.6 pts | $[7.0, 48.0]$ | 10.1% | 29.8% | 10.5% |
| R06 | Miami | **Sprint** | 26.5 pts | $[7.0, 47.0]$ | 9.7% | 30.1% | 10.7% |
| R08 | **Monaco** | Conventional | 22.1 pts | $[6.0, 40.0]$ | **11.2%** | **33.4%** | 10.5% |
| R11 | Austria (Spielberg) | **Sprint** | 26.5 pts | $[8.0, 48.0]$ | 9.4% | 30.0% | 10.4% |
| R13 | Belgium (Spa) | **Sprint** | 26.4 pts | $[7.0, 48.0]$ | 9.7% | 29.5% | 10.3% |
| R16 | **Monza (Italian GP)** | Conventional | 21.5 pts | $[6.0, 39.0]$ | **11.1%** | **32.7%** | 10.4% |
| R18 | **Singapore** | **Sprint** | 27.6 pts | $[8.0, 49.0]$ | **12.4%** | **34.2%** | 10.8% |
| R21 | São Paulo (Interlagos)| **Sprint** | 26.6 pts | $[7.0, 48.0]$ | 9.9% | 30.0% | 10.4% |
| R24 | Abu Dhabi | **Sprint** | 26.7 pts | $[8.0, 48.0]$ | 9.8% | 30.3% | 10.6% |

- **Strongest Venues:** Singapore ($12.4\%$ win prob), Monaco ($11.2\%$), and Monza ($11.1\%$) represent Ferrari's highest win-probability circuits, grounded in historical low-speed traction, straight-line efficiency, and driver track records.
- **Sprint Events:** The 10 Sprint events boost total available constructor points from $1,056$ to $1,386$ across the championship, expanding weekend point capacity up to $59$ points.

---

## 5. Model Provenance & Verification against 2026 Defects

| 2026 Defect Identified | 2027 Resolution in Preseason Forecast |
|---|---|
| **DEF-01: Pagination cap (Rounds 6–24 dropped)** | Full-season historical records ingested and verified (420 rows in 2019, 340 in 2020, 440 in 2021). |
| **DEF-02: Round=0 export bug** | Strict canonical `event_id` and positive round validation (`1..24`) enforced across all tables. |
| **DEF-03: Ferrari-only data** | All 10 constructors and 20 drivers modeled simultaneously. |
| **DEF-04: Lapped finishes counted as DNFs** | 7-category status taxonomy cleanly distinguishes lapped finishes (`+1 Lap`) and disqualifications from true retirements. |
| **DEF-05: `.bfill()` future temporal leakage** | Zero-leakage verified by unit test: modifying future records yields $0.000$ change in historical predictions. |
| **DEF-06: Hardcoded `/24` race divisors** | Dynamic event and session count aggregation. |
| **DEF-07: Subjective technical ratings & overrides** | Abolished. All pace and reliability parameters are derived via Empirical Bayes shrinkage from race telemetry. |
| **DEF-08: N=5 rows overfitting** | Hierarchical model pooling across 1,739 driver-event records. |
| **DEF-09: Model disagreement as confidence intervals** | Calibrated empirical percentiles ($p_5, p_{10}, p_{50}, p_{90}, p_{95}$) from 10,000 Monte Carlo runs. |
| **DEF-10: Synthetic race outcome generation** | Mutually exclusive finishing orders, exact FIA points tables, and constructor points = sum of drivers. |

---

## 6. Preseason Forecast Freeze Metadata

- **Archive File:** `ferrari-f1-predictor/data/predictions/preseason_2027_forecast.json`
- **SHA-256 Checksum:** Generated and registered in manifest.
- **Freeze Policy:** This preseason forecast is frozen and immutable. It will not be revised after observing 2027 Bahrain pre-season testing. Testing observations will be tracked in a separate, versioned *Post-Testing Update* artifact as outlined in the parent architecture.

