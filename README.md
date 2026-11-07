# 🏎️ FAHFERRARI — Formula 1 Performance Intelligence (2027 Season)

> **"Data, not delusion."** — Quantitative probabilistic modeling for Scuderia Ferrari's 2027 campaign.

Welcome to **FahFerrari**, an independent Formula 1 performance-intelligence platform engineered to forecast Scuderia Ferrari’s 2027 season using Bayesian hierarchical pace estimation, discrete-time reliability hazard modeling, and 10,000 paired Monte Carlo season simulations.

While Ferrari is the primary subject, all 10 constructors and driver contenders are fully modeled for competitive benchmarking. The 2026 implementation is formally preserved as an audited historical baseline and forensic retrospective.

---

## ⚡ 2027 Preseason Frozen Baseline (v2027.1)

The inaugural 2027 forecast is an immutable preseason baseline frozen prior to Bahrain pre-season testing:

*   **Scuderia Expected Points:** `488.8 pts` (Median: `488.0 pts`)
*   **90% High Density Credible Interval (HDI):** `[403.0 – 577.0 pts]`
*   **P1 Championship Win Probability:** `1.1%` (109 of 10,000 simulated seasons)
*   **Top-3 Podium Finish Probability:** `98.5%` (Expected finish: **P3** with 80.3% probability)
*   **Driver Paired Battle:**
    *   **Lewis Hamilton (#44):** `269.3 pts` expected, `2.6 wins`, `7.8 podiums`, `1.4 DNFs` (3.2% WDC probability)
    *   **Charles Leclerc (#16):** `219.5 pts` expected, `1.8 wins`, `5.9 podiums`, `3.0 DNFs` (0.2% WDC probability)
*   **Baseline Commit:** `ae91b36` | **Data Cutoff:** `FEB 18, 2027` | **Simulation Budget:** `10,000 seasons`

---

## 🖥️ Scuderia Command UI (8 Core Modules)

The frontend is built according to the **Scuderia Command** design system (`#08090B` canvas, `#111317` surface, `#E10600` Ferrari red, strict 0px sharp geometry, and OpenType tabular numerals `tnum`):

1.  **Overview (`/`)**: Cinematic Ferrari hero, season intelligence strip, Scuderia championship outlook, driver equation, 10-team competitive field, and race watchlist.
2.  **Championship (`/championship`)**: 10-constructor ranking with 90% HDI whiskers, finishing position probability heatmap (P1 to P10+), and top driver contender cards.
3.  **Inside the Scuderia (`/ferrari`)**: Performance targets, qualifying vs race pace decomposition, circuit archetype sensitivities, and Weibull reliability analysis.
4.  **Race Calendar (`/race-calendar`)**: 24-round master calendar with sprint indicators, circuit telemetry profiles, and interactive race inspector drawer.
5.  **Simulation Lab (`/simulation-lab`)**: Real-time counterfactual sliders ($\Delta$ aero, reliability multiplier, qualifying variance, tire deg) with dynamic points recalculation and points density histogram.
6.  **Model Intelligence (`/model-intelligence`)**: 8-stage predictive pipeline, mathematical latent pace formulations, and historical backtest validation matrix.
7.  **AI Strategist (`/ai-strategist`)**: Telemetry query terminal powered by `POST /api/chat`, strictly distinguishing verified model facts from AI tactical interpretation.
8.  **2026 Retrospective (`/2026-retrospective`)**: Forensic review of 2026 model accuracy drift, analysis of 5 critical lessons learned (DEF-01 to DEF-05), and toggleable preserved legacy 2026 raw UI.

---

## 🛠️ Quickstart

### Prerequisites
* Python 3.9+
* Node.js 18+ and npm

### 1. Launch the Frontend
```bash
cd ferrari-f1-predictor/frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

*(Note: The frontend operates completely self-contained out-of-the-box thanks to the embedded frozen 10,000-season preseason artifact).*

### 2. (Optional) Launch the Backend API
In a separate terminal:
```bash
cd ferrari-f1-predictor
python3 -m api.app
```
The REST API runs on `http://localhost:5000` exposing `/api/predictions/2027`, `/api/predictions/2027/provenance`, `/api/predictions/2026`, `/api/historical`, and `/api/chat`.

---

## 🧪 Testing & Validation

Run the automated data quality, schema verification, and model calibration tests:
```bash
cd ferrari-f1-predictor
python3 -m unittest discover -s tests/data_quality
```
*Result: 17 unit tests passing across canonical schema validation, zero future leakage audit, Plackett-Luce choice sampling, and Weibull hazard functions.*

To verify the frontend production build:
```bash
cd ferrari-f1-predictor/frontend
npm run build
```

---

## 📚 Technical Documentation

*   [2027 Architecture Specification](./docs/2027-architecture.md)
*   [2027 Preseason Forecast Freeze Report](./docs/2027-preseason-forecast.md)
*   [2027 Canonical Data Dictionary](./docs/2027-data-dictionary.md)
*   [2026 Model Retrospective & Defect Audit](./docs/2026-retrospective.md)
*   [Validation & Backtesting Plan](./docs/2027-validation-plan.md)
*   [Implementation Plan & Milestones](./docs/2027-implementation-plan.md)

---

## ⚖️ Legal & Governance Notice

FahFerrari is an independent quantitative research and performance-intelligence project. All team names, driver names, circuit names, and trademarks are the property of their respective owners (Formula One Licensing B.V. and Scuderia Ferrari S.p.A.). This platform is not affiliated with, sponsored by, or endorsed by Ferrari or Formula 1.
