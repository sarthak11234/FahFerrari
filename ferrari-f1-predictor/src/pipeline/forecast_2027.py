"""
2027 Preseason Forecast Pipeline (Stage 6).
Executes the frozen 10,000-season Monte Carlo simulation, applying:
  - Confirmed 2027 calendar (24 GPs, 10 Sprints)
  - 2027 Technical Regulations (420 kW ICE / 300 kW MGU-K, 290 km standard distance)
  - Confirmed 20-driver roster (Leclerc + Hamilton at Ferrari, Verstappen + Tsunoda at Red Bull, etc.)
  - Deterministic random seed 2027 for 100% reproducible forecast freeze
  - Zero artificial rival multipliers or handcoded driver splits
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.models.latent_pace import HierarchicalLatentPaceModel
from src.models.race_outcome import ProbabilisticRaceOutcomeModel
from src.models.reliability import ReliabilityHazardModel
from src.simulation.season_simulator import CALENDAR_2027, ROSTER_2027, SeasonSimulator


def run_2027_preseason_forecast(
    base_dir: Optional[Path] = None,
    num_seasons: int = 10000,
    random_seed: int = 2027,
) -> Dict[str, Any]:
    """
    Generate and save the frozen 2027 Preseason Forecast artifact.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]
    base_dir = Path(base_dir)

    staging_dir = base_dir / "data" / "staging"
    predictions_dir = base_dir / "data" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    with (staging_dir / "race_results.csv").open("r", encoding="utf-8") as f:
        historical_results = list(csv.DictReader(f))

    # 1. Fit Hierarchical Latent Pace and Reliability Models
    pace_model = HierarchicalLatentPaceModel().fit(historical_results)
    reliability_model = ReliabilityHazardModel().fit(historical_results)
    race_model = ProbabilisticRaceOutcomeModel(pace_model, reliability_model)

    # 2. Run 10,000 Monte Carlo Season Simulations
    simulator = SeasonSimulator(race_model=race_model, calendar=CALENDAR_2027, roster=ROSTER_2027)
    forecast_results = simulator.run_simulation(num_seasons=num_seasons, random_seed=random_seed)

    # 3. Add Forecast Provenance & Freeze Metadata
    freeze_meta = {
        "forecast_type": "preseason_frozen",
        "target_season": 2027,
        "freeze_date_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_event_cutoff": "pre_bahrain_testing_2027",
        "num_simulations": num_seasons,
        "random_seed": random_seed,
        "model_architecture": "HierarchicalLatentPace + DiscreteTimeReliabilityHazard + MonteCarlo",
        "code_commit_baseline": "ae91b36",
    }
    forecast_results["freeze_metadata"] = freeze_meta

    # 4. Save to data/predictions/preseason_2027_forecast.json
    output_path = predictions_dir / "preseason_2027_forecast.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(forecast_results, f, indent=2)

    return forecast_results


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    forecast = run_2027_preseason_forecast(repo_root, num_seasons=10000, random_seed=2027)
    fc = forecast["ferrari_constructors_forecast"]
    dl = forecast["driver_forecasts"]["leclerc"]
    dh = forecast["driver_forecasts"]["hamilton"]

    print("=" * 65)
    print("🏁 SCUDERIA FERRARI 2027 PRESEASON FORECAST (FROZEN 10k MONTE CARLO)")
    print("=" * 65)
    print(f"Constructors' Championship Points: {fc['mean_points']} (90% CI: {fc['ci_90'][0]} – {fc['ci_90'][1]})")
    print(f"Constructors' P1 Championship Probability: {fc['prob_championship_win']*100:.1f}%")
    print(f"Constructors' Top-3 Probability: {fc['prob_top3']*100:.1f}%")
    print(f"Rank Probabilities: {fc['position_probabilities']}")
    print("-" * 65)
    print(f"Charles Leclerc: {dl['mean_points']} pts (CI: {dl['ci_90'][0]}–{dl['ci_90'][1]}) | Exp Wins: {dl['expected_wins']} | Exp Podiums: {dl['expected_podiums']} | P1 Title Prob: {dl['prob_drivers_title']*100:.1f}%")
    print(f"Lewis Hamilton:  {dh['mean_points']} pts (CI: {dh['ci_90'][0]}–{dh['ci_90'][1]}) | Exp Wins: {dh['expected_wins']} | Exp Podiums: {dh['expected_podiums']} | P1 Title Prob: {dh['prob_drivers_title']*100:.1f}%")
    print("=" * 65)
    print("Constructors' Standings Expectations:")
    for team in forecast["constructors_standings"]:
        print(f"  P{team['expected_position']}: {team['constructor_id']:<15} {team['mean_points']:>5.1f} pts (P1 Prob: {team['prob_p1']*100:>4.1f}%, Top-3: {team['prob_top3']*100:>4.1f}%)")

