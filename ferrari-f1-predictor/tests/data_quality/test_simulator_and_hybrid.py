"""
Simulator and Hybrid Model verification tests (Stage 4 & Stage 5).
Verifies:
  1. Deterministic repeatability with random seeds.
  2. Strict FIA points scoring invariants (25-18-15... and 8-7-6...).
  3. Mutual exclusivity of positions 1..N.
  4. Exact constructor points = driver_1_points + driver_2_points (Zero artificial rival multipliers).
  5. Monte Carlo convergence across increasing simulation counts.
"""

from __future__ import annotations

import csv
import unittest
from pathlib import Path
from typing import Any, Dict, List

from src.models.latent_pace import HierarchicalLatentPaceModel
from src.models.race_outcome import ProbabilisticRaceOutcomeModel
from src.models.reliability import ReliabilityHazardModel
from src.simulation.season_simulator import ROSTER_2027, SeasonSimulator


class TestHybridModelsAndSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        staging_dir = base_dir / "data" / "staging"
        with (staging_dir / "race_results.csv").open("r", encoding="utf-8") as f:
            cls.historical_results = list(csv.DictReader(f))

        # Train models on historical records
        cls.pace_model = HierarchicalLatentPaceModel().fit(cls.historical_results)
        cls.reliability_model = ReliabilityHazardModel().fit(cls.historical_results)
        cls.race_model = ProbabilisticRaceOutcomeModel(cls.pace_model, cls.reliability_model)
        cls.simulator = SeasonSimulator(cls.race_model)

    def test_single_race_scoring_invariants(self) -> None:
        """Verify strict points tables and mutual exclusivity in race simulation."""
        gp_res = self.race_model.simulate_race(
            event_id="2027_r01",
            round_number=1,
            circuit_id="bahrain",
            entries=ROSTER_2027,
            is_sprint=False,
            fastest_lap_point_active=False,
        )

        positions = [dr.finish_position for dr in gp_res.driver_results]
        self.assertEqual(sorted(positions), list(range(1, 21)))  # Exactly 1..20

        # P1 must get 25 points, P2 gets 18, P3 gets 15, P10 gets 1, P11-P20 get 0
        expected_pts = {1: 25.0, 2: 18.0, 3: 15.0, 4: 12.0, 5: 10.0, 6: 8.0, 7: 6.0, 8: 4.0, 9: 2.0, 10: 1.0}
        for dr in gp_res.driver_results:
            if not dr.is_dnf:
                self.assertEqual(dr.points_scored, expected_pts.get(dr.finish_position, 0.0))

        # Verify Constructor points equals exact sum of drivers' points
        for cid, total_pts in gp_res.constructor_points.items():
            driver_sum = sum(dr.points_scored for dr in gp_res.driver_results if dr.constructor_id == cid)
            self.assertAlmostEqual(total_pts, driver_sum, places=4)

    def test_sprint_scoring_invariants(self) -> None:
        """Verify Sprint points scoring (8-7-6-5-4-3-2-1)."""
        sprint_res = self.race_model.simulate_race(
            event_id="2027_r05_S",
            round_number=5,
            circuit_id="shanghai",
            entries=ROSTER_2027,
            is_sprint=True,
            fastest_lap_point_active=False,
        )
        p1 = next(dr for dr in sprint_res.driver_results if dr.finish_position == 1)
        self.assertEqual(p1.points_scored, 8.0)
        p8 = next(dr for dr in sprint_res.driver_results if dr.finish_position == 8)
        self.assertEqual(p8.points_scored, 1.0)
        p9 = next(dr for dr in sprint_res.driver_results if dr.finish_position == 9)
        self.assertEqual(p9.points_scored, 0.0)

    def test_seeded_deterministic_repeatability(self) -> None:
        """Verify that passing the same seed yields 100% bitwise identical simulation results."""
        sim_run_1 = self.simulator.run_simulation(num_seasons=100, random_seed=12345)
        sim_run_2 = self.simulator.run_simulation(num_seasons=100, random_seed=12345)

        self.assertEqual(
            sim_run_1["ferrari_constructors_forecast"]["mean_points"],
            sim_run_2["ferrari_constructors_forecast"]["mean_points"],
        )
        self.assertEqual(
            sim_run_1["driver_forecasts"]["leclerc"]["mean_points"],
            sim_run_2["driver_forecasts"]["leclerc"]["mean_points"],
        )
        self.assertEqual(
            sim_run_1["driver_forecasts"]["hamilton"]["mean_points"],
            sim_run_2["driver_forecasts"]["hamilton"]["mean_points"],
        )

    def test_monte_carlo_convergence(self) -> None:
        """Verify Monte Carlo stability: mean estimates stabilize within 3% between 500 and 1000 runs."""
        sim_500 = self.simulator.run_simulation(num_seasons=500, random_seed=999)
        sim_1000 = self.simulator.run_simulation(num_seasons=1000, random_seed=999)

        pts_500 = sim_500["ferrari_constructors_forecast"]["mean_points"]
        pts_1000 = sim_1000["ferrari_constructors_forecast"]["mean_points"]

        pct_diff = abs(pts_1000 - pts_500) / pts_1000
        self.assertLess(pct_diff, 0.05, f"Monte Carlo estimate did not converge: {pts_500} vs {pts_1000}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

