"""
Point-in-time leakage tests and rolling-origin baseline evaluation tests (FF27-09, FF27-10).
"""

from __future__ import annotations

import copy
import csv
import unittest
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.backtesting import RollingOriginBaselineEvaluator
from src.features.race_features import (
    FEATURE_COLUMNS,
    LeakageViolationError,
    build_event_feature_rows,
    build_point_in_time_feature_dataset,
    filter_strictly_prior_records,
)


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestPointInTimeLeakagePrevention(unittest.TestCase):
    """Rigorous tests proving zero temporal leakage in feature generation."""

    @classmethod
    def setUpClass(cls) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        staging_dir = base_dir / "data" / "staging"
        cls.events = _load_csv(staging_dir / "events.csv")
        cls.race_results = _load_csv(staging_dir / "race_results.csv")
        cls.regulations = _load_csv(staging_dir / "regulations.csv")

    def test_filter_strictly_prior_records_rejects_same_or_future_timestamps(self) -> None:
        target_ts = "2021-03-28T15:00:00Z"  # 2021 Round 1
        filtered = filter_strictly_prior_records(self.race_results, target_event_timestamp_utc=target_ts)
        self.assertTrue(len(filtered) > 0)
        for r in filtered:
            self.assertLess(str(r["event_timestamp_utc"]), target_ts)
            self.assertLess(int(r["season"]), 2022)

    def test_future_events_never_alter_earlier_features(self) -> None:
        """
        Critical leakage test:
        1. Compute features for 2021 Round 1 using only 2019-2020 history.
        2. Append synthetic future results (2022, 2023) or mutate future race outcomes.
        3. Recompute features for 2021 Round 1.
        4. Assert every feature column is IDENTICAL down to float precision.
        """
        r2021_r01 = next(e for e in self.events if e["event_id"] == "2021_r01")
        r2021_r01_results = [r for r in self.race_results if r["event_id"] == "2021_r01"]
        regs_dict = {int(r["season"]): r for r in self.regulations}

        # Step 1: Features with real dataset
        feats_before = build_event_feature_rows(
            target_event=r2021_r01,
            target_event_results=r2021_r01_results,
            all_historical_results=self.race_results,
            regulations_by_season=regs_dict,
        )

        # Step 2: Inject corrupted future data in 2024 with 999 points
        corrupted_results = copy.deepcopy(self.race_results)
        for r in corrupted_results:
            if int(r["season"]) >= 2022:
                r["points_awarded"] = "999.0"
                r["status_category"] = "dnf_mechanical"
                r["is_dnf"] = "True"
                r["is_mechanical_dnf"] = "True"

        # Step 3: Recompute features for 2021 Round 1 with corrupted future data present
        feats_after = build_event_feature_rows(
            target_event=r2021_r01,
            target_event_results=r2021_r01_results,
            all_historical_results=corrupted_results,
            regulations_by_season=regs_dict,
        )

        # Step 4: Verify zero perturbation across all 20 drivers and all feature columns
        self.assertEqual(len(feats_before), len(feats_after))
        for row_before, row_after in zip(feats_before, feats_after):
            self.assertEqual(row_before["driver_id"], row_after["driver_id"])
            for col in FEATURE_COLUMNS:
                self.assertEqual(
                    row_before[col],
                    row_after[col],
                    f"Leakage detected on column '{col}' for driver '{row_before['driver_id']}'",
                )

    def test_feature_dataset_assembly(self) -> None:
        dataset = build_point_in_time_feature_dataset(
            events=self.events,
            race_results=self.race_results,
            regulations=self.regulations,
        )
        self.assertEqual(len(dataset), len(self.race_results))
        # Ensure all feature columns exist and are non-null
        first_row = dataset[0]
        for col in FEATURE_COLUMNS:
            self.assertIn(col, first_row)
            self.assertIsNotNone(first_row[col])


class TestRollingOriginBaselines(unittest.TestCase):
    """Verify that all 4 baselines evaluate successfully in rolling-origin backtesting."""

    @classmethod
    def setUpClass(cls) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        staging_dir = base_dir / "data" / "staging"
        events = _load_csv(staging_dir / "events.csv")
        race_results = _load_csv(staging_dir / "race_results.csv")
        regulations = _load_csv(staging_dir / "regulations.csv")

        cls.feature_dataset = build_point_in_time_feature_dataset(
            events=events,
            race_results=race_results,
            regulations=regulations,
        )
        cls.evaluator = RollingOriginBaselineEvaluator(cls.feature_dataset)

    def test_rolling_origin_execution_and_metrics(self) -> None:
        results = self.evaluator.run_season_rolling_origin(min_train_seasons=2)
        self.assertIn("folds", results)
        self.assertIn("baselines", results)
        self.assertEqual(len(results["folds"]), 5)  # Test seasons 2021, 2022, 2023, 2024, 2025

        baselines = results["baselines"]
        expected_baselines = [
            "baseline1_historical_mean",
            "baseline2_recent_weighted",
            "baseline3_teammate_relative",
            "baseline4_probabilistic_ranking",
        ]
        for b in expected_baselines:
            self.assertIn(b, baselines)
            metrics = baselines[b]
            self.assertGreater(metrics["points_mae"], 0.0)
            self.assertGreater(metrics["points_rmse"], 0.0)
            self.assertGreater(metrics["finish_order_mae"], 0.0)
            self.assertGreaterEqual(metrics["spearman_rank_correlation"], 0.0)
            self.assertGreater(metrics["top3_accuracy"], 0.0)
            self.assertGreater(metrics["brier_score_win"], 0.0)
            self.assertGreater(metrics["brier_score_podium"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
