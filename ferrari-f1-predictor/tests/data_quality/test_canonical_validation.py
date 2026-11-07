"""
Data-quality and canonical schema tests for Stage 1 (FF27-07, FF27-08).
Compatible with both `unittest` and `pytest`.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

from src.ingestion.fastf1_collector import FastF1Collector
from src.ingestion.fia_documents import get_canonical_regulations_table
from src.ingestion.jolpica_collector import JolpicaCollector
from src.preprocessing.normalization import (
    classify_race_status,
    infer_regulation_era,
    make_event_id,
    make_session_id,
    map_franchise_id,
    parse_lap_time_to_seconds,
)
from src.preprocessing.validation import DataValidationError, validate_canonical_bundle


class TestCanonicalNormalization(unittest.TestCase):
    """Unit tests for canonical IDs, franchise lineage, and status classification."""

    def test_make_event_id_and_reject_zero_round(self) -> None:
        self.assertEqual(make_event_id(2024, 1), "2024_r01")
        self.assertEqual(make_event_id(2027, 24), "2027_r24")
        with self.assertRaises(ValueError):
            make_event_id(2024, 0)  # DEF-02 guard: round=0 must fail

    def test_make_session_id(self) -> None:
        self.assertEqual(make_session_id("2024_r01", "r"), "2024_r01_R")
        self.assertEqual(make_session_id("2027_r05", "SQ"), "2027_r05_SQ")
        with self.assertRaises(ValueError):
            make_session_id("2024_r01", "INVALID")

    def test_franchise_lineage_mapping(self) -> None:
        self.assertEqual(map_franchise_id("toro_rosso"), "racing_bulls")
        self.assertEqual(map_franchise_id("alphatauri"), "racing_bulls")
        self.assertEqual(map_franchise_id("rb"), "racing_bulls")
        self.assertEqual(map_franchise_id("alfa"), "sauber_audi")
        self.assertEqual(map_franchise_id("sauber"), "sauber_audi")
        self.assertEqual(map_franchise_id("racing_point"), "aston_martin")
        self.assertEqual(map_franchise_id("renault"), "alpine")
        self.assertEqual(map_franchise_id("ferrari"), "ferrari")

    def test_status_classification_fixes_def04(self) -> None:
        # Lapped finishes must NOT be counted as DNFs (DEF-04)
        cat, is_cls, cls_pos, is_dnf, is_mech = classify_race_status("+1 Lap", "6", laps_completed=69)
        self.assertEqual(cat, "finished_lapped")
        self.assertTrue(is_cls)
        self.assertEqual(cls_pos, 6)
        self.assertFalse(is_dnf)
        self.assertFalse(is_mech)

        # Disqualifications must NOT be counted as DNFs (DEF-04)
        cat, is_cls, cls_pos, is_dnf, is_mech = classify_race_status("Disqualified", "D", laps_completed=56)
        self.assertEqual(cat, "disqualified")
        self.assertFalse(is_cls)
        self.assertIsNone(cls_pos)
        self.assertFalse(is_dnf)
        self.assertFalse(is_mech)

        # Mechanical failure retirement
        cat, is_cls, cls_pos, is_dnf, is_mech = classify_race_status("Driveshaft", "R", laps_completed=0)
        self.assertEqual(cat, "dnf_mechanical")
        self.assertFalse(is_cls)
        self.assertIsNone(cls_pos)
        self.assertTrue(is_dnf)
        self.assertTrue(is_mech)

        # Collision / incident retirement
        cat, is_cls, cls_pos, is_dnf, is_mech = classify_race_status("Collision damage", "R", laps_completed=12)
        self.assertEqual(cat, "dnf_incident")
        self.assertFalse(is_cls)
        self.assertTrue(is_dnf)
        self.assertFalse(is_mech)

        # Pre-race withdrawal (0 laps) vs mid-race retirement classified at >90% distance (e.g., 2020 Istanbul Magnussen)
        cat_dns, is_cls_dns, _, is_dnf_dns, _ = classify_race_status("Withdrew", "W", laps_completed=0)
        self.assertEqual(cat_dns, "did_not_start")
        self.assertFalse(is_cls_dns)
        self.assertFalse(is_dnf_dns)

        cat_mid, is_cls_mid, cls_pos_mid, is_dnf_mid, _ = classify_race_status("Withdrew", "17", laps_completed=55)
        self.assertEqual(cat_mid, "dnf_other")
        self.assertTrue(is_cls_mid)
        self.assertEqual(cls_pos_mid, 17)
        self.assertTrue(is_dnf_mid)

    def test_parse_lap_time_to_seconds(self) -> None:
        self.assertAlmostEqual(parse_lap_time_to_seconds("1:32.608") or 0.0, 92.608, places=3)
        self.assertIsNone(parse_lap_time_to_seconds(None))
        self.assertIsNone(parse_lap_time_to_seconds(""))

    def test_infer_regulation_era(self) -> None:
        self.assertEqual(infer_regulation_era(2019), "2017_2021_wide_aero")
        self.assertEqual(infer_regulation_era(2024), "2022_2025_ground_effect")
        self.assertEqual(infer_regulation_era(2026), "2026_active_aero")
        self.assertEqual(infer_regulation_era(2027), "2027_rebalanced_pu")


class TestStage1CanonicalDatasetValidation(unittest.TestCase):
    """Integration and data-quality tests over the cached multi-team dataset sample."""

    @classmethod
    def setUpClass(cls) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        cache_path = base_dir / "cache" / "fastf1_http_cache.sqlite"
        fastf1_col = FastF1Collector(cache_path)
        jolpica_col = JolpicaCollector(cache_path)

        sprint_ids, _ = fastf1_col.extract_cached_schedules()
        wrappers, _ = jolpica_col.load_cached_race_payloads(2019, 2025)
        (
            cls.events,
            cls.circuits,
            cls.entries,
            cls.race_results,
        ) = jolpica_col.normalize_race_bundle(wrappers, sprint_event_ids=sprint_ids)
        cls.regulations = get_canonical_regulations_table()

    def test_canonical_bundle_passes_all_contracts(self) -> None:
        report = validate_canonical_bundle(
            events=self.events,
            circuits=self.circuits,
            entries=self.entries,
            race_results=self.race_results,
            regulations=self.regulations,
            raise_on_error=True,
        )
        self.assertTrue(report.passed)
        self.assertEqual(len(report.errors), 0)
        self.assertEqual(report.table_counts["events"], 87)
        self.assertEqual(report.table_counts["race_results"], 1739)
        # Verify complete seasons 2019, 2020, 2021 match official Ferrari GP points totals
        self.assertEqual(report.season_coverage[2019]["ferrari_gp_points"], 504.0)
        self.assertEqual(report.season_coverage[2020]["ferrari_gp_points"], 131.0)
        self.assertEqual(report.season_coverage[2021]["ferrari_gp_points"], 322.5)
        # Verify all 10 constructors are present in every season
        for season in range(2019, 2026):
            self.assertEqual(report.season_coverage[season]["unique_constructors"], 10)

    def test_validation_rejects_zero_round_regression(self) -> None:
        bad_events = copy.deepcopy(self.events[:2])
        bad_events[0]["round"] = 0
        with self.assertRaises(DataValidationError):
            validate_canonical_bundle(
                events=bad_events,
                circuits=self.circuits,
                entries=self.entries[:40],
                race_results=self.race_results[:40],
                raise_on_error=True,
            )

    def test_validation_rejects_ferrari_only_single_team_regression(self) -> None:
        ferrari_only_results = [
            r for r in self.race_results if r["event_id"] == "2019_r01" and r["constructor_id"] == "ferrari"
        ]
        ferrari_only_entries = [
            e for e in self.entries if e["event_id"] == "2019_r01" and e["constructor_id"] == "ferrari"
        ]
        with self.assertRaises(DataValidationError):
            validate_canonical_bundle(
                events=[e for e in self.events if e["event_id"] == "2019_r01"],
                circuits=self.circuits,
                entries=ferrari_only_entries,
                race_results=ferrari_only_results,
                raise_on_error=True,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

