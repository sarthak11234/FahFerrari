"""
Data-quality validation and coverage reporting for canonical datasets (FF27-07).
Enforces schema contracts defined in docs/2027-data-dictionary.md.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REQUIRED_PROVENANCE_COLUMNS: Tuple[str, ...] = (
    "source_provider",
    "source_record_id",
    "source_url",
    "retrieved_at_utc",
    "event_timestamp_utc",
    "is_provisional",
    "schema_version",
)

EXPECTED_SEASON_ROUNDS: Dict[int, int] = {
    2019: 21,
    2020: 17,
    2021: 22,
    2022: 22,
    2023: 22,
    2024: 24,
    2025: 24,
    2026: 24,
    2027: 24,
}

VALID_STATUS_CATEGORIES: Set[str] = {
    "finished_lead_lap",
    "finished_lapped",
    "dnf_mechanical",
    "dnf_incident",
    "dnf_other",
    "disqualified",
    "did_not_start",
}


class DataValidationError(ValueError):
    """Raised when a canonical dataset violates a hard data-quality invariant."""


@dataclass
class ValidationReport:
    """Structured result of canonical dataset validation."""

    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    table_counts: Dict[str, int] = field(default_factory=dict)
    season_coverage: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    status_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": self.errors,
            "warnings": self.warnings,
            "table_counts": self.table_counts,
            "season_coverage": {str(k): v for k, v in sorted(self.season_coverage.items())},
            "status_distribution": dict(sorted(self.status_distribution.items())),
        }


def _check_provenance(row: Dict[str, Any], table_name: str, row_idx: int, errors: List[str]) -> None:
    for col in REQUIRED_PROVENANCE_COLUMNS:
        if col not in row or row[col] in (None, ""):
            if col == "is_provisional" and isinstance(row.get(col), bool):
                continue
            errors.append(f"[{table_name} row {row_idx}] Missing mandatory provenance field '{col}'")


def validate_events_table(events: Sequence[Dict[str, Any]], errors: List[str]) -> Set[str]:
    seen_ids: Set[str] = set()
    seen_season_round: Set[Tuple[int, int]] = set()

    for idx, row in enumerate(events):
        _check_provenance(row, "events", idx, errors)
        event_id = str(row.get("event_id", ""))
        season = int(row.get("season", 0))
        rnd = int(row.get("round", 0))

        if rnd <= 0:
            errors.append(f"[events row {idx}] Invalid round={rnd} for event_id='{event_id}' (DEF-02 guard)")
        if not event_id or event_id != f"{season}_r{rnd:02d}":
            errors.append(f"[events row {idx}] Non-canonical event_id='{event_id}' for season={season}, round={rnd}")
        if event_id in seen_ids:
            errors.append(f"[events row {idx}] Duplicate event_id='{event_id}'")
        seen_ids.add(event_id)

        sr = (season, rnd)
        if sr in seen_season_round:
            errors.append(f"[events row {idx}] Duplicate (season, round)={sr}")
        seen_season_round.add(sr)

    return seen_ids


def validate_circuits_table(circuits: Sequence[Dict[str, Any]], errors: List[str]) -> Set[str]:
    seen_ids: Set[str] = set()
    for idx, row in enumerate(circuits):
        _check_provenance(row, "circuits", idx, errors)
        cid = str(row.get("circuit_id", "")).strip()
        if not cid:
            errors.append(f"[circuits row {idx}] Empty circuit_id")
        if cid in seen_ids:
            errors.append(f"[circuits row {idx}] Duplicate circuit_id='{cid}'")
        seen_ids.add(cid)
    return seen_ids


def validate_entries_table(
    entries: Sequence[Dict[str, Any]],
    valid_event_ids: Set[str],
    errors: List[str],
) -> Set[Tuple[str, str]]:
    seen_keys: Set[Tuple[str, str]] = set()
    for idx, row in enumerate(entries):
        _check_provenance(row, "entries", idx, errors)
        event_id = str(row.get("event_id", ""))
        driver_id = str(row.get("driver_id", ""))
        constructor_id = str(row.get("constructor_id", ""))
        franchise_id = str(row.get("franchise_id", ""))
        rnd = int(row.get("round", 0))

        if rnd <= 0:
            errors.append(f"[entries row {idx}] Invalid round={rnd}")
        if event_id not in valid_event_ids:
            errors.append(f"[entries row {idx}] Unknown event_id='{event_id}'")
        if not driver_id or not constructor_id or not franchise_id:
            errors.append(f"[entries row {idx}] Missing driver_id/constructor_id/franchise_id")

        key = (event_id, driver_id)
        if key in seen_keys:
            errors.append(f"[entries row {idx}] Duplicate entry key {key}")
        seen_keys.add(key)
    return seen_keys


def validate_race_results_table(
    race_results: Sequence[Dict[str, Any]],
    valid_event_ids: Set[str],
    valid_entry_keys: Set[Tuple[str, str]],
    errors: List[str],
    warnings: List[str],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, int]]:
    seen_keys: Set[Tuple[str, str, str]] = set()
    by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_season: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    status_counts: Counter[str] = Counter()

    for idx, row in enumerate(race_results):
        _check_provenance(row, "race_results", idx, errors)
        event_id = str(row.get("event_id", ""))
        season = int(row.get("season", 0))
        rnd = int(row.get("round", 0))
        session_type = str(row.get("session_type", "R"))
        driver_id = str(row.get("driver_id", ""))
        constructor_id = str(row.get("constructor_id", ""))
        pos_text = str(row.get("position_text", ""))
        is_classified = bool(row.get("is_classified"))
        classified_pos = row.get("classified_position")
        status_cat = str(row.get("status_category", ""))
        is_dnf = bool(row.get("is_dnf"))
        is_mech_dnf = bool(row.get("is_mechanical_dnf"))
        points = float(row.get("points_awarded", -1.0))

        if rnd <= 0:
            errors.append(f"[race_results row {idx}] Invalid round={rnd} (DEF-02 guard)")
        if event_id not in valid_event_ids:
            errors.append(f"[race_results row {idx}] Foreign key violation: event_id='{event_id}'")
        if (event_id, driver_id) not in valid_entry_keys:
            errors.append(f"[race_results row {idx}] Foreign key violation: entry ({event_id}, {driver_id}) missing")
        if not constructor_id:
            errors.append(f"[race_results row {idx}] Missing constructor_id")

        key = (event_id, session_type, driver_id)
        if key in seen_keys:
            errors.append(f"[race_results row {idx}] Duplicate race_result key {key}")
        seen_keys.add(key)

        # Status & classification invariants (DEF-04 guards)
        if status_cat not in VALID_STATUS_CATEGORIES:
            errors.append(f"[race_results row {idx}] Invalid status_category='{status_cat}'")
        if status_cat in {"finished_lead_lap", "finished_lapped", "disqualified", "did_not_start"} and is_dnf:
            errors.append(
                f"[race_results row {idx}] status_category='{status_cat}' cannot have is_dnf=True (DEF-04 guard)"
            )
        if status_cat in {"dnf_mechanical", "dnf_incident", "dnf_other"} and not is_dnf:
            errors.append(f"[race_results row {idx}] status_category='{status_cat}' must have is_dnf=True")
        if is_mech_dnf and status_cat != "dnf_mechanical":
            errors.append(
                f"[race_results row {idx}] is_mechanical_dnf=True only valid for status_category='dnf_mechanical'"
            )

        # Classification position invariants
        if pos_text.isdigit():
            if not is_classified or classified_pos in (None, ""):
                errors.append(
                    f"[race_results row {idx}] Numeric position_text='{pos_text}' must be classified with integer position"
                )
        else:
            if is_classified or classified_pos not in (None, ""):
                errors.append(
                    f"[race_results row {idx}] Non-numeric position_text='{pos_text}' must have is_classified=False and classified_position=None"
                )

        # Points invariants
        max_pts = 26.0 if session_type == "R" else 8.0
        if points < 0.0 or points > max_pts:
            errors.append(
                f"[race_results row {idx}] Points out of bounds ({points}) for session_type='{session_type}'"
            )

        by_event[event_id].append(row)
        by_season[season].append(row)
        status_counts[status_cat] += 1

    # Event-level multi-team and driver count checks (DEF-03 guard)
    for event_id, rows in sorted(by_event.items()):
        constructors_in_event = {r["constructor_id"] for r in rows}
        if len(rows) < 18 or len(constructors_in_event) < 9:
            errors.append(
                f"[race_results event {event_id}] Incomplete field: {len(rows)} drivers across {len(constructors_in_event)} constructors (expected >=18 drivers, >=9 constructors; DEF-01/DEF-03 guard)"
            )

    # Season-level coverage summary
    season_coverage: Dict[int, Dict[str, Any]] = {}
    for season, rows in sorted(by_season.items()):
        rounds_present = sorted({int(r["round"]) for r in rows})
        constructors_present = sorted({str(r["constructor_id"]) for r in rows})
        drivers_present = sorted({str(r["driver_id"]) for r in rows})
        expected_rounds = EXPECTED_SEASON_ROUNDS.get(season, len(rounds_present))
        is_complete_season = len(rounds_present) == expected_rounds

        ferrari_rows = [r for r in rows if r["constructor_id"] == "ferrari"]
        ferrari_points = sum(float(r["points_awarded"]) for r in ferrari_rows)
        ferrari_wins = sum(1 for r in ferrari_rows if str(r.get("position_text")) == "1")
        ferrari_dnfs = sum(1 for r in ferrari_rows if bool(r.get("is_dnf")))

        if not is_complete_season:
            warnings.append(
                f"[Season {season}] Partial cached coverage: {len(rounds_present)}/{expected_rounds} rounds present in offline SQLite cache (rounds {rounds_present[0]}..{rounds_present[-1]}). Live pagination required for remaining {expected_rounds - len(rounds_present)} rounds."
            )

        season_coverage[season] = {
            "expected_rounds": expected_rounds,
            "ingested_rounds": len(rounds_present),
            "rounds_list": rounds_present,
            "is_complete_season": is_complete_season,
            "total_driver_event_rows": len(rows),
            "unique_constructors": len(constructors_present),
            "unique_drivers": len(drivers_present),
            "ferrari_rows": len(ferrari_rows),
            "ferrari_gp_points": round(ferrari_points, 1),
            "ferrari_gp_wins": ferrari_wins,
            "ferrari_true_dnfs": ferrari_dnfs,
        }

    return season_coverage, dict(status_counts)


def validate_canonical_bundle(
    events: Sequence[Dict[str, Any]],
    circuits: Sequence[Dict[str, Any]],
    entries: Sequence[Dict[str, Any]],
    race_results: Sequence[Dict[str, Any]],
    regulations: Optional[Sequence[Dict[str, Any]]] = None,
    raise_on_error: bool = False,
) -> ValidationReport:
    """Run all schema, primary/foreign key, status, and coverage validations."""
    errors: List[str] = []
    warnings: List[str] = []

    if not events:
        errors.append("[events] Table is empty")
    if not circuits:
        errors.append("[circuits] Table is empty")
    if not entries:
        errors.append("[entries] Table is empty")
    if not race_results:
        errors.append("[race_results] Table is empty")

    valid_circuit_ids = validate_circuits_table(circuits, errors)
    valid_event_ids = validate_events_table(events, errors)

    for idx, ev in enumerate(events):
        cid = str(ev.get("circuit_id", ""))
        if cid not in valid_circuit_ids:
            errors.append(f"[events row {idx}] Foreign key violation: circuit_id='{cid}' not in circuits")

    valid_entry_keys = validate_entries_table(entries, valid_event_ids, errors)
    season_cov, status_dist = validate_race_results_table(
        race_results, valid_event_ids, valid_entry_keys, errors, warnings
    )

    table_counts = {
        "events": len(events),
        "circuits": len(circuits),
        "entries": len(entries),
        "race_results": len(race_results),
        "regulations": len(regulations) if regulations is not None else 0,
    }

    report = ValidationReport(
        passed=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
        table_counts=table_counts,
        season_coverage=season_cov,
        status_distribution=status_dist,
    )

    if raise_on_error and not report.passed:
        raise DataValidationError(
            f"Canonical dataset validation failed with {len(errors)} error(s):\n"
            + "\n".join(errors[:15])
        )

    return report

