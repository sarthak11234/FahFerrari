"""
Point-in-time feature store assembler and leakage guard (FF27-09).
Guarantees that features for any target event E with timestamp t_E are computed
exclusively from records with event_timestamp_utc < t_E and <= cutoff_utc.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.features.circuit_features import compute_circuit_point_in_time_features
from src.features.driver_features import compute_driver_point_in_time_features
from src.features.team_features import compute_team_point_in_time_features
from src.preprocessing.normalization import SCHEMA_VERSION

FEATURE_COLUMNS: Tuple[str, ...] = (
    # Team / constructor features (11)
    "team_is_cold_start",
    "team_prior_entries",
    "team_pts_per_entry_ewma5",
    "team_pts_per_entry_season_to_date",
    "team_finish_pos_mean5",
    "team_grid_pos_mean5",
    "team_podium_rate_10",
    "team_win_rate_10",
    "team_dnf_rate_10",
    "team_mech_dnf_rate_10",
    "team_field_pts_share_5",
    # Driver & teammate-relative features (10)
    "driver_is_cold_start",
    "driver_prior_starts",
    "driver_pts_ewma5",
    "driver_finish_pos_mean5",
    "driver_grid_pos_mean5",
    "driver_positions_gained_mean5",
    "driver_teammate_grid_delta_mean5",
    "driver_teammate_finish_delta_mean5",
    "driver_teammate_pts_share_10",
    "driver_incident_dnf_rate_10",
    # Circuit suitability & attrition features (7)
    "circuit_is_cold_start",
    "circuit_team_prior_starts",
    "circuit_team_pts_per_entry_hist",
    "circuit_team_podium_rate_hist",
    "circuit_driver_prior_starts",
    "circuit_driver_mean_finish_hist",
    "circuit_field_dnf_rate_hist",
    # Regulation regime indicators (2)
    "reg_electric_power_share",
    "reg_fastest_lap_point_active",
)


class LeakageViolationError(RuntimeError):
    """Raised if any record with event_timestamp_utc >= target_event_timestamp_utc enters feature computation."""


def filter_strictly_prior_records(
    records: Sequence[Dict[str, Any]],
    target_event_timestamp_utc: str,
    as_of_cutoff_utc: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Return records strictly prior to `target_event_timestamp_utc` (and <= `as_of_cutoff_utc` if specified),
    sorted chronologically by `(event_timestamp_utc, event_id)`.
    """
    effective_cutoff = (
        min(target_event_timestamp_utc, as_of_cutoff_utc)
        if as_of_cutoff_utc
        else target_event_timestamp_utc
    )

    filtered = [
        r
        for r in records
        if str(r.get("event_timestamp_utc", "")) < target_event_timestamp_utc
        and str(r.get("event_timestamp_utc", "")) <= effective_cutoff
    ]

    # Explicit assertion guard against temporal leakage
    for r in filtered:
        if str(r.get("event_timestamp_utc", "")) >= target_event_timestamp_utc:
            raise LeakageViolationError(
                f"Temporal leakage detected: record {r.get('source_record_id')} has timestamp "
                f"{r.get('event_timestamp_utc')} >= target {target_event_timestamp_utc}"
            )

    filtered.sort(
        key=lambda x: (
            str(x.get("event_timestamp_utc", "")),
            str(x.get("event_id", "")),
            int(x.get("finish_position_order", 0)),
        )
    )
    return filtered


def build_event_feature_rows(
    target_event: Dict[str, Any],
    target_event_results: Sequence[Dict[str, Any]],
    all_historical_results: Sequence[Dict[str, Any]],
    regulations_by_season: Dict[int, Dict[str, Any]],
    as_of_cutoff_utc: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build point-in-time feature rows for all driver entries in `target_event`.
    """
    target_ts = str(target_event["event_timestamp_utc"])
    season = int(target_event["season"])
    rnd = int(target_event["round"])
    event_id = str(target_event["event_id"])
    circuit_id = str(target_event["circuit_id"])

    prior_results = filter_strictly_prior_records(
        all_historical_results,
        target_event_timestamp_utc=target_ts,
        as_of_cutoff_utc=as_of_cutoff_utc,
    )

    reg = regulations_by_season.get(season, {})
    reg_elec_share = float(reg.get("electric_power_share", 0.1727))
    reg_fl_active = 1 if str(reg.get("fastest_lap_point_active", "True")).lower() == "true" else 0

    output_rows: List[Dict[str, Any]] = []
    for res in target_event_results:
        driver_id = str(res["driver_id"])
        constructor_id = str(res["constructor_id"])
        franchise_id = str(res["franchise_id"])

        team_feats = compute_team_point_in_time_features(
            prior_race_results=prior_results,
            target_franchise_id=franchise_id,
            target_season=season,
        )
        driver_feats = compute_driver_point_in_time_features(
            prior_race_results=prior_results,
            target_driver_id=driver_id,
        )
        circuit_feats = compute_circuit_point_in_time_features(
            prior_race_results=prior_results,
            target_circuit_id=circuit_id,
            target_franchise_id=franchise_id,
            target_driver_id=driver_id,
        )

        cls_pos_raw = res.get("classified_position")
        cls_pos = int(cls_pos_raw) if cls_pos_raw not in (None, "") else None
        finish_order = int(res.get("finish_position_order", 20))
        pts = float(res.get("points_awarded", 0.0))
        is_dnf = 1 if str(res.get("is_dnf")).lower() == "true" else 0
        is_mech_dnf = 1 if str(res.get("is_mechanical_dnf")).lower() == "true" else 0

        row: Dict[str, Any] = {
            # Identifiers & point-in-time metadata
            "event_id": event_id,
            "season": season,
            "round": rnd,
            "circuit_id": circuit_id,
            "driver_id": driver_id,
            "constructor_id": constructor_id,
            "franchise_id": franchise_id,
            "event_timestamp_utc": target_ts,
            "feature_cutoff_utc": as_of_cutoff_utc or target_ts,
            "prior_records_used": len(prior_results),
            "schema_version": SCHEMA_VERSION,
        }
        row.update(team_feats)
        row.update(driver_feats)
        row.update(circuit_feats)
        row["reg_electric_power_share"] = reg_elec_share
        row["reg_fastest_lap_point_active"] = reg_fl_active

        # Observed targets (strictly for post-cutoff evaluation / training labels)
        row["target_points"] = pts
        row["target_finish_order"] = finish_order
        row["target_classified_position"] = cls_pos if cls_pos is not None else ""
        row["target_is_win"] = 1 if cls_pos == 1 else 0
        row["target_is_podium"] = 1 if (cls_pos is not None and cls_pos <= 3) else 0
        row["target_is_top5"] = 1 if (cls_pos is not None and cls_pos <= 5) else 0
        row["target_is_points_finish"] = 1 if pts > 0.0 else 0
        row["target_is_dnf"] = is_dnf
        row["target_is_mechanical_dnf"] = is_mech_dnf

        output_rows.append(row)

    return output_rows


def build_point_in_time_feature_dataset(
    events: Sequence[Dict[str, Any]],
    race_results: Sequence[Dict[str, Any]],
    regulations: Sequence[Dict[str, Any]],
    as_of_cutoff_utc: Optional[str] = None,
    output_csv_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Build the complete point-in-time feature table across all chronologically ordered events.
    """
    ordered_events = sorted(
        events,
        key=lambda e: (str(e["event_timestamp_utc"]), int(e["season"]), int(e["round"])),
    )
    results_by_event: Dict[str, List[Dict[str, Any]]] = {}
    for r in race_results:
        results_by_event.setdefault(str(r["event_id"]), []).append(r)

    regs_by_season = {int(r["season"]): r for r in regulations}

    all_feature_rows: List[Dict[str, Any]] = []
    for ev in ordered_events:
        eid = str(ev["event_id"])
        ev_results = results_by_event.get(eid, [])
        if not ev_results:
            continue
        rows = build_event_feature_rows(
            target_event=ev,
            target_event_results=ev_results,
            all_historical_results=race_results,
            regulations_by_season=regs_by_season,
            as_of_cutoff_utc=as_of_cutoff_utc,
        )
        all_feature_rows.extend(rows)

    if output_csv_path is not None and all_feature_rows:
        output_csv_path = Path(output_csv_path)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_feature_rows[0].keys())
        with output_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_feature_rows:
                writer.writerow(r)

    return all_feature_rows

