"""
Circuit suitability and historical venue attrition features (FF27-09).
Computed strictly from prior editions of the target circuit (event_timestamp_utc < target_ts).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from src.features.team_features import DEFAULT_FIELD_DNF_RATE, FIELD_MEDIAN_POSITION


def compute_circuit_point_in_time_features(
    prior_race_results: Sequence[Dict[str, Any]],
    target_circuit_id: str,
    target_franchise_id: str,
    target_driver_id: str,
) -> Dict[str, Any]:
    """
    Compute point-in-time circuit suitability and venue attrition features
    from strictly prior events held at `target_circuit_id`.
    """
    circuit_rows = [
        r for r in prior_race_results if str(r.get("circuit_id")) == target_circuit_id
    ]
    circuit_team_rows = [
        r for r in circuit_rows if str(r.get("franchise_id")) == target_franchise_id
    ]
    circuit_driver_rows = [
        r for r in circuit_rows if str(r.get("driver_id")) == target_driver_id
    ]

    if not circuit_rows:
        return {
            "circuit_is_cold_start": 1,
            "circuit_team_prior_starts": 0,
            "circuit_team_pts_per_entry_hist": 0.0,
            "circuit_team_podium_rate_hist": 0.0,
            "circuit_driver_prior_starts": 0,
            "circuit_driver_mean_finish_hist": FIELD_MEDIAN_POSITION,
            "circuit_field_dnf_rate_hist": DEFAULT_FIELD_DNF_RATE,
        }

    field_dnf_rate = sum(
        1 for r in circuit_rows if str(r.get("is_dnf")).lower() == "true"
    ) / len(circuit_rows)

    if circuit_team_rows:
        team_pts_hist = sum(float(r.get("points_awarded", 0.0)) for r in circuit_team_rows) / len(
            circuit_team_rows
        )
        team_podium_hist = sum(
            1
            for r in circuit_team_rows
            if r.get("classified_position") not in (None, "")
            and int(r["classified_position"]) <= 3
        ) / len(circuit_team_rows)
    else:
        team_pts_hist = 0.0
        team_podium_hist = 0.0

    if circuit_driver_rows:
        drv_finishes = [
            float(r["classified_position"])
            if r.get("classified_position") not in (None, "")
            else float(r.get("finish_position_order", 20))
            for r in circuit_driver_rows
        ]
        drv_mean_finish = sum(drv_finishes) / len(drv_finishes)
    else:
        drv_mean_finish = FIELD_MEDIAN_POSITION

    return {
        "circuit_is_cold_start": 0,
        "circuit_team_prior_starts": len(circuit_team_rows),
        "circuit_team_pts_per_entry_hist": round(team_pts_hist, 4),
        "circuit_team_podium_rate_hist": round(team_podium_hist, 4),
        "circuit_driver_prior_starts": len(circuit_driver_rows),
        "circuit_driver_mean_finish_hist": round(drv_mean_finish, 4),
        "circuit_field_dnf_rate_hist": round(field_dnf_rate, 4),
    }

