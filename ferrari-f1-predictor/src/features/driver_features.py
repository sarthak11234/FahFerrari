"""
Driver form and teammate-relative point-in-time features (FF27-09).
Separates individual driver skill/form from constructor car pace via same-event teammate deltas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from src.features.team_features import FIELD_MEDIAN_POSITION, _ewma


def compute_driver_point_in_time_features(
    prior_race_results: Sequence[Dict[str, Any]],
    target_driver_id: str,
) -> Dict[str, Any]:
    """
    Compute point-in-time driver and teammate-relative features using only strictly prior race results.
    """
    driver_rows = [
        r for r in prior_race_results if str(r.get("driver_id")) == target_driver_id
    ]
    prior_starts = len(driver_rows)

    if prior_starts == 0:
        return {
            "driver_is_cold_start": 1,
            "driver_prior_starts": 0,
            "driver_pts_ewma5": 0.0,
            "driver_finish_pos_mean5": FIELD_MEDIAN_POSITION,
            "driver_grid_pos_mean5": FIELD_MEDIAN_POSITION,
            "driver_positions_gained_mean5": 0.0,
            "driver_teammate_grid_delta_mean5": 0.0,
            "driver_teammate_finish_delta_mean5": 0.0,
            "driver_teammate_pts_share_10": 0.50,
            "driver_incident_dnf_rate_10": 0.05,
        }

    # Index all prior rows by (event_id, constructor_id) to pair teammates in the exact same car & event
    by_event_team: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in prior_race_results:
        key = (str(r["event_id"]), str(r["constructor_id"]))
        by_event_team.setdefault(key, []).append(r)

    last5 = driver_rows[-5:]
    last10 = driver_rows[-10:]

    pts_series = [float(r.get("points_awarded", 0.0)) for r in last5]
    pts_ewma5 = _ewma(pts_series, alpha=0.4)

    finish_positions_5 = [
        float(r["classified_position"])
        if r.get("classified_position") not in (None, "")
        else float(r.get("finish_position_order", 20))
        for r in last5
    ]
    grid_positions_5 = [
        float(r["grid_position"]) for r in last5 if int(r.get("grid_position", 0)) > 0
    ]

    pos_gained_5: List[float] = []
    for r in last5:
        grid_pos = int(r.get("grid_position", 0))
        cls_pos = r.get("classified_position")
        if grid_pos > 0 and cls_pos not in (None, ""):
            pos_gained_5.append(float(grid_pos - int(cls_pos)))

    # Teammate deltas (negative delta = driver started/finished AHEAD of teammate in lower position number)
    tm_grid_deltas: List[float] = []
    tm_finish_deltas: List[float] = []
    driver_pts_10 = 0.0
    pair_pts_10 = 0.0

    for r in last10:
        eid = str(r["event_id"])
        cid = str(r["constructor_id"])
        teammates = [
            tm
            for tm in by_event_team.get((eid, cid), [])
            if str(tm.get("driver_id")) != target_driver_id
        ]
        d_pts = float(r.get("points_awarded", 0.0))
        driver_pts_10 += d_pts
        pair_pts_10 += d_pts + sum(float(tm.get("points_awarded", 0.0)) for tm in teammates)

        if not teammates:
            continue
        tm = teammates[0]
        d_grid = int(r.get("grid_position", 0))
        t_grid = int(tm.get("grid_position", 0))
        if d_grid > 0 and t_grid > 0:
            tm_grid_deltas.append(float(d_grid - t_grid))

        d_fin = (
            int(r["classified_position"])
            if r.get("classified_position") not in (None, "")
            else int(r.get("finish_position_order", 20))
        )
        t_fin = (
            int(tm["classified_position"])
            if tm.get("classified_position") not in (None, "")
            else int(tm.get("finish_position_order", 20))
        )
        tm_finish_deltas.append(float(d_fin - t_fin))

    n10 = len(last10)
    incident_dnf_rate_10 = (
        sum(1 for r in last10 if str(r.get("status_category")) == "dnf_incident") / n10
        if n10 > 0
        else 0.05
    )

    return {
        "driver_is_cold_start": 0,
        "driver_prior_starts": prior_starts,
        "driver_pts_ewma5": round(pts_ewma5, 4),
        "driver_finish_pos_mean5": round(
            sum(finish_positions_5) / len(finish_positions_5)
            if finish_positions_5
            else FIELD_MEDIAN_POSITION,
            4,
        ),
        "driver_grid_pos_mean5": round(
            sum(grid_positions_5) / len(grid_positions_5)
            if grid_positions_5
            else FIELD_MEDIAN_POSITION,
            4,
        ),
        "driver_positions_gained_mean5": round(
            sum(pos_gained_5) / len(pos_gained_5) if pos_gained_5 else 0.0, 4
        ),
        "driver_teammate_grid_delta_mean5": round(
            sum(tm_grid_deltas[-5:]) / len(tm_grid_deltas[-5:]) if tm_grid_deltas else 0.0,
            4,
        ),
        "driver_teammate_finish_delta_mean5": round(
            sum(tm_finish_deltas[-5:]) / len(tm_finish_deltas[-5:])
            if tm_finish_deltas
            else 0.0,
            4,
        ),
        "driver_teammate_pts_share_10": round(
            driver_pts_10 / pair_pts_10 if pair_pts_10 > 0 else 0.50, 4
        ),
        "driver_incident_dnf_rate_10": round(incident_dnf_rate_10, 4),
    }

