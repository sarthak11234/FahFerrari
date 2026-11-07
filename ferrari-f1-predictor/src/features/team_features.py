"""
Constructor / franchise form, competitor-relative pace, and reliability features (FF27-09).
All features are computed strictly from historical events with event_timestamp_utc < target_ts.
Fixes DEF-05 (no backward fill), DEF-06 (no fixed /24 or /48 divisors), and DEF-07 (no subjective scores).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

FIELD_MEDIAN_POSITION = 10.5
DEFAULT_FIELD_DNF_RATE = 0.12
DEFAULT_MECH_DNF_RATE = 0.06


def _ewma(values: Sequence[float], alpha: float = 0.35) -> float:
    """Compute exponentially weighted moving average over chronologically ordered values."""
    if not values:
        return 0.0
    acc = float(values[0])
    for v in values[1:]:
        acc = alpha * float(v) + (1.0 - alpha) * acc
    return acc


def compute_team_point_in_time_features(
    prior_race_results: Sequence[Dict[str, Any]],
    target_franchise_id: str,
    target_season: int,
) -> Dict[str, Any]:
    """
    Compute point-in-time constructor/franchise features using only strictly prior race results.

    Args:
        prior_race_results: Chronologically sorted race_results rows with event_timestamp_utc < target_ts.
        target_franchise_id: Canonical longitudinal franchise_id (e.g. 'ferrari').
        target_season: Target event season year.
    """
    team_rows = [
        r for r in prior_race_results if str(r.get("franchise_id")) == target_franchise_id
    ]
    season_team_rows = [r for r in team_rows if int(r.get("season", 0)) == target_season]

    # Group prior rows by event_id to compute event-level constructor aggregates and field shares
    events_order: List[str] = []
    seen_events = set()
    by_event: Dict[str, List[Dict[str, Any]]] = {}
    for r in prior_race_results:
        eid = str(r["event_id"])
        if eid not in seen_events:
            seen_events.add(eid)
            events_order.append(eid)
        by_event.setdefault(eid, []).append(r)

    team_events_order = [
        eid
        for eid in events_order
        if any(str(r.get("franchise_id")) == target_franchise_id for r in by_event[eid])
    ]

    prior_team_entries = len(team_rows)
    if prior_team_entries == 0:
        return {
            "team_is_cold_start": 1,
            "team_prior_entries": 0,
            "team_pts_per_entry_ewma5": 0.0,
            "team_pts_per_entry_season_to_date": 0.0,
            "team_finish_pos_mean5": FIELD_MEDIAN_POSITION,
            "team_grid_pos_mean5": FIELD_MEDIAN_POSITION,
            "team_podium_rate_10": 0.0,
            "team_win_rate_10": 0.0,
            "team_dnf_rate_10": DEFAULT_FIELD_DNF_RATE,
            "team_mech_dnf_rate_10": DEFAULT_MECH_DNF_RATE,
            "team_field_pts_share_5": 0.10,
        }

    # Last 5 and last 10 team events (~10 and ~20 driver-race entries)
    last5_eids = set(team_events_order[-5:])
    last10_eids = set(team_events_order[-10:])

    last5_team_rows = [r for r in team_rows if str(r["event_id"]) in last5_eids]
    last10_team_rows = [r for r in team_rows if str(r["event_id"]) in last10_eids]

    # Event-level points per entry series for EWMA
    event_pts_per_entry: List[float] = []
    event_field_shares: List[float] = []
    for eid in team_events_order[-10:]:
        ev_rows = by_event[eid]
        ev_team = [r for r in ev_rows if str(r.get("franchise_id")) == target_franchise_id]
        team_pts = sum(float(r.get("points_awarded", 0.0)) for r in ev_team)
        field_pts = sum(float(r.get("points_awarded", 0.0)) for r in ev_rows)
        if ev_team:
            event_pts_per_entry.append(team_pts / len(ev_team))
        if field_pts > 0:
            event_field_shares.append(team_pts / field_pts)

    pts_ewma5 = _ewma(event_pts_per_entry[-5:], alpha=0.4)
    if season_team_rows:
        pts_std = sum(float(r.get("points_awarded", 0.0)) for r in season_team_rows) / len(
            season_team_rows
        )
    else:
        pts_std = pts_ewma5

    # Finish position & grid position over last 5 events (using finish_position_order for all entries)
    finish_positions_5 = [
        float(r["classified_position"])
        if r.get("classified_position") not in (None, "")
        else float(r.get("finish_position_order", 20))
        for r in last5_team_rows
    ]
    grid_positions_5 = [
        float(r["grid_position"])
        for r in last5_team_rows
        if int(r.get("grid_position", 0)) > 0
    ]

    finish_mean5 = (
        sum(finish_positions_5) / len(finish_positions_5)
        if finish_positions_5
        else FIELD_MEDIAN_POSITION
    )
    grid_mean5 = (
        sum(grid_positions_5) / len(grid_positions_5)
        if grid_positions_5
        else FIELD_MEDIAN_POSITION
    )

    n10 = len(last10_team_rows)
    podium_rate_10 = (
        sum(
            1
            for r in last10_team_rows
            if r.get("classified_position") not in (None, "")
            and int(r["classified_position"]) <= 3
        )
        / n10
        if n10 > 0
        else 0.0
    )
    win_rate_10 = (
        sum(
            1
            for r in last10_team_rows
            if r.get("classified_position") not in (None, "")
            and int(r["classified_position"]) == 1
        )
        / n10
        if n10 > 0
        else 0.0
    )
    dnf_rate_10 = (
        sum(1 for r in last10_team_rows if str(r.get("is_dnf")).lower() == "true") / n10
        if n10 > 0
        else DEFAULT_FIELD_DNF_RATE
    )
    mech_dnf_rate_10 = (
        sum(1 for r in last10_team_rows if str(r.get("is_mechanical_dnf")).lower() == "true")
        / n10
        if n10 > 0
        else DEFAULT_MECH_DNF_RATE
    )
    field_share_5 = (
        sum(event_field_shares[-5:]) / len(event_field_shares[-5:])
        if event_field_shares
        else 0.10
    )

    return {
        "team_is_cold_start": 0,
        "team_prior_entries": prior_team_entries,
        "team_pts_per_entry_ewma5": round(pts_ewma5, 4),
        "team_pts_per_entry_season_to_date": round(pts_std, 4),
        "team_finish_pos_mean5": round(finish_mean5, 4),
        "team_grid_pos_mean5": round(grid_mean5, 4),
        "team_podium_rate_10": round(podium_rate_10, 4),
        "team_win_rate_10": round(win_rate_10, 4),
        "team_dnf_rate_10": round(dnf_rate_10, 4),
        "team_mech_dnf_rate_10": round(mech_dnf_rate_10, 4),
        "team_field_pts_share_5": round(field_share_5, 4),
    }

