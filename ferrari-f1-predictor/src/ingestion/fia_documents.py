"""
FIA Sporting & Technical Regulation metadata registry (FF27-04, FF27-06).
Versions season-level regulatory parameters, scoring rules, and power-unit specifications
with explicit FIA source citations and effective timestamps.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.preprocessing.normalization import SCHEMA_VERSION, infer_regulation_era


def get_canonical_regulations_table(
    retrieved_at_utc: str = "2026-11-07T00:00:00Z",
) -> List[Dict[str, Any]]:
    """
    Return the versioned canonical `regulations` dataset for 2019–2027.
    All parameters are grounded in published FIA Sporting and Technical Regulations.
    """
    seasons_spec = [
        # (season, total_rounds, sprint_events, fl_point_active, ice_kw, mguk_kw, race_dist_km, effective_date, doc_url, is_provisional)
        (
            2019,
            21,
            0,
            True,
            570.0,
            120.0,
            305.0,
            "2019-03-15T00:00:00Z",
            "https://www.fia.com/regulation/category/110",
            False,
        ),
        (
            2020,
            17,
            0,
            True,
            570.0,
            120.0,
            305.0,
            "2020-07-03T00:00:00Z",
            "https://www.fia.com/regulation/category/110",
            False,
        ),
        (
            2021,
            22,
            3,
            True,
            570.0,
            120.0,
            305.0,
            "2021-03-26T00:00:00Z",
            "https://www.fia.com/regulation/category/110",
            False,
        ),
        (
            2022,
            22,
            3,
            True,
            575.0,
            120.0,
            305.0,
            "2022-03-18T00:00:00Z",
            "https://www.fia.com/regulation/category/110",
            False,
        ),
        (
            2023,
            22,
            6,
            True,
            575.0,
            120.0,
            305.0,
            "2023-03-03T00:00:00Z",
            "https://www.fia.com/regulation/category/110",
            False,
        ),
        (
            2024,
            24,
            6,
            True,
            575.0,
            120.0,
            305.0,
            "2024-02-29T00:00:00Z",
            "https://www.fia.com/regulation/category/110",
            False,
        ),
        (
            2025,
            24,
            6,
            False,
            575.0,
            120.0,
            305.0,
            "2025-03-14T00:00:00Z",
            "https://www.fia.com/regulation/category/110",
            False,
        ),
        (
            2026,
            24,
            6,
            False,
            400.0,
            350.0,
            305.0,
            "2026-03-06T00:00:00Z",
            "https://www.fia.com/regulation/category/110/technical-regulations",
            True,
        ),
        (
            2027,
            24,
            10,
            False,
            420.0,
            300.0,
            290.0,
            "2026-11-01T00:00:00Z",
            "https://www.fia.com/news/agreement-reached-proposed-regulatory-changes-2027-and-2028",
            True,
        ),
    ]

    rows: List[Dict[str, Any]] = []
    for (
        season,
        scheduled_rounds,
        sprint_count,
        fl_active,
        ice_kw,
        mguk_kw,
        dist_km,
        eff_date,
        url,
        is_prov,
    ) in seasons_spec:
        total_kw = ice_kw + mguk_kw
        rows.append(
            {
                "season": season,
                "regulation_version": f"{season}.v1",
                "regulation_era": infer_regulation_era(season),
                "scheduled_rounds": scheduled_rounds,
                "sprint_events_count": sprint_count,
                "fastest_lap_point_active": fl_active,
                "ice_max_power_kw": ice_kw,
                "mguk_max_power_kw": mguk_kw,
                "electric_power_share": round(mguk_kw / total_kw, 4),
                "standard_race_distance_km": dist_km,
                "effective_date_utc": eff_date,
                "source_provider": "fia",
                "source_record_id": f"fia:regulations:{season}.v1",
                "source_url": url,
                "retrieved_at_utc": retrieved_at_utc,
                "event_timestamp_utc": eff_date,
                "is_provisional": is_prov,
                "schema_version": SCHEMA_VERSION,
            }
        )
    return rows

