"""
Canonical ID normalization, constructor franchise lineage mapping, and status classification.
Implements contracts defined in docs/2027-data-dictionary.md (FF27-05, FF27-07).
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

SCHEMA_VERSION = "1.0.0"

# Longitudinal team lineage mapping across commercial rebrands (2014-2027)
CONSTRUCTOR_TO_FRANCHISE: Dict[str, str] = {
    "ferrari": "ferrari",
    "red_bull": "red_bull",
    "mercedes": "mercedes",
    "mclaren": "mclaren",
    "williams": "williams",
    "haas": "haas",
    # Aston Martin lineage
    "force_india": "aston_martin",
    "racing_point": "aston_martin",
    "aston_martin": "aston_martin",
    # Alpine / Renault lineage
    "lotus_f1": "alpine",
    "renault": "alpine",
    "alpine": "alpine",
    # Racing Bulls / AlphaTauri / Toro Rosso lineage
    "toro_rosso": "racing_bulls",
    "alphatauri": "racing_bulls",
    "rb": "racing_bulls",
    # Sauber / Alfa Romeo / Audi lineage
    "sauber": "sauber_audi",
    "alfa": "sauber_audi",
    "kick_sauber": "sauber_audi",
    "audi": "sauber_audi",
    # Historical / expansion teams
    "manor": "manor",
    "marussia": "manor",
    "caterham": "caterham",
    "cadillac": "cadillac",
}

_LAPPED_RE = re.compile(r"^\+\d+\s+Laps?$", re.IGNORECASE)

_INCIDENT_STATUSES = {
    "accident",
    "collision",
    "collision damage",
    "spun off",
    "damage",
    "debris",
    "puncture",
    "front wing",
    "rear wing",
    "undertray",
    "contact",
    "barrier",
    "stuck in gravel",
}

_MECHANICAL_KEYWORDS = (
    "brake",
    "clutch",
    "cooling",
    "differential",
    "driveshaft",
    "electrical",
    "electronics",
    "engine",
    "ers",
    "exhaust",
    "fire",
    "fuel",
    "gearbox",
    "handling",
    "hydraulic",
    "ignition",
    "mechanical",
    "mguk",
    "mguh",
    "oil",
    "overheating",
    "pneumatic",
    "power loss",
    "power unit",
    "radiator",
    "seat",
    "spark plug",
    "steering",
    "suspension",
    "throttle",
    "transmission",
    "turbo",
    "vibration",
    "water",
    "wheel",
)

_DNS_STATUSES = {
    "withdrew",
    "did not start",
    "did not qualify",
    "did not prequalify",
    "illness",
    "injury",
    "injured",
    "driver unwell",
    "107% rule",
}

_DQ_STATUSES = {
    "disqualified",
    "excluded",
}


def make_event_id(season: int, round_number: int) -> str:
    """Format canonical event_id as '{season}_r{round:02d}'."""
    if season < 1950 or season > 2100:
        raise ValueError(f"Invalid season: {season}")
    if round_number < 1 or round_number > 35:
        raise ValueError(f"Invalid round_number (must be >= 1, never 0): {round_number}")
    return f"{season}_r{round_number:02d}"


def make_session_id(event_id: str, session_code: str) -> str:
    """Format canonical session_id as '{event_id}_{session_code}'."""
    valid_codes = {"FP1", "FP2", "FP3", "SQ", "S", "Q", "R"}
    code = session_code.upper().strip()
    if code not in valid_codes:
        raise ValueError(f"Invalid session_code '{session_code}'. Expected one of {sorted(valid_codes)}")
    return f"{event_id}_{code}"


def map_franchise_id(constructor_id: str) -> str:
    """Map provider constructor_id to longitudinal franchise_id."""
    cid = constructor_id.strip().lower()
    return CONSTRUCTOR_TO_FRANCHISE.get(cid, cid)


def infer_regulation_era(season: int) -> str:
    """Return canonical regulation era label for a given season."""
    if 2014 <= season <= 2016:
        return "2014_2016_v6_hybrid"
    if 2017 <= season <= 2021:
        return "2017_2021_wide_aero"
    if 2022 <= season <= 2025:
        return "2022_2025_ground_effect"
    if season == 2026:
        return "2026_active_aero"
    if season >= 2027:
        return "2027_rebalanced_pu"
    return "pre_2014_historical"


def classify_race_status(
    status_raw: str, position_text: str, laps_completed: int = 0
) -> Tuple[str, bool, Optional[int], bool, bool]:
    """
    Normalize FIA/Jolpica status, positionText, and laps_completed into canonical status semantics.
    Fixes DEF-04 so lapped finishes ('+1 Lap') and disqualifications ('Disqualified')
    are never counted as mechanical or incident DNFs, and distinguishes pre-race withdrawals
    (laps_completed == 0) from mid-race withdrawals/retirements (laps_completed > 0).

    Returns:
        (status_category, is_classified, classified_position, is_dnf, is_mechanical_dnf)
    """
    status_clean = (status_raw or "").strip()
    status_lower = status_clean.lower()
    pos_text_clean = (position_text or "").strip().upper()

    is_numeric_pos = pos_text_clean.isdigit()
    classified_pos: Optional[int] = int(pos_text_clean) if is_numeric_pos else None

    # 1. Disqualified / Excluded
    if pos_text_clean in {"D", "E"} or status_lower in _DQ_STATUSES:
        return ("disqualified", False, None, False, False)

    # 2. Did Not Start / Withdrew before start (0 laps completed)
    if laps_completed == 0 and (pos_text_clean in {"W", "F"} or status_lower in _DNS_STATUSES):
        return ("did_not_start", False, None, False, False)

    # 3. Finished on lead lap
    if status_lower == "finished":
        return ("finished_lead_lap", is_numeric_pos, classified_pos, False, False)

    # 4. Finished lapped (+1 Lap, +2 Laps, Lapped)
    if _LAPPED_RE.match(status_clean) or status_lower == "lapped":
        return ("finished_lapped", is_numeric_pos, classified_pos, False, False)

    # 5. Incident / collision / driver error retirement
    if status_lower in _INCIDENT_STATUSES:
        return ("dnf_incident", is_numeric_pos, classified_pos, True, False)

    # 6. Mechanical / reliability retirement
    if any(kw in status_lower for kw in _MECHANICAL_KEYWORDS):
        return ("dnf_mechanical", is_numeric_pos, classified_pos, True, True)

    # 7. Unspecified / other retirement (e.g., 'Retired', 'Out of fuel', or mid-race 'Withdrew'/'Illness' with laps > 0)
    return ("dnf_other", is_numeric_pos, classified_pos, True, False)


def parse_lap_time_to_seconds(lap_time_str: Optional[str]) -> Optional[float]:
    """
    Convert a lap time string like '1:32.608' or '89.123' into float seconds.
    Returns None if missing or malformed.
    """
    if not lap_time_str or not isinstance(lap_time_str, str):
        return None
    raw = lap_time_str.strip()
    if not raw:
        return None
    try:
        if ":" in raw:
            parts = raw.split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return round(minutes * 60.0 + seconds, 3)
            if len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return round(hours * 3600.0 + minutes * 60.0 + seconds, 3)
        return round(float(raw), 3)
    except (ValueError, TypeError):
        return None


def make_event_timestamp_utc(event_date: str, event_time: Optional[str]) -> str:
    """Combine YYYY-MM-DD and HH:MM:SSZ into a standard ISO-8601 UTC timestamp."""
    date_part = (event_date or "").strip()
    time_part = (event_time or "14:00:00Z").strip()
    if not time_part.endswith("Z") and "+" not in time_part:
        time_part = f"{time_part}Z"
    return f"{date_part}T{time_part}"
