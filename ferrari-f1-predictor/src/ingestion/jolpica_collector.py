"""
Jolpica F1 API adapter with explicit pagination, SQLite cache extraction,
and canonical normalization (FF27-04, FF27-06).
Fixes DEF-01 (limit=100 truncation), DEF-02 (round=0), and DEF-03 (Ferrari-only filtering).
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.preprocessing.normalization import (
    SCHEMA_VERSION,
    classify_race_status,
    infer_regulation_era,
    make_event_id,
    make_event_timestamp_utc,
    map_franchise_id,
    parse_lap_time_to_seconds,
)

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
JOLPICA_MAX_PAGE_LIMIT = 100


class JolpicaCollector:
    """
    Collects multi-team F1 historical data from Jolpica (Ergast-compatible API)
    either from the committed requests-cache SQLite database or via paginated HTTP requests.
    """

    def __init__(self, sqlite_cache_path: Optional[Path] = None, base_url: str = JOLPICA_BASE_URL):
        self.base_url = base_url.rstrip("/")
        if sqlite_cache_path is None:
            repo_dir = Path(__file__).resolve().parents[2]
            sqlite_cache_path = repo_dir / "cache" / "fastf1_http_cache.sqlite"
        self.sqlite_cache_path = Path(sqlite_cache_path)

    @staticmethod
    def _sha256_bytes(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def load_cached_race_payloads(
        self, start_season: int = 2019, end_season: int = 2025
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract and deduplicate race payloads from the committed SQLite HTTP cache.
        Prefers complete per-round payloads over truncated season-level limit=100 pages,
        and discards incomplete boundary race fragments (<18 drivers, e.g. 2024 R06 1-row truncation).

        Returns:
            (deduplicated_race_dicts, manifest_records)
        """
        if not self.sqlite_cache_path.exists():
            raise FileNotFoundError(f"SQLite cache not found at {self.sqlite_cache_path}")

        conn = sqlite3.connect(str(self.sqlite_cache_path))
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM responses")

        # Keyed by (season, round) -> (race_dict, source_url, retrieved_at_utc, payload_sha256)
        best_races: Dict[Tuple[int, int], Tuple[Dict[str, Any], str, str, str]] = {}
        manifest_records: List[Dict[str, Any]] = []

        for cache_key, blob in cur.fetchall():
            obj = pickle.loads(blob)
            url = str(obj.get("url", ""))
            if "jolpi.ca/ergast/f1" not in url or "results.json" not in url:
                continue

            raw_bytes: bytes = obj.get("_content", b"")
            if not raw_bytes:
                continue

            payload_hash = self._sha256_bytes(raw_bytes)
            retrieved_at = str(obj.get("created_at", datetime.now(timezone.utc).isoformat()))
            data = json.loads(raw_bytes.decode("utf-8"))
            mr_data = data.get("MRData", {})
            races = mr_data.get("RaceTable", {}).get("Races", [])

            total_available = int(mr_data.get("total", 0))
            limit_used = int(mr_data.get("limit", 0))
            offset_used = int(mr_data.get("offset", 0))
            returned_rows = sum(len(r.get("Results", [])) for r in races)

            manifest_records.append(
                {
                    "cache_key": cache_key,
                    "source_provider": "jolpica",
                    "source_url": url,
                    "retrieved_at_utc": retrieved_at,
                    "sha256_checksum": payload_hash,
                    "mrdata_total": total_available,
                    "mrdata_limit": limit_used,
                    "mrdata_offset": offset_used,
                    "returned_result_rows": returned_rows,
                    "is_paginated_complete": (offset_used + returned_rows >= total_available),
                }
            )

            for race in races:
                season = int(race.get("season", 0))
                rnd = int(race.get("round", 0))
                if not (start_season <= season <= end_season):
                    continue
                results = race.get("Results", [])
                # Guard against limit=100 mid-race truncation (e.g., 2024 Round 6 having only 1 row)
                if len(results) < 18:
                    continue

                sr_key = (season, rnd)
                existing = best_races.get(sr_key)
                # Prefer per-round endpoint URLs or payloads with more driver entries
                if existing is None or len(results) > len(existing[0].get("Results", [])) or (
                    "?limit=" not in url and "?limit=" in existing[1]
                ):
                    best_races[sr_key] = (race, url, retrieved_at, payload_hash)

        conn.close()

        ordered_races: List[Dict[str, Any]] = []
        for (season, rnd) in sorted(best_races.keys()):
            race_obj, src_url, retrieved_at, payload_hash = best_races[(season, rnd)]
            ordered_races.append(
                {
                    "race": race_obj,
                    "source_url": src_url,
                    "retrieved_at_utc": retrieved_at,
                    "payload_sha256": payload_hash,
                }
            )

        return ordered_races, manifest_records

    def fetch_season_race_results_paginated(
        self, season: int, rate_limit_sleep_sec: float = 0.3
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Live paginated fetch from Jolpica respecting the server-side limit=100 cap (fixes DEF-01).
        Iterates offset=0, 100, 200... until offset + count >= total, merging split race rounds across page boundaries.
        """
        offset = 0
        total: Optional[int] = None
        races_by_round: Dict[int, Dict[str, Any]] = {}
        manifest_records: List[Dict[str, Any]] = []
        last_url = ""
        last_retrieved_at = ""
        last_hash = ""

        while total is None or offset < total:
            url = f"{self.base_url}/{season}/results.json?limit={JOLPICA_MAX_PAGE_LIMIT}&offset={offset}"
            req = urllib.request.Request(url, headers={"User-Agent": "FahFerrari2027/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw_bytes = resp.read()
            retrieved_at = datetime.now(timezone.utc).isoformat()
            payload_hash = self._sha256_bytes(raw_bytes)
            data = json.loads(raw_bytes.decode("utf-8"))

            mr = data.get("MRData", {})
            total = int(mr.get("total", 0))
            races = mr.get("RaceTable", {}).get("Races", [])
            page_rows = sum(len(r.get("Results", [])) for r in races)

            manifest_records.append(
                {
                    "cache_key": f"live_{season}_{offset}",
                    "source_provider": "jolpica",
                    "source_url": url,
                    "retrieved_at_utc": retrieved_at,
                    "sha256_checksum": payload_hash,
                    "mrdata_total": total,
                    "mrdata_limit": int(mr.get("limit", JOLPICA_MAX_PAGE_LIMIT)),
                    "mrdata_offset": int(mr.get("offset", offset)),
                    "returned_result_rows": page_rows,
                    "is_paginated_complete": (offset + page_rows >= total),
                }
            )

            for r in races:
                rnd = int(r["round"])
                if rnd not in races_by_round:
                    races_by_round[rnd] = r
                else:
                    # Merge results split across page boundary
                    seen_drivers = {
                        res["Driver"]["driverId"] for res in races_by_round[rnd]["Results"]
                    }
                    for res in r["Results"]:
                        if res["Driver"]["driverId"] not in seen_drivers:
                            races_by_round[rnd]["Results"].append(res)

            last_url = url
            last_retrieved_at = retrieved_at
            last_hash = payload_hash

            if page_rows == 0:
                break
            offset += page_rows
            if offset < total and rate_limit_sleep_sec > 0:
                time.sleep(rate_limit_sleep_sec)

        ordered = [
            {
                "race": races_by_round[rnd],
                "source_url": last_url,
                "retrieved_at_utc": last_retrieved_at,
                "payload_sha256": last_hash,
            }
            for rnd in sorted(races_by_round.keys())
        ]
        return ordered, manifest_records

    @staticmethod
    def normalize_race_bundle(
        race_wrappers: List[Dict[str, Any]],
        sprint_event_ids: Optional[Set[str]] = None,
    ) -> Tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
    ]:
        """
        Convert raw Jolpica race payloads into canonical tables:
        (events, circuits, entries, race_results)
        """
        sprint_event_ids = sprint_event_ids or set()
        events: List[Dict[str, Any]] = []
        circuits_by_id: Dict[str, Dict[str, Any]] = {}
        entries: List[Dict[str, Any]] = []
        race_results: List[Dict[str, Any]] = []

        for item in race_wrappers:
            race = item["race"]
            source_url = item["source_url"]
            retrieved_at_utc = item["retrieved_at_utc"]

            season = int(race["season"])
            rnd = int(race["round"])
            event_id = make_event_id(season, rnd)

            circuit = race.get("Circuit", {})
            circuit_id = str(circuit.get("circuitId", "")).strip().lower()
            loc = circuit.get("Location", {})

            event_date = str(race.get("date", ""))
            event_time = race.get("time")
            event_ts_utc = make_event_timestamp_utc(event_date, event_time)
            is_provisional = season >= 2026

            if circuit_id not in circuits_by_id:
                lat_raw = loc.get("lat")
                long_raw = loc.get("long")
                circuits_by_id[circuit_id] = {
                    "circuit_id": circuit_id,
                    "circuit_name": str(circuit.get("circuitName", "")),
                    "locality": str(loc.get("locality", "")),
                    "country": str(loc.get("country", "")),
                    "latitude": float(lat_raw) if lat_raw not in (None, "") else None,
                    "longitude": float(long_raw) if long_raw not in (None, "") else None,
                    "source_provider": "jolpica",
                    "source_record_id": f"jolpica:circuit:{circuit_id}",
                    "source_url": source_url,
                    "retrieved_at_utc": retrieved_at_utc,
                    "event_timestamp_utc": event_ts_utc,
                    "is_provisional": False,
                    "schema_version": SCHEMA_VERSION,
                }

            events.append(
                {
                    "event_id": event_id,
                    "season": season,
                    "round": rnd,
                    "circuit_id": circuit_id,
                    "event_name": str(race.get("raceName", "")),
                    "event_date": event_date,
                    "event_start_time_utc": str(event_time) if event_time else None,
                    "event_format": "sprint" if event_id in sprint_event_ids else "conventional",
                    "regulation_era": infer_regulation_era(season),
                    "source_provider": "jolpica",
                    "source_record_id": f"jolpica:{season}:r{rnd:02d}:event",
                    "source_url": source_url,
                    "retrieved_at_utc": retrieved_at_utc,
                    "event_timestamp_utc": event_ts_utc,
                    "is_provisional": is_provisional,
                    "schema_version": SCHEMA_VERSION,
                }
            )

            for res in race.get("Results", []):
                drv = res.get("Driver", {})
                cons = res.get("Constructor", {})
                driver_id = str(drv.get("driverId", "")).strip().lower()
                constructor_id = str(cons.get("constructorId", "")).strip().lower()
                franchise_id = map_franchise_id(constructor_id)

                car_num_raw = res.get("number")
                car_number = int(car_num_raw) if str(car_num_raw).isdigit() else None

                entries.append(
                    {
                        "event_id": event_id,
                        "season": season,
                        "round": rnd,
                        "driver_id": driver_id,
                        "driver_code": drv.get("code"),
                        "driver_given_name": str(drv.get("givenName", "")),
                        "driver_family_name": str(drv.get("familyName", "")),
                        "driver_dob": drv.get("dateOfBirth"),
                        "driver_nationality": drv.get("nationality"),
                        "constructor_id": constructor_id,
                        "franchise_id": franchise_id,
                        "car_number": car_number,
                        "source_provider": "jolpica",
                        "source_record_id": f"jolpica:{season}:r{rnd:02d}:entry:{driver_id}",
                        "source_url": source_url,
                        "retrieved_at_utc": retrieved_at_utc,
                        "event_timestamp_utc": event_ts_utc,
                        "is_provisional": is_provisional,
                        "schema_version": SCHEMA_VERSION,
                    }
                )

                pos_text = str(res.get("positionText", "")).strip()
                status_raw = str(res.get("status", "")).strip()
                laps_completed = int(res.get("laps", 0))
                (
                    status_category,
                    is_classified,
                    classified_position,
                    is_dnf,
                    is_mechanical_dnf,
                ) = classify_race_status(status_raw, pos_text, laps_completed=laps_completed)

                fl_obj = res.get("FastestLap", {}) or {}
                fl_rank_raw = fl_obj.get("rank")
                fl_rank = int(fl_rank_raw) if str(fl_rank_raw).isdigit() else None
                fl_time_str = (fl_obj.get("Time") or {}).get("time")
                fl_time_sec = parse_lap_time_to_seconds(fl_time_str)

                race_time_obj = res.get("Time") or {}
                race_time_ms_raw = race_time_obj.get("millis")
                race_time_ms = int(race_time_ms_raw) if str(race_time_ms_raw).isdigit() else None

                race_results.append(
                    {
                        "event_id": event_id,
                        "season": season,
                        "round": rnd,
                        "session_type": "R",
                        "driver_id": driver_id,
                        "constructor_id": constructor_id,
                        "franchise_id": franchise_id,
                        "circuit_id": circuit_id,
                        "grid_position": int(res.get("grid", 0)),
                        "finish_position_order": int(res.get("position", 0)),
                        "classified_position": classified_position,
                        "position_text": pos_text,
                        "is_classified": is_classified,
                        "status_raw": status_raw,
                        "status_category": status_category,
                        "is_dnf": is_dnf,
                        "is_mechanical_dnf": is_mechanical_dnf,
                        "laps_completed": int(res.get("laps", 0)),
                        "points_awarded": float(res.get("points", 0.0)),
                        "fastest_lap_rank": fl_rank,
                        "fastest_lap_time_sec": fl_time_sec,
                        "race_time_ms": race_time_ms,
                        "source_provider": "jolpica",
                        "source_record_id": f"jolpica:{season}:r{rnd:02d}:race_result:{driver_id}",
                        "source_url": source_url,
                        "retrieved_at_utc": retrieved_at_utc,
                        "event_timestamp_utc": event_ts_utc,
                        "is_provisional": is_provisional,
                        "schema_version": SCHEMA_VERSION,
                    }
                )

        circuits = [circuits_by_id[k] for k in sorted(circuits_by_id.keys())]
        return events, circuits, entries, race_results
