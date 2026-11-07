"""
FastF1 schedule and LiveTiming stream collector (FF27-04, FF27-06).
Extracts cached schedule metadata and session streams from fastf1_http_cache.sqlite
and provides session-level adapters for lap, stint, and weather extraction.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.preprocessing.normalization import SCHEMA_VERSION, make_event_id


class FastF1Collector:
    """Adapter for FastF1 event schedules and LiveTiming session streams."""

    def __init__(self, sqlite_cache_path: Optional[Path] = None):
        if sqlite_cache_path is None:
            repo_dir = Path(__file__).resolve().parents[2]
            sqlite_cache_path = repo_dir / "cache" / "fastf1_http_cache.sqlite"
        self.sqlite_cache_path = Path(sqlite_cache_path)

    def extract_cached_schedules(self) -> Tuple[Set[str], List[Dict[str, Any]]]:
        """
        Inspect cached FastF1 schedule_{year}.json payloads inside fastf1_http_cache.sqlite.

        Returns:
            (sprint_event_ids, manifest_records)
        """
        if not self.sqlite_cache_path.exists():
            return set(), []

        conn = sqlite3.connect(str(self.sqlite_cache_path))
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM responses")

        sprint_event_ids: Set[str] = set()
        manifests: List[Dict[str, Any]] = []

        for cache_key, blob in cur.fetchall():
            obj = pickle.loads(blob)
            url = str(obj.get("url", ""))
            if "f1schedule" not in url or not url.endswith(".json"):
                continue

            raw_bytes: bytes = obj.get("_content", b"")
            if not raw_bytes:
                continue

            # Extract season from schedule_YYYY.json
            fname = url.split("/")[-1]
            year_str = fname.replace("schedule_", "").replace(".json", "")
            if not year_str.isdigit():
                continue
            season = int(year_str)

            sched = json.loads(raw_bytes.decode("utf-8"))
            rounds = sched.get("round_number", {})
            formats = sched.get("event_format", {})

            round_count = 0
            for idx_key, rnd_val in rounds.items():
                rnd = int(rnd_val)
                if rnd <= 0:
                    continue
                round_count += 1
                fmt = str(formats.get(idx_key, "conventional")).lower()
                if "sprint" in fmt:
                    sprint_event_ids.add(make_event_id(season, rnd))

            manifests.append(
                {
                    "cache_key": cache_key,
                    "source_provider": "fastf1",
                    "source_url": url,
                    "retrieved_at_utc": str(obj.get("created_at", "")),
                    "sha256_checksum": hashlib.sha256(raw_bytes).hexdigest(),
                    "season": season,
                    "scheduled_events_count": round_count,
                }
            )

        conn.close()

        # Known FIA Sprint weekends in 2023-2025 for event_format tagging when schedule_2023..2025 is not in cache
        known_sprint_rounds = {
            2023: [4, 10, 12, 17, 18, 20],
            2024: [5, 6, 11, 19, 21, 23],
            2025: [2, 6, 13, 19, 21, 23],
        }
        for season, rnds in known_sprint_rounds.items():
            for rnd in rnds:
                sprint_event_ids.add(make_event_id(season, rnd))

        return sprint_event_ids, manifests

    def summarize_cached_livetiming_streams(self) -> List[Dict[str, Any]]:
        """
        Inventory all cached F1 LiveTiming .jsonStream payloads in fastf1_http_cache.sqlite
        for provenance tracking.
        """
        if not self.sqlite_cache_path.exists():
            return []

        conn = sqlite3.connect(str(self.sqlite_cache_path))
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM responses")

        streams: List[Dict[str, Any]] = []
        for cache_key, blob in cur.fetchall():
            obj = pickle.loads(blob)
            url = str(obj.get("url", ""))
            if "livetiming" not in url or ".jsonStream" not in url:
                continue
            raw_bytes: bytes = obj.get("_content", b"")
            streams.append(
                {
                    "cache_key": cache_key,
                    "source_provider": "fastf1_livetiming",
                    "source_url": url,
                    "retrieved_at_utc": str(obj.get("created_at", "")),
                    "sha256_checksum": hashlib.sha256(raw_bytes).hexdigest(),
                    "payload_bytes": len(raw_bytes),
                    "stream_type": url.split("/")[-1],
                    "schema_version": SCHEMA_VERSION,
                }
            )
        conn.close()
        return streams

