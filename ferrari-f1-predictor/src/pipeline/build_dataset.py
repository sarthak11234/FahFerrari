"""
Stage 1 End-to-End Canonical Dataset Builder (FF27-06, FF27-07, FF27-08).
Builds the multi-team historical dataset sample with canonical IDs, source provenance,
ingestion manifests, and automated data-quality validation while preserving 2026 baseline files.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.ingestion.fastf1_collector import FastF1Collector
from src.ingestion.fia_documents import get_canonical_regulations_table
from src.ingestion.jolpica_collector import JolpicaCollector
from src.preprocessing.normalization import SCHEMA_VERSION
from src.preprocessing.validation import ValidationReport, validate_canonical_bundle


def _write_csv_table(rows: Sequence[Dict[str, Any]], output_path: Path) -> str:
    """Write a sequence of dicts to CSV and return the SHA-256 hex digest of the file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return hashlib.sha256(b"").hexdigest()

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return hashlib.sha256(output_path.read_bytes()).hexdigest()


def build_stage1_canonical_dataset(
    base_dir: Optional[Path] = None,
    start_season: int = 2019,
    end_season: int = 2025,
) -> Dict[str, Any]:
    """
    Execute the Stage 1 ingestion, normalization, and validation pipeline.

    Returns:
        Dictionary containing table counts, file paths, checksums, and validation report.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2]
    base_dir = Path(base_dir)

    staging_dir = base_dir / "data" / "staging"
    manifests_dir = base_dir / "data" / "manifests"
    staging_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    fastf1_collector = FastF1Collector(base_dir / "cache" / "fastf1_http_cache.sqlite")
    jolpica_collector = JolpicaCollector(base_dir / "cache" / "fastf1_http_cache.sqlite")

    # 1. Extract cached FastF1 schedules & stream inventory
    sprint_event_ids, fastf1_schedule_manifests = fastf1_collector.extract_cached_schedules()
    livetiming_manifests = fastf1_collector.summarize_cached_livetiming_streams()

    # 2. Extract and deduplicate Jolpica race payloads
    race_wrappers, jolpica_manifests = jolpica_collector.load_cached_race_payloads(
        start_season=start_season, end_season=end_season
    )

    # 3. Normalize into canonical tables
    events, circuits, entries, race_results = jolpica_collector.normalize_race_bundle(
        race_wrappers, sprint_event_ids=sprint_event_ids
    )
    regulations = get_canonical_regulations_table()

    # 4. Run hard data-quality validation
    report: ValidationReport = validate_canonical_bundle(
        events=events,
        circuits=circuits,
        entries=entries,
        race_results=race_results,
        regulations=regulations,
        raise_on_error=True,
    )

    # 5. Persist canonical tables to data/staging/
    tables_to_write = {
        "events": events,
        "circuits": circuits,
        "entries": entries,
        "race_results": race_results,
        "regulations": regulations,
    }
    output_artifacts: Dict[str, Dict[str, Any]] = {}
    for table_name, table_rows in tables_to_write.items():
        csv_path = staging_dir / f"{table_name}.csv"
        sha256_hex = _write_csv_table(table_rows, csv_path)
        output_artifacts[table_name] = {
            "path": str(csv_path.relative_to(base_dir)),
            "row_count": len(table_rows),
            "sha256": sha256_hex,
            "schema_version": SCHEMA_VERSION,
        }

    # 6. Write immutable ingestion manifest and validation report
    build_timestamp = datetime.now(timezone.utc).isoformat()
    manifest_payload = {
        "manifest_version": SCHEMA_VERSION,
        "generated_at_utc": build_timestamp,
        "season_range": [start_season, end_season],
        "canonical_tables": output_artifacts,
        "source_summary": {
            "jolpica_cached_responses": len(jolpica_manifests),
            "fastf1_schedule_responses": len(fastf1_schedule_manifests),
            "fastf1_livetiming_streams": len(livetiming_manifests),
            "fia_regulation_seasons": len(regulations),
        },
        "jolpica_payloads": jolpica_manifests,
        "fastf1_schedules": fastf1_schedule_manifests,
    }

    manifest_path = manifests_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    quality_report_path = manifests_dir / "stage1_quality_report.json"
    quality_report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")

    return {
        "manifest_path": str(manifest_path),
        "quality_report_path": str(quality_report_path),
        "output_artifacts": output_artifacts,
        "validation_report": report.to_dict(),
    }


if __name__ == "__main__":
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    summary = build_stage1_canonical_dataset(repo_root)
    val = summary["validation_report"]
    print(f"✅ Stage 1 canonical dataset built and validated (passed={val['passed']})")
    for tname, meta in summary["output_artifacts"].items():
        print(f"   - {tname}: {meta['row_count']} rows -> {meta['path']} (sha256={meta['sha256'][:12]}...)")
    print(f"   - Manifest: {summary['manifest_path']}")
    print(f"   - Quality report: {summary['quality_report_path']}")

