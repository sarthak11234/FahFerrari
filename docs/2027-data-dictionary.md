# FahFerrari 2027 — Source-Provider Evaluation & Canonical Data Dictionary

**Tasks covered:** `FF27-04` (Source-provider evaluation spike), `FF27-05` (Canonical IDs, schema contracts, and provenance)  
**Parent architecture:** [2027-architecture.md](./2027-architecture.md)  
**Implementation plan:** [2027-implementation-plan.md](./2027-implementation-plan.md)  
**Audit baseline:** [2026-retrospective.md](./2026-retrospective.md)  
**Schema version:** `1.0.0`

---

## 1. Source-Provider Comparison & Spike Findings (`FF27-04`)

### 1.1 Provider Matrix

| Dimension | Jolpica F1 API (`api.jolpi.ca/ergast/f1`) | FastF1 LiveTiming (`livetiming.formula1.com` / mirror) | FIA Official Publications (`fia.com`) |
|---|---|---|---|
| **Primary Role** | Canonical historical schedules, entries, race/qualifying/sprint classifications, standings, drivers, constructors, circuits (1950–present). | High-frequency session timing: lap times, sector splits, tyre stints, compound/age, pit-stop durations, weather streams, race control messages (2018–present). | Authoritative sporting/technical regulations, official entry lists, stewards' penalties, and post-session classifications. |
| **Pagination & Rate Limits** | **Server-side max `limit=100` per request** (even if `?limit=1000` is requested). Requires explicit `offset` pagination (`offset=0, 100, 200...`) until `offset + len(items) >= int(MRData.total)`. Rate limit: ~4 req/sec burst, 500 req/hr unauthenticated. | Static `.jsonStream` files per session (~6–12 HTTP requests per session). Heavy downloads; rate-limited or throttled if uncached. | HTML/PDF documents; manually curated and versioned into structured JSON/YAML regulation manifests (`regulations`). |
| **Caching Behaviour** | Cached via `requests-cache` SQLite (`fastf1_http_cache.sqlite`) or immutable raw JSON snapshots per `(endpoint, season, offset)`. | Cached via `fastf1.Cache.enable_cache()`. Stream responses are UTF-8-BOM line-delimited JSON prefixed by session timestamp (`HH:MM:SS.mmm{...}`). | Stored as versioned static metadata files with publication date, document title, and URL/checksum. |
| **Schema Stability** | High (`MRData` JSON envelope). Caveat: FastF1's `Ergast` wrapper splits race metadata into `.description` and results into `.content`, which caused `DEF-02` (`round=0`). Direct JSON parsing of `MRData.RaceTable.Races` preserves full hierarchy. | Moderate-to-High for 2018+. Pre-2018 lacks full telemetry/weather streams (`f1_api_support: false`). | Regulatory rules change by season/directive; represented via explicit rule-version tables. |
| **Coverage Gaps** | No lap-by-lap tyre compound, stint, weather, or sector time data; Sprint results live at a separate endpoint (`/sprint.json`); `status="Retired"` does not distinguish mechanical vs collision when `Ergast` lacks detail. | Practice/qualifying/race streams occasionally have missing lap/sector timestamps on red-flagged or aborted laps; historical telemetry before 2018 unavailable. | PDFs require structured human-verified transcription; never scraped automatically without validation. |
| **Usage Constraints** | Open-source Ergast-compatible service; must respect pagination and rate limits (≥0.25s delay between uncached requests). | For non-commercial research/analytical use; never fabricate or interpolate missing telemetry channels. | Public regulatory documents; cite exact issue number and publication date. |

### 1.2 Spike Verification against `fastf1_http_cache.sqlite`
Inspection of the 299 cached HTTP responses captured on `2026-01-08T04:52:51Z–04:58:06Z` confirmed:
1. **Jolpica JSON payloads (`79` responses):**
   - Full-season per-round responses (`/ergast/f1/{year}/{round}/results.json`) are already cached in `fastf1_http_cache.sqlite` for **2019 (all 21 rounds, 420 driver-event rows)**, **2020 (all 17 rounds, 340 rows)**, **2021 (all 22 rounds, 440 rows)**, and **2022 (Rounds 1–12, 240 rows)**, plus **Rounds 1–5 of 2023–2025** (`1,740` total driver-race records across all 10 teams).
   - Direct parsing of `MRData.RaceTable.Races` preserves `season`, `round`, `raceName`, `date`, `time`, `Circuit`, `Driver`, `Constructor`, `grid`, `position`, `positionText`, `points`, `laps`, `status`, `Time`, and `FastestLap`.
2. **FastF1 LiveTiming Streams (`216` responses across 72 races, 2019–2022):**
   - `SessionInfo.jsonStream` provides `Meeting.Key`, `OfficialName`, `Location`, `Country`, `Circuit.Key`, and session start/end timestamps.
   - `DriverList.jsonStream` provides `RacingNumber`, `Tla` (3-letter code), `FirstName`, `LastName`, `TeamName`, and `TeamColour`.
   - `RaceControlMessages.jsonStream` provides timestamped track status, safety car, VSC, red flag, and steward investigation messages.
3. **FastF1 Schedule Manifests (`4` responses, 2019–2022):**
   - `schedule_{year}.json` provides `round_number`, `country`, `location`, `official_event_name`, `event_date`, `event_format` (`conventional`, `sprint`, `sprint_shootout`, `sprint_qualifying`, `testing`), session names/dates (`session1`..`session5`), and `f1_api_support`.

---

## 2. Canonical Identifiers & Lineage Rules (`FF27-05`)

Every canonical table uses deterministic, human-readable primary and foreign keys while retaining provider-native identifiers.

### 2.1 Identifier Formats

| Entity | Canonical ID Column | Format / Pattern | Example | Provider-Native Fields Retained |
|---|---|---|---|---|
| **Season** | `season` | 4-digit integer (`YYYY`) | `2024` | `jolpica_season` |
| **Event (Grand Prix)** | `event_id` | `{season}_r{round:02d}` | `2024_r01` | `jolpica_round`, `fastf1_meeting_key`, `event_name` |
| **Session** | `session_id` | `{event_id}_{session_code}` (`FP1`, `FP2`, `FP3`, `SQ`, `S`, `Q`, `R`) | `2024_r01_R` | `fastf1_session_key` |
| **Driver** | `driver_id` | Lowercase canonical slug (aligned with Jolpica `driverId`) | `leclerc`, `hamilton`, `bearman` | `jolpica_driver_id`, `driver_code` (`LEC`), `permanent_number` (`16`) |
| **Constructor (Legal Entry)** | `constructor_id` | Lowercase season constructor slug (aligned with Jolpica `constructorId`) | `ferrari`, `red_bull`, `rb`, `sauber` | `jolpica_constructor_id`, `fastf1_team_name` |
| **Franchise (Team Lineage)** | `franchise_id` | Stable multi-season team identity across rebrands | `ferrari`, `racing_bulls`, `sauber_audi`, `alpine`, `aston_martin` | Mapped via `constructor_lineage` table |
| **Circuit** | `circuit_id` | Lowercase canonical circuit slug (aligned with Jolpica `circuitId`) | `bahrain`, `monaco`, `monza`, `jeddah` | `jolpica_circuit_id`, `fastf1_circuit_key` |

### 2.2 Constructor Franchise Lineage Mapping (2014–2027)
Because F1 teams change commercial names while retaining factory, chassis lineage, and personnel, both `constructor_id` (season-specific legal constructor) and `franchise_id` (longitudinal team lineage for hierarchical latent pace pooling) are recorded:

| `franchise_id` | `constructor_id` values mapped |
|---|---|
| `ferrari` | `ferrari` |
| `red_bull` | `red_bull` |
| `mercedes` | `mercedes` |
| `mclaren` | `mclaren` |
| `aston_martin` | `force_india` (2014–2018), `racing_point` (2019–2020), `aston_martin` (2021–2027) |
| `alpine` | `lotus_f1` (2014–2015), `renault` (2016–2020), `alpine` (2021–2027) |
| `williams` | `williams` |
| `racing_bulls` | `toro_rosso` (2014–2019), `alphatauri` (2020–2023), `rb` (2024–2027) |
| `sauber_audi` | `sauber` (2014–2018, 2024–2025), `alfa` (2019–2023), `audi` (2026–2027) |
| `haas` | `haas` (2016–2027) |
| `cadillac` | `cadillac` (where applicable in confirmed entry lists) |

---

## 3. Mandatory Provenance & Point-in-Time Columns

Every row in every canonical table MUST include the following provenance and point-in-time metadata columns:

| Column | Type | Nullable | Description |
|---|---|---|---|
| `source_provider` | `string` | `False` | Provider identifier: `'jolpica'`, `'fastf1'`, or `'fia'`. |
| `source_record_id` | `string` | `False` | Deterministic key within the source payload (e.g., `'jolpica:2024:r02:race_result:leclerc'`). |
| `source_url` | `string` | `False` | Exact HTTP endpoint or document URI from which the record was retrieved. |
| `retrieved_at_utc` | `string` (ISO-8601 UTC) | `False` | Exact UTC timestamp when the raw payload was fetched (e.g., `'2026-01-08T04:58:06.002056+00:00'`). |
| `event_timestamp_utc` | `string` (ISO-8601 UTC) | `True` | Official UTC completion/start timestamp of the session or event (`YYYY-MM-DDTHH:MM:SSZ`). Used as the point-in-time availability cutoff (`available_at_utc`) for backtesting. |
| `is_provisional` | `bool` | `False` | `True` if the season or classification is in-progress/unratified at snapshot time; `False` once the season/event is finalized. |
| `schema_version` | `string` | `False` | Semantic version of the canonical schema (`'1.0.0'`). |

---

## 4. Canonical Dataset Schemas

### 4.1 `events` (Grain: `event_id` — 1 row per Grand Prix weekend)
| Column | Type | Nullable | Units / Domain | Description |
|---|---|---|---|---|
| `event_id` | `string` | `False` | PK (`{season}_r{round:02d}`) | Canonical event identifier. |
| `season` | `int64` | `False` | `2014..2027` | Championship year (`> 0`). |
| `round` | `int64` | `False` | `1..25` | Official championship round number (`>= 1`; **never `0`**). |
| `circuit_id` | `string` | `False` | FK → `circuits.circuit_id` | Canonical circuit slug. |
| `event_name` | `string` | `False` | Text | Official Grand Prix name (e.g., `'Bahrain Grand Prix'`). |
| `event_date` | `string` | `False` | `YYYY-MM-DD` | Scheduled date of the Grand Prix race session. |
| `event_start_time_utc` | `string` | `True` | `HH:MM:SSZ` | Scheduled UTC race start time (`None` for pre-2005 or unannounced times). |
| `event_format` | `string` | `False` | `'conventional'`, `'sprint'` | Weekend format indicator. |
| `regulation_era` | `string` | `False` | `'2014_2016_v6_hybrid'`, `'2017_2021_wide_aero'`, `'2022_2025_ground_effect'`, `'2026_active_aero'`, `'2027_rebalanced_pu'` | Technical regulation regime indicator. |
| *+ Provenance columns* | — | — | See §3 | `source_provider`, `source_record_id`, `source_url`, `retrieved_at_utc`, `event_timestamp_utc`, `is_provisional`, `schema_version`. |

### 4.2 `circuits` (Grain: `circuit_id` — 1 row per circuit)
| Column | Type | Nullable | Units / Domain | Description |
|---|---|---|---|---|
| `circuit_id` | `string` | `False` | PK | Canonical circuit slug (e.g., `'monaco'`, `'monza'`). |
| `circuit_name` | `string` | `False` | Text | Official venue name (e.g., `'Autodromo Nazionale di Monza'`). |
| `locality` | `string` | `False` | Text | City / locality name. |
| `country` | `string` | `False` | Text | Country name. |
| `latitude` | `float64` | `True` | Degrees (`-90..90`) | WGS84 latitude from provider metadata. |
| `longitude` | `float64` | `True` | Degrees (`-180..180`) | WGS84 longitude from provider metadata. |
| *+ Provenance columns* | — | — | See §3 | Mandatory provenance metadata. |

### 4.3 `entries` (Grain: `(event_id, driver_id)` — 1 row per driver entered in an event)
| Column | Type | Nullable | Units / Domain | Description |
|---|---|---|---|---|
| `event_id` | `string` | `False` | PK, FK → `events` | Canonical event ID. |
| `season` | `int64` | `False` | `2014..2027` | Season year. |
| `round` | `int64` | `False` | `1..25` | Round number (`>= 1`). |
| `driver_id` | `string` | `False` | PK | Canonical driver slug. |
| `driver_code` | `string` | `True` | 3-char uppercase (`LEC`) | FIA 3-letter driver TLA. |
| `driver_given_name` | `string` | `False` | Text | Driver first name. |
| `driver_family_name` | `string` | `False` | Text | Driver last name. |
| `driver_dob` | `string` | `True` | `YYYY-MM-DD` | Driver date of birth. |
| `driver_nationality` | `string` | `True` | Text | Driver nationality. |
| `constructor_id` | `string` | `False` | FK | Legal constructor slug for this event. |
| `franchise_id` | `string` | `False` | FK | Stable multi-season franchise slug (§2.2). |
| `car_number` | `int64` | `True` | `1..99` | Racing number on the car for that event. |
| *+ Provenance columns* | — | — | See §3 | Mandatory provenance metadata. |

### 4.4 `race_results` & `sprint_results` (Grain: `(event_id, driver_id)`)
| Column | Type | Nullable | Units / Domain | Description |
|---|---|---|---|---|
| `event_id` | `string` | `False` | PK, FK → `events` | Canonical event ID. |
| `season` | `int64` | `False` | `2014..2027` | Season year. |
| `round` | `int64` | `False` | `1..25` | Round number (`>= 1`). |
| `session_type` | `string` | `False` | `'R'` or `'S'` | `'R'` (Grand Prix) or `'S'` (Sprint). |
| `driver_id` | `string` | `False` | PK, FK → `entries` | Canonical driver slug. |
| `constructor_id` | `string` | `False` | FK | Legal constructor slug. |
| `franchise_id` | `string` | `False` | FK | Longitudinal franchise slug. |
| `circuit_id` | `string` | `False` | FK → `circuits` | Canonical circuit slug. |
| `grid_position` | `int64` | `True` | `0..24` (`0` = pit-lane/unassigned start) | Starting grid position. |
| `finish_position_order` | `int64` | `False` | `1..24` | Official order in race classification table (`position` in Jolpica). |
| `classified_position` | `int64` | `True` | `1..24` or `None` | Numeric position **only if officially classified** (`positionText.isdigit()`); `None` (`NaN`) if `position_text` is `'R'`, `'D'`, `'W'`, `'N'`, `'E'`, or `'F'`. |
| `position_text` | `string` | `False` | `'1'..'24'`, `'R'`, `'D'`, `'W'`, `'N'` | Raw classification symbol from FIA/Jolpica. |
| `is_classified` | `bool` | `False` | `True` / `False` | `True` iff driver received an official numeric classification (`position_text.isdigit()`). |
| `status_raw` | `string` | `False` | Provider string | Exact status string (e.g., `'Finished'`, `'+1 Lap'`, `'Driveshaft'`, `'Disqualified'`). |
| `status_category` | `string` | `False` | Enum (see §4.5) | Normalized outcome category (`'finished_lead_lap'`, `'finished_lapped'`, `'dnf_mechanical'`, `'dnf_incident'`, `'dnf_other'`, `'disqualified'`, `'did_not_start'`). |
| `is_dnf` | `bool` | `False` | `True` / `False` | `True` **only** for `status_category` in `{'dnf_mechanical', 'dnf_incident', 'dnf_other'}`. Explicitly `False` for `finished_lapped` and `disqualified` (fixing `DEF-04`). |
| `is_mechanical_dnf` | `bool` | `False` | `True` / `False` | `True` iff `status_category == 'dnf_mechanical'`. |
| `laps_completed` | `int64` | `False` | `>= 0` | Number of race laps completed. |
| `points_awarded` | `float64` | `False` | `0.0..26.0` (GP) / `0.0..8.0` (Sprint) | Total official championship points awarded for this session. |
| `fastest_lap_rank` | `int64` | `True` | `1..24` or `None` | Rank of driver's fastest lap in the race (`None` if no lap completed). |
| `fastest_lap_time_sec` | `float64` | `True` | Seconds (`> 0`) | Driver's fastest lap duration in seconds (`None` if unavailable). |
| `race_time_ms` | `int64` | `True` | Milliseconds (`> 0`) | Total race elapsed time in milliseconds (`None` if lapped or retired). |
| *+ Provenance columns* | — | — | See §3 | Mandatory provenance metadata. |

### 4.5 Status Normalization Taxonomy (Fixing `DEF-04`)
To prevent conflating lapped finishes (`+1 Lap`), withdrawals (`Withdrew`), and post-race disqualifications (`Disqualified`) with retirements:

| `status_category` | `is_classified` | `is_dnf` | `is_mechanical_dnf` | Matching `status_raw` patterns |
|---|---|---|---|---|
| `'finished_lead_lap'` | `True` | `False` | `False` | `'Finished'` |
| `'finished_lapped'` | `True` | `False` | `False` | `'^\+\d+ Laps?$'`, `'Lapped'` |
| `'disqualified'` | `False` | `False` | `False` | `'Disqualified'`, `'Excluded'` (or `position_text in {'D', 'E'}`) |
| `'did_not_start'` | `False` | `False` | `False` | `'Withdrew'`, `'Did not start'`, `'Did not qualify'`, `'Illness'`, `'Injury'` (or `position_text in {'W', 'F'}`) |
| `'dnf_incident'` | `False` (or `True` if ≥90% distance) | `True` | `False` | `'Accident'`, `'Collision'`, `'Collision damage'`, `'Spun off'`, `'Damage'`, `'Debris'`, `'Puncture'`, `'Front wing'`, `'Rear wing'`, `'Undertray'` |
| `'dnf_mechanical'` | `False` (or `True` if ≥90% distance) | `True` | `True` | `'Brakes'`, `'Cooling system'`, `'Driveshaft'`, `'Electrical'`, `'Electronics'`, `'Engine'`, `'Exhaust'`, `'Fuel.*'`, `'Gearbox'`, `'Hydraulics'`, `'Mechanical'`, `'Oil.*'`, `'Overheating'`, `'Power Unit'`, `'Power loss'`, `'Radiator'`, `'Suspension'`, `'Transmission'`, `'Turbo'`, `'Vibrations'`, `'Water.*'`, `'Wheel.*'` |
| `'dnf_other'` | `False` (or `True` if ≥90% distance) | `True` | `False` | `'Retired'`, `'Out of fuel'`, or any unclassified retirement string not matching mechanical/incident patterns. |

### 4.6 Additional Canonical Tables (Stage 2–4 Expansion)
- **`qualifying_results` (Grain: `(event_id, driver_id)`):** `qualifying_position`, `q1_time_sec`, `q2_time_sec`, `q3_time_sec`, `best_quali_time_sec`, `teammate_quali_delta_sec` + provenance.
- **`lap_times` (Grain: `(session_id, driver_id, lap_number)`):** `lap_time_sec`, `sector1_sec`, `sector2_sec`, `sector3_sec`, `position`, `is_pit_in_lap`, `is_pit_out_lap`, `track_status`, `is_accurate` + provenance.
- **`stints` (Grain: `(session_id, driver_id, stint_number)`):** `compound` (`SOFT`, `MEDIUM`, `HARD`, `INTERMEDIATE`, `WET`), `start_lap`, `end_lap`, `tyre_age_at_start`, `stint_laps` + provenance.
- **`weather` (Grain: `(session_id, sample_offset_sec)`):** `air_temp_c`, `track_temp_c`, `humidity_pct`, `pressure_mbar`, `rainfall`, `wind_speed_ms`, `wind_direction_deg` + provenance.
- **`regulations` (Grain: `(season, regulation_version)`):** `ice_max_power_kw`, `mguk_max_power_kw`, `standard_race_distance_km`, `fastest_lap_point_active`, `sprint_events_count`, `effective_date_utc`, `source_document_url` + provenance.
- **`source_manifest` (Grain: `manifest_id`):** Tracks every ingestion run with `run_id`, `created_at_utc`, `provider`, `query_parameters`, `raw_payload_path`, `sha256_checksum`, `row_count`, and `is_complete`.

---

## 5. Null Semantics, Units & Data-Quality Invariants

1. **Explicit Nulls:** Missing numeric measurements (e.g., `classified_position` for a retired driver, `race_time_ms` for a lapped driver, `fastest_lap_time_sec` for a Lap-1 retirement) are stored as `NaN` / `null`—**never encoded as `0`, `-1`, or string `'D'` in numeric columns**.
2. **Zero vs Missing:** `round` must be an integer `>= 1` (`0` is rejected by validation). `grid_position == 0` is permitted only when the provider explicitly indicates a pit-lane or unclassified grid start; `points_awarded == 0.0` means zero points scored.
3. **Standardized Units:**
   - Lap and sector durations: `float64` **seconds** (`_sec`).
   - Total race elapsed time: `int64` **milliseconds** (`_ms`).
   - Temperatures: `float64` **degrees Celsius** (`_c`).
   - Distances: `float64` **kilometers** (`_km`).
   - Power: `float64` **kilowatts** (`_kw`).
   - Timestamps: **UTC ISO-8601** strings (`_utc`).
4. **Leakage Prevention Rule:** Any feature computed for target event `E` with start time $t_E$ may only read canonical records where `event_timestamp_utc < t_E` and `retrieved_at_utc <= cutoff_utc`.

