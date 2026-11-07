"""
Monte Carlo Season Simulator (Stage 5).
Executes probabilistic full-season simulations (10,000+ runs) applying versioned points rules,
Sprint formats, driver rosters, and deterministic random seeds.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.models.race_outcome import ProbabilisticRaceOutcomeModel, SimulatedRaceResult


# 2027 Official Calendar Specification (24 Grands Prix, 10 Sprints)
CALENDAR_2027: List[Dict[str, Any]] = [
    {"round": 1, "circuit_id": "bahrain", "name": "Bahrain Grand Prix", "is_sprint": False},
    {"round": 2, "circuit_id": "jeddah", "name": "Saudi Arabian Grand Prix", "is_sprint": False},
    {"round": 3, "circuit_id": "albert_park", "name": "Australian Grand Prix", "is_sprint": False},
    {"round": 4, "circuit_id": "suzuka", "name": "Japanese Grand Prix", "is_sprint": False},
    {"round": 5, "circuit_id": "shanghai", "name": "Chinese Grand Prix", "is_sprint": True},
    {"round": 6, "circuit_id": "miami", "name": "Miami Grand Prix", "is_sprint": True},
    {"round": 7, "circuit_id": "imola", "name": "Emilia Romagna Grand Prix", "is_sprint": False},
    {"round": 8, "circuit_id": "monaco", "name": "Monaco Grand Prix", "is_sprint": False},
    {"round": 9, "circuit_id": "catalunya", "name": "Spanish Grand Prix", "is_sprint": False},
    {"round": 10, "circuit_id": "villeneuve", "name": "Canadian Grand Prix", "is_sprint": True},
    {"round": 11, "circuit_id": "red_bull_ring", "name": "Austrian Grand Prix", "is_sprint": True},
    {"round": 12, "circuit_id": "silverstone", "name": "British Grand Prix", "is_sprint": False},
    {"round": 13, "circuit_id": "spa", "name": "Belgian Grand Prix", "is_sprint": True},
    {"round": 14, "circuit_id": "hungaroring", "name": "Hungarian Grand Prix", "is_sprint": False},
    {"round": 15, "circuit_id": "zandvoort", "name": "Dutch Grand Prix", "is_sprint": False},
    {"round": 16, "circuit_id": "monza", "name": "Italian Grand Prix", "is_sprint": False},
    {"round": 17, "circuit_id": "baku", "name": "Azerbaijan Grand Prix", "is_sprint": False},
    {"round": 18, "circuit_id": "marina_bay", "name": "Singapore Grand Prix", "is_sprint": True},
    {"round": 19, "circuit_id": "americas", "name": "United States Grand Prix", "is_sprint": True},
    {"round": 20, "circuit_id": "rodriguez", "name": "Mexico City Grand Prix", "is_sprint": False},
    {"round": 21, "circuit_id": "interlagos", "name": "São Paulo Grand Prix", "is_sprint": True},
    {"round": 22, "circuit_id": "las_vegas", "name": "Las Vegas Grand Prix", "is_sprint": False},
    {"round": 23, "circuit_id": "losail", "name": "Qatar Grand Prix", "is_sprint": True},
    {"round": 24, "circuit_id": "yas_marina", "name": "Abu Dhabi Grand Prix", "is_sprint": True},
]

# 2027 Confirmed Full-Field Grid (10 Constructors, 20 Drivers)
ROSTER_2027: List[Dict[str, str]] = [
    {"driver_id": "leclerc", "constructor_id": "ferrari", "franchise_id": "ferrari"},
    {"driver_id": "hamilton", "constructor_id": "ferrari", "franchise_id": "ferrari"},
    {"driver_id": "max_verstappen", "constructor_id": "red_bull", "franchise_id": "red_bull"},
    {"driver_id": "tsunoda", "constructor_id": "red_bull", "franchise_id": "red_bull"},
    {"driver_id": "norris", "constructor_id": "mclaren", "franchise_id": "mclaren"},
    {"driver_id": "piastri", "constructor_id": "mclaren", "franchise_id": "mclaren"},
    {"driver_id": "russell", "constructor_id": "mercedes", "franchise_id": "mercedes"},
    {"driver_id": "antonelli", "constructor_id": "mercedes", "franchise_id": "mercedes"},
    {"driver_id": "alonso", "constructor_id": "aston_martin", "franchise_id": "aston_martin"},
    {"driver_id": "stroll", "constructor_id": "aston_martin", "franchise_id": "aston_martin"},
    {"driver_id": "gasly", "constructor_id": "alpine", "franchise_id": "alpine"},
    {"driver_id": "doohan", "constructor_id": "alpine", "franchise_id": "alpine"},
    {"driver_id": "albon", "constructor_id": "williams", "franchise_id": "williams"},
    {"driver_id": "sainz", "constructor_id": "williams", "franchise_id": "williams"},
    {"driver_id": "hadjar", "constructor_id": "rb", "franchise_id": "racing_bulls"},
    {"driver_id": "lawson", "constructor_id": "rb", "franchise_id": "racing_bulls"},
    {"driver_id": "hulkenberg", "constructor_id": "audi", "franchise_id": "sauber_audi"},
    {"driver_id": "bortoleto", "constructor_id": "audi", "franchise_id": "sauber_audi"},
    {"driver_id": "bearman", "constructor_id": "haas", "franchise_id": "haas"},
    {"driver_id": "ocon", "constructor_id": "haas", "franchise_id": "haas"},
]


def _percentile(sorted_values: Sequence[float], p: float) -> float:
    """Calculate percentile value on sorted sequence (p in [0, 100])."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[int(f)] * (c - k)
    d1 = sorted_values[int(c)] * (k - f)
    return round(d0 + d1, 1)


class SeasonSimulator:
    """
    Monte Carlo full-season simulation engine.
    Runs N full-season realizations and returns calibrated distributions,
    percentiles, and event-level expectations.
    """

    def __init__(
        self,
        race_model: ProbabilisticRaceOutcomeModel,
        calendar: Optional[List[Dict[str, Any]]] = None,
        roster: Optional[List[Dict[str, str]]] = None,
    ):
        self.race_model = race_model
        self.calendar = calendar or CALENDAR_2027
        self.roster = roster or ROSTER_2027

    def run_simulation(
        self,
        num_seasons: int = 10000,
        random_seed: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """
        Run `num_seasons` full Monte Carlo simulations with seeded pseudo-random generator.
        """
        rng = random.Random(random_seed)

        # Accumulators
        ferrari_constructor_pts: List[float] = []
        ferrari_constructor_ranks: List[int] = []

        leclerc_pts: List[float] = []
        leclerc_ranks: List[int] = []
        leclerc_wins: List[int] = []
        leclerc_podiums: List[int] = []
        leclerc_dnfs: List[int] = []

        hamilton_pts: List[float] = []
        hamilton_ranks: List[int] = []
        hamilton_wins: List[int] = []
        hamilton_podiums: List[int] = []
        hamilton_dnfs: List[int] = []

        all_constructor_totals: Dict[str, List[float]] = defaultdict(list)
        all_constructor_ranks: Dict[str, List[int]] = defaultdict(list)

        # Race-by-race tracking for Ferrari
        race_stats: Dict[int, Dict[str, Any]] = {
            ev["round"]: {
                "round": ev["round"],
                "circuit_id": ev["circuit_id"],
                "name": ev["name"],
                "is_sprint": ev["is_sprint"],
                "ferrari_points_samples": [],
                "ferrari_wins_count": 0,
                "ferrari_podiums_count": 0,
                "ferrari_dnf_count": 0,
            }
            for ev in self.calendar
        }

        for _ in range(num_seasons):
            season_driver_pts: Dict[str, float] = defaultdict(float)
            season_driver_wins: Dict[str, int] = defaultdict(int)
            season_driver_podiums: Dict[str, int] = defaultdict(int)
            season_driver_dnfs: Dict[str, int] = defaultdict(int)
            season_constructor_pts: Dict[str, float] = defaultdict(float)

            for event in self.calendar:
                rnd = event["round"]
                cid = event["circuit_id"]
                is_sprint = event["is_sprint"]
                eid = f"2027_r{rnd:02d}"

                # 1. Simulate Sprint race if scheduled
                if is_sprint:
                    sprint_res = self.race_model.simulate_race(
                        event_id=f"{eid}_S",
                        round_number=rnd,
                        circuit_id=cid,
                        entries=self.roster,
                        is_sprint=True,
                        fastest_lap_point_active=False,
                        rng=rng,
                    )
                    for dr in sprint_res.driver_results:
                        season_driver_pts[dr.driver_id] += dr.points_scored
                        season_constructor_pts[dr.constructor_id] += dr.points_scored

                # 2. Simulate Grand Prix race
                gp_res = self.race_model.simulate_race(
                    event_id=eid,
                    round_number=rnd,
                    circuit_id=cid,
                    entries=self.roster,
                    is_sprint=False,
                    fastest_lap_point_active=False,  # Abolished from 2025+
                    rng=rng,
                )

                ev_ferrari_pts = 0.0
                ev_ferrari_win = 0
                ev_ferrari_podium = 0
                ev_ferrari_dnf = 0

                for dr in gp_res.driver_results:
                    did = dr.driver_id
                    season_driver_pts[did] += dr.points_scored
                    season_constructor_pts[dr.constructor_id] += dr.points_scored
                    if dr.finish_position == 1:
                        season_driver_wins[did] += 1
                    if dr.finish_position <= 3:
                        season_driver_podiums[did] += 1
                    if dr.is_dnf:
                        season_driver_dnfs[did] += 1

                    if dr.constructor_id == "ferrari":
                        ev_ferrari_pts += dr.points_scored
                        if dr.finish_position == 1:
                            ev_ferrari_win = 1
                        if dr.finish_position <= 3:
                            ev_ferrari_podium += 1
                        if dr.is_dnf:
                            ev_ferrari_dnf += 1

                race_stats[rnd]["ferrari_points_samples"].append(ev_ferrari_pts)
                race_stats[rnd]["ferrari_wins_count"] += ev_ferrari_win
                race_stats[rnd]["ferrari_podiums_count"] += ev_ferrari_podium
                race_stats[rnd]["ferrari_dnf_count"] += ev_ferrari_dnf

            # Rank drivers
            sorted_drivers = sorted(
                season_driver_pts.items(), key=lambda x: -x[1]
            )
            driver_ranks = {did: idx + 1 for idx, (did, _) in enumerate(sorted_drivers)}

            # Rank constructors
            sorted_constructors = sorted(
                season_constructor_pts.items(), key=lambda x: -x[1]
            )
            constructor_ranks = {
                cid: idx + 1 for idx, (cid, _) in enumerate(sorted_constructors)
            }

            # Record season samples
            for cid, pts in season_constructor_pts.items():
                all_constructor_totals[cid].append(pts)
                all_constructor_ranks[cid].append(constructor_ranks[cid])

            ferrari_constructor_pts.append(season_constructor_pts["ferrari"])
            ferrari_constructor_ranks.append(constructor_ranks["ferrari"])

            leclerc_pts.append(season_driver_pts["leclerc"])
            leclerc_ranks.append(driver_ranks["leclerc"])
            leclerc_wins.append(season_driver_wins["leclerc"])
            leclerc_podiums.append(season_driver_podiums["leclerc"])
            leclerc_dnfs.append(season_driver_dnfs["leclerc"])

            hamilton_pts.append(season_driver_pts["hamilton"])
            hamilton_ranks.append(driver_ranks["hamilton"])
            hamilton_wins.append(season_driver_wins["hamilton"])
            hamilton_podiums.append(season_driver_podiums["hamilton"])
            hamilton_dnfs.append(season_driver_dnfs["hamilton"])

        # Aggregate distributions & summary statistics
        ferrari_pts_sorted = sorted(ferrari_constructor_pts)
        leclerc_pts_sorted = sorted(leclerc_pts)
        hamilton_pts_sorted = sorted(hamilton_pts)

        # Rank position probability tables
        ferrari_rank_probs = {
            f"P{p}": round(ferrari_constructor_ranks.count(p) / num_seasons, 4)
            for p in range(1, 11)
        }
        leclerc_rank_probs = {
            f"P{p}": round(leclerc_ranks.count(p) / num_seasons, 4)
            for p in range(1, 11)
        }
        hamilton_rank_probs = {
            f"P{p}": round(hamilton_ranks.count(p) / num_seasons, 4)
            for p in range(1, 11)
        }

        # Constructor Standings Summary
        constructors_summary = []
        for cid, pts_list in all_constructor_totals.items():
            pts_sorted = sorted(pts_list)
            ranks_list = all_constructor_ranks[cid]
            p1_prob = ranks_list.count(1) / num_seasons
            p3_prob = sum(1 for r in ranks_list if r <= 3) / num_seasons
            mean_pts = sum(pts_list) / num_seasons
            constructors_summary.append(
                {
                    "constructor_id": cid,
                    "mean_points": round(mean_pts, 1),
                    "median_points": round(_percentile(pts_sorted, 50), 1),
                    "ci_90": [round(_percentile(pts_sorted, 5), 1), round(_percentile(pts_sorted, 95), 1)],
                    "prob_p1": round(p1_prob, 4),
                    "prob_top3": round(p3_prob, 4),
                }
            )
        constructors_summary.sort(key=lambda x: -x["mean_points"])
        for idx, item in enumerate(constructors_summary):
            item["expected_position"] = idx + 1

        # Race-by-race Ferrari forecasts
        race_forecasts = []
        for rnd in sorted(race_stats.keys()):
            stat = race_stats[rnd]
            pts_samples = sorted(stat["ferrari_points_samples"])
            mean_pts = sum(pts_samples) / num_seasons
            race_forecasts.append(
                {
                    "round": rnd,
                    "circuit": stat["name"],
                    "circuit_id": stat["circuit_id"],
                    "is_sprint": stat["is_sprint"],
                    "expected_points": round(mean_pts, 1),
                    "points_ci_90": [round(_percentile(pts_samples, 5), 1), round(_percentile(pts_samples, 95), 1)],
                    "win_probability": round(stat["ferrari_wins_count"] / num_seasons, 4),
                    "podium_probability": round(stat["ferrari_podiums_count"] / (num_seasons * 2.0), 4),
                    "dnf_risk": round(stat["ferrari_dnf_count"] / (num_seasons * 2.0), 4),
                }
            )

        return {
            "simulation_meta": {
                "num_seasons": num_seasons,
                "random_seed": random_seed,
                "calendar_races": len(self.calendar),
                "sprint_events": sum(1 for e in self.calendar if e["is_sprint"]),
                "regulation_era": "2027_rebalanced_pu",
            },
            "ferrari_constructors_forecast": {
                "mean_points": round(sum(ferrari_constructor_pts) / num_seasons, 1),
                "median_points": round(_percentile(ferrari_pts_sorted, 50), 1),
                "ci_90": [round(_percentile(ferrari_pts_sorted, 5), 1), round(_percentile(ferrari_pts_sorted, 95), 1)],
                "ci_80": [round(_percentile(ferrari_pts_sorted, 10), 1), round(_percentile(ferrari_pts_sorted, 90), 1)],
                "position_probabilities": ferrari_rank_probs,
                "prob_championship_win": ferrari_rank_probs.get("P1", 0.0),
                "prob_top3": round(sum(ferrari_rank_probs.get(f"P{p}", 0.0) for p in range(1, 4)), 4),
            },
            "driver_forecasts": {
                "leclerc": {
                    "mean_points": round(sum(leclerc_pts) / num_seasons, 1),
                    "median_points": round(_percentile(leclerc_pts_sorted, 50), 1),
                    "ci_90": [round(_percentile(leclerc_pts_sorted, 5), 1), round(_percentile(leclerc_pts_sorted, 95), 1)],
                    "expected_wins": round(sum(leclerc_wins) / num_seasons, 1),
                    "expected_podiums": round(sum(leclerc_podiums) / num_seasons, 1),
                    "expected_dnfs": round(sum(leclerc_dnfs) / num_seasons, 1),
                    "position_probabilities": leclerc_rank_probs,
                    "prob_drivers_title": leclerc_rank_probs.get("P1", 0.0),
                },
                "hamilton": {
                    "mean_points": round(sum(hamilton_pts) / num_seasons, 1),
                    "median_points": round(_percentile(hamilton_pts_sorted, 50), 1),
                    "ci_90": [round(_percentile(hamilton_pts_sorted, 5), 1), round(_percentile(hamilton_pts_sorted, 95), 1)],
                    "expected_wins": round(sum(hamilton_wins) / num_seasons, 1),
                    "expected_podiums": round(sum(hamilton_podiums) / num_seasons, 1),
                    "expected_dnfs": round(sum(hamilton_dnfs) / num_seasons, 1),
                    "position_probabilities": hamilton_rank_probs,
                    "prob_drivers_title": hamilton_rank_probs.get("P1", 0.0),
                },
            },
            "constructors_standings": constructors_summary,
            "race_by_race_forecast": race_forecasts,
        }

