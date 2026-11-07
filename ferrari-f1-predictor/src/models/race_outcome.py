"""
Probabilistic Race Outcome Model (FF27-09, FF27-10, Stage 4).
Simulates full-field race classifications, order statistics, and points scoring
by combining the Hierarchical Latent Pace model and Reliability Hazard model.

Guarantees:
  - ZERO subjective rival multipliers (DEF-07, DEF-10).
  - ZERO hand-authored driver splits; Leclerc and Hamilton race on track against the full grid.
  - Strict FIA points table invariants (25-18-15-12-10-8-6-4-2-1).
  - Constructor points equal the exact sum of its drivers' points.
  - Mutual exclusivity: each finishing position 1..N is occupied by exactly one driver.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.models.latent_pace import HierarchicalLatentPaceModel
from src.models.reliability import ReliabilityHazardModel

GP_POINTS_TABLE = {
    1: 25.0,
    2: 18.0,
    3: 15.0,
    4: 12.0,
    5: 10.0,
    6: 8.0,
    7: 6.0,
    8: 4.0,
    9: 2.0,
    10: 1.0,
}

SPRINT_POINTS_TABLE = {
    1: 8.0,
    2: 7.0,
    3: 6.0,
    4: 5.0,
    5: 4.0,
    6: 3.0,
    7: 2.0,
    8: 1.0,
}


@dataclass
class SimulatedDriverResult:
    driver_id: str
    constructor_id: str
    franchise_id: str
    finish_position: int
    points_scored: float
    is_dnf: bool
    is_mechanical_dnf: bool
    is_incident_dnf: bool


@dataclass
class SimulatedRaceResult:
    event_id: str
    round_number: int
    circuit_id: str
    is_sprint: bool
    driver_results: List[SimulatedDriverResult]
    constructor_points: Dict[str, float] = field(default_factory=dict)
    constructor_positions: Dict[str, int] = field(default_factory=dict)


class ProbabilisticRaceOutcomeModel:
    """
    Simulates Grand Prix and Sprint race classifications by integrating
    latent performance distributions and reliability hazard rates.
    """

    def __init__(
        self,
        pace_model: HierarchicalLatentPaceModel,
        reliability_model: ReliabilityHazardModel,
    ):
        self.pace_model = pace_model
        self.reliability_model = reliability_model

    def simulate_race(
        self,
        event_id: str,
        round_number: int,
        circuit_id: str,
        entries: Sequence[Dict[str, Any]],
        is_sprint: bool = False,
        fastest_lap_point_active: bool = False,
        rng: Optional[random.Random] = None,
    ) -> SimulatedRaceResult:
        """
        Simulate a single realization of a Grand Prix or Sprint weekend.
        """
        if rng is None:
            rng = random.Random()

        surviving_drivers: List[Tuple[float, Dict[str, Any]]] = []
        retired_drivers: List[Tuple[float, Dict[str, Any], bool, bool]] = []

        # 1. Evaluate DNF status and latent pace perturbation for each driver
        for entry in entries:
            did = str(entry["driver_id"])
            fid = str(entry["franchise_id"])

            hazard = self.reliability_model.predict_hazard(
                franchise_id=fid, driver_id=did, circuit_id=circuit_id
            )

            # Sample DNF events
            mech_dnf = rng.random() < hazard.prob_mechanical_dnf
            incident_dnf = (not mech_dnf) and (rng.random() < hazard.prob_incident_dnf)

            expected_pace = self.pace_model.predict_expected_latent_pace(
                franchise_id=fid, driver_id=did, circuit_id=circuit_id
            )

            # Add stochastic variance: N(0, residual_std^2)
            noise = rng.gauss(0.0, self.pace_model.residual_std)
            realized_pace = expected_pace + noise

            if mech_dnf or incident_dnf:
                retired_drivers.append((realized_pace, entry, mech_dnf, incident_dnf))
            else:
                surviving_drivers.append((realized_pace, entry))

        # 2. Sort finishers by realized pace (lowest score = P1)
        surviving_drivers.sort(key=lambda x: x[0])
        # Retired drivers are ordered after finishers
        retired_drivers.sort(key=lambda x: x[0])

        driver_results: List[SimulatedDriverResult] = []
        current_pos = 1

        pts_table = SPRINT_POINTS_TABLE if is_sprint else GP_POINTS_TABLE

        # Assign positions to finishers
        for _, entry in surviving_drivers:
            did = str(entry["driver_id"])
            cid = str(entry["constructor_id"])
            fid = str(entry["franchise_id"])
            pts = pts_table.get(current_pos, 0.0)

            driver_results.append(
                SimulatedDriverResult(
                    driver_id=did,
                    constructor_id=cid,
                    franchise_id=fid,
                    finish_position=current_pos,
                    points_scored=pts,
                    is_dnf=False,
                    is_mechanical_dnf=False,
                    is_incident_dnf=False,
                )
            )
            current_pos += 1

        # Award fastest lap bonus point if active (top 10 finisher only)
        if fastest_lap_point_active and not is_sprint and len(driver_results) >= 1:
            # Fastest lap goes with highest probability to the leading cars
            top10 = driver_results[: min(10, len(driver_results))]
            fl_winner = rng.choice(top10[: min(5, len(top10))])
            fl_winner.points_scored += 1.0

        # Assign remaining positions to retired drivers
        for _, entry, mech_dnf, inc_dnf in retired_drivers:
            did = str(entry["driver_id"])
            cid = str(entry["constructor_id"])
            fid = str(entry["franchise_id"])
            driver_results.append(
                SimulatedDriverResult(
                    driver_id=did,
                    constructor_id=cid,
                    franchise_id=fid,
                    finish_position=current_pos,
                    points_scored=0.0,
                    is_dnf=True,
                    is_mechanical_dnf=mech_dnf,
                    is_incident_dnf=inc_dnf,
                )
            )
            current_pos += 1

        # 3. Calculate exact Constructor Championship points (sum of drivers)
        constructor_pts: Dict[str, float] = {}
        for dr in driver_results:
            cid = dr.constructor_id
            constructor_pts[cid] = constructor_pts.get(cid, 0.0) + dr.points_scored

        # Rank constructors by points descending
        sorted_constructors = sorted(
            constructor_pts.items(), key=lambda x: -x[1]
        )
        constructor_positions = {
            cid: idx + 1 for idx, (cid, _) in enumerate(sorted_constructors)
        }

        return SimulatedRaceResult(
            event_id=event_id,
            round_number=round_number,
            circuit_id=circuit_id,
            is_sprint=is_sprint,
            driver_results=driver_results,
            constructor_points=constructor_pts,
            constructor_positions=constructor_positions,
        )

