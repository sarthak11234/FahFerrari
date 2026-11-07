"""
Reliability Hazard Model (FF27-09, FF27-10, Stage 4).
Estimates retirement risks using discrete-time hazard formulation,
separating mechanical reliability from collision/incident risk.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class ReliabilityEstimate:
    driver_id: str
    franchise_id: str
    circuit_id: str
    prob_mechanical_dnf: float
    prob_incident_dnf: float
    prob_total_dnf: float


class ReliabilityHazardModel:
    """
    Discrete-time hazard model for Formula 1 retirements.
    Decomposes DNF risk into independent hazards:
        1 - P(DNF) = (1 - h_mechanical) * (1 - h_incident)
    """

    def __init__(
        self,
        base_mech_hazard: float = 0.055,
        base_incident_hazard: float = 0.050,
        team_prior_weight: float = 15.0,
        driver_prior_weight: float = 12.0,
        circuit_prior_weight: float = 8.0,
    ):
        self.base_mech_hazard = base_mech_hazard
        self.base_incident_hazard = base_incident_hazard
        self.team_prior_weight = team_prior_weight
        self.driver_prior_weight = driver_prior_weight
        self.circuit_prior_weight = circuit_prior_weight

        # Learned hazard rates
        self.team_mech_hazards: Dict[str, float] = {}
        self.driver_incident_hazards: Dict[str, float] = {}
        self.circuit_attrition_multipliers: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, historical_rows: Sequence[Dict[str, Any]]) -> ReliabilityHazardModel:
        """
        Fit mechanical and incident hazards using empirical Bayes shrinkage.
        """
        if not historical_rows:
            raise ValueError("historical_rows cannot be empty")

        # 1. Constructor Mechanical Hazard
        team_total: Dict[str, int] = defaultdict(int)
        team_mech: Dict[str, int] = defaultdict(int)
        for r in historical_rows:
            fid = str(r["franchise_id"])
            team_total[fid] += 1
            if str(r.get("target_is_mechanical_dnf")) == "1" or str(r.get("is_mechanical_dnf")).lower() == "true":
                team_mech[fid] += 1

        self.team_mech_hazards.clear()
        for fid, n in team_total.items():
            k = team_mech[fid]
            # Shrink toward base mechanical hazard
            shrunk_h = (k + (self.team_prior_weight * self.base_mech_hazard)) / (
                n + self.team_prior_weight
            )
            self.team_mech_hazards[fid] = round(shrunk_h, 4)

        # 2. Driver Incident Hazard
        driver_total: Dict[str, int] = defaultdict(int)
        driver_incidents: Dict[str, int] = defaultdict(int)
        for r in historical_rows:
            did = str(r["driver_id"])
            driver_total[did] += 1
            if str(r.get("status_category")) == "dnf_incident":
                driver_incidents[did] += 1

        self.driver_incident_hazards.clear()
        for did, n in driver_total.items():
            k = driver_incidents[did]
            # Shrink toward base incident hazard
            shrunk_h = (k + (self.driver_prior_weight * self.base_incident_hazard)) / (
                n + self.driver_prior_weight
            )
            self.driver_incident_hazards[did] = round(shrunk_h, 4)

        # 3. Circuit Attrition Multiplier
        circuit_total: Dict[str, int] = defaultdict(int)
        circuit_dnf: Dict[str, int] = defaultdict(int)
        for r in historical_rows:
            cid = str(r["circuit_id"])
            circuit_total[cid] += 1
            if str(r.get("target_is_dnf")) == "1" or str(r.get("is_dnf")).lower() == "true":
                circuit_dnf[cid] += 1

        global_dnf_rate = (
            sum(circuit_dnf.values()) / sum(circuit_total.values())
            if sum(circuit_total.values()) > 0
            else 0.11
        )

        self.circuit_attrition_multipliers.clear()
        for cid, n in circuit_total.items():
            k = circuit_dnf[cid]
            shrunk_rate = (k + (self.circuit_prior_weight * global_dnf_rate)) / (
                n + self.circuit_prior_weight
            )
            mult = shrunk_rate / global_dnf_rate if global_dnf_rate > 0 else 1.0
            self.circuit_attrition_multipliers[cid] = round(max(0.5, min(2.0, mult)), 4)

        self.is_fitted = True
        return self

    def predict_hazard(
        self, franchise_id: str, driver_id: str, circuit_id: str
    ) -> ReliabilityEstimate:
        """
        Predict mechanical, incident, and combined DNF probability for a driver-entry.
        """
        # Circuit multiplier modulates baseline track difficulty
        circuit_mult = self.circuit_attrition_multipliers.get(circuit_id, 1.0)

        # Base team mechanical hazard adjusted by circuit severity
        h_team = self.team_mech_hazards.get(franchise_id, self.base_mech_hazard)
        h_mech = 1.0 - math.pow(1.0 - h_team, circuit_mult)
        h_mech = round(max(0.01, min(0.35, h_mech)), 4)

        # Driver incident hazard adjusted by circuit street/track danger
        h_driver = self.driver_incident_hazards.get(driver_id, self.base_incident_hazard)
        h_inc = 1.0 - math.pow(1.0 - h_driver, circuit_mult)
        h_inc = round(max(0.01, min(0.35, h_inc)), 4)

        # Total DNF probability assuming independent hazards
        total_dnf = 1.0 - ((1.0 - h_mech) * (1.0 - h_inc))
        total_dnf = round(max(0.02, min(0.50, total_dnf)), 4)

        return ReliabilityEstimate(
            driver_id=driver_id,
            franchise_id=franchise_id,
            circuit_id=circuit_id,
            prob_mechanical_dnf=h_mech,
            prob_incident_dnf=h_inc,
            prob_total_dnf=total_dnf,
        )

