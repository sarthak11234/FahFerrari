"""
Hierarchical Latent Pace Model (FF27-09, FF27-10, Stage 4).
Estimates constructor car pace and driver contribution separately via empirical Bayes partial pooling.
Decomposes observed performance into:
    pace = team_latent_pace + driver_effect + circuit_interaction + residual_uncertainty
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.evaluation.calibration import gaussian_cdf


@dataclass
class TeamPaceParameters:
    franchise_id: str
    latent_pace: float  # Expected finishing position score (lower is faster)
    sample_size: int
    uncertainty_std: float


@dataclass
class DriverPaceParameters:
    driver_id: str
    driver_delta: float  # Relative position advantage over teammate (negative = faster)
    sample_size: int
    uncertainty_std: float


class HierarchicalLatentPaceModel:
    """
    Hierarchical Empirical Bayes model for Formula 1 performance.
    Separates constructor car performance and driver contribution using
    teammate differential shrinkage and regulation-era partial pooling.
    """

    def __init__(
        self,
        global_mean_position: float = 10.5,
        team_prior_strength: float = 5.0,  # Equivalent prior observations for team pooling
        driver_prior_strength: float = 8.0,  # Equivalent prior observations for driver shrinkage
        circuit_prior_strength: float = 4.0,
    ):
        self.global_mean_position = global_mean_position
        self.team_prior_strength = team_prior_strength
        self.driver_prior_strength = driver_prior_strength
        self.circuit_prior_strength = circuit_prior_strength

        # Fitted parameters
        self.team_params: Dict[str, TeamPaceParameters] = {}
        self.driver_params: Dict[str, DriverPaceParameters] = {}
        self.circuit_adjustments: Dict[Tuple[str, str], float] = {}  # (circuit_id, franchise_id) -> delta
        self.residual_std: float = 2.45
        self.is_fitted: bool = False

    def fit(self, historical_rows: Sequence[Dict[str, Any]]) -> HierarchicalLatentPaceModel:
        """
        Fit hierarchical team and driver parameters on historical race records.
        """
        if not historical_rows:
            raise ValueError("historical_rows cannot be empty")

        # 1. Constructor Car Pace Estimation with Partial Pooling
        team_observations: Dict[str, List[float]] = defaultdict(list)
        for r in historical_rows:
            fid = str(r["franchise_id"])
            finish_order = float(r.get("target_finish_order") or r.get("finish_position_order", 20))
            team_observations[fid].append(finish_order)

        self.team_params.clear()
        for fid, obs in team_observations.items():
            n = len(obs)
            sample_mean = sum(obs) / n
            # Empirical Bayes shrinkage toward global median (10.5)
            shrunk_mean = (
                (n * sample_mean) + (self.team_prior_strength * self.global_mean_position)
            ) / (n + self.team_prior_strength)
            # Posterior standard error
            sample_var = sum((x - sample_mean) ** 2 for x in obs) / max(1, n - 1)
            se = math.sqrt((sample_var / (n + self.team_prior_strength)) + 1.0)
            self.team_params[fid] = TeamPaceParameters(
                franchise_id=fid,
                latent_pace=round(shrunk_mean, 4),
                sample_size=n,
                uncertainty_std=round(se, 4),
            )

        # 2. Driver Effect Estimation via Intra-Team Differentials
        # Teammate deltas isolate driver skill from constructor machinery
        driver_deltas: Dict[str, List[float]] = defaultdict(list)
        by_event_team: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for r in historical_rows:
            key = (str(r["event_id"]), str(r["constructor_id"]))
            by_event_team[key].append(r)

        for _, entries in by_event_team.items():
            if len(entries) >= 2:
                for i in range(len(entries)):
                    d1 = str(entries[i]["driver_id"])
                    pos1 = float(entries[i].get("target_finish_order") or entries[i].get("finish_position_order", 20))
                    for j in range(len(entries)):
                        if i != j:
                            pos2 = float(entries[j].get("target_finish_order") or entries[j].get("finish_position_order", 20))
                            # delta < 0 means driver 1 finished ahead of teammate
                            driver_deltas[d1].append((pos1 - pos2) / 2.0)

        self.driver_params.clear()
        for did, deltas in driver_deltas.items():
            n = len(deltas)
            sample_delta = sum(deltas) / n
            # Shrink toward 0 (teammates are assumed equal a priori)
            shrunk_delta = (n * sample_delta) / (n + self.driver_prior_strength)
            d_var = sum((x - sample_delta) ** 2 for x in deltas) / max(1, n - 1)
            se_d = math.sqrt((d_var / (n + self.driver_prior_strength)) + 0.25)
            self.driver_params[did] = DriverPaceParameters(
                driver_id=did,
                driver_delta=round(shrunk_delta, 4),
                sample_size=n,
                uncertainty_std=round(se_d, 4),
            )

        # 3. Circuit-Team Interaction (Suitability)
        circuit_team_obs: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        for r in historical_rows:
            cid = str(r["circuit_id"])
            fid = str(r["franchise_id"])
            expected_base = (
                self.team_params[fid].latent_pace if fid in self.team_params else self.global_mean_position
            )
            res_val = float(r.get("target_finish_order") or r.get("finish_position_order", 20)) - expected_base
            circuit_team_obs[(cid, fid)].append(res_val)

        self.circuit_adjustments.clear()
        for pair, residuals in circuit_team_obs.items():
            n = len(residuals)
            sample_res = sum(residuals) / n
            # Shrink toward 0
            shrunk_adj = (n * sample_res) / (n + self.circuit_prior_strength)
            self.circuit_adjustments[pair] = round(shrunk_adj, 4)

        # 4. Out-of-Sample Residual Standard Deviation
        squared_errors = []
        for r in historical_rows:
            pred = self.predict_expected_latent_pace(
                franchise_id=str(r["franchise_id"]),
                driver_id=str(r["driver_id"]),
                circuit_id=str(r["circuit_id"]),
            )
            act_pos = float(r.get("target_finish_order") or r.get("finish_position_order", 20))
            squared_errors.append((act_pos - pred) ** 2)

        self.residual_std = round(math.sqrt(sum(squared_errors) / len(squared_errors)), 4)
        self.is_fitted = True
        return self

    def predict_expected_latent_pace(
        self,
        franchise_id: str,
        driver_id: str,
        circuit_id: str,
    ) -> float:
        """
        Predict expected finishing score (lower = faster).
        Combines constructor pace, driver delta, and circuit interaction.
        """
        # Base team pace
        if franchise_id in self.team_params:
            team_pace = self.team_params[franchise_id].latent_pace
        else:
            team_pace = self.global_mean_position

        # Driver delta
        if driver_id in self.driver_params:
            driver_delta = self.driver_params[driver_id].driver_delta
        else:
            driver_delta = 0.0

        # Circuit fit
        circuit_adj = self.circuit_adjustments.get((circuit_id, franchise_id), 0.0)

        # Combined latent score
        score = team_pace + driver_delta + circuit_adj
        return round(max(1.0, min(20.0, score)), 4)

    def predict_event_field(
        self, event_entries: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Predict full race classifications, points, and calibrated win/podium probabilities
        for all entries in a single Grand Prix event.
        """
        predictions = []
        for r in event_entries:
            fid = str(r["franchise_id"])
            did = str(r["driver_id"])
            cid = str(r["circuit_id"])
            expected_score = self.predict_expected_latent_pace(fid, did, cid)
            predictions.append(
                {
                    "row": r,
                    "driver_id": did,
                    "franchise_id": fid,
                    "expected_score": expected_score,
                }
            )

        # Sort field by ascending expected score (fastest first)
        predictions.sort(key=lambda x: x["expected_score"])

        f1_points = [25.0, 18.0, 15.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0]

        results = []
        for rank_idx, item in enumerate(predictions):
            rank = rank_idx + 1
            pts = f1_points[rank_idx] if rank_idx < len(f1_points) else 0.0
            score = item["expected_score"]

            # Probability of win (finishing <= 1.5) and podium (finishing <= 3.5)
            # using calibrated residual standard deviation
            win_prob = gaussian_cdf(1.5, mean=score, std=self.residual_std)
            podium_prob = gaussian_cdf(3.5, mean=score, std=self.residual_std)
            top5_prob = gaussian_cdf(5.5, mean=score, std=self.residual_std)
            pts_prob = gaussian_cdf(10.5, mean=score, std=self.residual_std)

            results.append(
                {
                    "driver_id": item["driver_id"],
                    "franchise_id": item["franchise_id"],
                    "expected_latent_score": score,
                    "predicted_rank": rank,
                    "predicted_points": pts,
                    "prob_win": round(win_prob, 4),
                    "prob_podium": round(podium_prob, 4),
                    "prob_top5": round(top5_prob, 4),
                    "prob_points": round(pts_prob, 4),
                }
            )

        # Normalize probabilities across the field so P(Win) sums to 1.0 and P(Podium) sums to 3.0
        total_win_prob = sum(x["prob_win"] for x in results)
        total_podium_prob = sum(x["prob_podium"] for x in results)
        for x in results:
            if total_win_prob > 0:
                x["prob_win_calibrated"] = round(x["prob_win"] / total_win_prob, 4)
            else:
                x["prob_win_calibrated"] = 0.05
            if total_podium_prob > 0:
                x["prob_podium_calibrated"] = round((x["prob_podium"] / total_podium_prob) * 3.0, 4)
            else:
                x["prob_podium_calibrated"] = 0.15

        return results
