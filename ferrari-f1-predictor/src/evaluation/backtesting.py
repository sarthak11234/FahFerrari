"""
Rolling-origin baseline backtesting framework (FF27-10).
Implements the 4 baseline models required by Section 4.6 of docs/2027-architecture.md:
  1. Historical points-per-event baseline
  2. Recent-season weighted baseline
  3. Teammate-relative pace baseline
  4. Probabilistic ranking baseline
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.evaluation.calibration import expected_calibration_error, gaussian_cdf
from src.evaluation.metrics import (
    brier_score,
    mae,
    rmse,
    spearman_rank_correlation,
    top_k_overlap_accuracy,
)

F1_POINTS_SYSTEM = [25.0, 18.0, 15.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 1.0]


def _f1_points_for_rank(rank_1_based: int) -> float:
    if 1 <= rank_1_based <= 10:
        return F1_POINTS_SYSTEM[rank_1_based - 1]
    return 0.0


class RollingOriginBaselineEvaluator:
    """
    Executes expanding-window / rolling-origin backtests over point-in-time feature rows.
    Guarantees no future event enters training for a test fold.
    """

    def __init__(self, feature_rows: Sequence[Dict[str, Any]]):
        # Sort chronologically by (event_timestamp_utc, event_id, finish_position_order)
        self.rows = sorted(
            feature_rows,
            key=lambda r: (
                str(r["event_timestamp_utc"]),
                str(r["event_id"]),
                int(r.get("target_finish_order", 20)),
            ),
        )
        self.seasons = sorted({int(r["season"]) for r in self.rows})

    def run_season_rolling_origin(
        self, min_train_seasons: int = 2
    ) -> Dict[str, Any]:
        """
        Expanding-window backtest over seasons:
        For test_season in seasons[min_train_seasons:]:
          Train on all events from seasons < test_season.
          Evaluate each event in test_season using point-in-time features.
        """
        if len(self.seasons) <= min_train_seasons:
            raise ValueError(
                f"Insufficient seasons for rolling-origin backtest ({len(self.seasons)} available, {min_train_seasons} required for warm-up)"
            )

        test_seasons = self.seasons[min_train_seasons:]
        baseline_names = [
            "baseline1_historical_mean",
            "baseline2_recent_weighted",
            "baseline3_teammate_relative",
            "baseline4_probabilistic_ranking",
        ]

        per_baseline_records: Dict[str, Dict[str, List[float]]] = {
            b: {
                "true_points": [],
                "pred_points": [],
                "true_finish": [],
                "pred_finish": [],
                "true_wins": [],
                "pred_win_prob": [],
                "true_podiums": [],
                "pred_podium_prob": [],
                "true_dnfs": [],
                "pred_dnf_prob": [],
            }
            for b in baseline_names
        }

        # Event-level rank metrics
        per_baseline_event_corrs: Dict[str, List[float]] = {b: [] for b in baseline_names}
        per_baseline_top3_accs: Dict[str, List[float]] = {b: [] for b in baseline_names}

        fold_summaries: List[Dict[str, Any]] = []

        for test_s in test_seasons:
            train_rows = [r for r in self.rows if int(r["season"]) < test_s]
            test_rows = [r for r in self.rows if int(r["season"]) == test_s]
            if not test_rows:
                continue

            # Compute historical aggregates across training set
            team_train_pts: Dict[str, List[float]] = defaultdict(list)
            team_season_pts: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
            for r in train_rows:
                fid = str(r["franchise_id"])
                pts = float(r["target_points"])
                s = int(r["season"])
                team_train_pts[fid].append(pts)
                team_season_pts[fid][s].append(pts)

            # Baseline 1 historical mean points per team
            b1_team_means = {
                fid: sum(pts_list) / len(pts_list) if pts_list else 1.0
                for fid, pts_list in team_train_pts.items()
            }

            # Baseline 2 recent-weighted points per team (3-season exponential decay)
            b2_team_weighted: Dict[str, float] = {}
            prior_seasons_sorted = sorted({int(r["season"]) for r in train_rows})
            weights = [0.5, 0.3, 0.2]  # Most recent to oldest
            for fid, s_dict in team_season_pts.items():
                recent_s = prior_seasons_sorted[-3:]
                accum = 0.0
                weight_sum = 0.0
                for w, s in zip(weights, reversed(recent_s)):
                    if s in s_dict and s_dict[s]:
                        season_mean = sum(s_dict[s]) / len(s_dict[s])
                        accum += w * season_mean
                        weight_sum += w
                b2_team_weighted[fid] = accum / weight_sum if weight_sum > 0 else b1_team_means.get(fid, 1.0)

            # Group test rows by event to evaluate per-event predictions
            test_by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for r in test_rows:
                test_by_event[str(r["event_id"])].append(r)

            for eid, ev_rows in test_by_event.items():
                ev_true_pts = [float(r["target_points"]) for r in ev_rows]
                ev_true_finish = [float(r["target_finish_order"]) for r in ev_rows]
                ev_true_wins = [int(r["target_is_win"]) for r in ev_rows]
                ev_true_podiums = [int(r["target_is_podium"]) for r in ev_rows]
                ev_true_dnfs = [int(r["target_is_dnf"]) for r in ev_rows]

                # --- 1. Baseline 1: Historical Team Mean ---
                b1_pts = [b1_team_means.get(str(r["franchise_id"]), 1.0) for r in ev_rows]
                # Invert points to estimate finishing order
                b1_pred_finish = [
                    float(idx + 1)
                    for idx, _ in sorted(enumerate(b1_pts), key=lambda x: -x[1])
                ]
                b1_win_prob = [min(0.95, max(0.01, p / 25.0)) for p in b1_pts]
                b1_podium_prob = [min(0.95, max(0.02, p / 15.0)) for p in b1_pts]
                b1_dnf_prob = [float(r.get("team_dnf_rate_10", 0.12)) for r in ev_rows]

                # --- 2. Baseline 2: Recent-Season Weighted ---
                b2_pts = [b2_team_weighted.get(str(r["franchise_id"]), 1.0) for r in ev_rows]
                b2_pred_finish = [
                    float(idx + 1)
                    for idx, _ in sorted(enumerate(b2_pts), key=lambda x: -x[1])
                ]
                b2_win_prob = [min(0.95, max(0.01, p / 25.0)) for p in b2_pts]
                b2_podium_prob = [min(0.95, max(0.02, p / 15.0)) for p in b2_pts]
                b2_dnf_prob = [float(r.get("team_dnf_rate_10", 0.12)) for r in ev_rows]

                # --- 3. Baseline 3: Teammate-Relative Pace ---
                # Adjusted pace score = team EWMA finish + 0.5 * teammate finish delta
                b3_scores = [
                    float(r.get("team_finish_pos_mean5", 10.5))
                    + 0.5 * float(r.get("driver_teammate_finish_delta_mean5", 0.0))
                    for r in ev_rows
                ]
                b3_pred_finish = [
                    float(idx + 1)
                    for idx, _ in sorted(enumerate(b3_scores), key=lambda x: x[1])
                ]
                b3_pts = [_f1_points_for_rank(int(rk)) for rk in b3_pred_finish]
                b3_win_prob = [0.65 if rk == 1.0 else (0.20 if rk == 2.0 else 0.02) for rk in b3_pred_finish]
                b3_podium_prob = [0.85 if rk <= 3.0 else (0.30 if rk <= 5.0 else 0.05) for rk in b3_pred_finish]
                b3_dnf_prob = [
                    0.5 * float(r.get("team_dnf_rate_10", 0.12)) + 0.5 * float(r.get("driver_incident_dnf_rate_10", 0.05))
                    for r in ev_rows
                ]

                # --- 4. Baseline 4: Probabilistic Ranking Baseline ---
                # Latent latent pace proxy combining team, driver form, and circuit fit
                b4_pace = [
                    0.50 * float(r.get("team_finish_pos_mean5", 10.5))
                    + 0.30 * float(r.get("driver_finish_pos_mean5", 10.5))
                    + 0.20 * float(r.get("circuit_driver_mean_finish_hist", 10.5))
                    for r in ev_rows
                ]
                b4_pred_ranks = [
                    float(idx + 1)
                    for idx, _ in sorted(enumerate(b4_pace), key=lambda x: x[1])
                ]
                b4_pts = [_f1_points_for_rank(int(rk)) for rk in b4_pred_ranks]
                # Gaussian CDF probability of finishing <= 1 (win) or <= 3 (podium) given expected finish & std
                b4_win_prob = [gaussian_cdf(1.5, mean=p, std=2.5) for p in b4_pace]
                b4_podium_prob = [gaussian_cdf(3.5, mean=p, std=3.0) for p in b4_pace]
                b4_dnf_prob = [
                    float(r.get("team_mech_dnf_rate_10", 0.06)) + float(r.get("driver_incident_dnf_rate_10", 0.05))
                    for r in ev_rows
                ]

                preds = {
                    "baseline1_historical_mean": (b1_pts, b1_pred_finish, b1_win_prob, b1_podium_prob, b1_dnf_prob),
                    "baseline2_recent_weighted": (b2_pts, b2_pred_finish, b2_win_prob, b2_podium_prob, b2_dnf_prob),
                    "baseline3_teammate_relative": (b3_pts, b3_pred_finish, b3_win_prob, b3_podium_prob, b3_dnf_prob),
                    "baseline4_probabilistic_ranking": (b4_pts, b4_pred_ranks, b4_win_prob, b4_podium_prob, b4_dnf_prob),
                }

                for bname, (p_pts, p_fin, p_win, p_pod, p_dnf) in preds.items():
                    per_baseline_records[bname]["true_points"].extend(ev_true_pts)
                    per_baseline_records[bname]["pred_points"].extend(p_pts)
                    per_baseline_records[bname]["true_finish"].extend(ev_true_finish)
                    per_baseline_records[bname]["pred_finish"].extend(p_fin)
                    per_baseline_records[bname]["true_wins"].extend(ev_true_wins)
                    per_baseline_records[bname]["pred_win_prob"].extend(p_win)
                    per_baseline_records[bname]["true_podiums"].extend(ev_true_podiums)
                    per_baseline_records[bname]["pred_podium_prob"].extend(p_pod)
                    per_baseline_records[bname]["true_dnfs"].extend(ev_true_dnfs)
                    per_baseline_records[bname]["pred_dnf_prob"].extend(p_dnf)

                    corr = spearman_rank_correlation(ev_true_finish, p_fin)
                    top3 = top_k_overlap_accuracy(ev_true_finish, p_fin, k=3)
                    per_baseline_event_corrs[bname].append(corr)
                    per_baseline_top3_accs[bname].append(top3)

            fold_summaries.append(
                {
                    "test_season": test_s,
                    "train_seasons": [s for s in self.seasons if s < test_s],
                    "train_entries": len(train_rows),
                    "test_entries": len(test_rows),
                    "test_events": len(test_by_event),
                }
            )

        # Aggregate overall metrics across all test folds
        results: Dict[str, Any] = {"folds": fold_summaries, "baselines": {}}
        for bname in baseline_names:
            rec = per_baseline_records[bname]
            p_mae = mae(rec["true_points"], rec["pred_points"])
            p_rmse = rmse(rec["true_points"], rec["pred_points"])
            f_mae = mae(rec["true_finish"], rec["pred_finish"])
            mean_corr = sum(per_baseline_event_corrs[bname]) / len(per_baseline_event_corrs[bname])
            mean_top3 = sum(per_baseline_top3_accs[bname]) / len(per_baseline_top3_accs[bname])

            brier_win = brier_score(rec["true_wins"], rec["pred_win_prob"])
            brier_pod = brier_score(rec["true_podiums"], rec["pred_podium_prob"])
            brier_dnf = brier_score(rec["true_dnfs"], rec["pred_dnf_prob"])

            ece_win = expected_calibration_error(rec["true_wins"], rec["pred_win_prob"], n_bins=5)
            ece_pod = expected_calibration_error(rec["true_podiums"], rec["pred_podium_prob"], n_bins=5)

            results["baselines"][bname] = {
                "points_mae": round(p_mae, 3),
                "points_rmse": round(p_rmse, 3),
                "finish_order_mae": round(f_mae, 3),
                "spearman_rank_correlation": round(mean_corr, 3),
                "top3_accuracy": round(mean_top3, 3),
                "brier_score_win": round(brier_win, 4),
                "brier_score_podium": round(brier_pod, 4),
                "brier_score_dnf": round(brier_dnf, 4),
                "ece_win": round(ece_win, 4),
                "ece_podium": round(ece_pod, 4),
            }

        return results

