"""
vrp_state.py — Model-free variance risk premium term-structure state.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


class ModelFreeVRPState:
    """
    Snapshot estimator for VRP term-structure state:

      VRP(T) = RN_var(T) - E[RV(T)]
      VRP_slope = VRP(short) - VRP(long)
    """

    HORIZONS = (7, 30, 60)

    def __init__(self):
        self.last_state: Dict[str, float] = {}

    @staticmethod
    def _safe_var(returns: np.ndarray) -> float:
        if returns is None or len(returns) < 2:
            return 0.0
        return float(np.var(np.asarray(returns, dtype=float), ddof=1))

    def expected_realized_var(self, returns_history: np.ndarray, horizon_days: int) -> float:
        """
        Lightweight HAR-RV style expectation (daily data).
        """
        arr = np.asarray(returns_history if returns_history is not None else [], dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 5:
            return 0.0001

        rv1 = self._safe_var(arr[-1:])
        rv5 = self._safe_var(arr[-5:]) if len(arr) >= 5 else rv1
        rv22 = self._safe_var(arr[-22:]) if len(arr) >= 22 else rv5

        # HAR-inspired convex combination, then annualize
        daily_var = 0.55 * rv1 + 0.30 * rv5 + 0.15 * rv22
        annual_var = float(max(daily_var * 252.0, 1e-8))

        # Horizon-adjusted expectation (mild mean reversion)
        h = max(int(horizon_days), 1)
        blend = min(h / 60.0, 1.0)
        long_run = float(max(rv22 * 252.0, 1e-8))
        exp_var = (1.0 - 0.35 * blend) * annual_var + (0.35 * blend) * long_run
        return float(max(exp_var, 1e-8))

    def compute_state(
        self,
        rn_var_term: Optional[Dict[int, float]],
        returns_history: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """
        Parameters
        ----------
        rn_var_term : dict {7: RN_var_7d, 30: RN_var_30d, 60: RN_var_60d}
                      Values are annualized variances in decimal^2.
        returns_history : recent log returns
        """
        rn_var_term = dict(rn_var_term or {})
        rv_exp = {}
        vrp = {}

        for h in self.HORIZONS:
            rn = rn_var_term.get(h, np.nan)
            erv = self.expected_realized_var(returns_history, h)
            rv_exp[h] = float(erv)
            if np.isfinite(rn) and rn > 0:
                vrp[h] = float(rn - erv)
            else:
                vrp[h] = np.nan

        # Missing RN variance fallback: use nearest available
        for h in self.HORIZONS:
            if not np.isfinite(vrp[h]):
                nearest = None
                best_dist = 10**9
                for hh in self.HORIZONS:
                    if np.isfinite(vrp.get(hh, np.nan)):
                        d = abs(h - hh)
                        if d < best_dist:
                            best_dist = d
                            nearest = vrp[hh]
                vrp[h] = float(nearest) if nearest is not None else 0.0

        vrp_level = float(vrp[30])
        vrp_slope = float(vrp[7] - vrp[60])

        # State logic
        if vrp_level > 0.01 and vrp_slope > 0.002:
            regime = "FEAR_STEEP"
        elif vrp_level > 0.0:
            regime = "FEAR"
        elif vrp_level < -0.005 and vrp_slope < -0.002:
            regime = "COMPLACENCY_STEEP"
        elif vrp_level < 0.0:
            regime = "COMPLACENCY"
        else:
            regime = "NEUTRAL"

        out = {
            "vrp_7d": float(vrp[7]),
            "vrp_30d": float(vrp[30]),
            "vrp_60d": float(vrp[60]),
            "vrp_level": vrp_level,
            "vrp_slope": vrp_slope,
            "rn_var_7d": float(rn_var_term.get(7, np.nan)) if 7 in rn_var_term else np.nan,
            "rn_var_30d": float(rn_var_term.get(30, np.nan)) if 30 in rn_var_term else np.nan,
            "rn_var_60d": float(rn_var_term.get(60, np.nan)) if 60 in rn_var_term else np.nan,
            "exp_rv_7d": float(rv_exp[7]),
            "exp_rv_30d": float(rv_exp[30]),
            "exp_rv_60d": float(rv_exp[60]),
            "state_label": regime,
        }
        self.last_state = out
        return out

    @staticmethod
    def parameter_adjustments(state: Dict[str, float]) -> Dict[str, float]:
        """
        Conservative bounded multipliers for Heston params.
        """
        lvl = float(state.get("vrp_level", 0.0))
        slope = float(state.get("vrp_slope", 0.0))

        # Tight bounds for regression safety
        kappa_mult = float(np.clip(1.0 - 2.0 * slope, 0.85, 1.15))
        theta_mult = float(np.clip(1.0 + 2.5 * lvl, 0.85, 1.20))
        sigma_v_mult = float(np.clip(1.0 + 1.8 * max(lvl, 0.0) + 0.8 * max(slope, 0.0), 0.85, 1.25))

        return {
            "kappa_mult": kappa_mult,
            "theta_mult": theta_mult,
            "sigma_v_mult": sigma_v_mult,
        }

