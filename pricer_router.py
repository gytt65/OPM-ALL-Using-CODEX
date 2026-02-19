"""
pricer_router.py — Tiered CPU-budgeted pricer routing.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
from scipy.stats import norm


def _bsm_price(spot, strike, T, r, q, sigma, option_type):
    s = max(float(spot), 1e-8)
    k = max(float(strike), 1e-8)
    t = max(float(T), 1e-10)
    v = max(float(sigma), 1e-8)
    if t <= 0:
        if option_type.upper() in ("CE", "CALL"):
            return max(s - k, 0.0)
        return max(k - s, 0.0)
    sqrt_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r - q + 0.5 * v * v) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t
    if option_type.upper() in ("CE", "CALL"):
        return float(s * np.exp(-q * t) * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))
    return float(k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp(-q * t) * norm.cdf(-d1))


class TieredPricerRouter:
    """
    Route pricing across:
      Tier 0: BS from surface IV (fastest)
      Tier 1: Heston COS (fast refinement)
      Tier 2: Heston+Jump QMC MC (selective, budget-aware)
    """

    def __init__(self, default_cpu_budget_ms: float = 20.0):
        self.default_cpu_budget_ms = float(default_cpu_budget_ms)

    def route_price(
        self,
        *,
        spot: float,
        strike: float,
        T: float,
        r: float,
        q: float,
        option_type: str,
        surface_iv: float,
        regime_params: Optional[Dict] = None,
        india_features: Optional[Dict] = None,
        quant_engine=None,
        mc_pricer=None,
        full_chain_mode: bool = False,
        cpu_budget_ms: Optional[float] = None,
        liquidity_score: float = 1.0,
        anomaly_score: float = 0.0,
        mispricing_hint: float = 0.0,
    ) -> Dict[str, float]:
        budget_ms = float(cpu_budget_ms if cpu_budget_ms is not None else self.default_cpu_budget_ms)
        t0 = time.perf_counter()

        out = {
            "price": np.nan,
            "tier_used": "tier0_surface_bs",
            "surface_iv": float(surface_iv),
            "latency_ms": 0.0,
            "bs_price": np.nan,
            "cos_price": np.nan,
            "mc_price": np.nan,
            "mc_std_error": np.nan,
        }

        bs_price = _bsm_price(spot, strike, T, r, q, max(surface_iv, 1e-4), option_type)
        out["bs_price"] = float(bs_price)
        out["price"] = float(bs_price)

        def _elapsed_ms():
            return (time.perf_counter() - t0) * 1000.0

        def _within_budget(extra_ms: float = 0.0):
            return (_elapsed_ms() + extra_ms) <= budget_ms

        # Tier 1: COS refinement
        cos_price = None
        if quant_engine is not None and hasattr(quant_engine, "heston_cos") and _within_budget(1.0):
            try:
                V0 = max(surface_iv ** 2, 1e-8)
                rp = regime_params or {}
                cos_price = quant_engine.heston_cos.price(
                    spot, strike, T, r, q, V0,
                    rp.get("kappa", 2.0),
                    rp.get("theta_v", V0),
                    rp.get("sigma_v", 0.3),
                    rp.get("rho_sv", -0.5),
                    option_type,
                )
                if np.isfinite(cos_price) and cos_price > 0:
                    out["cos_price"] = float(cos_price)
                    out["price"] = float(cos_price)
                    out["tier_used"] = "tier1_heston_cos"
            except Exception:
                pass

        # Tier 2 eligibility: selective and CPU-budget controlled
        run_tier2 = False
        if mc_pricer is not None and _within_budget(3.0):
            if full_chain_mode:
                # Chain scans: Tier 2 only for exceptional candidates
                run_tier2 = (
                    liquidity_score >= 0.5
                    and abs(mispricing_hint) >= 0.08
                    and anomaly_score <= 0.8
                )
            else:
                run_tier2 = (
                    liquidity_score >= 0.4
                    and anomaly_score <= 0.9
                    and abs(mispricing_hint) >= 0.03
                )

        if run_tier2 and _within_budget(5.0):
            rp = regime_params or {}
            old_n_paths = getattr(mc_pricer, "n_paths", None)
            old_use_sobol = getattr(mc_pricer, "use_sobol", None)
            try:
                # Keep Tier 2 path counts small for CPU-friendliness
                if old_n_paths is not None:
                    mc_pricer.n_paths = int(max(1500, min(6000, old_n_paths // 8)))
                if old_use_sobol is not None:
                    mc_pricer.use_sobol = True

                mc_price, mc_se, _ = mc_pricer.price(
                    spot, strike, T, r, q, max(surface_iv, 1e-4),
                    rp, option_type, india_features or {},
                )
                if np.isfinite(mc_price) and mc_price > 0:
                    out["mc_price"] = float(mc_price)
                    out["mc_std_error"] = float(max(mc_se, 0.0))

                    # Dual-control-variate style blend:
                    # base BS + COS + MC (MC gets largest weight).
                    c = float(cos_price) if (cos_price is not None and np.isfinite(cos_price) and cos_price > 0) else float(bs_price)
                    blended = 0.70 * float(mc_price) + 0.20 * c + 0.10 * float(bs_price)
                    out["price"] = float(max(blended, 0.0))
                    out["tier_used"] = "tier2_qmc_mc"
            except Exception:
                pass
            finally:
                if old_n_paths is not None:
                    mc_pricer.n_paths = old_n_paths
                if old_use_sobol is not None:
                    mc_pricer.use_sobol = old_use_sobol

        out["latency_ms"] = float(_elapsed_ms())
        return out

