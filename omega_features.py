#!/usr/bin/env python3
"""
omega_features.py — Centralized Feature Flag Configuration for OMEGA
========================================================================

All feature upgrades default to OFF unless explicitly enabled.
Backward compatibility is preserved via legacy v5 flag names.

Override via environment variable:
    export OMEGA_FEATURES_JSON='{"india_vix_synth": true, "arbfree_surface": true}'

Or programmatically:
    from omega_features import OmegaFeatures
    features = OmegaFeatures(india_vix_synth=True)
"""

import os
import json
from typing import Optional


class OmegaFeatures:
    """
    Immutable feature-flag container.

    Attributes
    ----------
    Legacy v5 flags (kept for backward compatibility):
    india_vix_synth : bool
        Compute synthetic India VIX from option chain.
    arb_free_surface : bool
        Arbitrage-aware volatility surface construction.
    vrr_state : bool
        Variance risk premium / risk-aversion state filter.
    surface_shock : bool
        Surface shock scenario generator.

    OMEGA v6 flags (new):
    USE_NSE_CONTRACT_SPECS : bool
    USE_NSE_VIX_ENGINE : bool
    USE_TAIL_CORRECTED_VARIANCE : bool
    USE_ESSVI_SURFACE : bool
    USE_SVI_FIXED_POINT_WARMSTART : bool
    USE_MODEL_FREE_VRP : bool
    USE_TIERED_PRICER : bool
    USE_CONFORMAL_INTERVALS : bool
    USE_RESEARCH_HIGH_CONVICTION : bool
    USE_OOS_RELIABILITY_GATE : bool
    """

    # All known flags with defaults.
    # New v6 flags are OFF by default to preserve baseline outputs.
    _DEFAULTS = {
        # v5 (legacy)
        'india_vix_synth': False,
        'arb_free_surface': False,
        'vrr_state': False,
        'surface_shock': False,
        # v6 (new)
        'USE_NSE_CONTRACT_SPECS': False,
        'USE_NSE_VIX_ENGINE': False,
        'USE_TAIL_CORRECTED_VARIANCE': False,
        'USE_ESSVI_SURFACE': False,
        'USE_SVI_FIXED_POINT_WARMSTART': False,
        'USE_MODEL_FREE_VRP': False,
        'USE_TIERED_PRICER': False,
        'USE_CONFORMAL_INTERVALS': False,
        'USE_RESEARCH_HIGH_CONVICTION': False,
        'USE_OOS_RELIABILITY_GATE': False,
    }

    _ALIASES = {
        # Common lowercase aliases for v6 flags
        'use_nse_contract_specs': 'USE_NSE_CONTRACT_SPECS',
        'use_nse_vix_engine': 'USE_NSE_VIX_ENGINE',
        'use_tail_corrected_variance': 'USE_TAIL_CORRECTED_VARIANCE',
        'use_essvi_surface': 'USE_ESSVI_SURFACE',
        'use_svi_fixed_point_warmstart': 'USE_SVI_FIXED_POINT_WARMSTART',
        'use_model_free_vrp': 'USE_MODEL_FREE_VRP',
        'use_tiered_pricer': 'USE_TIERED_PRICER',
        'use_conformal_intervals': 'USE_CONFORMAL_INTERVALS',
        'use_research_high_conviction': 'USE_RESEARCH_HIGH_CONVICTION',
        'use_oos_reliability_gate': 'USE_OOS_RELIABILITY_GATE',
    }

    def __init__(self, **overrides):
        """
        Parameters
        ----------
        **overrides : bool
            Keyword arguments matching flag names (or aliases) to override defaults.
            Unknown keys are silently ignored.
        """
        # Start with defaults
        flags = dict(self._DEFAULTS)

        def _canonical_key(key: str) -> str:
            if key in flags:
                return key
            return self._ALIASES.get(str(key), '')

        # Apply environment variable overrides first
        env_json = os.environ.get('OMEGA_FEATURES_JSON', '')
        if env_json:
            try:
                env_overrides = json.loads(env_json)
                for key, val in env_overrides.items():
                    ckey = _canonical_key(key)
                    if ckey:
                        flags[ckey] = bool(val)
            except (json.JSONDecodeError, TypeError):
                pass  # Silently ignore malformed JSON

        # Apply programmatic overrides (highest priority)
        for key, val in overrides.items():
            ckey = _canonical_key(key)
            if ckey:
                flags[ckey] = bool(val)

        # Set as attributes
        for key, val in flags.items():
            object.__setattr__(self, key, val)

        # Freeze
        object.__setattr__(self, '_frozen', True)

    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False):
            raise AttributeError(
                f"OmegaFeatures is immutable. Create a new instance to change flags."
            )
        object.__setattr__(self, name, value)

    def to_dict(self) -> dict:
        """Return all flags as a plain dict."""
        return {k: getattr(self, k) for k in self._DEFAULTS}

    def __repr__(self) -> str:
        flags_str = ', '.join(f'{k}={getattr(self, k)}' for k in self._DEFAULTS)
        return f'OmegaFeatures({flags_str})'

    @classmethod
    def all_on(cls) -> 'OmegaFeatures':
        """Return an instance with every flag ON (for testing)."""
        return cls(**{k: True for k in cls._DEFAULTS})

    @classmethod
    def all_off(cls) -> 'OmegaFeatures':
        """Return an instance with every flag OFF (baseline)."""
        return cls()

    @classmethod
    def best_mode_macbook(cls) -> 'OmegaFeatures':
        """
        Recommended CPU-friendly production profile for OMEGA v6.

        Notes
        -----
        - Keeps expensive tails/exotic paths disabled by default.
        - Enables accuracy-improving components that are stable for scans.
        """
        return cls(
            # legacy
            india_vix_synth=False,        # prefer NSE engine
            arb_free_surface=True,        # extra safety checks/repair
            vrr_state=False,              # superseded by model-free VRP state
            surface_shock=False,
            # v6
            USE_NSE_CONTRACT_SPECS=True,
            USE_NSE_VIX_ENGINE=True,
            USE_TAIL_CORRECTED_VARIANCE=False,
            USE_ESSVI_SURFACE=True,
            USE_SVI_FIXED_POINT_WARMSTART=True,
            USE_MODEL_FREE_VRP=True,
            USE_TIERED_PRICER=True,
            USE_CONFORMAL_INTERVALS=True,
            USE_RESEARCH_HIGH_CONVICTION=False,
            USE_OOS_RELIABILITY_GATE=False,
        )

    @classmethod
    def best_mode_max_accuracy(cls) -> 'OmegaFeatures':
        """
        Accuracy-first profile for research/deep scans.

        This mode is intentionally CPU-intensive and enables conservative
        variance tail correction in addition to the MacBook profile stack.
        """
        return cls(
            # legacy
            india_vix_synth=False,
            arb_free_surface=True,
            vrr_state=False,
            surface_shock=False,
            # v6
            USE_NSE_CONTRACT_SPECS=True,
            USE_NSE_VIX_ENGINE=True,
            USE_TAIL_CORRECTED_VARIANCE=True,
            USE_ESSVI_SURFACE=True,
            USE_SVI_FIXED_POINT_WARMSTART=True,
            USE_MODEL_FREE_VRP=True,
            USE_TIERED_PRICER=True,
            USE_CONFORMAL_INTERVALS=True,
            USE_RESEARCH_HIGH_CONVICTION=True,
            USE_OOS_RELIABILITY_GATE=True,
        )


# Module-level singleton (default: all OFF)
FEATURES = OmegaFeatures()


def get_features() -> OmegaFeatures:
    """Return the module-level feature flags singleton."""
    return FEATURES


def set_features(**kwargs) -> OmegaFeatures:
    """
    Replace the module-level singleton with new feature flags.

    Usage:
        set_features(india_vix_synth=True)
    """
    global FEATURES
    FEATURES = OmegaFeatures(**kwargs)
    return FEATURES


def set_best_mode_macbook() -> OmegaFeatures:
    """
    Convenience helper to switch to the recommended v6 CPU profile.
    """
    global FEATURES
    FEATURES = OmegaFeatures.best_mode_macbook()
    return FEATURES


def set_best_mode_max_accuracy() -> OmegaFeatures:
    """
    Convenience helper to switch to the accuracy-first v6 profile.
    """
    global FEATURES
    FEATURES = OmegaFeatures.best_mode_max_accuracy()
    return FEATURES
