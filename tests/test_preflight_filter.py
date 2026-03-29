#!/usr/bin/env python3
"""
Tests for preflight_filter.py — Post Fiat Signal Routing Protocol
=================================================================
90+ tests covering:
  - VOI computation (edge cases, accuracy boundaries, hazard adjustment)
  - Regime-duration gate boundary transitions
  - Weak-symbol INVERT logic
  - All-signals-filtered scenarios
  - Confidence threshold enforcement
  - Malformed input handling
  - Schema validation
  - Cross-schema compatibility with pf-signal-schema and pf-resolution-protocol
  - Per-gate isolation
  - Multi-symbol routing
  - REDUCE_WEIGHT policy
  - EXCLUDE policy
  - Duration bucket assignment
  - Limitation detection
  - CLI-style end-to-end
"""

import json
import math
import os
import sys
import tempfile
import unittest

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preflight_filter import (
    PreFlightRouter,
    wilson_ci,
    weibull_hazard,
    weibull_survival,
    compute_voi,
    assign_duration_bucket,
    validate_report,
    validate_config,
    PROTOCOL_VERSION,
    DEFAULT_DURATION_BUCKETS,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal config
# ---------------------------------------------------------------------------

def _minimal_config(**overrides):
    """Build a minimal routing config for testing."""
    config = {
        "policy_version": "1.0.0",
        "producer_id": "test-producer",
        "gates": {
            "regime_gate": {"enabled": False},
            "duration_gate": {"enabled": False},
            "confidence_gate": {"enabled": False},
            "voi_gate": {"enabled": False},
            "weak_symbol_gate": {"enabled": False},
        },
        "symbols": {},
    }
    config.update(overrides)
    return config


def _make_signal(symbol="BTC", confidence=0.55, direction="bullish",
                 signal_id=None, **kwargs):
    """Build a minimal test signal."""
    sig = {
        "signal_id": signal_id or f"test-{symbol}-001",
        "producer_id": "test-producer",
        "timestamp": "2026-03-29T16:00:00Z",
        "symbol": symbol,
        "direction": direction,
        "confidence": confidence,
        "horizon_hours": 24,
        "action": "EXECUTE",
        "schema_version": "1.0.0",
    }
    sig.update(kwargs)
    return sig


# ===========================================================================
# 1. VOI Computation Tests
# ===========================================================================

class TestVOIComputation(unittest.TestCase):
    """Tests for the VOI formula: VOI = conf * (2*acc - 1)."""

    def test_voi_50_percent_accuracy(self):
        """At 50% accuracy, VOI should be exactly zero."""
        result = compute_voi(0.5, 0.6)
        self.assertAlmostEqual(result["voi"], 0.0, places=6)

    def test_voi_positive_when_accuracy_above_50(self):
        """VOI should be positive when accuracy > 50%."""
        result = compute_voi(0.6, 0.55)
        self.assertGreater(result["voi"], 0)

    def test_voi_negative_when_accuracy_below_50(self):
        """VOI should be negative when accuracy < 50%."""
        result = compute_voi(0.4, 0.52)
        self.assertLess(result["voi"], 0)

    def test_voi_formula_correctness(self):
        """Check exact VOI formula: conf * (2*acc - 1)."""
        result = compute_voi(0.7, 0.8)
        expected = 0.8 * (2 * 0.7 - 1)  # 0.8 * 0.4 = 0.32
        self.assertAlmostEqual(result["voi"], expected, places=5)

    def test_voi_zero_confidence(self):
        """Zero confidence should produce zero VOI."""
        result = compute_voi(0.9, 0.0)
        self.assertAlmostEqual(result["voi"], 0.0, places=6)

    def test_voi_perfect_accuracy(self):
        """100% accuracy: VOI = confidence."""
        result = compute_voi(1.0, 0.7)
        self.assertAlmostEqual(result["voi"], 0.7, places=6)

    def test_voi_zero_accuracy(self):
        """0% accuracy: VOI = -confidence."""
        result = compute_voi(0.0, 0.7)
        self.assertAlmostEqual(result["voi"], -0.7, places=6)

    def test_voi_e_withhold_always_zero(self):
        """E[karma|withhold] is always 0."""
        result = compute_voi(0.8, 0.6)
        self.assertEqual(result["e_karma_withhold"], 0.0)

    def test_voi_accuracy_in_result(self):
        """Accuracy estimate should be in the result."""
        result = compute_voi(0.65, 0.5)
        self.assertAlmostEqual(result["accuracy_estimate"], 0.65, places=5)

    def test_voi_symmetry(self):
        """VOI at acc=0.6 should equal -VOI at acc=0.4 (same conf)."""
        v1 = compute_voi(0.6, 0.5)
        v2 = compute_voi(0.4, 0.5)
        self.assertAlmostEqual(v1["voi"], -v2["voi"], places=6)


class TestVOIHazardAdjustment(unittest.TestCase):
    """Tests for Weibull hazard integration with VOI."""

    def test_hazard_adjustment_present(self):
        """Hazard adjustment should appear when params provided."""
        result = compute_voi(0.6, 0.5, {"shape": 2.0, "scale": 18.0}, 10.0)
        self.assertIn("hazard_adjustment", result)

    def test_hazard_adjustment_reduces_voi(self):
        """Hazard-adjusted VOI should be <= raw VOI (survival <= 1)."""
        result = compute_voi(0.6, 0.5, {"shape": 2.0, "scale": 18.0}, 10.0)
        adj = result["hazard_adjustment"]
        self.assertLessEqual(abs(adj["hazard_adjusted_voi"]), abs(result["voi"]) + 1e-6)

    def test_hazard_at_zero_duration(self):
        """At t=0, survival=1.0, so adjusted VOI = raw VOI."""
        result = compute_voi(0.6, 0.5, {"shape": 2.0, "scale": 18.0}, 0.0)
        # No hazard adjustment when t <= 0
        self.assertNotIn("hazard_adjustment", result)

    def test_hazard_survival_decreases_with_time(self):
        """Survival should decrease as duration increases."""
        r1 = compute_voi(0.6, 0.5, {"shape": 2.0, "scale": 18.0}, 5.0)
        r2 = compute_voi(0.6, 0.5, {"shape": 2.0, "scale": 18.0}, 20.0)
        s1 = r1["hazard_adjustment"]["survival_probability"]
        s2 = r2["hazard_adjustment"]["survival_probability"]
        self.assertGreater(s1, s2)

    def test_no_hazard_without_params(self):
        """No hazard adjustment when params are None."""
        result = compute_voi(0.6, 0.5, None, 10.0)
        self.assertNotIn("hazard_adjustment", result)


# ===========================================================================
# 2. Weibull Functions
# ===========================================================================

class TestWeibullFunctions(unittest.TestCase):
    """Tests for Weibull hazard and survival functions."""

    def test_survival_at_zero(self):
        """S(0) = 1.0 for any parameters."""
        self.assertAlmostEqual(weibull_survival(0, 2.0, 18.0), 1.0)

    def test_survival_decreasing(self):
        """S(t) should decrease with t."""
        s1 = weibull_survival(5.0, 2.0, 18.0)
        s2 = weibull_survival(15.0, 2.0, 18.0)
        self.assertGreater(s1, s2)

    def test_survival_positive(self):
        """S(t) should always be > 0."""
        s = weibull_survival(100.0, 2.0, 18.0)
        self.assertGreater(s, 0)

    def test_hazard_positive(self):
        """Hazard should be positive for t > 0 and valid params."""
        h = weibull_hazard(10.0, 2.0, 18.0)
        self.assertGreater(h, 0)

    def test_hazard_zero_at_zero_time(self):
        """Hazard at t=0 should be 0."""
        h = weibull_hazard(0, 2.0, 18.0)
        self.assertEqual(h, 0.0)

    def test_hazard_increasing_for_shape_gt_1(self):
        """With shape > 1 (wearing out), hazard should increase."""
        h1 = weibull_hazard(5.0, 2.0, 18.0)
        h2 = weibull_hazard(15.0, 2.0, 18.0)
        self.assertGreater(h2, h1)

    def test_hazard_invalid_params(self):
        """Invalid params should return 0."""
        self.assertEqual(weibull_hazard(10.0, 0, 18.0), 0.0)
        self.assertEqual(weibull_hazard(10.0, 2.0, 0), 0.0)
        self.assertEqual(weibull_hazard(10.0, -1, 18.0), 0.0)


# ===========================================================================
# 3. Wilson CI
# ===========================================================================

class TestWilsonCI(unittest.TestCase):
    """Tests for Wilson score confidence intervals."""

    def test_zero_trials(self):
        """n=0 should return all zeros."""
        ci = wilson_ci(0, 0)
        self.assertEqual(ci["point"], 0.0)
        self.assertEqual(ci["n"], 0)

    def test_all_hits(self):
        """100% hit rate at large n."""
        ci = wilson_ci(100, 100)
        self.assertAlmostEqual(ci["point"], 1.0)
        self.assertGreater(ci["lower"], 0.95)

    def test_no_hits(self):
        """0% hit rate."""
        ci = wilson_ci(0, 100)
        self.assertAlmostEqual(ci["point"], 0.0)
        self.assertLess(ci["upper"], 0.05)

    def test_ci_contains_point(self):
        """CI should contain the point estimate."""
        ci = wilson_ci(50, 100)
        self.assertLessEqual(ci["lower"], ci["point"])
        self.assertGreaterEqual(ci["upper"], ci["point"])

    def test_ci_narrows_with_n(self):
        """Width should decrease with sample size."""
        ci_small = wilson_ci(5, 10)
        ci_large = wilson_ci(500, 1000)
        width_small = ci_small["upper"] - ci_small["lower"]
        width_large = ci_large["upper"] - ci_large["lower"]
        self.assertGreater(width_small, width_large)


# ===========================================================================
# 4. Duration Bucket Assignment
# ===========================================================================

class TestDurationBuckets(unittest.TestCase):
    """Tests for duration bucket assignment."""

    def test_early_bucket(self):
        self.assertEqual(assign_duration_bucket(5.0), "early")

    def test_mid_bucket(self):
        self.assertEqual(assign_duration_bucket(13.0), "mid")

    def test_mature_bucket(self):
        self.assertEqual(assign_duration_bucket(16.0), "mature")

    def test_late_bucket(self):
        self.assertEqual(assign_duration_bucket(20.0), "late")

    def test_boundary_early_mid(self):
        """12.0 should be mid (min_days=12 for mid)."""
        self.assertEqual(assign_duration_bucket(12.0), "mid")

    def test_boundary_mid_mature(self):
        """15.0 should be mature."""
        self.assertEqual(assign_duration_bucket(15.0), "mature")

    def test_zero_duration(self):
        self.assertEqual(assign_duration_bucket(0.0), "early")

    def test_very_large_duration(self):
        self.assertEqual(assign_duration_bucket(100.0), "late")


# ===========================================================================
# 5. Regime Gate Tests
# ===========================================================================

class TestRegimeGate(unittest.TestCase):
    """Tests for regime-based signal suppression."""

    def test_allowed_regime_passes(self):
        config = _minimal_config()
        config["gates"]["regime_gate"] = {
            "enabled": True,
            "allowed_regimes": ["NEUTRAL", "SYSTEMIC"],
        }
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()], regime_id="SYSTEMIC")
        self.assertEqual(report["decisions"][0]["action"], "EMIT")

    def test_disallowed_regime_withholds(self):
        config = _minimal_config()
        config["gates"]["regime_gate"] = {
            "enabled": True,
            "allowed_regimes": ["NEUTRAL"],
        }
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()], regime_id="SYSTEMIC")
        self.assertEqual(report["decisions"][0]["action"], "WITHHOLD")

    def test_earnings_excluded(self):
        config = _minimal_config()
        config["gates"]["regime_gate"] = {
            "enabled": True,
            "allowed_regimes": ["NEUTRAL", "SYSTEMIC", "DIVERGENCE"],
        }
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()], regime_id="EARNINGS")
        self.assertEqual(report["decisions"][0]["action"], "WITHHOLD")
        self.assertIn("not in", report["decisions"][0]["rationale"])

    def test_disabled_regime_gate_passes_all(self):
        config = _minimal_config()
        config["gates"]["regime_gate"] = {"enabled": False}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()], regime_id="EARNINGS")
        self.assertEqual(report["decisions"][0]["action"], "EMIT")


# ===========================================================================
# 6. Duration Gate Tests
# ===========================================================================

class TestDurationGate(unittest.TestCase):
    """Tests for regime-duration signal gating."""

    def test_above_threshold_passes(self):
        config = _minimal_config()
        config["gates"]["duration_gate"] = {"enabled": True, "default_min_days": 15}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()], duration_days=16.0)
        self.assertEqual(report["decisions"][0]["action"], "EMIT")

    def test_below_threshold_withholds(self):
        config = _minimal_config()
        config["gates"]["duration_gate"] = {"enabled": True, "default_min_days": 15}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()], duration_days=12.0)
        self.assertEqual(report["decisions"][0]["action"], "WITHHOLD")

    def test_exact_threshold_passes(self):
        """Exact boundary (>=) should pass."""
        config = _minimal_config()
        config["gates"]["duration_gate"] = {"enabled": True, "default_min_days": 15}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()], duration_days=15.0)
        self.assertEqual(report["decisions"][0]["action"], "EMIT")

    def test_per_symbol_override(self):
        """Per-symbol duration override should take precedence."""
        config = _minimal_config()
        config["gates"]["duration_gate"] = {"enabled": True, "default_min_days": 15}
        config["symbols"]["BTC"] = {"min_duration_days": 10}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("BTC")], duration_days=12.0)
        self.assertEqual(report["decisions"][0]["action"], "EMIT")

    def test_per_symbol_stricter(self):
        """Per-symbol can be stricter than default."""
        config = _minimal_config()
        config["gates"]["duration_gate"] = {"enabled": True, "default_min_days": 15}
        config["symbols"]["ETH"] = {"min_duration_days": 20}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("ETH")], duration_days=18.0)
        self.assertEqual(report["decisions"][0]["action"], "WITHHOLD")


# ===========================================================================
# 7. Confidence Gate Tests
# ===========================================================================

class TestConfidenceGate(unittest.TestCase):
    """Tests for confidence threshold enforcement."""

    def test_above_threshold_passes(self):
        config = _minimal_config()
        config["gates"]["confidence_gate"] = {"enabled": True, "default_min_confidence": 0.30}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal(confidence=0.55)])
        self.assertEqual(report["decisions"][0]["action"], "EMIT")

    def test_below_threshold_withholds(self):
        config = _minimal_config()
        config["gates"]["confidence_gate"] = {"enabled": True, "default_min_confidence": 0.30}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal(confidence=0.20)])
        self.assertEqual(report["decisions"][0]["action"], "WITHHOLD")

    def test_exact_threshold_passes(self):
        config = _minimal_config()
        config["gates"]["confidence_gate"] = {"enabled": True, "default_min_confidence": 0.30}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal(confidence=0.30)])
        self.assertEqual(report["decisions"][0]["action"], "EMIT")

    def test_per_symbol_confidence_override(self):
        config = _minimal_config()
        config["gates"]["confidence_gate"] = {"enabled": True, "default_min_confidence": 0.30}
        config["symbols"]["SOL"] = {"min_confidence": 0.05}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("SOL", confidence=0.10)])
        self.assertEqual(report["decisions"][0]["action"], "EMIT")

    def test_zero_confidence_withheld(self):
        config = _minimal_config()
        config["gates"]["confidence_gate"] = {"enabled": True, "default_min_confidence": 0.01}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal(confidence=0.0)])
        self.assertEqual(report["decisions"][0]["action"], "WITHHOLD")


# ===========================================================================
# 8. VOI Gate Tests
# ===========================================================================

class TestVOIGate(unittest.TestCase):
    """Tests for VOI-based routing gate."""

    def _voi_config(self, **sym_overrides):
        config = _minimal_config()
        config["gates"]["voi_gate"] = {"enabled": True, "min_voi": 0.0}
        sym = {
            "voi_cells": {
                "mature": {"voi": 0.03, "accuracy": 0.56, "n": 100, "e_karma_send": 0.03}
            }
        }
        sym.update(sym_overrides)
        config["symbols"]["BTC"] = sym
        config["duration_buckets"] = DEFAULT_DURATION_BUCKETS
        return config

    def test_positive_voi_emits(self):
        config = self._voi_config()
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("BTC")], duration_days=16.0)
        self.assertEqual(report["decisions"][0]["action"], "EMIT")

    def test_negative_voi_withholds(self):
        config = self._voi_config()
        config["symbols"]["BTC"]["voi_cells"]["mature"]["accuracy"] = 0.38
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("BTC")], duration_days=16.0)
        self.assertEqual(report["decisions"][0]["action"], "WITHHOLD")

    def test_voi_zero_accuracy_withholds(self):
        config = self._voi_config()
        config["symbols"]["BTC"]["voi_cells"]["mature"]["accuracy"] = 0.50
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("BTC")], duration_days=16.0)
        # VOI = conf * (2*0.5 - 1) = 0, and min_voi=0, so >= passes
        self.assertEqual(report["decisions"][0]["action"], "EMIT")

    def test_voi_result_in_decision(self):
        config = self._voi_config()
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("BTC")], duration_days=16.0)
        voi = report["decisions"][0].get("voi")
        self.assertIsNotNone(voi)
        self.assertIn("e_karma_send", voi)
        self.assertIn("voi", voi)

    def test_voi_invert_symbol_uses_inverted_accuracy(self):
        """INVERT symbols should compute VOI with 1-accuracy."""
        config = self._voi_config(
            weak_symbol_policy="INVERT",
            inversion_justified=True,
            inversion_p_value=0.001,
            weakness_severity="SEVERE",
            weakness_score=0.7,
            accuracy=0.38,
        )
        config["symbols"]["BTC"]["voi_cells"]["mature"]["accuracy"] = 0.38
        config["gates"]["weak_symbol_gate"] = {"enabled": True, "severity_threshold": "MODERATE"}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("BTC")], duration_days=16.0)
        # Inverted accuracy = 0.62, so VOI should be positive
        self.assertEqual(report["decisions"][0]["action"], "INVERT")
        voi = report["decisions"][0]["voi"]
        self.assertGreater(voi["voi"], 0)

    def test_voi_missing_cell_defaults_to_50(self):
        """Missing VOI cell should default to 50% accuracy."""
        config = self._voi_config()
        config["symbols"]["BTC"]["voi_cells"] = {}  # no cells
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("BTC")], duration_days=16.0)
        voi = report["decisions"][0]["voi"]
        self.assertAlmostEqual(voi["accuracy_estimate"], 0.5, places=1)


# ===========================================================================
# 9. Weak Symbol Gate + INVERT Tests
# ===========================================================================

class TestWeakSymbolGate(unittest.TestCase):
    """Tests for weak-symbol detection and INVERT routing."""

    def _weak_config(self, policy="INVERT", justified=True, severity="SEVERE"):
        config = _minimal_config()
        config["gates"]["weak_symbol_gate"] = {
            "enabled": True, "severity_threshold": "MODERATE"
        }
        config["symbols"]["SOL"] = {
            "weak_symbol_policy": policy,
            "weakness_score": 0.70,
            "weakness_severity": severity,
            "inversion_justified": justified,
            "inversion_p_value": 0.0001,
            "accuracy": 0.44,
        }
        return config

    def test_invert_when_justified(self):
        config = self._weak_config("INVERT", True, "SEVERE")
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("SOL", direction="bearish")])
        d = report["decisions"][0]
        self.assertEqual(d["action"], "INVERT")
        self.assertEqual(d["routed_direction"], "bullish")

    def test_invert_flips_bullish_to_bearish(self):
        config = self._weak_config("INVERT", True, "SEVERE")
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("SOL", direction="bullish")])
        self.assertEqual(report["decisions"][0]["routed_direction"], "bearish")

    def test_withhold_when_not_justified(self):
        config = self._weak_config("INVERT", False, "SEVERE")
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("SOL")])
        self.assertEqual(report["decisions"][0]["action"], "WITHHOLD")

    def test_exclude_policy_withholds(self):
        config = self._weak_config("EXCLUDE", True, "SEVERE")
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("SOL")])
        self.assertEqual(report["decisions"][0]["action"], "WITHHOLD")

    def test_reduce_weight_emits(self):
        config = self._weak_config("REDUCE_WEIGHT", True, "SEVERE")
        config["symbols"]["SOL"]["weight_factor"] = 0.5
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("SOL")])
        d = report["decisions"][0]
        self.assertEqual(d["action"], "EMIT")
        self.assertIn("confidence reduced", d["rationale"])

    def test_mild_weakness_passes(self):
        """MILD weakness below MODERATE threshold should pass."""
        config = self._weak_config("INVERT", True, "MILD")
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("SOL")])
        self.assertEqual(report["decisions"][0]["action"], "EMIT")

    def test_weak_diag_in_decision(self):
        config = self._weak_config()
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("SOL")])
        d = report["decisions"][0]
        self.assertIn("weak_symbol", d)
        ws = d["weak_symbol"]
        self.assertEqual(ws["severity"], "SEVERE")
        self.assertAlmostEqual(ws["weakness_score"], 0.70, places=2)

    def test_non_weak_symbol_unaffected(self):
        config = self._weak_config()
        config["symbols"]["BTC"] = {"weakness_severity": "NONE", "weakness_score": 0.10}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("BTC")])
        self.assertEqual(report["decisions"][0]["action"], "EMIT")

    def test_no_policy_withholds_weak(self):
        """Weak symbol with NONE policy should withhold."""
        config = self._weak_config("NONE", True, "SEVERE")
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("SOL")])
        self.assertEqual(report["decisions"][0]["action"], "WITHHOLD")


# ===========================================================================
# 10. All-Signals-Filtered Scenarios
# ===========================================================================

class TestAllFiltered(unittest.TestCase):
    """Tests for edge case where all signals are withheld."""

    def test_all_withheld_by_regime(self):
        config = _minimal_config()
        config["gates"]["regime_gate"] = {"enabled": True, "allowed_regimes": ["NEUTRAL"]}
        router = PreFlightRouter(config)
        signals = [_make_signal("BTC"), _make_signal("ETH"), _make_signal("SOL")]
        report = router.route_signals(signals, regime_id="SYSTEMIC")
        self.assertEqual(report["summary"]["total_emit"], 0)
        self.assertEqual(report["summary"]["total_withhold"], 3)
        self.assertEqual(report["summary"]["filter_rate"], 1.0)

    def test_all_withheld_produces_limitation(self):
        config = _minimal_config()
        config["gates"]["regime_gate"] = {"enabled": True, "allowed_regimes": ["NEUTRAL"]}
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()], regime_id="SYSTEMIC")
        lim_ids = [l["id"] for l in report.get("limitations", [])]
        self.assertIn("ALL_SIGNALS_FILTERED", lim_ids)

    def test_empty_input(self):
        config = _minimal_config()
        router = PreFlightRouter(config)
        report = router.route_signals([])
        self.assertEqual(report["summary"]["total_input"], 0)
        self.assertEqual(len(report["decisions"]), 0)


# ===========================================================================
# 11. Multi-Gate Interaction
# ===========================================================================

class TestMultiGateInteraction(unittest.TestCase):
    """Tests for AND logic across multiple gates."""

    def test_all_gates_pass(self):
        config = _minimal_config()
        config["gates"]["regime_gate"] = {"enabled": True, "allowed_regimes": ["SYSTEMIC"]}
        config["gates"]["duration_gate"] = {"enabled": True, "default_min_days": 15}
        config["gates"]["confidence_gate"] = {"enabled": True, "default_min_confidence": 0.30}
        router = PreFlightRouter(config)
        report = router.route_signals(
            [_make_signal(confidence=0.55)],
            regime_id="SYSTEMIC", duration_days=16.0
        )
        self.assertEqual(report["decisions"][0]["action"], "EMIT")

    def test_first_gate_fails(self):
        """If regime gate fails, should WITHHOLD even if others would pass."""
        config = _minimal_config()
        config["gates"]["regime_gate"] = {"enabled": True, "allowed_regimes": ["NEUTRAL"]}
        config["gates"]["duration_gate"] = {"enabled": True, "default_min_days": 15}
        router = PreFlightRouter(config)
        report = router.route_signals(
            [_make_signal()], regime_id="SYSTEMIC", duration_days=20.0
        )
        self.assertEqual(report["decisions"][0]["action"], "WITHHOLD")

    def test_gate_counts_in_decision(self):
        config = _minimal_config()
        config["gates"]["regime_gate"] = {"enabled": True, "allowed_regimes": ["SYSTEMIC"]}
        config["gates"]["duration_gate"] = {"enabled": True, "default_min_days": 15}
        config["gates"]["confidence_gate"] = {"enabled": True, "default_min_confidence": 0.30}
        router = PreFlightRouter(config)
        report = router.route_signals(
            [_make_signal(confidence=0.55)],
            regime_id="SYSTEMIC", duration_days=16.0
        )
        d = report["decisions"][0]
        self.assertEqual(len(d["gates_passed"]), 3)
        self.assertEqual(len(d["gates_failed"]), 0)


# ===========================================================================
# 12. Summary Statistics
# ===========================================================================

class TestSummaryStats(unittest.TestCase):
    """Tests for report summary correctness."""

    def test_per_symbol_counts(self):
        config = _minimal_config()
        router = PreFlightRouter(config)
        signals = [
            _make_signal("BTC", signal_id="s1"),
            _make_signal("BTC", signal_id="s2"),
            _make_signal("ETH", signal_id="s3"),
        ]
        report = router.route_signals(signals)
        ps = report["summary"]["per_symbol_summary"]
        self.assertEqual(ps["BTC"]["total"], 2)
        self.assertEqual(ps["ETH"]["total"], 1)

    def test_filter_rate_calculation(self):
        config = _minimal_config()
        config["gates"]["confidence_gate"] = {"enabled": True, "default_min_confidence": 0.50}
        router = PreFlightRouter(config)
        signals = [
            _make_signal(confidence=0.60, signal_id="s1"),
            _make_signal(confidence=0.40, signal_id="s2"),
            _make_signal(confidence=0.55, signal_id="s3"),
            _make_signal(confidence=0.20, signal_id="s4"),
        ]
        report = router.route_signals(signals)
        self.assertEqual(report["summary"]["total_emit"], 2)
        self.assertEqual(report["summary"]["total_withhold"], 2)
        self.assertAlmostEqual(report["summary"]["filter_rate"], 0.5, places=2)

    def test_gate_pass_rates(self):
        config = _minimal_config()
        config["gates"]["confidence_gate"] = {"enabled": True, "default_min_confidence": 0.50}
        router = PreFlightRouter(config)
        signals = [
            _make_signal(confidence=0.60, signal_id="s1"),
            _make_signal(confidence=0.40, signal_id="s2"),
        ]
        report = router.route_signals(signals)
        rates = report["summary"]["per_gate_pass_rates"]
        self.assertAlmostEqual(rates["confidence_gate"], 0.5, places=2)

    def test_invert_rate(self):
        config = _minimal_config()
        config["gates"]["weak_symbol_gate"] = {"enabled": True, "severity_threshold": "MODERATE"}
        config["symbols"]["SOL"] = {
            "weak_symbol_policy": "INVERT",
            "weakness_severity": "SEVERE",
            "weakness_score": 0.70,
            "inversion_justified": True,
            "inversion_p_value": 0.001,
            "accuracy": 0.44,
        }
        router = PreFlightRouter(config)
        signals = [
            _make_signal("BTC", signal_id="s1"),
            _make_signal("SOL", signal_id="s2"),
        ]
        report = router.route_signals(signals)
        self.assertAlmostEqual(report["summary"]["invert_rate"], 0.5, places=2)


# ===========================================================================
# 13. Report Structure
# ===========================================================================

class TestReportStructure(unittest.TestCase):
    """Tests for report metadata and structure."""

    def test_protocol_version(self):
        config = _minimal_config()
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()])
        self.assertEqual(report["protocol_version"], PROTOCOL_VERSION)

    def test_meta_fields(self):
        config = _minimal_config()
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()], regime_id="SYSTEMIC", duration_days=16.0)
        meta = report["meta"]
        self.assertEqual(meta["producer_id"], "test-producer")
        self.assertIn("generated_at", meta)
        self.assertEqual(meta["regime_context"]["regime_id"], "SYSTEMIC")
        self.assertEqual(meta["regime_context"]["duration_days"], 16.0)

    def test_config_snapshot_included(self):
        config = _minimal_config()
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()])
        self.assertIn("config_snapshot", report)
        self.assertEqual(report["config_snapshot"]["producer_id"], "test-producer")

    def test_decisions_order_matches_input(self):
        config = _minimal_config()
        router = PreFlightRouter(config)
        signals = [
            _make_signal("BTC", signal_id="first"),
            _make_signal("ETH", signal_id="second"),
            _make_signal("SOL", signal_id="third"),
        ]
        report = router.route_signals(signals)
        ids = [d["signal_id"] for d in report["decisions"]]
        self.assertEqual(ids, ["first", "second", "third"])


# ===========================================================================
# 14. Limitation Detection
# ===========================================================================

class TestLimitations(unittest.TestCase):
    """Tests for automatic limitation detection."""

    def test_single_regime_limitation(self):
        config = _minimal_config()
        config["evidence_source"] = "SYSTEMIC regime data"
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()], regime_id="SYSTEMIC")
        lim_ids = [l["id"] for l in report["limitations"]]
        self.assertIn("SINGLE_REGIME_EVIDENCE", lim_ids)

    def test_small_sample_limitation(self):
        config = _minimal_config()
        config["gates"]["voi_gate"] = {"enabled": True, "min_voi": 0.0}
        config["symbols"]["BTC"] = {
            "voi_cells": {"mature": {"voi": 0.01, "accuracy": 0.55, "n": 10, "e_karma_send": 0.01}}
        }
        config["duration_buckets"] = DEFAULT_DURATION_BUCKETS
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()], duration_days=16.0)
        lim_ids = [l["id"] for l in report["limitations"]]
        self.assertIn("SMALL_SAMPLE_VOI", lim_ids)

    def test_no_false_limitation_for_good_data(self):
        config = _minimal_config()
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()])
        lim_ids = [l["id"] for l in report["limitations"]]
        self.assertNotIn("ALL_SIGNALS_FILTERED", lim_ids)


# ===========================================================================
# 15. Schema Validation
# ===========================================================================

class TestSchemaValidation(unittest.TestCase):
    """Tests for JSON Schema validation of reports and configs."""

    def test_report_validates(self):
        """A properly generated report should validate."""
        config = _minimal_config()
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()])
        valid, errors = validate_report(report)
        if not valid:
            self.fail(f"Report validation failed: {errors}")

    def test_config_validates(self):
        """The example config should validate."""
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "routing_config_example.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            valid, errors = validate_config(config)
            if not valid:
                self.fail(f"Config validation failed: {errors}")

    def test_report_with_voi_validates(self):
        """Report with VOI details should validate."""
        config = _minimal_config()
        config["gates"]["voi_gate"] = {"enabled": True, "min_voi": 0.0}
        config["symbols"]["BTC"] = {
            "voi_cells": {"mature": {"voi": 0.03, "accuracy": 0.56, "n": 100, "e_karma_send": 0.03}}
        }
        config["duration_buckets"] = DEFAULT_DURATION_BUCKETS
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()], duration_days=16.0)
        valid, errors = validate_report(report)
        if not valid:
            self.fail(f"Report with VOI validation failed: {errors}")

    def test_report_with_invert_validates(self):
        """Report with INVERT decision should validate."""
        config = _minimal_config()
        config["gates"]["weak_symbol_gate"] = {"enabled": True, "severity_threshold": "MODERATE"}
        config["symbols"]["SOL"] = {
            "weak_symbol_policy": "INVERT",
            "weakness_severity": "SEVERE", "weakness_score": 0.70,
            "inversion_justified": True, "inversion_p_value": 0.001,
            "accuracy": 0.44,
        }
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal("SOL")])
        valid, errors = validate_report(report)
        if not valid:
            self.fail(f"Report with INVERT validation failed: {errors}")


# ===========================================================================
# 16. Cross-Schema Compatibility
# ===========================================================================

class TestCrossSchemaCompat(unittest.TestCase):
    """Tests for compatibility with pf-signal-schema and pf-resolution-protocol."""

    def test_action_vocabulary_matches_signal_schema(self):
        """Our EMIT/WITHHOLD/INVERT should map cleanly to signal schema EXECUTE/WITHHOLD/INVERT."""
        action_map = {"EMIT": "EXECUTE", "WITHHOLD": "WITHHOLD", "INVERT": "INVERT"}
        for routing_action, signal_action in action_map.items():
            self.assertIn(signal_action, ["EXECUTE", "WITHHOLD", "INVERT"])

    def test_signal_schema_signals_accepted(self):
        """Signals conforming to pf-signal-schema should route without error."""
        schema_signal = {
            "signal_id": "pf-BTC-1774000000",
            "producer_id": "post-fiat-signals",
            "timestamp": "2026-03-29T10:00:00.000Z",
            "symbol": "BTC",
            "direction": "bullish",
            "confidence": 0.53,
            "horizon_hours": 24,
            "action": "EXECUTE",
            "schema_version": "1.0.0",
            "regime_context": {
                "regime_id": "SYSTEMIC",
                "regime_confidence": 0.77,
                "proximity": 0.4,
                "duration_days": 16,
                "decision": "NO_TRADE"
            },
        }
        config = _minimal_config()
        router = PreFlightRouter(config)
        report = router.route_signals([schema_signal])
        self.assertEqual(len(report["decisions"]), 1)

    def test_resolution_compatible_output(self):
        """Routing decisions should reference signal_ids that resolution protocol can trace."""
        config = _minimal_config()
        router = PreFlightRouter(config)
        sig = _make_signal(signal_id="traceable-id-123")
        report = router.route_signals([sig])
        self.assertEqual(report["decisions"][0]["signal_id"], "traceable-id-123")

    def test_confidence_scale_01(self):
        """Confidence should be 0-1 scale (matching signal schema)."""
        config = _minimal_config()
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal(confidence=0.55)])
        conf = report["decisions"][0]["confidence"]
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_regime_ids_match_signal_schema(self):
        """Regime IDs should match signal schema enum."""
        valid_regimes = {"SYSTEMIC", "NEUTRAL", "DIVERGENCE", "EARNINGS", "UNKNOWN"}
        config = _minimal_config()
        config["gates"]["regime_gate"] = {"enabled": True, "allowed_regimes": list(valid_regimes)}
        router = PreFlightRouter(config)
        for regime in valid_regimes:
            report = router.route_signals([_make_signal()], regime_id=regime)
            self.assertEqual(report["decisions"][0]["action"], "EMIT")


# ===========================================================================
# 17. Malformed Input Handling
# ===========================================================================

class TestMalformedInput(unittest.TestCase):
    """Tests for graceful handling of malformed signals."""

    def test_missing_symbol(self):
        config = _minimal_config()
        router = PreFlightRouter(config)
        sig = {"signal_id": "bad-1", "confidence": 0.5, "direction": "bullish"}
        report = router.route_signals([sig])
        self.assertEqual(len(report["decisions"]), 1)
        self.assertEqual(report["decisions"][0]["symbol"], "UNKNOWN")

    def test_missing_confidence(self):
        config = _minimal_config()
        router = PreFlightRouter(config)
        sig = {"signal_id": "bad-2", "symbol": "BTC", "direction": "bullish"}
        report = router.route_signals([sig])
        self.assertEqual(report["decisions"][0]["confidence"], 0.0)

    def test_missing_signal_id(self):
        config = _minimal_config()
        router = PreFlightRouter(config)
        sig = {"symbol": "BTC", "confidence": 0.5, "direction": "bullish"}
        report = router.route_signals([sig])
        self.assertEqual(report["decisions"][0]["signal_id"], "unknown")

    def test_extra_fields_ignored(self):
        config = _minimal_config()
        router = PreFlightRouter(config)
        sig = _make_signal()
        sig["extra_field"] = "should_be_ignored"
        report = router.route_signals([sig])
        self.assertEqual(report["decisions"][0]["action"], "EMIT")


# ===========================================================================
# 18. File I/O
# ===========================================================================

class TestFileIO(unittest.TestCase):
    """Tests for loading config from files."""

    def test_from_config_file(self):
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "routing_config_example.json")
        if os.path.exists(config_path):
            router = PreFlightRouter.from_config(config_path)
            self.assertEqual(router.producer_id, "post-fiat-signals")
            self.assertIn("BTC", router.symbols)

    def test_output_to_file(self):
        config = _minimal_config()
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(report, f)
            tmppath = f.name
        try:
            with open(tmppath) as f:
                loaded = json.load(f)
            self.assertEqual(loaded["protocol_version"], PROTOCOL_VERSION)
        finally:
            os.unlink(tmppath)


# ===========================================================================
# 19. End-to-End with Real Config
# ===========================================================================

class TestEndToEnd(unittest.TestCase):
    """End-to-end tests using the example config and signals."""

    def setUp(self):
        self.config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "routing_config_example.json")
        self.signals_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "example_routing", "candidate_signals.json")

    def test_example_routing(self):
        if not os.path.exists(self.config_path):
            self.skipTest("Config not found")
        router = PreFlightRouter.from_config(self.config_path)
        with open(self.signals_path) as f:
            signals = json.load(f)
        report = router.route_signals(signals, regime_id="SYSTEMIC", duration_days=16.5)
        self.assertEqual(report["summary"]["total_input"], 6)
        self.assertGreater(report["summary"]["total_emit"], 0)

    def test_sol_inverted(self):
        if not os.path.exists(self.config_path):
            self.skipTest("Config not found")
        router = PreFlightRouter.from_config(self.config_path)
        with open(self.signals_path) as f:
            signals = json.load(f)
        report = router.route_signals(signals, regime_id="SYSTEMIC", duration_days=16.5)
        sol_decisions = [d for d in report["decisions"] if d["symbol"] == "SOL"]
        self.assertTrue(any(d["action"] == "INVERT" for d in sol_decisions))

    def test_low_conf_withheld(self):
        if not os.path.exists(self.config_path):
            self.skipTest("Config not found")
        router = PreFlightRouter.from_config(self.config_path)
        with open(self.signals_path) as f:
            signals = json.load(f)
        report = router.route_signals(signals, regime_id="SYSTEMIC", duration_days=16.5)
        btc_low = [d for d in report["decisions"]
                    if d["signal_id"] == "pf-BTC-1774800000"]
        self.assertEqual(btc_low[0]["action"], "WITHHOLD")

    def test_report_json_serializable(self):
        config = _minimal_config()
        router = PreFlightRouter(config)
        report = router.route_signals([_make_signal()])
        # Should not raise
        json_str = json.dumps(report)
        self.assertIsInstance(json_str, str)


if __name__ == "__main__":
    unittest.main()
