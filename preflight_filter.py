#!/usr/bin/env python3
"""
Post Fiat Signal Pre-Flight Router
====================================
Standalone CLI + library that applies routing policy to candidate signals,
producing auditable EMIT / WITHHOLD / INVERT decisions per signal.

Companion to:
- pf-signal-schema  (what a signal looks like)
- pf-resolution-protocol  (how a signal is scored)

This protocol defines the decision layer BETWEEN generation and emission.

Dependencies: jsonschema (pip install jsonschema)
No other external dependencies required.

Usage:
    # Route a batch of signals using a policy config
    python preflight_filter.py signals.json routing_config.json

    # Route with explicit regime context
    python preflight_filter.py signals.json routing_config.json \
        --regime SYSTEMIC --duration 16.5

    # JSON output
    python preflight_filter.py signals.json routing_config.json --json

    # Save to file
    python preflight_filter.py signals.json routing_config.json -o report.json

    # Validate output against routing protocol schema
    python preflight_filter.py signals.json routing_config.json --validate

    # Read signals from stdin
    echo '[...]' | python preflight_filter.py - routing_config.json
"""

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    import jsonschema
    from jsonschema import Draft202012Validator
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "1.0.0"
SIGNAL_SCHEMA_VERSION = "1.0.0"

SCHEMA_DIR = os.path.dirname(os.path.abspath(__file__))
ROUTING_PROTOCOL_SCHEMA_PATH = os.path.join(SCHEMA_DIR, "routing_protocol.json")
ROUTING_POLICY_SCHEMA_PATH = os.path.join(SCHEMA_DIR, "routing_policy.json")

DEFAULT_DURATION_BUCKETS = [
    {"label": "early", "min_days": 0, "max_days": 12},
    {"label": "mid", "min_days": 12, "max_days": 15},
    {"label": "mature", "min_days": 15, "max_days": 18},
    {"label": "late", "min_days": 18, "max_days": 9999},
]

SEVERITY_ORDER = {"NONE": 0, "MILD": 1, "MODERATE": 2, "SEVERE": 3}


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def wilson_ci(hits: int, n: int, z: float = 1.645) -> Dict[str, float]:
    """Wilson score confidence interval (90% default)."""
    if n == 0:
        return {"point": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return {
        "point": round(p, 6),
        "lower": round(max(0, center - spread), 6),
        "upper": round(min(1, center + spread), 6),
        "n": n,
    }


def weibull_hazard(t: float, shape: float, scale: float) -> float:
    """Weibull hazard function h(t) = (k/lambda) * (t/lambda)^(k-1)."""
    if t <= 0 or scale <= 0 or shape <= 0:
        return 0.0
    return (shape / scale) * ((t / scale) ** (shape - 1))


def weibull_survival(t: float, shape: float, scale: float) -> float:
    """Weibull survival S(t) = exp(-(t/lambda)^k)."""
    if t <= 0:
        return 1.0
    if scale <= 0 or shape <= 0:
        return 0.0
    return math.exp(-((t / scale) ** shape))


def compute_voi(accuracy: float, confidence: float,
                 hazard_params: Optional[Dict] = None,
                 duration_days: float = 0) -> Dict[str, Any]:
    """
    Compute Value-of-Information for a signal.

    VOI = E[karma|send] - E[karma|withhold]
        = confidence * (2 * accuracy - 1) - 0
        = confidence * (2 * accuracy - 1)

    Positive when accuracy > 50%. Negative when accuracy < 50%.
    Zero-confidence signals produce VOI=0 and are flagged.
    """
    e_send = confidence * (2 * accuracy - 1)
    e_withhold = 0.0
    voi = e_send - e_withhold

    result = {
        "e_karma_send": round(e_send, 6),
        "e_karma_withhold": round(e_withhold, 6),
        "voi": round(voi, 6),
        "accuracy_estimate": round(accuracy, 6),
    }

    # Flag zero-confidence as carrying no information
    if confidence <= 0.0:
        result["zero_confidence"] = True

    if hazard_params and duration_days > 0:
        # Validate Weibull params before use
        if "shape" not in hazard_params or "scale" not in hazard_params:
            result["hazard_skipped"] = True
            result["hazard_skip_reason"] = (
                f"Incomplete Weibull params: missing "
                f"{[k for k in ('shape', 'scale') if k not in hazard_params]}"
            )
        else:
            h = weibull_hazard(duration_days, hazard_params["shape"], hazard_params["scale"])
            s = weibull_survival(duration_days, hazard_params["shape"], hazard_params["scale"])
            result["hazard_adjustment"] = {
                "hazard_rate": round(h, 6),
                "survival_probability": round(s, 6),
                "hazard_adjusted_voi": round(voi * s, 6),
            }

    return result


def assign_duration_bucket(duration_days: float,
                           buckets: Optional[List[Dict]] = None) -> str:
    """Map regime duration to bucket label.

    Returns 'unknown' if duration falls in a gap between buckets rather
    than silently falling through to the last bucket.
    """
    if buckets is None:
        buckets = DEFAULT_DURATION_BUCKETS
    if not buckets:
        return "unknown"
    for bucket in buckets:
        if bucket["min_days"] <= duration_days < bucket["max_days"]:
            return bucket["label"]
    # Check if duration exceeds all buckets (legitimate overflow)
    max_bucket = max(buckets, key=lambda b: b["max_days"])
    if duration_days >= max_bucket["max_days"]:
        return max_bucket["label"]
    # Duration falls in a gap between buckets — return unknown
    return "unknown"


# ---------------------------------------------------------------------------
# Pre-Flight Router
# ---------------------------------------------------------------------------

class PreFlightRouter:
    """
    Applies a routing policy to candidate signals, producing per-signal
    EMIT / WITHHOLD / INVERT decisions with full audit trail.

    Usage:
        router = PreFlightRouter.from_config("routing_config.json")
        report = router.route_signals(signals, regime_id="SYSTEMIC",
                                       duration_days=16.5)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Routing policy configuration (conforms to routing_policy.json).
        """
        self.config = config
        self.gates = config.get("gates", {})
        self.symbols = config.get("symbols", {})
        self.duration_buckets = config.get("duration_buckets", DEFAULT_DURATION_BUCKETS)
        self.weibull_params = config.get("weibull_params", {})
        self.producer_id = config.get("producer_id", "unknown")

    @classmethod
    def from_config(cls, path: str) -> "PreFlightRouter":
        """Load from a routing policy JSON file."""
        with open(path) as f:
            config = json.load(f)
        return cls(config)

    @classmethod
    def from_dict(cls, config: Dict) -> "PreFlightRouter":
        """Create from an already-parsed config dict."""
        return cls(config)

    def route_signals(self, signals: List[Dict],
                      regime_id: str = "UNKNOWN",
                      duration_days: float = 0,
                      regime_confidence: float = 0.5) -> Dict[str, Any]:
        """
        Route a batch of candidate signals through the pre-flight filter.

        Args:
            signals: List of candidate signal dicts (pf-signal-schema format).
            regime_id: Current regime classification.
            duration_days: Days since current regime started.
            regime_confidence: Confidence in the regime classification.

        Returns:
            Routing report conforming to routing_protocol.json.
        """
        decisions = []
        per_symbol = defaultdict(lambda: {"total": 0, "emit": 0, "withhold": 0, "invert": 0})
        gate_pass_counts = defaultdict(lambda: {"passed": 0, "total": 0})

        for sig in signals:
            decision = self._route_single(sig, regime_id, duration_days, regime_confidence)
            decisions.append(decision)

            sym = decision["symbol"]
            action = decision["action"]
            per_symbol[sym]["total"] += 1
            per_symbol[sym][action.lower()] += 1

            for g in decision["gates_passed"]:
                gate_pass_counts[g["gate_id"]]["passed"] += 1
                gate_pass_counts[g["gate_id"]]["total"] += 1
            for g in decision["gates_failed"]:
                gate_pass_counts[g["gate_id"]]["total"] += 1

        total = len(signals)
        n_emit = sum(1 for d in decisions if d["action"] == "EMIT")
        n_withhold = sum(1 for d in decisions if d["action"] == "WITHHOLD")
        n_invert = sum(1 for d in decisions if d["action"] == "INVERT")

        gate_rates = {}
        for gid, counts in gate_pass_counts.items():
            gate_rates[gid] = round(counts["passed"] / max(1, counts["total"]), 4)

        limitations = self._detect_limitations(signals, decisions, regime_id, duration_days)

        report = {
            "meta": {
                "producer_id": self.producer_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "protocol_version": PROTOCOL_VERSION,
                "signal_schema_version": SIGNAL_SCHEMA_VERSION,
                "config_version": self.config.get("policy_version", "0.0.0"),
                "regime_context": {
                    "regime_id": regime_id,
                    "duration_days": duration_days,
                    "regime_confidence": regime_confidence,
                },
            },
            "summary": {
                "total_input": total,
                "total_emit": n_emit,
                "total_withhold": n_withhold,
                "total_invert": n_invert,
                "filter_rate": round(n_withhold / max(1, total), 4),
                "invert_rate": round(n_invert / max(1, total), 4),
                "per_gate_pass_rates": dict(gate_rates),
                "per_symbol_summary": dict(per_symbol),
            },
            "decisions": decisions,
            "config_snapshot": self.config,
            "protocol_version": PROTOCOL_VERSION,
            "limitations": limitations,
        }

        return report

    def _route_single(self, signal: Dict, regime_id: str,
                      duration_days: float,
                      regime_confidence: float) -> Dict[str, Any]:
        """Route a single candidate signal through all configured gates."""
        sig_id = signal.get("signal_id", "unknown")
        symbol = signal.get("symbol", "UNKNOWN")
        confidence = signal.get("confidence", 0.0)
        direction = signal.get("direction", "bullish")
        sym_config = self.symbols.get(symbol, {})

        gates_passed = []
        gates_failed = []
        voi_result = None
        weak_diag = None

        # --- Gate 1: Regime Gate ---
        regime_gate = self.gates.get("regime_gate", {})
        if regime_gate.get("enabled", False):
            allowed = regime_gate.get("allowed_regimes", [])
            passed = regime_id in allowed
            gate = {
                "gate_id": "regime_gate",
                "passed": passed,
                "threshold": allowed,
                "actual": regime_id,
                "description": f"regime {regime_id} {'in' if passed else 'not in'} allowed list {allowed}",
            }
            (gates_passed if passed else gates_failed).append(gate)

        # --- Gate 2: Duration Gate ---
        dur_gate = self.gates.get("duration_gate", {})
        if dur_gate.get("enabled", False):
            min_dur = sym_config.get("min_duration_days",
                                     dur_gate.get("default_min_days", 0))
            passed = duration_days >= min_dur
            gate = {
                "gate_id": "duration_gate",
                "passed": passed,
                "threshold": min_dur,
                "actual": duration_days,
                "description": f"duration {duration_days:.1f}d {'>='}  {min_dur}d" if passed
                    else f"duration {duration_days:.1f}d < {min_dur}d minimum",
            }
            (gates_passed if passed else gates_failed).append(gate)

        # --- Gate 3: Confidence Gate ---
        conf_gate = self.gates.get("confidence_gate", {})
        if conf_gate.get("enabled", False):
            min_conf = sym_config.get("min_confidence",
                                      conf_gate.get("default_min_confidence", 0))
            passed = confidence >= min_conf
            gate = {
                "gate_id": "confidence_gate",
                "passed": passed,
                "threshold": min_conf,
                "actual": confidence,
                "description": f"confidence {confidence:.3f} {'>='}  {min_conf:.3f}" if passed
                    else f"confidence {confidence:.3f} < {min_conf:.3f} minimum",
            }
            (gates_passed if passed else gates_failed).append(gate)

        # --- Gate 4: VOI Gate ---
        voi_gate = self.gates.get("voi_gate", {})
        if voi_gate.get("enabled", False):
            min_voi = voi_gate.get("min_voi", 0.0)
            use_hazard = voi_gate.get("use_hazard_adjustment", False)

            # Get accuracy estimate from symbol config VOI cells
            dur_label = assign_duration_bucket(duration_days, self.duration_buckets)
            voi_cells = sym_config.get("voi_cells", {})
            cell = voi_cells.get(dur_label, {})
            accuracy_est = cell.get("accuracy", 0.5)
            sample_n = cell.get("n", 0)

            # For INVERT symbols, use inverted accuracy for VOI computation
            # An INVERT symbol with 38% accuracy becomes 62% after inversion
            is_invert_symbol = (sym_config.get("weak_symbol_policy") == "INVERT"
                                and sym_config.get("inversion_justified", False))
            if is_invert_symbol:
                accuracy_est = 1.0 - accuracy_est

            hazard_params = None
            if use_hazard and regime_id in self.weibull_params:
                hazard_params = self.weibull_params[regime_id]

            voi_result = compute_voi(accuracy_est, confidence,
                                     hazard_params, duration_days)
            voi_result["sample_size"] = sample_n

            effective_voi = voi_result["voi"]
            if use_hazard and "hazard_adjustment" in voi_result:
                effective_voi = voi_result["hazard_adjustment"]["hazard_adjusted_voi"]

            # Strict inequality: VOI must be strictly positive to pass.
            # Zero-confidence signals produce VOI=0 and should not be emitted.
            passed = effective_voi > min_voi
            gate = {
                "gate_id": "voi_gate",
                "passed": passed,
                "threshold": {"min_voi": min_voi},
                "actual": effective_voi,
                "description": f"VOI {effective_voi:+.4f} > {min_voi}" if passed
                    else f"VOI {effective_voi:+.4f} <= {min_voi} threshold",
            }
            if voi_result.get("zero_confidence"):
                gate["description"] += " (zero-confidence signal)"
            if voi_result.get("hazard_skipped"):
                gate["description"] += f" (hazard skipped: {voi_result.get('hazard_skip_reason', 'incomplete params')})"
            (gates_passed if passed else gates_failed).append(gate)

        # --- Gate 5: Weak Symbol Gate ---
        ws_gate = self.gates.get("weak_symbol_gate", {})
        if ws_gate.get("enabled", False):
            severity_thresh = ws_gate.get("severity_threshold", "MODERATE")
            sym_severity = sym_config.get("weakness_severity", "NONE")
            sym_weakness = sym_config.get("weakness_score", 0.0)
            sym_accuracy = sym_config.get("accuracy", 0.5)

            if SEVERITY_ORDER.get(sym_severity, 0) >= SEVERITY_ORDER.get(severity_thresh, 2):
                # Symbol IS weak — this gate "fails" in the normal sense
                # but may trigger INVERT instead of WITHHOLD
                weak_diag = {
                    "weakness_score": sym_weakness,
                    "severity": sym_severity,
                    "accuracy": sym_accuracy,
                    "ci_lower": sym_config.get("accuracy", 0.5) - 0.05,  # approximate
                    "ci_upper": sym_config.get("accuracy", 0.5) + 0.05,
                    "ci_excludes_50": sym_accuracy < 0.45,
                    "inversion_justified": sym_config.get("inversion_justified", False),
                    "inversion_p_value": sym_config.get("inversion_p_value", 1.0),
                }

                gate = {
                    "gate_id": "weak_symbol_gate",
                    "passed": False,
                    "threshold": severity_thresh,
                    "actual": sym_severity,
                    "description": f"{symbol} weakness {sym_severity} >= {severity_thresh} threshold",
                }
                gates_failed.append(gate)
            else:
                gate = {
                    "gate_id": "weak_symbol_gate",
                    "passed": True,
                    "threshold": severity_thresh,
                    "actual": sym_severity,
                    "description": f"{symbol} weakness {sym_severity} below {severity_thresh} threshold",
                }
                gates_passed.append(gate)

        # --- Determine final action ---
        action, routed_direction, rationale = self._decide_action(
            symbol, direction, confidence, gates_passed, gates_failed,
            sym_config, voi_result, weak_diag
        )

        decision = {
            "signal_id": sig_id,
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "direction": direction,
            "gates_passed": gates_passed,
            "gates_failed": gates_failed,
            "rationale": rationale,
        }

        if routed_direction:
            decision["routed_direction"] = routed_direction
        if voi_result:
            decision["voi"] = voi_result
        if weak_diag:
            decision["weak_symbol"] = weak_diag

        return decision

    def _decide_action(self, symbol: str, direction: str, confidence: float,
                       gates_passed: List, gates_failed: List,
                       sym_config: Dict, voi_result: Optional[Dict],
                       weak_diag: Optional[Dict]) -> Tuple[str, Optional[str], str]:
        """
        Determine final routing action from gate results.

        Logic:
        1. If weak_symbol_gate failed AND inversion justified → INVERT
        2. If weak_symbol_gate failed AND policy=EXCLUDE → WITHHOLD
        3. If any other gate failed → WITHHOLD
        4. Otherwise → EMIT
        """
        policy = sym_config.get("weak_symbol_policy", "NONE")
        failed_ids = {g["gate_id"] for g in gates_failed}
        non_weak_failures = failed_ids - {"weak_symbol_gate"}

        # Check for non-weak gate failures first
        if non_weak_failures:
            first_fail = next(g for g in gates_failed if g["gate_id"] in non_weak_failures)
            return "WITHHOLD", None, f"WITHHOLD: {first_fail['description']}"

        # Check weak symbol gate
        if "weak_symbol_gate" in failed_ids:
            if policy == "INVERT" and sym_config.get("inversion_justified", False):
                flipped = "bearish" if direction == "bullish" else "bullish"
                p_val = sym_config.get("inversion_p_value", 1.0)
                ws = sym_config.get("weakness_score", 0)
                return ("INVERT", flipped,
                        f"INVERT: {symbol} weakness {sym_config.get('weakness_severity', '?')} "
                        f"({ws:.2f}), inversion justified p={p_val:.4f}")

            elif policy == "EXCLUDE":
                return ("WITHHOLD", None,
                        f"WITHHOLD: {symbol} excluded by weak-symbol policy "
                        f"(severity {sym_config.get('weakness_severity', '?')})")

            elif policy == "REDUCE_WEIGHT":
                wf = sym_config.get("weight_factor", 0.5)
                new_conf = 0.5 + wf * (confidence - 0.5)
                return ("EMIT", direction,
                        f"EMIT: {symbol} confidence reduced {confidence:.3f} -> "
                        f"{new_conf:.3f} (weight_factor={wf:.1f})")

            else:
                # Weak but no policy configured — withhold by default
                return ("WITHHOLD", None,
                        f"WITHHOLD: {symbol} weakness detected "
                        f"({sym_config.get('weakness_severity', '?')}) "
                        f"but no intervention policy configured")

        # All gates passed
        n_gates = len(gates_passed)
        voi_str = ""
        if voi_result:
            voi_str = f", VOI {voi_result['voi']:+.4f}"
        return "EMIT", direction, f"EMIT: all {n_gates} gates passed{voi_str}"

    def _detect_limitations(self, signals: List, decisions: List,
                            regime_id: str, duration_days: float) -> List[Dict]:
        """Detect and report structural limitations of the routing."""
        limitations = []

        # Check for UNKNOWN regime (may indicate null/missing regime_context)
        if regime_id == "UNKNOWN":
            limitations.append({
                "id": "UNKNOWN_REGIME",
                "description": (
                    "Routing used regime_id=UNKNOWN. This typically means "
                    "regime_context was null or missing from the input signals. "
                    "All regime-dependent gates (regime_gate, VOI hazard adjustment) "
                    "may produce incorrect results."
                ),
                "bias_direction": "INDETERMINATE",
                "bias_magnitude": "High — routing decisions not calibrated for actual regime."
            })

        # Check if all signals were filtered
        n_emit = sum(1 for d in decisions if d["action"] == "EMIT")
        n_invert = sum(1 for d in decisions if d["action"] == "INVERT")
        if n_emit == 0 and n_invert == 0 and len(signals) > 0:
            limitations.append({
                "id": "ALL_SIGNALS_FILTERED",
                "description": f"All {len(signals)} signals were withheld. "
                    f"No signals will be emitted in regime {regime_id} at duration {duration_days:.1f}d.",
                "bias_direction": "UNDERSTATED",
                "bias_magnitude": "Producer appears inactive but is actually suppressing for quality."
            })

        # Check for single-regime config
        evidence = self.config.get("evidence_source", "")
        if "SYSTEMIC" in evidence or regime_id == "SYSTEMIC":
            limitations.append({
                "id": "SINGLE_REGIME_EVIDENCE",
                "description": "Routing thresholds derived from SYSTEMIC-regime evidence. "
                    "Thresholds may not be optimal for other regimes.",
                "bias_direction": "INDETERMINATE",
                "bias_magnitude": "High for non-SYSTEMIC regimes."
            })

        # Check for small VOI sample sizes
        small_n_symbols = []
        for sym, sym_cfg in self.symbols.items():
            for label, cell in sym_cfg.get("voi_cells", {}).items():
                if cell.get("n", 0) < 30:
                    small_n_symbols.append(f"{sym}/{label}")
        if small_n_symbols:
            limitations.append({
                "id": "SMALL_SAMPLE_VOI",
                "description": f"VOI cells with n < 30: {', '.join(small_n_symbols[:5])}. "
                    f"Accuracy estimates in these cells are unreliable.",
                "bias_direction": "INDETERMINATE",
                "bias_magnitude": "Moderate for affected cells."
            })

        # Check for incomplete Weibull params (hazard adjustment may be silently skipped)
        voi_gate = self.gates.get("voi_gate", {})
        if voi_gate.get("use_hazard_adjustment", False):
            missing_regimes = []
            incomplete_regimes = []
            for rid in ["SYSTEMIC", "NEUTRAL", "DIVERGENCE", "EARNINGS"]:
                wp = self.weibull_params.get(rid)
                if wp is None:
                    missing_regimes.append(rid)
                elif "shape" not in wp or "scale" not in wp:
                    incomplete_regimes.append(rid)
            if missing_regimes or incomplete_regimes:
                parts = []
                if missing_regimes:
                    parts.append(f"missing for: {', '.join(missing_regimes)}")
                if incomplete_regimes:
                    parts.append(f"incomplete for: {', '.join(incomplete_regimes)}")
                limitations.append({
                    "id": "INCOMPLETE_HAZARD_PARAMS",
                    "description": (
                        f"Weibull hazard params {'; '.join(parts)}. "
                        f"Hazard adjustment will be skipped for these regimes, "
                        f"producing non-hazard-adjusted VOI."
                    ),
                    "bias_direction": "OVERSTATED",
                    "bias_magnitude": "Moderate — VOI not adjusted for regime duration risk."
                })

        return limitations


# ---------------------------------------------------------------------------
# Schema Validation
# ---------------------------------------------------------------------------

def validate_report(report: Dict) -> Tuple[bool, List[str]]:
    """Validate a routing report against the routing protocol schema."""
    if not HAS_JSONSCHEMA:
        return True, ["jsonschema not installed, skipping validation"]

    try:
        with open(ROUTING_PROTOCOL_SCHEMA_PATH) as f:
            schema = json.load(f)
    except FileNotFoundError:
        return False, [f"Schema not found: {ROUTING_PROTOCOL_SCHEMA_PATH}"]

    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(report))

    if not errors:
        return True, []

    messages = []
    for err in errors[:10]:
        path = ".".join(str(p) for p in err.absolute_path)
        messages.append(f"{path}: {err.message}")

    return False, messages


def validate_config(config: Dict) -> Tuple[bool, List[str]]:
    """Validate a routing policy config against the routing policy schema."""
    if not HAS_JSONSCHEMA:
        return True, ["jsonschema not installed, skipping validation"]

    try:
        with open(ROUTING_POLICY_SCHEMA_PATH) as f:
            schema = json.load(f)
    except FileNotFoundError:
        return False, [f"Schema not found: {ROUTING_POLICY_SCHEMA_PATH}"]

    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(config))

    if not errors:
        return True, []

    messages = []
    for err in errors[:10]:
        path = ".".join(str(p) for p in err.absolute_path)
        messages.append(f"{path}: {err.message}")

    return False, messages


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_text_report(report: Dict):
    """Print human-readable routing report to stdout."""
    meta = report["meta"]
    summary = report["summary"]

    print("=" * 70)
    print("Post Fiat Signal Pre-Flight Routing Report")
    print("=" * 70)
    print(f"Producer:  {meta['producer_id']}")
    print(f"Generated: {meta['generated_at']}")
    print(f"Protocol:  v{meta['protocol_version']}")
    regime = meta.get("regime_context", {})
    print(f"Regime:    {regime.get('regime_id', '?')} "
          f"(day {regime.get('duration_days', '?')}, "
          f"conf {regime.get('regime_confidence', '?')})")
    print()

    print("--- Summary ---")
    print(f"Input:    {summary['total_input']} signals")
    print(f"EMIT:     {summary['total_emit']}")
    print(f"WITHHOLD: {summary['total_withhold']}")
    print(f"INVERT:   {summary['total_invert']}")
    print(f"Filter:   {summary['filter_rate']:.1%}")
    print()

    # Per-gate pass rates
    gate_rates = summary.get("per_gate_pass_rates", {})
    if gate_rates:
        print("--- Gate Pass Rates ---")
        for gid, rate in sorted(gate_rates.items()):
            print(f"  {gid:<20} {rate:.1%}")
        print()

    # Per-symbol
    per_sym = summary.get("per_symbol_summary", {})
    if per_sym:
        print("--- Per Symbol ---")
        print(f"  {'Symbol':<8} {'Total':>6} {'EMIT':>6} {'WHLD':>6} {'INV':>6}")
        for sym, counts in sorted(per_sym.items()):
            print(f"  {sym:<8} {counts['total']:>6} {counts['emit']:>6} "
                  f"{counts['withhold']:>6} {counts['invert']:>6}")
        print()

    # Decisions
    print("--- Per-Signal Decisions ---")
    for d in report["decisions"]:
        action = d["action"]
        marker = {"EMIT": "+", "WITHHOLD": "-", "INVERT": "~"}[action]
        print(f"  [{marker}] {d['signal_id']:<30} {action:<10} {d['rationale']}")
    print()

    # Limitations
    lims = report.get("limitations", [])
    if lims:
        print("--- Limitations ---")
        for lim in lims:
            print(f"  [{lim['id']}] {lim['description']}")
            print(f"    Bias: {lim.get('bias_direction', '?')} "
                  f"({lim.get('bias_magnitude', '?')})")
        print()

    print("=" * 70)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Post Fiat Signal Pre-Flight Router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python preflight_filter.py signals.json config.json
  python preflight_filter.py signals.json config.json --regime SYSTEMIC --duration 16.5
  python preflight_filter.py signals.json config.json --json -o report.json
  echo '[...]' | python preflight_filter.py - config.json --json
""")

    parser.add_argument("signals", help="Signals JSON file (array of signals, or '-' for stdin)")
    parser.add_argument("config", help="Routing policy configuration JSON file")
    parser.add_argument("--regime", default=None,
                        help="Override regime_id (default: from signal regime_context or UNKNOWN)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override regime duration in days")
    parser.add_argument("--regime-confidence", type=float, default=0.5,
                        help="Regime classification confidence (0-1)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("-o", "--output", help="Write output to file")
    parser.add_argument("--validate", action="store_true",
                        help="Validate output against routing protocol schema")

    args = parser.parse_args()

    # Load signals
    if args.signals == "-":
        signals_data = json.load(sys.stdin)
    else:
        with open(args.signals) as f:
            signals_data = json.load(f)

    # Handle single signal vs array
    if isinstance(signals_data, dict):
        if "signals" in signals_data:
            signals = signals_data["signals"]
        else:
            signals = [signals_data]
    elif isinstance(signals_data, list):
        signals = signals_data
    else:
        print("ERROR: signals must be a JSON array or object with 'signals' key", file=sys.stderr)
        sys.exit(1)

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Determine regime context
    regime_id = args.regime or "UNKNOWN"
    duration_days = args.duration if args.duration is not None else 0

    # If not explicitly set, try to infer from first signal's regime_context
    if args.regime is None and signals:
        rc = signals[0].get("regime_context", {})
        if rc.get("regime_id"):
            regime_id = rc["regime_id"]
        if args.duration is None and rc.get("duration_days") is not None:
            duration_days = rc["duration_days"]

    # Route
    router = PreFlightRouter(config)
    report = router.route_signals(
        signals,
        regime_id=regime_id,
        duration_days=duration_days,
        regime_confidence=args.regime_confidence,
    )

    # Validate
    if args.validate:
        valid, errors = validate_report(report)
        if valid:
            print("PASS: Report validates against routing_protocol.json", file=sys.stderr)
        else:
            print("FAIL: Validation errors:", file=sys.stderr)
            for e in errors:
                print(f"  {e}", file=sys.stderr)

    # Output
    if args.json or args.output:
        output_json = json.dumps(report, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_json)
            if not args.json:
                print(f"Report saved to {args.output}")
        if args.json:
            print(output_json)
    else:
        _print_text_report(report)


if __name__ == "__main__":
    main()
