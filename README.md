# Post Fiat Signal Routing Protocol

Pre-flight routing contract for the Post Fiat network. Defines how signals conforming to [pf-signal-schema](https://github.com/sendoeth/pf-signal-schema) are filtered, gated, and routed before emission — using decision-theoretic Value of Information (VOI), regime awareness, duration gating, and weak-symbol intervention.

## Why this exists

The signal schema defines **what** to emit. The resolution protocol defines **how** to score. This protocol defines **whether** to emit at all — and if so, whether to modify the signal first.

```
pf-signal-schema          pf-routing-protocol        pf-resolution-protocol
────────────────          ───────────────────         ──────────────────────
Defines signals    →      Routes signals       →      Resolves signals
"what to emit"           "whether to emit"           "how to score"
JSON Schema              JSON Schema + CLI           JSON Schema + CLI
v1.0.0                   v1.0.0                      v1.0.0
```

Without a routing protocol, producers emit every signal and hope for the best. With it, a producer can systematically suppress negative-VOI signals (improving karma), invert weak-symbol signals (turning anti-edge into edge), and gate on regime duration (where accuracy structurally improves from 38.6% to 58.9%).

## Contents

| File | Description |
|------|-------------|
| `routing_protocol.json` | JSON Schema (draft 2020-12) defining the routing report output format |
| `routing_policy.json` | JSON Schema defining the routing configuration (policy input) |
| `preflight_filter.py` | Standalone CLI + library — routes signals through 5 gates, outputs decision report |
| `routing_config_example.json` | Worked example config derived from 2,128 resolved signals |
| `example_routing/` | Candidate signals + generated routing report |
| `tests/` | 100 tests covering all components |

## Quick start

```bash
# Route candidate signals through the filter
python preflight_filter.py example_routing/candidate_signals.json \
  --config routing_config_example.json \
  --regime SYSTEMIC --duration 16

# JSON output
python preflight_filter.py example_routing/candidate_signals.json \
  --config routing_config_example.json \
  --regime SYSTEMIC --duration 16 --json

# Save to file and validate against schema
python preflight_filter.py example_routing/candidate_signals.json \
  --config routing_config_example.json \
  --regime SYSTEMIC --duration 16 \
  -o report.json --validate

# Validate an existing report
python preflight_filter.py example_routing/candidate_signals.json \
  --config routing_config_example.json --validate
```

### Install dependency

```bash
pip install jsonschema
```

No other external dependencies required.

---

## The 5-Gate Model

Every candidate signal passes through 5 sequential gates. If any non-weak gate fails, the signal is WITHHOLD. The weak-symbol gate has special semantics (can trigger INVERT or EXCLUDE instead of simple WITHHOLD).

```
candidate signal
      │
      ▼
┌─────────────┐
│ regime_gate  │──fail──→ WITHHOLD (regime not actionable)
└──────┬──────┘
       │ pass
       ▼
┌─────────────┐
│ duration_gate│──fail──→ WITHHOLD (regime too young)
└──────┬──────┘
       │ pass
       ▼
┌─────────────┐
│confidence_gate│──fail──→ WITHHOLD (confidence too low)
└──────┬──────┘
       │ pass
       ▼
┌─────────────┐
│   voi_gate   │──fail──→ WITHHOLD (negative expected karma)
└──────┬──────┘
       │ pass
       ▼
┌────────────────┐
│weak_symbol_gate│──INVERT──→ INVERT (flip direction, emit)
│                │──EXCLUDE──→ WITHHOLD (too weak, no fix)
│                │──REDUCE──→ EMIT (reduced confidence)
└──────┬─────────┘
       │ pass
       ▼
     EMIT
```

### Gate details

| Gate | What it checks | Key parameter | Evidence |
|------|---------------|---------------|----------|
| `regime_gate` | Is the current regime actionable? | `allowed_regimes` | EARNINGS excluded (short duration, event-driven) |
| `duration_gate` | Has the regime been active long enough? | `min_duration_days` | 38.6% accuracy at 10-15d vs 58.9% at 15d+ |
| `confidence_gate` | Is the signal confident enough to matter? | `min_confidence` | Below 0.30, even correct signals generate negligible karma |
| `voi_gate` | Is E[karma\|send] > E[karma\|withhold]? | `min_voi` | Decision-theoretic: `VOI = confidence * (2*accuracy - 1)` |
| `weak_symbol_gate` | Is this a historically weak symbol? | `severity_threshold` | SOL accuracy 44%, CI excludes 50%, INVERT justified (p=0.0001) |

### Action vocabulary

| Action | Meaning | Consumer behavior |
|--------|---------|-------------------|
| `EMIT` | Signal passes all gates | Act on direction + confidence as stated |
| `WITHHOLD` | Signal fails a gate | Do not trade. Record only. |
| `INVERT` | Weak symbol with justified inversion | Act **opposite** to stated direction |

These map directly to the `action` field in [pf-signal-schema](https://github.com/sendoeth/pf-signal-schema): `EMIT` → `EXECUTE`, `WITHHOLD` → `WITHHOLD`, `INVERT` → `INVERT`.

---

## VOI computation

The core decision gate uses Value of Information:

```
E[karma|send]    = confidence * (2 * accuracy - 1)
E[karma|withhold] = 0  (no signal, no karma)
VOI              = E[karma|send] - E[karma|withhold]
                 = confidence * (2 * accuracy - 1)
```

Where `accuracy` comes from the VOI cell matching the signal's symbol + duration bucket.

**Key insight for INVERT symbols**: SOL has 44% native accuracy, which gives negative VOI. But after inversion, accuracy becomes 56% (= 1 - 0.44), giving positive VOI. The router detects INVERT-justified symbols and computes VOI with inverted accuracy. Without this, the VOI gate would incorrectly WITHHOLD signals that should be INVERT.

### Weibull hazard adjustment (optional)

When `use_hazard_adjustment` is enabled and `weibull_params` are provided:

```
survival    = exp(-(duration / scale)^shape)
hazard_adj  = VOI * survival
```

This discounts VOI for signals emitted late in a regime's expected lifetime. Disabled by default because the duration gate handles this more directly.

---

## Duration buckets

Accuracy is not uniform across regime duration. The protocol defines 4 canonical buckets:

| Bucket | Days | Typical accuracy | Evidence |
|--------|------|-----------------|----------|
| `early` | 0-12 | ~48% | Insufficient regime stability |
| `mid` | 12-15 | ~50% | Marginal — near coin-flip |
| `mature` | 15-18 | ~54-57% | Structural accuracy improvement |
| `late` | 18+ | ~56-62% | Best accuracy, but regime end risk |

Duration gating at 15 days is the single largest accuracy lever in the forward-test evidence.

---

## Weak symbol intervention

The protocol supports 4 intervention policies for symbols that fail the weakness threshold:

| Policy | Effect | When to use |
|--------|--------|-------------|
| `NONE` | No intervention | Symbol accuracy is adequate |
| `EXCLUDE` | WITHHOLD all signals | Symbol too weak, inversion not justified |
| `REDUCE_WEIGHT` | Emit at reduced confidence | Marginal weakness, want some exposure |
| `INVERT` | Flip direction, emit | Accuracy significantly below 50%, CI excludes 50%, p < 0.01 |

Weakness is measured by a 5-component composite score:

| Component | What it measures |
|-----------|-----------------|
| `accuracy_deficit` | How far accuracy is below baseline |
| `reliability_gap` | Calibration degradation vs pool |
| `karma_drag` | Negative karma contribution |
| `brier_excess` | Brier score above pool average |
| `directional_bias` | Systematic directional error |

See [weak_symbol_evaluator.py](https://github.com/sendoeth/pf-routing-protocol) for the full evaluation framework (extracted from the `post-fiat-signals` producer's operational module).

---

## Routing configuration

The routing config is a JSON file defining all gate parameters + per-symbol overrides. See `routing_config_example.json` for a complete worked example derived from 2,128 resolved signals.

### Minimal config

```json
{
  "policy_version": "1.0.0",
  "producer_id": "my-producer",
  "gates": {
    "regime_gate": { "enabled": true, "allowed_regimes": ["NEUTRAL", "SYSTEMIC"] },
    "duration_gate": { "enabled": true, "default_min_days": 15 },
    "confidence_gate": { "enabled": true, "default_min_confidence": 0.30 },
    "voi_gate": { "enabled": true, "min_voi": 0.0 },
    "weak_symbol_gate": { "enabled": true, "severity_threshold": "MODERATE" }
  },
  "symbols": {
    "BTC": {
      "accuracy": 0.52,
      "weak_symbol_policy": "NONE",
      "weakness_severity": "NONE"
    }
  },
  "duration_buckets": [
    {"label": "early",  "min_days": 0,  "max_days": 12},
    {"label": "mid",    "min_days": 12, "max_days": 15},
    {"label": "mature", "min_days": 15, "max_days": 18},
    {"label": "late",   "min_days": 18, "max_days": 9999}
  ]
}
```

### Per-symbol overrides

Each symbol can override global gate thresholds:

```json
{
  "SOL": {
    "min_duration_days": 0,
    "min_confidence": 0.05,
    "weak_symbol_policy": "INVERT",
    "inversion_justified": true,
    "inversion_p_value": 0.0001,
    "accuracy": 0.440,
    "weakness_score": 0.6979,
    "weakness_severity": "SEVERE",
    "voi_cells": {
      "early":  {"voi": -0.010, "accuracy": 0.42, "n": 196},
      "mature": {"voi": -0.022, "accuracy": 0.38, "n": 120}
    }
  }
}
```

---

## Library usage

```python
from preflight_filter import PreFlightRouter
import json

# Load config and signals
with open("routing_config_example.json") as f:
    config = json.load(f)
with open("candidate_signals.json") as f:
    signals = json.load(f)

# Create router
router = PreFlightRouter(config)

# Route signals
report = router.route_signals(signals, regime="SYSTEMIC", duration_days=16)

# Inspect decisions
for d in report["decisions"]:
    print(f"{d['signal_id']}: {d['action']} ({d['gates_passed']}/{d['gates_total']} gates)")

# Validate output against schema
from preflight_filter import validate_report
errors = validate_report(report)
assert len(errors) == 0, f"Validation failed: {errors}"
```

### Integration with pf-signal-schema

```python
# Route first, then stamp the action into the signal before emission
for decision in report["decisions"]:
    signal = find_signal(decision["signal_id"])

    if decision["action"] == "EMIT":
        signal["action"] = "EXECUTE"
    elif decision["action"] == "INVERT":
        signal["action"] = "INVERT"
        # Flip direction for consumer clarity
        signal["direction"] = "bearish" if signal["direction"] == "bullish" else "bullish"
        signal["weak_symbol"] = decision.get("weak_symbol_diagnosis")
    elif decision["action"] == "WITHHOLD":
        signal["action"] = "WITHHOLD"

    emit_to_oracle(signal)
```

### Integration with pf-resolution-protocol

```python
# After signals are emitted and resolved, use resolution protocol to score
# The routing report's VOI predictions can be compared against realized karma
# to validate the router's accuracy over time

from resolve_signals import SignalResolver

resolver = SignalResolver()
resolution = resolver.resolve(emitted_signals, price_data)

# Compare predicted VOI vs realized karma
for decision in report["decisions"]:
    if decision["action"] in ("EMIT", "INVERT"):
        predicted_voi = decision["voi_computation"]["voi"]
        realized = resolution["per_signal"][decision["signal_id"]]
        print(f"{decision['signal_id']}: predicted VOI={predicted_voi:.4f}, realized karma={realized['karma']:.4f}")
```

---

## Report structure

The routing report (defined in `routing_protocol.json`) contains:

| Section | Contents |
|---------|----------|
| `meta` | Timestamp, producer_id, regime, duration, config snapshot hash |
| `summary` | Total signals, emit/withhold/invert counts, filter rate, per-symbol breakdown |
| `decisions` | Per-signal decision with gate results, VOI computation, weak symbol diagnosis |
| `limitations` | Auto-detected caveats (all filtered, single-regime evidence, small sample VOI) |
| `protocol_version` | Semver of this protocol |

### Limitations (auto-detected)

| Code | Trigger | Meaning |
|------|---------|---------|
| `ALL_SIGNALS_FILTERED` | 100% withhold rate | Something is misconfigured or regime is fully suppressive |
| `SINGLE_REGIME_EVIDENCE` | Only 1 regime in evidence | VOI cells may not generalize to other regimes |
| `SMALL_SAMPLE_VOI` | Any VOI cell has n < 30 | Accuracy estimates have wide CIs |

---

## Running tests

```bash
pip install pytest jsonschema
python -m pytest tests/ -v
```

100 tests across 19 test classes covering:
- VOI computation (formula, symmetry, edge cases, hazard adjustment)
- Weibull hazard and survival functions
- Wilson confidence intervals
- Duration bucket assignment (boundaries, edge cases)
- All 5 gates independently (regime, duration, confidence, VOI, weak symbol)
- Gate interaction (multi-gate, first-fail semantics)
- INVERT handling (direction flip, VOI with inverted accuracy, unjustified fallback)
- Summary statistics (filter rate, per-symbol counts, invert rate)
- Report structure (meta fields, decision ordering, config snapshot)
- Limitations detection (all filtered, single regime, small sample)
- Schema validation (report + config against JSON Schemas)
- Cross-schema compatibility (action vocabulary, regime IDs, confidence scale)
- Malformed input handling (missing fields, extra fields)
- File I/O (config loading, report output)
- End-to-end integration (example data through full pipeline)

---

## Design decisions

1. **5 gates, AND logic** — a signal must pass all gates to EMIT. This is conservative by design: false positives (emitting bad signals) cost karma; false negatives (withholding good signals) cost nothing.

2. **VOI is decision-theoretic, not heuristic** — the formula `E[karma|send] = confidence * (2*accuracy - 1)` derives directly from the karma formula in pf-resolution-protocol. A signal is worth emitting if and only if its expected karma contribution is positive.

3. **INVERT symbols use inverted accuracy for VOI** — a symbol with 44% accuracy has -VOI naively, but after inversion it has 56% accuracy and +VOI. The router must detect this to avoid incorrectly withholding signals that should be inverted. This interaction between the VOI gate and the weak-symbol gate is the most subtle design decision in the protocol.

4. **Duration gating is structural, not overfit** — the 38.6% → 58.9% accuracy jump at 15 days reflects regime stability physics: early in a regime, the classification is uncertain and signals are noisy. This is robust across the full forward-test dataset.

5. **Per-symbol overrides > global thresholds** — SOL has min_duration_days=0 because inversion changes its accuracy profile. Forcing a global 15-day gate on SOL would suppress its inverted signals unnecessarily.

6. **Pure Python, no ML** — the router is a transparent decision tree. Every gate has an explicit threshold, every decision is traceable. No hidden weights, no training, no opaque scoring.

7. **Self-validating** — output can be validated against `routing_protocol.json` via `--validate`. The schema enforces required fields, enum values, and structural consistency.

8. **Companion, not standalone** — this protocol sits between signal emission and resolution. It requires signals conforming to pf-signal-schema and produces decisions that can be validated by pf-resolution-protocol.

---

## Versioning

- **MAJOR** (e.g. 1.0.0 → 2.0.0): new gate added or gate semantics changed. Existing configs may need updates.
- **MINOR** (e.g. 1.0.0 → 1.1.0): new optional fields in config or report. Backwards compatible.
- **PATCH** (e.g. 1.0.0 → 1.0.1): bug fixes, description updates. No structural changes.

---

## License

MIT
