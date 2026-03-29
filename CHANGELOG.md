# Changelog

## 1.0.0 (2026-03-29)

Initial release.

- 5-gate pre-flight routing: regime, duration, confidence, VOI, weak symbol
- Decision-theoretic VOI computation: `E[karma|send] = confidence * (2*accuracy - 1)`
- INVERT support with inverted-accuracy VOI (SOL: 44% native → 56% inverted)
- Weibull hazard adjustment (optional)
- Wilson score confidence intervals
- Duration bucketing (early/mid/mature/late)
- Weak symbol diagnosis with 4 intervention policies (NONE/EXCLUDE/REDUCE_WEIGHT/INVERT)
- Self-validating output (`--validate` flag)
- Cross-schema compatibility with pf-signal-schema and pf-resolution-protocol
- 100 tests across 19 test classes
- Worked example from 2,128 resolved signals (SYSTEMIC regime, March 19-26 2026)
