"""VIGIL Lab - benchmark-grade research harness for Kessler's risk models.

Isolated from serving: nothing here is imported by the API. It reproduces
the ESA Kelvins Collision Avoidance Challenge protocol exactly (2-day input
cutoff, official test-event filters, the official score L = MSE_HR / F2 and
its latest-risk baseline), races a pool of candidate models under repeated,
event-grouped cross-validation with paired bootstrap confidence intervals,
evaluates the sealed official test set once, and decides shipping with a
pre-registered, power-aware gate.

Modules:
    data       - Kelvins loading, the official protocol, per-event inputs
    metric     - the official score and baselines, VIGIL-E metrics
    physics    - encounter-plane geometry, dilution, covariance projection
    features   - per-event feature table (benchmark and serving tracks)
    candidates - the model pool (regression + early-warning)
    protocol   - repeated grouped CV, bootstrap, sealed-test evaluation
    gate       - pre-registered ship rules
    report     - JSON + Markdown report

CLI: python -m backend.app.models.lab --help
"""
