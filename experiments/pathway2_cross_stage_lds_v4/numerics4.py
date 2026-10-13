"""Numerical diagnostics and replay checks for the v4 runner.

Finite differences of the non-smooth CNN are diagnostics, not acceptance tests.
Replay comparisons concern the same computation and retain their original tolerances.
"""
import warnings
import numpy as np


def finite_difference_diagnostic(name, rows, relative_key, reference_tol):
    if not rows or not all(np.isfinite(float(v)) for row in rows for v in row.values()):
        raise FloatingPointError(f'{name}: non-finite finite-difference diagnostic')
    best = min(float(row[relative_key]) for row in rows)
    above = best > reference_tol
    result = dict(policy='diagnostic_only', best_relative_error=best,
                  reference_relative_tolerance=reference_tol, above_reference=above)
    if above:
        warnings.warn(f'{name}: best relative error {best:.6g} exceeds the diagnostic reference '
                      f'{reference_tol:.6g}; recorded for review, continuing.', RuntimeWarning)
    return result


def require_replay_close(name, actual, reference, atol):
    """Preserve absolute replay tolerances and reject NaN/Inf explicitly."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        raise ValueError(f'{name}: replay shapes differ')
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise FloatingPointError(f'{name}: non-finite replay value')
    error = float(np.max(np.abs(actual - reference)))
    if error > atol:
        raise ValueError(f'{name}: max absolute difference {error:.6g} exceeds {atol:.6g}')
    return error
