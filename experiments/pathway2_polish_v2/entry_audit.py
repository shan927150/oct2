"""Replay the legacy CE under its precision policy, then verify the polish policy.

These checks run before any optimizer exists. The legacy reference tolerance is
unchanged. TF32-off optimization is never compared directly to a TF32-on CE gate.
"""
from contextlib import contextmanager
import copy

import numpy as np
import torch


@contextmanager
def precision(cudnn_tf32):
    old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old


def verify_endpoint(model, snap, reference_ce, expected_vectors, write_report,
                    *, check_identity=False):
    report = dict(status='started', reference_ce=float(reference_ce),
        ce_tolerance=1e-8 + 5e-5 * abs(reference_ce),
        legacy_policy=dict(matmul_tf32=False, cudnn_tf32=True),
        polish_policy=dict(matmul_tf32=False, cudnn_tf32=False),
        torch=str(torch.__version__), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU test')
    try:
        with precision(True):
            legacy, lv = snap(model)
        with precision(False):
            strict, sv = snap(model, check_identity=check_identity)
            repeated, rv = snap(model)
            m64 = copy.deepcopy(model).double()
            double, dv = snap(m64)
            del m64
        if expected_vectors.shape != lv.shape:
            raise ValueError('Audited vector dimensions differ')
        legacy_error = np.linalg.norm(lv - expected_vectors, axis=1)
        legacy_limits = 1e-8 + 5e-5 * np.linalg.norm(expected_vectors, axis=1)
        strict_error = np.linalg.norm(sv - dv, axis=1)
        double_norms = np.linalg.norm(dv, axis=1)
        legacy_ce_error = abs(legacy['ce'] - reference_ce)
        strict_ce_error = abs(strict['ce'] - reference_ce)
        finite = all(np.isfinite(v).all() for v in (lv, sv, rv, dv, expected_vectors))
        finite = finite and all(np.isfinite(x[k]) for x in (legacy, strict, repeated, double)
                                for k in ('ce', 'objective'))
        checks = dict(finite=bool(finite),
            legacy_ce_matches=bool(legacy_ce_error <= report['ce_tolerance']),
            legacy_vectors_match=bool(np.all(legacy_error <= legacy_limits)),
            strict_repeat_exact=bool(strict['ce'] == repeated['ce'] and np.array_equal(sv, rv)),
            strict_ce_agrees_fp64=bool(abs(strict['ce'] - double['ce']) <=
                                     1e-8 + 5e-5 * abs(double['ce'])),
            strict_vectors_agree_fp64=bool(np.all(strict_error <= .1 * double_norms)))
        report.update(legacy_ce=legacy['ce'], strict_ce=strict['ce'], fp64_ce=double['ce'],
            legacy_ce_error=legacy_ce_error, strict_vs_old_ce_error=strict_ce_error,
            strict_would_pass_old_gate=bool(strict_ce_error <= report['ce_tolerance']),
            legacy_vector_error=legacy_error.tolist(), legacy_vector_tolerance=legacy_limits.tolist(),
            strict_fp64_vector_error=strict_error.tolist(), fp64_vector_norm=double_norms.tolist(),
            checks=checks, status='passed' if all(checks.values()) else 'failed')
        report['precision_mismatch_explains_old_failure'] = bool(
            all(checks.values()) and strict_ce_error > report['ce_tolerance'])
        if not all(checks.values()):
            raise ValueError('Endpoint audit failed: ' + ', '.join(k for k, v in checks.items() if not v))
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_report(report)
        raise
    write_report(report)
    return strict, sv, double, dv, report
