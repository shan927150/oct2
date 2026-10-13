"""Regression cases for the two observed Delta failures and retained replay checks."""
import unittest
import numpy as np
from numerics4 import finite_difference_diagnostic, require_replay_close


class NumericalChecksTests(unittest.TestCase):
    def test_observed_h_finite_difference_discrepancy_is_reported(self):
        analytic = -0.00012930326380655103
        fd = [-0.000128092592185311, -0.00012877639974040456,
              -0.00012949999299016746, -0.0001294519160477403]
        rows = [dict(fd=x, analytic=analytic, relative_error=abs(x - analytic) / abs(analytic)) for x in fd]
        with self.assertWarns(RuntimeWarning):
            result = finite_difference_diagnostic('h', rows, 'relative_error', 1e-5)
        self.assertTrue(result['above_reference'])
        self.assertEqual(result['policy'], 'diagnostic_only')

    def test_observed_jvp_discrepancy_is_reported(self):
        scale = 12.48154591158384
        rows = [dict(max_abs_error=e, scale=scale, max_relative_error=e / scale)
                for e in [0.0282969129314381, 0.013858125913906007, 0.013114396291573738]]
        with self.assertWarns(RuntimeWarning):
            result = finite_difference_diagnostic('jvp', rows, 'max_relative_error', 1e-5)
        self.assertAlmostEqual(result['best_relative_error'], 0.013114396291573738 / scale)

    def test_nonfinite_diagnostics_are_rejected(self):
        for bad in [float('nan'), float('inf')]:
            with self.subTest(value=bad), self.assertRaises(FloatingPointError):
                finite_difference_diagnostic('fd', [dict(value=bad, relative_error=0.)], 'relative_error', 1e-5)

    def test_replay_tolerance_and_nonfinite_values(self):
        self.assertLess(require_replay_close('J00', .5 + 1e-7, .5, 1e-6), 1e-6)
        with self.assertRaises(ValueError):
            require_replay_close('J00', .501, .5, 1e-6)
        with self.assertRaises(FloatingPointError):
            require_replay_close('J00', np.nan, .5, 1e-6)

    def test_unqualified_l3_shift_does_not_abort_collection(self):
        import analysis4
        data = dict(scores={42: dict(arrays={}, result=dict(gammas=[
            dict(gamma=.03, qualified=False, reasons=['true relative residual above tolerance'])]))})
        predictions, status = analysis4.l3_predictions(data, dict(score_seeds=[42]), .03)
        self.assertIsNone(predictions)
        self.assertIn('unqualified', status)


if __name__ == '__main__':
    unittest.main()
