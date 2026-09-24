#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import route_a_smallpanel as c

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,HERE/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
report=load('report_test','24_A_smallpanel_report.py')
control=load('control_test','24_A_smallpanel_control.py')

class Contracts(unittest.TestCase):
    def test_isolated_preflight_keeps_parent_plan_immutable(self):
        parent = {"role": "A", "branch": "exp/pathway2-a-stationarity", "x": [1]}
        child = control.isolated_preflight_plan(parent)
        self.assertEqual(parent["branch"], "exp/pathway2-a-stationarity")
        self.assertEqual(child["branch"], "exp/pathway2-a-smallpanel")
        self.assertEqual(child["x"], [1])

    def test_isolated_preflight_refuses_unexpected_parent(self):
        with self.assertRaisesRegex(ValueError, "Unexpected frozen Route A plan"):
            control.isolated_preflight_plan({"role": "A", "branch": "another-branch"})

    def test_panel_is_fixed_40_unique_cells(self):
        self.assertEqual(len(set(c.expected_cells())),40)
        self.assertEqual([c.expected_cells()[i] for i in (0,2,4,6)],
          ['seed42_patient807_full_control','seed42_patient807_dose01_control','seed42_patient2085_full_control','seed42_patient2085_dose01_control'])

    def test_missing_entire_groups_refuses_report(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name in c.expected_cells()[:4]: (root/name).mkdir()
            with self.assertRaisesRegex(ValueError,'Panel incomplete'): report.load_runs(root,{})

    def test_duplicate_or_extra_cell_refuses_report(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name in c.expected_cells()+['seed42_duplicate']: (root/name).mkdir()
            with self.assertRaisesRegex(ValueError,'extra='): report.load_runs(root,{})

    def test_missing_baseline_metadata_refused(self):
        with self.assertRaises(ValueError): c.check_baseline_metadata({'metadata':{}},42,'decay',dict(epoch_order_sha256='x',final_rng_sha256='y'))

    def test_probe_wrong_recipe_refused(self):
        with self.assertRaises(ValueError): c.validate_probe_manifest(dict(status='complete',seed=42,arm='decay'),42,'decay')

    def test_optimizer_certificate_failed_or_duplicate_refused(self):
        row=dict(role='loo',seed=42,training_instance='shadow_3_seed42_patient807',passed=True,continued_optimizer_sha256_at_E_star='f'*64)
        cert=dict(status='complete',extended_epochs=100,checks=[row])
        self.assertEqual(c.optimizer_reference(cert,42,807),'f'*64)
        cert['checks']=[row,row]
        with self.assertRaises(ValueError): c.optimizer_reference(cert,42,807)

    def test_config_change_refused_before_execution(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/c.CONFIG).write_text('{}');(root/'config.sha256').write_text('bad')
            with self.assertRaisesRegex(ValueError,'configuration changed'):c.load_config(root)

    def test_run_requires_successful_test_and_does_not_submit(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(control,'load_config',return_value=(Path(td),{})),mock.patch.object(control,'require_test',side_effect=ValueError('test failed')),mock.patch.object(control,'command') as cmd:
                with self.assertRaisesRegex(ValueError,'test failed'):control.submit(Path(td),'run')
                cmd.assert_not_called()

    def test_duplicate_submission_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'run_submission.json').write_text('{}')
            with mock.patch.object(control,'load_config',return_value=(root,{})),mock.patch.object(control,'command') as cmd:
                with self.assertRaisesRegex(ValueError,'already submitted'):control.submit(root,'run')
                cmd.assert_not_called()

    def test_linear_geometry(self):
        got=report.geometry([1.,2.],[.1,.2],[4.,5.])
        self.assertAlmostEqual(got['R'],1);self.assertAlmostEqual(got['cosine'],1)
        self.assertAlmostEqual(got['relative_linear_error'],0)

if __name__=='__main__':unittest.main(verbosity=2)
