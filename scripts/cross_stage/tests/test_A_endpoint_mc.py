#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('mc_test_module',HERE/'25_A_endpoint_mc.py')
mc=importlib.util.module_from_spec(spec);spec.loader.exec_module(mc)

class EndpointMC(unittest.TestCase):
    def test_vector_difference_and_standard_error(self):
        model=torch.nn.Linear(1,2)
        vectors=[torch.tensor([1.,0.]),torch.tensor([0.,2.]),torch.tensor([2.,2.])]
        with mock.patch.object(mc.core,'objective_snapshot',side_effect=[({},v) for v in vectors]):
            got=mc.gradient_comparison(model,None,None,None,2,2,100)
        self.assertAlmostEqual(got['diff_norm_dropout_minus_eval'],2.)
        self.assertAlmostEqual(got['mc_mean_vector_standard_error'],1.)
        self.assertAlmostEqual(got['diff_norm_over_mc_error'],2.)
        self.assertAlmostEqual(got['cosine_dropout_eval'],1/np.sqrt(5))

    def test_real_objective_rng_mode_and_no_parameter_changes(self):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(88)
        model=torch.nn.Sequential(torch.nn.Linear(3,5),torch.nn.ReLU(),torch.nn.Dropout(.5),torch.nn.Linear(5,2)).to(device)
        model.train(); model[0].eval()
        modes=[m.training for m in model.modules()]
        params=[p.clone() for p in model.parameters()]
        rng=mc.core.rng_fingerprint()
        got=mc.gradient_comparison(model,np.arange(12,dtype=np.float32).reshape(4,3)/12,np.array([0,1,0,1]),np.arange(4),2,4,100)
        self.assertEqual(rng,mc.core.rng_fingerprint())
        self.assertEqual(modes,[m.training for m in model.modules()])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(params,model.parameters())))
        self.assertTrue(all(v is None or np.isfinite(v) for v in got.values()))

    def test_zero_mc_error_does_not_format_none_as_float(self):
        model=torch.nn.Linear(1,2)
        with mock.patch.object(mc.core,'objective_snapshot',return_value=({},torch.tensor([1.,0.]))):
            got=mc.gradient_comparison(model,None,None,None,2,2,100)
        self.assertIsNone(got['diff_norm_over_mc_error'])

if __name__=='__main__': unittest.main(verbosity=2)
