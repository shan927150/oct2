#!/usr/bin/env python3
import argparse
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import runtime as rt
try:
    import numpy as np
    import torch
    from torch import nn
    import polish_core as pc
    from audit_torch import snapshot
except ImportError:
    torch = None


class ControllerTests(unittest.TestCase):
    def test_output_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            for out, source in [(d,d), (d+'/new',d), (d,d+'/old')]:
                with self.assertRaises(ValueError):
                    rt.disjoint(out, source)

    def test_duplicate_submit_and_partial_ledger(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'logs').mkdir()
            c = dict(out=d, account='test', cells=[{},{}], slurm_time='00:01:00')
            a = argparse.Namespace(out=d, after='55')
            with patch.object(rt, 'config', return_value=c), patch.object(rt.subprocess, 'check_output', side_effect=['111;cluster\n', RuntimeError('test failure')]) as call:
                with self.assertRaises(RuntimeError):
                    rt.submit(a, 'test', 'test')
                self.assertIn('--dependency=afterany:55', call.call_args_list[0].args[0])
                self.assertIn('--dependency=afterok:111', call.call_args_list[1].args[0])
                self.assertEqual((Path(d)/'smoke_job.txt').read_text().strip(), '111')
                with self.assertRaises(FileExistsError):
                    rt.submit(a, 'test', 'test')
                self.assertEqual(call.call_count, 2)

    def test_completed_checks_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'value.txt'
            p.write_text('a')
            rt.write(Path(d)/'complete.json', dict(status='complete', config_sha256='c', output_sha256={'value.txt':rt.sha(p)}))
            rt.completed(d,'c')
            p.write_text('b')
            with self.assertRaises(ValueError):
                rt.completed(d,'c')


@unittest.skipIf(torch is None, 'PyTorch unavailable; Delta smoke requires it')
class MathTests(unittest.TestCase):
    def test_real_cnn_fp64_and_checkpoint_roundtrip(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('polish_cnn_test',rt.REPO/'src/models.py')
        models=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models)
        torch.manual_seed(3)
        m=models.SmallCNN(1,128,4).cpu()
        self.assertEqual(sum(p.numel() for p in m.parameters()),503044)
        X=np.random.default_rng(1).normal(size=(2,1,32,32)).astype('float32')
        y=np.array([0,1])
        info,v=snapshot(copy.deepcopy(m).double(),X,y,[0,1],{807:[0],2085:[1]},1,1e-5,check_identity=True)
        self.assertTrue(np.isfinite(v).all())
        with tempfile.TemporaryDirectory() as d:
            torch.save(m.state_dict(),Path(d)/'weights.pt')
            n=models.SmallCNN(1,128,4).cpu()
            n.load_state_dict(torch.load(Path(d)/'weights.pt',weights_only=True))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(m.parameters(),n.parameters())))

    def toy(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(23,4))
        y = rng.integers(0,3,size=23)
        torch.manual_seed(4)
        m = nn.Sequential(nn.Linear(4,5), nn.Tanh(), nn.Dropout(.3), nn.Linear(5,3)).double()
        return m, X, y, np.arange(23), {807: [0,4], 2085: [2,8,10]}

    def test_gradient_matches_direct_loss_and_patient_identity(self):
        m,X,y,idx,patients = self.toy()
        m.eval()
        logits = m(torch.tensor(X))
        ce = nn.functional.cross_entropy(logits, torch.tensor(y), reduction='none')
        direct = ce.mean()+.01/2*sum((p*p).sum() for p in m.parameters())
        grad = torch.cat([g.reshape(-1) for g in torch.autograd.grad(direct,m.parameters())]).detach().numpy()
        info, vec = snapshot(m,X,y,idx,patients,batch=7,wd=.01,check_identity=True)
        np.testing.assert_allclose(vec[0], grad, rtol=1e-10,atol=1e-12)
        for alpha in (0.,.1,1.):
            obj = pc.FullBatchObjective(m,X,y,idx,.01,7,patient_rows=patients,alpha=alpha,deleted_patient=807)
            opt = torch.optim.SGD(m.parameters(),lr=.1)
            obj.make_closure(opt)()
            actual = torch.cat([p.grad.reshape(-1) for p in m.parameters()]).numpy()
            np.testing.assert_allclose(actual, vec[0]-alpha*vec[1],rtol=1e-10,atol=1e-12)

    def test_hard_budget_rolls_back_trial(self):
        m,X,y,idx,_ = self.toy()
        old = copy.deepcopy(m.state_dict())
        r = pc.polish(m,X,y,idx,.01,7,max_outer=5,eval_budget=2)
        self.assertLessEqual(r['closure_calls'],2)
        self.assertEqual(r['accepted_iterations'],0)
        self.assertEqual(r['stop_reason'],'eval_budget_exhausted_rolled_back')
        for k,v in old.items():
            self.assertTrue(torch.equal(v,m.state_dict()[k]))

    def test_accepted_endpoint_grad_and_rng(self):
        m,X,y,idx,patients = self.toy()
        m.train()
        state = torch.get_rng_state().clone()
        initial = snapshot(m,X,y,idx,patients,7,.01)[0]['objective']
        r = pc.polish(m,X,y,idx,.01,7,max_outer=8,eval_budget=60,diagnostic_every=3)
        info, v = snapshot(m,X,y,idx,patients,7,.01)
        self.assertLess(info['objective'],initial)
        self.assertAlmostEqual(r['final']['fp32_grad_norm'],float(np.linalg.norm(v[0])),places=9)
        self.assertAlmostEqual(r['final']['objective'],info['objective'],places=10)
        self.assertEqual(r['final']['outer'],r['accepted_iterations'])
        self.assertTrue(m.training)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))


if __name__ == '__main__':
    require = '--require-torch' in sys.argv
    if require:
        sys.argv.remove('--require-torch')
    if require and torch is None:
        raise SystemExit('PyTorch numerical tests MUST run in smoke')
    if torch is not None:
        torch.set_num_threads(1)
    unittest.main(verbosity=2)
