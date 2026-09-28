#!/usr/bin/env python3
import argparse
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode = True
import runtime as rt
from stable_core import learning_rate
try:
    import numpy as np
    import torch
    from torch import nn
    from stable_core import train, fixed_mask_loss
except ImportError:
    torch = None


class ControllerTests(unittest.TestCase):
    def test_schedule_matches_mentor_convention(self):
        self.assertEqual(learning_rate(0,100),.0001)
        self.assertEqual(learning_rate(9,100),.001)
        self.assertEqual(learning_rate(10,100),.001)
        self.assertAlmostEqual(learning_rate(55,100),.0005)
        self.assertGreater(learning_rate(99,100),0.)
        with self.assertRaises(ValueError):
            learning_rate(100,100)

    def test_disjoint(self):
        with self.assertRaises(ValueError):
            rt.disjoint('/tmp/a/new','/tmp/a')

    def test_submit_is_exclusive_and_gated(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'logs').mkdir()
            c = dict(out=d,account='a',cells=[{},{}],slurm_time='00:01:00')
            a = argparse.Namespace(out=d,after=None)
            with patch.object(rt,'config',return_value=c), patch.object(rt.subprocess,'check_output',side_effect=['11\n','12\n']) as sb:
                rt.submit(a,'s','s')
                self.assertIn('--dependency=afterok:11',sb.call_args_list[1].args[0])
                self.assertIn('--array=0-1%1',sb.call_args_list[1].args[0])
                with self.assertRaises(FileExistsError):
                    rt.submit(a,'s','s')


@unittest.skipIf(torch is None,'PyTorch unavailable; mandatory in Delta smoke')
class NumericTests(unittest.TestCase):
    def test_real_cnn_replay_and_preserved_observation(self):
        import importlib.util
        from audit_torch import snapshot
        spec=importlib.util.spec_from_file_location('stable_cnn_test',rt.REPO/'src/models.py')
        models=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(models)
        torch.manual_seed(3)
        m=models.SmallCNN(1,128,4).cpu()
        n=copy.deepcopy(m)
        X=np.random.default_rng(1).normal(size=(2,1,32,32)).astype('float32')
        y=np.array([0,1]); orders=np.array([[0,1],[1,0]])
        def observe(raw,ema,epoch):
            snapshot(raw,X,y,[0,1],{807:[0],2085:[1]},1,1e-5)
            for p in ema.parameters():
                p.requires_grad_(True)
            snapshot(ema,X,y,[0,1],{807:[0],2085:[1]},1,1e-5)
            return {}
        raw,ema,_,_=train(m,X,y,orders,[],0.,batch=1,observe=observe)
        raw2,ema2,_,_=train(n,X,y,orders,[],0.,batch=1)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(raw.parameters(),raw2.parameters())))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(ema.parameters(),ema2.parameters())))

    def setup_data(self):
        torch.manual_seed(42)
        m = nn.Sequential(nn.Linear(3,5),nn.Tanh(),nn.Dropout(.2),nn.Linear(5,2)).double()
        rng = np.random.default_rng(42)
        X = rng.normal(size=(11,3))
        y = rng.integers(0,2,11)
        orders = np.stack([rng.permutation(11) for _ in range(4)])
        return m,X,y,orders

    def test_fixed_denominator_against_explicit_gradient(self):
        m,X,y,orders = self.setup_data()
        m.eval()
        idx = orders[0][:4]
        mask = torch.tensor(np.isin(idx,[1,2]))
        logits = m(torch.tensor(X[idx]))
        weighted,_ = fixed_mask_loss(logits,torch.tensor(y[idx]),mask,1.)
        ref = nn.functional.cross_entropy(logits,torch.tensor(y[idx]),reduction='none')[~mask].sum()/4
        self.assertTrue(torch.equal(weighted,ref))

    def test_training_matches_independent_recipe_and_ema(self):
        m,X,y,orders = self.setup_data()
        ref = copy.deepcopy(m)
        ref[2].p=0
        avg = [p.detach().clone() for p in ref.parameters()]
        opt = torch.optim.Adam(ref.parameters(),lr=.001,weight_decay=1e-5)
        total=12
        step=0
        for order in orders:
            for start in range(0,11,4):
                idx=order[start:start+4]
                # Independent spelling of the attached mentor schedule.
                rate=.001*(step+1) if step<1 else .001*.5*(1+math.cos(math.pi*(step-1)/11))
                opt.param_groups[0]['lr']=rate
                opt.zero_grad(set_to_none=True)
                losses=nn.functional.cross_entropy(ref(torch.tensor(X[idx])),torch.tensor(y[idx]),reduction='none')
                weights=torch.tensor([.9 if i in [1,2] else 1. for i in idx])
                (losses*weights).mean().backward()
                opt.step()
                with torch.no_grad():
                    for v,p in zip(avg,ref.parameters()):
                        v.mul_(.99).add_(p,alpha=.01)
                step+=1
        raw,ema,_,curve=train(m,X,y,orders,[1,2],.1,batch=4)
        for p,q in zip(raw.parameters(),ref.parameters()):
            torch.testing.assert_close(p,q,rtol=1e-8,atol=1e-10)
        for p,q in zip(ema.parameters(),avg):
            torch.testing.assert_close(p,q,rtol=1e-8,atol=1e-10)
        self.assertEqual(curve[-1]['step'],total)

    def test_alpha_zero_is_exact_noop_and_ema_has_no_feedback(self):
        m,X,y,orders=self.setup_data()
        m2=copy.deepcopy(m)
        a,e,_,_=train(m,X,y,orders,[],0.,batch=4,ema_decay=.9)
        b,f,_,_=train(m2,X,y,orders,[1,2],0.,batch=4,ema_decay=.99)
        for p,q in zip(a.parameters(),b.parameters()):
            self.assertTrue(torch.equal(p,q))
        self.assertTrue(any(not torch.equal(p,q) for p,q in zip(e.parameters(),f.parameters())))


if __name__=='__main__':
    require='--require-torch' in sys.argv
    if require:
        sys.argv.remove('--require-torch')
    if require and torch is None:
        raise SystemExit('PyTorch tests MUST run in smoke')
    if torch is not None:
        torch.set_num_threads(1)
    unittest.main(verbosity=2)
