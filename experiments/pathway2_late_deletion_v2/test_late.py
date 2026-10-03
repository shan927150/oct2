"""CPU regression gates for snapshot isolation, stable fidelity and real workflow."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import numpy as np
import torch
import kernel as k
import metrics
import runtime_late as rt
import run
import engine
torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
stable=engine.old.module('test_stable_exact',rt.REPO/'experiments/pathway2_stable_v1/stable_core.py')


def fixture():
    rng=np.random.default_rng(917);X=rng.normal(size=(64,5)).astype('float32')
    groups=np.repeat(np.array([807,2085]+list(range(1002,1016))),4);y=np.repeat(np.arange(16)%4,4).astype('int64')
    train=np.arange(48);test=np.arange(48,64)
    ctx=dict(X=X,y=y,groups=groups,train=train,test=test,patient_rows={807:np.arange(4)})
    d=engine.make_design(train,test,groups,y,8,.5,20260929)
    def factory():
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        return torch.nn.Sequential(torch.nn.Linear(5,12),torch.nn.ReLU(),torch.nn.Dropout(.2),torch.nn.Linear(12,4))
    orders=[np.random.default_rng(300+e).permutation(train) for e in range(6)]
    recipe=dict(batch=8,lr=.01,warmup_frac=.1,ema_decay=.9,wd=1e-5)
    c=dict(seed=42,patients=[807],epochs=6,recipe=recipe,T0_grid=[0,2,4],window_epochs=2,alphas=[.01,.1,1.],batch=8)
    return ctx,d,factory,orders,c


class KernelTests(unittest.TestCase):
    def setUp(self):
        self.ctx,self.d,self.factory,self.orders,self.c=fixture()
        self.start=k.fresh(self.factory,self.c['recipe'],36)
    def train(self,s,alpha,**kw):
        return k.train(s,self.factory,self.ctx['X'],self.ctx['y'],self.orders,np.arange(4),alpha,stable,**kw)

    def test_same_native_stable_kernel(self):
        for alpha in [0.,.01,.1,1.]:
            ours,_=self.train(self.start,alpha)
            inherited=stable.train(self.factory(),self.ctx['X'],self.ctx['y'],self.orders,np.arange(4),alpha,**self.c['recipe'])
            for name,value in zip(['raw','ema','optimizer'],inherited[:3]):self.assertTrue(k.same(ours[name],value.state_dict()),name)

    def test_immutable_snapshot_repeated_reversed_and_disk_restoration(self):
        prefix,_=self.train(self.start,0.,stop=2);frozen=k.digest(prefix)
        states={a:self.train(prefix,a)[0] for a in [.1,1.]}
        for alpha in [1.,.1,.1]:engine.require_same(self.train(prefix,alpha)[0],states[alpha],'repeat')
        self.assertEqual(k.digest(prefix),frozen)
        with tempfile.TemporaryDirectory() as t:
            torch.save(prefix,Path(t)/'s.pt');loaded=engine.load(Path(t)/'s.pt')
            engine.require_same(self.train(loaded,.1)[0],states[.1],'disk restore')

    def test_full_state_replay_at_every_point_and_terminal_noop(self):
        full,_=self.train(self.start,0.)
        for epoch in [0,2,4,6]:
            source,_=self.train(self.start,0.,stop=epoch)
            for _ in range(2):engine.require_same(self.train(source,0.)[0],full,'replay')
        engine.require_same(self.train(full,1.)[0],full,'terminal')
        bad=copy.deepcopy(full);next(iter(bad['ema'].values())).add_(1)
        with self.assertRaises(ValueError):engine.require_same(bad,full,'EMA corruption')
        bad=copy.deepcopy(full);next(iter(bad['optimizer']['state'].values()))['exp_avg'].add_(1)
        with self.assertRaises(ValueError):engine.require_same(bad,full,'moment corruption')

    def test_observation_cannot_change_rng_trajectory(self):
        plain,_=self.train(self.start,.1)
        def noisy(*args):random.random();np.random.random(4);torch.randn(4)
        observed,_=self.train(self.start,.1,observe=noisy)
        engine.require_same(plain,observed,'observer RNG')

    def test_geometry_zero_and_linear_and_nonlinear(self):
        d=np.array([1.,2.,3.]);g=metrics.geometry(.1*d,d)
        self.assertAlmostEqual(g['R'],1);self.assertAlmostEqual(g['cosine'],1);self.assertEqual(g['relative_linear_error'],0)
        g=metrics.geometry(d,d);self.assertAlmostEqual(g['R'],10)
        g=metrics.geometry(d*0,d*0);self.assertIsNone(g['R']);self.assertIsNone(g['cosine'])


class IntegrationTests(unittest.TestCase):
    def test_real_oct_cnn_kernel_and_restore(self):
        models=engine.old.module('late_real_oct_models',rt.REPO/'src/models.py')
        def factory():
            torch.manual_seed(42)
            return models.SmallCNN(1,128,4).cpu()
        rng=np.random.default_rng(991);X=rng.normal(size=(4,1,32,32)).astype('float32');y=np.arange(4,dtype='int64')
        orders=[np.arange(4),np.array([3,1,0,2])];recipe=dict(batch=2,lr=.001,warmup_frac=.1,ema_decay=.99,wd=1e-5)
        s=k.fresh(factory,recipe,4)
        full,_=k.train(s,factory,X,y,orders,[],0.,stable)
        ref=stable.train(factory(),X,y,orders,[],0.,**recipe)
        self.assertTrue(all(k.same(full[key],m.state_dict()) for key,m in zip(['raw','ema','optimizer'],ref[:3])))
        mid,_=k.train(s,factory,X,y,orders,[],0.,stable,stop=1)
        engine.require_same(k.train(mid,factory,X,y,orders,[],0.,stable)[0],full,'real CNN replay')

    def test_smoke_calibration_all_cells_and_collect(self):
        ctx,d,factory,orders,c=fixture()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c.update(out=str(root/'out'),baseline_out=str(root/'parent'))
            out=Path(c['out']);out.mkdir();rt.write(out/'config.json',c)
            ref=Path(c['baseline_out'])/'seed42_baseline';ref.mkdir(parents=True)
            inherited=stable.train(factory(),ctx['X'],ctx['y'],orders,[],0.,**c['recipe'])
            torch.save({key:k.clone_cpu(m.state_dict()) for key,m in zip(['raw','ema','optimizer'],inherited[:3])},ref/'final.pt')
            smoke=out/'smoke';smoke.mkdir();engine.smoke(c,smoke,ctx,stable,d,orders,factory);rt.finish(smoke,c)
            base=out/'baseline';base.mkdir();engine.make_baseline(c,base,ctx,stable,d,orders,factory);rt.finish(base,c,elapsed_seconds=1.)
            c['cells']=[dict(name=f'T0{t:03d}_p807',T0=t,patient=807,seed=42) for t in [4,2,0]]
            for cell in c['cells']:
                dest=out/cell['name'];dest.mkdir();engine.branch_cell(c,dest,base,cell,ctx,stable,d,orders,factory)
                rt.finish(dest,c,elapsed_seconds=1.)
                self.assertTrue(rt.read(dest/'result.json')['source_snapshot_unchanged'])
            import report
            report.collect(c)
            self.assertTrue((out/'report/late_deletion_geometry.pdf').exists())
            self.assertEqual(rt.read(out/'report/summary.json')['geometry_rows'],40)
            run.archive(out)
            import tarfile
            with tarfile.open(str(out)+'.tar.gz') as tf:
                self.assertTrue(any(n.endswith('.npz') for n in tf.getnames()))
                self.assertFalse(any(n.endswith('.pt') for n in tf.getnames()))


class ControlTests(unittest.TestCase):
    def test_prepare_carries_fingerprint_and_pins_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);base=root/'parent';base.mkdir();(base/'report').mkdir()
            old=dict(recipe=dict(lr=.001,warmup_frac=.1,ema_decay=.99,wd=1e-5,batch=128),epochs=140,seeds=[42],patients=[807],
                     input_sha256={},full=str(root/'full'),data_dir=str(root/'data'),audit=str(root/'audit'),
                     dataset_fingerprint=dict(X_sha256='fixture-only'),account='test')
            rt.write(base/'config.json',old);h=rt.sha(base/'config.json');(base/'config.sha256').write_text(h+'\n')
            rt.write(base/'report/complete.json',dict(status='complete',config_sha256=h,noop_passed=True))
            d=base/'seed42_baseline';d.mkdir();(d/'final.pt').write_bytes(b'prepare contract fixture, not a checkpoint')
            rt.write(d/'complete.json',dict(status='complete',config_sha256=h,output_sha256={'final.pt':rt.sha(d/'final.pt')}))
            args=SimpleNamespace(baseline_out=str(base),out=str(root/'out'),seed=42,patient=807,concurrency=1)
            with patch.object(rt,'check_repo',return_value='fixture-commit'),patch.object(rt,'BASELINE_HASH',h):rt.prepare(args)
            c=rt.read(root/'out/config.json');self.assertEqual(c['dataset_fingerprint'],old['dataset_fingerprint'])
            self.assertEqual(c['cells'][0]['T0'],130);self.assertIn(str(d/'final.pt'),c['input_sha256'])
            self.assertIn(str(rt.HERE/'job.slurm'),c['source_sha256'])
            (d/'final.pt').write_bytes(b'tampered')
            with self.assertRaises(ValueError):rt.verify(c['input_sha256'])

    def test_partial_submission_is_recorded_and_not_repeated(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'logs').mkdir();c=dict(out=t,account='example')
            answers=[SimpleNamespace(returncode=0,stdout='123;delta\n',stderr=''),SimpleNamespace(returncode=1,stdout='',stderr='queue error')]
            with patch.object(run.rt,'config',return_value=c),patch.object(run.subprocess,'run',side_effect=answers):
                with self.assertRaises(RuntimeError):run.submit(SimpleNamespace(out=t))
                self.assertEqual((root/'smoke_job.txt').read_text(),'123\n')
                with self.assertRaises(FileExistsError):run.submit(SimpleNamespace(out=t))

    def test_rest_requires_artifacts_and_uses_no_old_dependency(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'logs').mkdir();c=dict(out=t,account='example',concurrency=1,cells=[dict(name='pilot'),dict(name='next')])
            rt.write(root/'config.json',c)
            with patch.object(run.rt,'config',return_value=c),patch.object(run.subprocess,'run') as command:
                with self.assertRaises(FileNotFoundError):run.submit_rest(SimpleNamespace(out=t))
                self.assertFalse(command.called)
                for name,extra in [('smoke',{}),('baseline',dict(exact_parent_replay=True)),('pilot',dict(quality_gate_passed=True))]:
                    dest=root/name;dest.mkdir();rt.write(dest/'result.json',extra);rt.finish(dest,c)
                command.return_value=SimpleNamespace(returncode=0,stdout='456\n',stderr='')
                run.submit_rest(SimpleNamespace(out=t))
                self.assertFalse(any('dependency' in a for a in command.call_args.args[0]))
                (root/'pilot/result.json').write_text('{}')
                with self.assertRaises(ValueError):run.prerequisites(c,True)


if __name__=='__main__':unittest.main(verbosity=2)
