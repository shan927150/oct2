"""Meaningful protocol, numerical and CPU end-to-end checks; no OCT GPU claim."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import design
import analysis_stats as stats


class ProtocolTests(unittest.TestCase):
    def panel(self):
        groups=np.repeat(np.arange(32),2)
        labels=np.repeat(np.arange(32)%4,2)
        return design.make_design(np.arange(32),np.arange(32,64),groups,labels,8,.5,20260929)

    def test_masks_and_query_units(self):
        d=self.panel()
        self.assertEqual(len(set(tuple(m) for m in d['masks'])),8)
        self.assertTrue(all(sum(m)==8 for m in d['masks']))
        self.assertAlmostEqual(sum(d['query_weights']),1.)
        self.assertEqual(d,self.panel())

    def test_objective_normalization_and_sign(self):
        d=self.panel()
        removed,w=design.intervention(d,0,'fixed_mask',.1,32)
        self.assertEqual(len(removed),16)
        np.testing.assert_allclose(w,-.1*(1-np.asarray(d['masks'][0])))
        _,w2=design.intervention(d,0,'subset',1.,32)
        np.testing.assert_allclose(w2,2*np.asarray(d['masks'][0])-1)
        # Hessian=2, h=3, b=5 => upweight -7.5; alpha=.1 deletion +.75.
        tau=-3*5/2
        self.assertAlmostEqual(-.1*tau,.75)

    def test_interface_derivative(self):
        p=np.array([[.3,.7],[.6,.4]])
        labels=np.array([1,0]);weights=np.array([.2,.8])
        _,v=design.output_and_v(p,labels,weights)
        direction=np.array([[.1,-.1],[-.2,.2]])
        eps=1e-5
        fd=(design.output_and_v(p+eps*direction,labels,weights)[0]-design.output_and_v(p-eps*direction,labels,weights)[0])/(2*eps)
        self.assertAlmostEqual(fd,float((v*direction).sum()),places=8)

    def test_stats_constant_and_no_seed_pseudoreplication(self):
        self.assertIsNone(stats.spearman([1,1,1],[1,2,3]))
        y=np.arange(8)[:,None]+np.array([[10.,30.,50.]])
        r=stats.reliability(y)
        self.assertAlmostEqual(r['reliability_of_seed_mean'],1.)
        self.assertTrue(all(x['spearman']>0.999 for x in r['seed_pairwise_rank_correlations']))
        self.assertEqual(stats.evaluate(np.arange(8),np.arange(8),draws=50)['n_subsets'],8)

    def test_crossfit_never_sees_own_label(self):
        rng=np.random.default_rng(11)
        X=rng.normal(size=(8,10));y=rng.normal(size=8)
        p1=stats.crossfit_ridge(X,y)
        y[3]+=1000
        p2=stats.crossfit_ridge(X,y)
        self.assertEqual(p1[3],p2[3])

    def test_two_stage_implicit_formula(self):
        from formula_checks import implicit_check
        self.assertLess(implicit_check()['relative_error'],1e-6)

    def test_slurm_dependencies_and_duplicate_protection(self):
        import run
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            c=dict(out=tmp,account='example',slurm_time='03:00:00',concurrency=2,cells=list(range(24)))
            calls=[]
            def fake(cmd,**kwargs):
                calls.append(cmd)
                return str(100+len(calls))+'\n'
            with patch.object(run,'config',return_value=c),patch.object(run.subprocess,'check_output',side_effect=fake):
                run.submit(SimpleNamespace(out=tmp))
                self.assertIn('--dependency=afterok:101',calls[1])
                self.assertIn('--dependency=afterok:102',calls[2])
                self.assertIn('--dependency=afterok:102:103',calls[3])
                self.assertIn('--array=0-23%2',calls[3])
                with self.assertRaises(FileExistsError):
                    run.submit(SimpleNamespace(out=tmp))

    def test_collect_from_complete_artifacts(self):
        import run
        import runtime as rt
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'experiment';root.mkdir()
            d=self.panel()
            c=dict(out=str(root),m=8,seeds=[42,43,44],mode='fixed_mask',alpha=1.,retain=.5,
                   primary='synthetic check only',gamma=1.,cells=[dict(name=f'subset{j:03d}_seed{s}',subset=j,seed=s) for j in range(8) for s in (42,43,44)])
            rt.write(root/'config.json',c)
            for name in ('smoke','noop'):
                dst=root/name;dst.mkdir();rt.write(dst/'result.json',dict(ok=True));rt.finish(dst,c,dict(name=name))
            tau=np.random.default_rng(121).normal(size=16)/100
            for seed in c['seeds']:
                dst=root/f'score_seed{seed}';dst.mkdir()
                rt.write(dst/'design.json',d)
                rt.write(dst/'result.json',dict(solver_qualified=True))
                rt.csv_write(dst/'scores.csv',[dict(patient=p['patient'],tau_upweight=t,frozen_h_upweight=t) for p,t in zip(d['train_patients'],tau)])
                rt.finish(dst,c,dict(seed=seed))
            for cell in c['cells']:
                j,s=cell['subset'],cell['seed']
                dy=float(-(1-np.asarray(d['masks'][j]))@tau)
                dst=root/cell['name'];dst.mkdir()
                rt.write(dst/'result.json',dict(Y=10+s+dy,delta_Y=dy,L1_true_P=dy,L2_true_theta=dy))
                rt.finish(dst,c,cell)
            with patch.object(run,'config',return_value=c):
                run.collect(SimpleNamespace(out=str(root)))
            report=rt.read(root/'report/summary.json')
            self.assertAlmostEqual(report['L3_stage1_normalization_matched']['spearman'],1.)
            self.assertAlmostEqual(report['LDS_canonical_retained_sum']['spearman'],1.)
            self.assertTrue(Path(str(root)+'.tar.gz').exists())


class TorchTests(unittest.TestCase):
    def test_two_stage_adam(self):
        import torch
        from formula_checks import trajectory_check
        try:
            self.assertTrue(trajectory_check()['pytorch_update_matched'])
        finally:
            torch.set_default_dtype(torch.float32)

    def test_core_score_and_truth(self):
        import torch
        import core
        import runtime as rt
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        score=core.module('test_actual_score',rt.REPO/'scripts/cross_stage/07_cross_stage_score_ladder.py')
        stable=core.module('test_actual_stable',rt.REPO/'experiments/pathway2_stable_v1/stable_core.py')
        # Two rows/patient; patient 807 included to exercise the actual FD path.
        groups=np.repeat(np.r_[807,2085,np.arange(2,32)],2)
        labels=np.repeat(np.arange(32)%4,2)
        X=np.random.default_rng(11).normal(size=(64,3)).astype('float32')
        train,test=np.arange(32),np.arange(32,64)
        ctx=dict(X=X,y=labels,train=train,test=test,interface=np.arange(64),groups=groups)
        def seed_everything(seed,det):
            torch.manual_seed(seed)
        def orders(idx,epochs,seed):
            rng=np.random.RandomState(seed)
            return np.stack([rng.permutation(idx) for _ in range(epochs)])
        pilot=SimpleNamespace(DEVICE=torch.device('cpu'),seed_everything=seed_everything,
            build_model=lambda *a: torch.nn.Linear(3,4),make_epoch_orders=orders)
        d=design.make_design(train,test,groups,labels,8,.5,20260929)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);base=root/'baseline';(base/'seed42_baseline').mkdir(parents=True)
            full=root/'full';full.mkdir()
            out=root/'out';out.mkdir()
            c=dict(out=str(out),baseline_out=str(base),full=str(full),batch=16,weight_decay=1e-5,
                   gamma=1.,cg_steps=60,cg_tol=1e-8,cg_fail_tol=1e-5,lanczos_steps=10,
                   epochs=2,mode='subset',alpha=1.,recipe=dict(lr=.001,warmup_frac=.1,ema_decay=.99,wd=1e-5,batch=8))
            order=orders(train,2,700042)
            np.savez(full/'stage1_order_seed42.npz',raw_index_order=order)
            m,e,opt,_=stable.train(core.initialize(ctx,pilot,42),X,labels,order,[],1.,**c['recipe'])
            torch.save(dict(raw=m.state_dict(),ema=e.state_dict(),optimizer=opt.state_dict()),base/'seed42_baseline/final.pt')
            np.savez(base/'seed42_baseline/interface.npz',raw=core.predict(m,X,np.arange(64)),raw_index=np.arange(64))
            with patch.object(core,'context',return_value=(ctx,pilot,None,score,stable,d)):
                sd=out/'score_seed42';sd.mkdir();core.score_seed(c,sd,42)
                self.assertTrue(rt.read(sd/'result.json')['solver_qualified'])
                for mode in ('subset','fixed_mask'):
                    c['mode']=mode
                    td=out/mode;td.mkdir()
                    core.truth_cell(c,td,dict(seed=42,subset=0,name=mode))
                    self.assertTrue(np.isfinite(rt.read(td/'result.json')['L2_true_theta']))
                nd=out/'noop';nd.mkdir()
                core.truth_cell(c,nd,dict(seed=42,subset=None,name='noop'))
                self.assertTrue(rt.read(nd/'result.json')['full_horizon_noop_exact'])
                import local_response
                ld=out/'local_response';ld.mkdir()
                with patch.object(local_response,'context',return_value=(ctx,pilot,None,score,stable,d)):
                    local_response.run(c,ld)
                lr=rt.read(ld/'result.json')
                qualified=[r for r in lr['finite_difference'] if r['qualified']]
                self.assertEqual(len(qualified),4)
                self.assertLess(max(r['relative_error'] for r in qualified),.03)


if __name__=='__main__':
    unittest.main()
