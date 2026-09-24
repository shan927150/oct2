#!/usr/bin/env python3
"""Exercise the complete 40-cell report and paired ΔP path with synthetic artifacts."""
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE))
import route_a_smallpanel as c
spec=importlib.util.spec_from_file_location('report_integration',HERE/'24_A_smallpanel_report.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);torch.save(value,p)

def curve(path,column,values):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['epoch',column]);w.writeheader()
        for ep,v in zip((100,140),values):w.writerow({'epoch':ep,column:v})

def fixture(root):
    e100=root/'truth';probe=root/'probe'
    cfg=dict(root=str(root),e100_root=str(e100),probe_root=str(probe),git_commit='test',resume_epoch=100,total_epochs=140,probe_facts={})
    (root/c.CONFIG).write_text('{}');cfg['config_sha256']=c.sha(root/c.CONFIG)
    base=np.array([3.,4.]);P=np.array([[.4,.6],[.3,.7]])
    for seed in c.SEEDS:
        for dose in c.DOSES:
            save(e100/f'shadow3_{dose}/checkpoints/shadow_3_baseline_seed{seed}.pt',dict(state_dict={'w':torch.tensor(base)}))
            for pid in c.PATIENTS:
                v=base+np.array([1.,2.])*(1 if dose=='full' else .1)
                save(e100/f'shadow3_{dose}/checkpoints/shadow_3_seed{seed}_patient{pid}.pt',dict(state_dict={'w':torch.tensor(v)}))
        for arm in c.ARMS:
            fact=dict(epoch_order_sha256='order',final_rng_sha256='rng');cfg['probe_facts'][f'{seed}/{arm}']=fact
            b=probe/f'shadow3_seed{seed}_{arm}';b.mkdir(parents=True)
            md=dict(seed=seed,arm=arm,epoch=140,resume_epoch=100,total_epochs=140,base_lr=.001,
                decay_schedule=c.SCHEDULE if arm=='decay' else None,source='pathway2_A_lr_continuation_v1',
                epoch_order_sha256='order',rng_fingerprint='rng',optimizer_state={'x':1},rng_states={'x':1})
            save(b/'checkpoints/epoch140.pt',dict(state_dict={'w':torch.tensor(base)},metadata=md));curve(b/'curve.csv','eval_grad_norm',[.1,.01])
            for pid in c.PATIENTS:
                for dose in c.DOSES:
                    name=c.cell_name(seed,pid,dose,arm);d=root/name;d.mkdir()
                    m=dict(schema=c.SCHEMA,cell=name,seed=seed,patient_id=pid,dose=dose,arm=arm,all_cells=c.expected_cells(),
                        panel=c.PATIENTS,affected_shadow=3,resume_epoch=100,total_epochs=140,git_commit='test',config_sha256=cfg['config_sha256'],
                        replay_e100_passed=True,input_files_unchanged=True,source_files_unchanged=True,dataset_arrays_unchanged=True,
                        status='complete',base_lr=.001,weight_decay=1e-5,deletion_weight=c.DOSES[dose],decay_schedule=c.SCHEDULE if arm=='decay' else None)
                    for epoch in (100,140):
                        md={k:m[k] for k in ('seed','patient_id','dose','arm','deletion_weight','resume_epoch','total_epochs','base_lr','decay_schedule')}
                        md.update(epoch=epoch,source=c.SCHEMA,optimizer_state={'x':1},rng_states={'x':1},epoch_order_sha256='order',rng_fingerprint='rng')
                        v=base+np.array([1.,2.])*(1 if dose=='full' else .1)
                        save(d/f'checkpoints/epoch{epoch:03d}.pt',dict(state_dict={'w':torch.tensor(v)},metadata=md))
                    curve(d/'curve.csv','weighted_eval_grad_norm',[.1,.01])
                    c.write_new(d/'replay_checks.json',[{k:True for k in ('passed','parameters_equal','rng_fingerprint_equal','optimizer_hash_equal','recorded_orders_equal')}])
                    dp=np.array([[.1,-.1],[.1,-.1]])*(1 if dose=='full' else .1)
                    np.savez(d/'interface.npz',raw_indices=np.array([0,1]),n_member=np.array(1),baseline_100=P,baseline_140=P,loo_100=P+dp,loo_140=P+dp)
                    m['checkpoint_sha256']={p.name:c.sha(p) for p in (d/'checkpoints').glob('*.pt')}
                    m['artifact_sha256']={n:c.sha(d/n) for n in ('curve.csv','replay_checks.json','interface.npz')}
                    c.write_new(d/'manifest.json',m)
    return cfg

class ReportIntegration(unittest.TestCase):
    def test_full_report_and_corrupt_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg=fixture(root)
            with mock.patch.object(c,'load_config',return_value=(root,cfg)),mock.patch('builtins.print'):
                r.main(['--root',str(root)])
            summary=c.read(root/'report/summary.json')
            self.assertEqual(summary['complete_cells'],40)
            self.assertEqual(summary['geometry_rows'],30)
            for s in summary['geometry_medians'].values():self.assertAlmostEqual(s['R'],1.)
            with (root/'report/gradient_rows.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),100)
            d=root/c.expected_cells()[0];(d/'curve.csv').write_text('bad')
            with self.assertRaisesRegex(ValueError,'Artifact changed'):r.load_runs(root,cfg)

if __name__=='__main__':unittest.main(verbosity=2)
