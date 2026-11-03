"""Local checks for methods A, B and C (CPU, synthetic data; the real attack4.adjoint is used where it matters).

    python3 test_removal_abc.py [--attack4-dir DIR]

DIR must contain attack4.py (experiments/pathway2_cross_stage_lds_v4 of the frozen checkout); the default
looks for it under ~/oct2-interface-lds-v1r1 and the repository next to this package.
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import common as C
import method_a_score_drop as MA
import method_b_collect as MB
import method_c_query as MC


def find_attack4():
    if '--attack4-dir' in sys.argv:
        i = sys.argv.index('--attack4-dir'); d = sys.argv[i + 1]; del sys.argv[i:i + 2]
        return Path(d)
    for d in [Path.home() / 'oct2-interface-lds-v1r1/experiments/pathway2_cross_stage_lds_v4',
              HERE.parent / 'shan927150/oct2/experiments/pathway2_cross_stage_lds_v4',
              HERE.parent / 'oct2/experiments/pathway2_cross_stage_lds_v4']:
        if (d / 'attack4.py').exists():
            return d
    return None


A4DIR = find_attack4()


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)


# ------------------------------------------------------------------ method A
class TestA(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.J, self.N, self.Rn = 8, 30, 20
        R = self.N + self.Rn
        self.M = (rng.random((self.J, self.N)) < .5).astype(float)
        self.cls = rng.integers(0, 4, R)
        self.data = dict(M=self.M, rows=dict(raw_index=np.arange(R)))
        self.fixed = dict(classes=np.r_[np.zeros(5, int), self.cls], shadow_id=np.r_[np.zeros(5, int), np.full(R, 3)])
        self.contrib = rng.normal(size=(self.J, R)) * (np.isin(self.cls, [1, 2]))[None]
        self.truth = rng.normal(size=self.J)
        self.images = [dict(train_position=i, cls=int(self.cls[i]), attack_row_active=str(self.cls[i] in (1, 2)),
                            margin=float(rng.random()), a_if=float(rng.normal()), a_true=float(rng.normal()))
                       for i in range(self.N)]

    def test_groups_partition_and_identities(self):
        with patch.object(MA, 'SIZES', (3,)):
            sets, cls, pool = MA.row_sets(self.data, self.fixed, self.images)
        d, r, n = sets['deleted'][1], sets['retained'][1], sets['nonmember'][1]
        self.assertTrue(np.all(d.astype(int) + r + n == 1))
        self.assertTrue(np.array_equal(sets['deleted_DME'][1], d & (cls == 1)[None]))
        true = self.contrib + np.random.default_rng(4).normal(size=self.contrib.shape)
        tg = {'combined': {'if': self.contrib, 'true': true, 'truth': self.truth}}
        member = np.arange(len(cls)) < self.N
        rows, _ = MA.analyze(tg, {k: sets[k] for k in ('deleted', 'retained', 'nonmember')}, cls, pool, 40, 1, member)
        got = {(r_['set'], r_['quantity']): r_ for r_ in rows}
        kd, kt = (self.contrib * d).sum(1), (true * d).sum(1)
        want = dict(keep_if=(kd, self.truth), keep_true=(kt, self.truth), drop_if=(self.contrib.sum(1) - kd, self.truth),
                    drop_true=(true.sum(1) - kt, self.truth), replace_if_by_true=(self.contrib.sum(1) - kd + kt, self.truth),
                    replace_true_by_if=(true.sum(1) - kt + kd, self.truth), within_S=(kd, kt))
        self.assertEqual(set(want), set(MA.QUANTITIES))
        for q, (x, y) in want.items():
            self.assertAlmostEqual(got[('deleted', q)]['lds'], C.spearman(x, y), msg=q)
            self.assertIn('random_mean', got[('deleted', q)])
        # replacing every group by its retrained value gives the all-retrained score
        groups = ('deleted', 'retained', 'nonmember')
        allt = self.contrib.sum(1) - sum((self.contrib * sets[g][1]).sum(1) for g in groups) \
            + sum((true * sets[g][1]).sum(1) for g in groups)
        self.assertTrue(np.allclose(allt, true.sum(1)))

    def test_random_controls_match_class_counts(self):
        with patch.object(MA, 'SIZES', (3,)):
            sets, cls, pool = MA.row_sets(self.data, self.fixed, self.images)
        ones = np.ones_like(self.contrib)
        member = np.arange(len(cls)) < self.N
        for name in ('deleted_DME', 'nonmember', 'low_margin_3'):
            _, mask, matching = sets[name]
            for k in range(4):
                for m in (False, True):
                    st = (cls == k) & (member == m)
                    v = ones * st[None]
                    got = MA.random_sums(mask, matching, cls, pool, [v], 50, np.random.default_rng(3), member)[0]
                    self.assertTrue(np.allclose(got, (mask & st[None]).sum(1)[None]), (name, k, m))
        _, mask, matching = sets['class_DME']        # whole class: every draw is the set itself
        got = MA.random_sums(mask, matching, cls, pool, [self.contrib], 5, np.random.default_rng(3), member)[0]
        self.assertTrue(np.allclose(got, (self.contrib * mask).sum(1)[None]))


# ------------------------------------------------------------------ method C: per-query decomposition
class TestCStartup(unittest.TestCase):
    def test_worker_loads_checkout_before_imports(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        class ContextReached(Exception):
            pass
        a = types.SimpleNamespace(repo='unused', out='unused', dest=tmp.name, task=0)
        with patch.dict(sys.modules, {'attack4': None, 'core_if': None}), \
             patch.object(C, 'load_context', side_effect=ContextReached), \
             self.assertRaises(ContextReached):
            MC.worker(a)


@unittest.skipIf(A4DIR is None, 'attack4.py not found; pass --attack4-dir')
class TestCAdjoint(unittest.TestCase):
    def test_query_scores_match_attack4_adjoint(self):
        import torch
        sys.path.insert(0, str(A4DIR)); import attack4
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
        rng = np.random.default_rng(1)
        P = softmax(rng.normal(size=(60, 4)) * 3).astype(np.float32); y = rng.integers(0, 2, 60)
        Pq = softmax(rng.normal(size=(25, 4)) * 3).astype(np.float32); yq = rng.integers(0, 2, 25)
        D = {'a': rng.normal(size=(5, 60, 4)) * 1e-2, 'b': rng.normal(size=(5, 60, 4)) * 1e-3}
        s, diag = MC.query_scores(model, P, y, Pq, yq, .3, D, 'cpu')
        ref = attack4.adjoint(model, P, y, Pq, yq, [.3], 'cpu')['per_gamma'][.3]['v']
        for n, Dn in D.items():
            want = np.einsum('jrk,rk->j', Dn, ref)
            self.assertTrue(np.allclose(s[n].mean(0), want, rtol=1e-9, atol=1e-15), n)
        for i in (0, 7, 24):          # each query alone is the same adjoint with a one-query objective
            vi = attack4.adjoint(model, P, y, Pq[i:i + 1], yq[i:i + 1], [.3], 'cpu')['per_gamma'][.3]['v']
            self.assertTrue(np.allclose(s['a'][i], np.einsum('jrk,rk->j', D['a'], vi), rtol=1e-9, atol=1e-15))
        self.assertLess(diag['solve_rel_residual'], 1e-10)


# ------------------------------------------------------------------ method C: worker end to end (fake trainer)
def fake_world(tmp, J=4, K=2):
    """A tiny attack table with shadows 0 and 3, four classes, target queries, stored 'v6' truth and v."""
    import torch
    sys.path.insert(0, str(A4DIR)); import attack4 as A
    rng = np.random.default_rng(5)
    n0, n3 = 40, 48
    N = 24                                                   # first 24 shadow-3 rows are training rows
    cls3 = np.tile([0, 1, 2, 3], n3 // 4); cls0 = np.tile([0, 1, 2, 3], n0 // 4)
    fixed = dict(shadow_id=np.r_[np.zeros(n0, int), np.full(n3, 3)], classes=np.r_[cls0, cls3],
                 membership=np.r_[rng.integers(0, 2, n0), np.ones(N, int), np.zeros(n3 - N, int)],
                 raw_index=np.r_[np.arange(n0), 1000 + np.arange(n3)])
    pfull = softmax(rng.normal(size=(n3, 4)) * 2)
    fixed['x'] = np.r_[softmax(rng.normal(size=(n0, 4)) * 2), pfull].astype(np.float32)
    nq = 40
    fixed.update(q_x=softmax(rng.normal(size=(nq, 4)) * 2).astype(np.float32), q_classes=np.tile([0, 1, 2, 3], nq // 4),
                 q_membership=np.tile([0, 0, 1, 1, 1, 0, 1, 0], nq // 8))
    M = np.zeros((J, N)); [M.__setitem__((j, rng.permutation(N)[:N // 2]), 1) for j in range(J)]
    truth = np.zeros((3, J, n3, 4))
    for j in range(J):
        truth[:, j] = softmax(np.log(pfull) + rng.normal(size=(n3, 4)) * .3) - pfull
    p0 = pfull
    dz_if = rng.normal(size=(J, 1, n3, 4)) * 1e-2
    dz_gall = rng.normal(size=(1, n3, 4)) * 1e-3

    def train(pilot, hp, x, y, seed):
        torch.manual_seed(int(seed))
        m = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
        opt = torch.optim.Adam(m.parameters(), lr=.05)
        X = torch.as_tensor(np.asarray(x, np.float32)); Y = torch.as_tensor(np.asarray(y), dtype=torch.long)
        for _ in range(15):
            opt.zero_grad(); torch.nn.functional.cross_entropy(m(X), Y).backward(); opt.step()
        return m.eval()

    def evaluate_attack_queries(model, X, yq):
        with torch.no_grad():
            logp = torch.log_softmax(model(torch.as_tensor(np.asarray(X, np.float32))), 1).numpy()
        return dict(cross_entropy=float(-logp[np.arange(len(yq)), yq].astype(np.float64).mean())), np.exp(logp[:, 1]), logp

    pilot = types.SimpleNamespace(evaluate_attack_queries=evaluate_attack_queries)
    seeds = [11, 12][:K]
    ai = types.SimpleNamespace(matched_dz=lambda dzi, dzg, Kj, Nn: (Nn / Kj).reshape(-1, 1, 1) * dzi - ((Nn / Kj).reshape(-1, 1, 1) - 1) * dzg[None],
                               softmax_jvp=lambda p, dz: p * (dz - (p * dz).sum(-1, keepdims=True)),
                               v3_interface=lambda c, name, rows: pfull + truth[0, int(name.split('_s')[1][:3])])
    c = dict(gammas=[.03], primary_gamma=.03, seeds=[42, 43, 44], design_name='img', v6_out=str(tmp / 'v6'))
    data = dict(M=M, full={42: pfull}, truth=truth, rows=dict(raw_index=fixed['raw_index'][n0:]),
                solves={42: dict(arrays=dict(dz_if=dz_if, dz_gall=dz_gall, p0=p0))},
                fr6=dict(attack_seeds=seeds, attack_gamma_star=.3, qualified_classes=[1, 2], omega={'1': .5, '2': .5}))
    # stored "v6" truth and v, made the way v6 makes them
    dp = ai.softmax_jvp(p0, ai.matched_dz(dz_if[:, 0], dz_gall[0], M.sum(1), N))
    V = np.zeros((K, 4, 1, n3, 4)); data['v6_truth'] = {(j, 42): [] for j in range(J)}
    xfull = A.assemble(fixed, pfull)
    for ki, k in enumerate(seeds):
        for cl in range(4):
            tr, aff, where = A.class_rows(fixed, cl); te = fixed['q_classes'] == cl
            m0 = train(None, None, xfull[tr], fixed['membership'][tr], k)
            j00 = evaluate_attack_queries(m0, fixed['q_x'][te], fixed['q_membership'][te])[0]['cross_entropy']
            v = A.adjoint(m0, xfull[tr], fixed['membership'][tr], fixed['q_x'][te], fixed['q_membership'][te], [.3], 'cpu')['per_gamma'][.3]['v']
            V[ki, cl, 0, where] = v[aff]
            for j in range(J):
                xj = A.assemble(fixed, pfull + truth[0, j])
                mj = train(None, None, xj[tr], fixed['membership'][tr], k)
                j10 = evaluate_attack_queries(mj, fixed['q_x'][te], fixed['q_membership'][te])[0]['cross_entropy']
                data['v6_truth'][(j, 42)].append(dict(cls=cl, attack_seed=k, J00=j00, J10=j10))
    vd = tmp / 'v6/base/seed42'; vd.mkdir(parents=True)
    np.savez(vd / 'v.npz', v=V, classes=[0, 1, 2, 3], attack_gammas=[.3], attack_seeds=seeds)
    fake_a4 = types.ModuleType('attack4')
    for name in ('assemble', 'class_rows', 'adjoint', 'AFFECTED'):
        setattr(fake_a4, name, getattr(A, name))
    fake_a4.train = train
    core = types.ModuleType('core_if'); core.ISession = lambda c: types.SimpleNamespace(pilot=pilot, hp=None, dev='cpu')
    return c, data, ai, fixed, fake_a4, core, dp, V


@unittest.skipIf(A4DIR is None, 'attack4.py not found; pass --attack4-dir')
class TestCWorker(unittest.TestCase):
    def test_worker_replays_and_decomposes(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); root = Path(tmp.name)
        c, data, ai, fixed, fa4, core, dp, V = fake_world(root)
        with patch.object(C, 'load_context', lambda *x: (c, data, ai, fixed)), \
                patch.dict(sys.modules, {'attack4': fa4, 'core_if': core}):
            MC.worker(types.SimpleNamespace(repo='', out='', dest=root / 'c', task=0, classes='0,1,2,3',
                                            replay_tol=1e-12, score_tol=1e-8))
        meta = json.loads((root / 'c/seed42_checks.json').read_text())
        for cl, ck in meta['checks'].items():
            self.assertEqual(ck['J00_max_abs'], 0.0); self.assertEqual(ck['J10_max_abs'], 0.0)
            self.assertLess(ck['score_if_max_rel'], 1e-8, cl)
        with np.load(root / 'c/seed42.npz') as z:
            l00, l10, s = z['c1_l00'], z['c1_l10'], z['c1_score_if']
            truth = np.array([[r['J10'] - r['J00'] for r in data['v6_truth'][(j, 42)] if r['cls'] == 1] for j in range(4)]).T
            self.assertTrue(np.allclose((l10 - l00[:, None]).mean(-1), truth, atol=1e-15))
            want = np.einsum('jrc,arc->aj', dp, V[:, 1, 0])
            self.assertTrue(np.allclose(s.mean(1), want, rtol=1e-8))
        with patch.object(C, 'load_context', lambda *x: (c, data, ai, fixed)), \
                patch.dict(sys.modules, {'attack4': fa4, 'core_if': core}), self.assertRaises(FileExistsError):
            MC.worker(types.SimpleNamespace(repo='', out='', dest=root / 'c', task=0, classes='1', replay_tol=1e-12, score_tol=1e-8))


# ------------------------------------------------------------------ method C: analysis
def synthetic_results(root, nq=(30, 26), J=12, K=6, rng=None):
    rng = rng or np.random.default_rng(9)
    qx = {cls: softmax(rng.normal(size=(n, 4))) for cls, n in zip((1, 2), nq)}
    for s in C.PANEL:
        arrays = {}
        for cls, n in zip((1, 2), nq):
            p = f'c{cls}_'
            base = rng.normal(size=(n, J))
            arrays.update({p + 'query_index': np.arange(n) + 100 * cls, p + 'membership': np.tile([0, 1], n // 2),
                           p + 'q_x': qx[cls], p + 'l00': rng.random((K, n)), p + 'p00': rng.random((K, n))})
            arrays[p + 'l10'] = arrays[p + 'l00'][:, None, :] + np.transpose(base[None] + rng.normal(size=(K, n, J)) * .5, (0, 2, 1))
            for v in MC.VARIANTS:
                arrays[p + 'score_' + v] = base[None] * (1 + .1 * rng.random()) + rng.normal(size=(K, n, J))
        meta = dict(seed=s, gamma_a=.3, classes=[1, 2], qualified=[1, 2], omega={'1': nq[0] / sum(nq), '2': nq[1] / sum(nq)},
                    attack_seeds=list(range(K)), checks={})
        np.savez(root / f'seed{s}.npz', meta=json.dumps(meta), **arrays)


class TestCAnalysis(unittest.TestCase):
    def test_four_classes_csv_and_identity(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); root = Path(tmp.name)
        synthetic_results(root)
        for seed in C.PANEL:
            f = root / f'seed{seed}.npz'
            with np.load(f) as z:
                arrays = {k: z[k] for k in z.files}
            meta = json.loads(str(arrays['meta']))
            for cls, src in ((0, 1), (3, 2)):
                for key in list(arrays):
                    if key.startswith(f'c{src}_'):
                        arrays[f'c{cls}_' + key[3:]] = arrays[key].copy()
            meta['classes'] = [0, 1, 2, 3]; arrays['meta'] = json.dumps(meta)
            np.savez(f, **arrays)
        res, _ = MC.load_results(root)
        pool = MC.Pool(res, [1, 2]); sc, tr = pool.target(np.ones(len(pool.cls), bool))
        a = types.SimpleNamespace(dest=root, draws=2, seed=1, fracs=[.1])
        with patch.object(MC, 'REFERENCE', C.spearman(sc['if'], tr)):
            MC.analyze(a)
        rows = C.readcsv(root / 'c_results.csv')
        self.assertTrue(any(r['rule'] == 'class_only_CNV' and r['lds_if_CNV'] for r in rows))
        self.assertTrue(any(r['rule'] == 'class_only_NORMAL' and r['lds_if_NORMAL'] for r in rows))
        f = root / 'seed43.npz'
        with np.load(f) as z:
            arrays = {k: z[k] for k in z.files}
        meta = json.loads(str(arrays['meta'])); meta['seed'] = 42; arrays['meta'] = json.dumps(meta)
        np.savez(f, **arrays)
        with self.assertRaisesRegex(ValueError, 'declares seed'):
            MC.load_results(root)

    def test_pool_drop_random_and_run(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); root = Path(tmp.name)
        synthetic_results(root)
        res, meta = MC.load_results(root)
        pool = MC.Pool(res, [1, 2])
        sc, tr = pool.target(np.ones(len(pool.cls), bool))
        om = {1: 30 / 56, 2: 26 / 56}
        scf, trf = pool.target(np.ones(len(pool.cls), bool), om)
        self.assertTrue(np.allclose(sc['if'], scf['if']) and np.allclose(tr, trf))       # recount = original omega
        d1 = res[1]['d'].mean((0, 1)).mean(0); d2 = res[2]['d'].mean((0, 1)).mean(0)
        self.assertTrue(np.allclose(tr, om[1] * d1 + om[2] * d2))
        keep = MC.drop_within_strata(pool, pool.loss, .25, True)
        for st in np.unique(pool.strata):
            idx = pool.strata == st
            self.assertEqual(int((~keep[idx]).sum()), int(round(.25 * idx.sum())))
        rk = MC.random_keeps(pool, keep, 20, np.random.default_rng(0), pool.strata)
        for st in np.unique(pool.strata):
            idx = pool.strata == st
            self.assertTrue(np.all((~rk[:, idx]).sum(1) == (~keep[idx]).sum()))
        ref = C.spearman(sc['if'], tr)
        a = types.SimpleNamespace(dest=root, draws=50, seed=1, fracs=[.1, .5])
        with patch.object(MC, 'REFERENCE', ref):          # the real-data gate value, replaced for synthetic data
            MC.analyze(a)
        with patch.object(MC, 'REFERENCE', ref + .01), self.assertRaises(ValueError):
            MC.analyze(a)
        rows = C.readcsv(root / 'c_results.csv')
        self.assertTrue(any(r['rule'] == 'crossfit_drop_low_agreement' for r in rows))
        self.assertAlmostEqual(float(rows[0]['lds_if']), ref)
        one = next(r for r in rows if r['rule'] == 'class_only_DME')
        s1, t1 = MC.Pool(res, [1]).target(np.ones(30, bool))
        self.assertAlmostEqual(float(one['lds_if']), C.spearman(s1['if'], t1))


# ------------------------------------------------------------------ method B
class TestB(unittest.TestCase):
    def test_v2_result_identity(self):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup); root = Path(tmp.name)
        d = root / 'g'; d.mkdir(); w = np.array([.6, .4]); h = 'abc'
        def write(file_seed, **kw):
            fields = dict(group='g', seed=file_seed, n_removed=5, manifest_sha256=h, weights=w,
                          score_by_class=np.ones((2, 32)), truth_by_class=np.ones((2, 32)))
            fields.update(kw); np.savez(d / f'seed{file_seed}.npz', **fields)
        for seed in C.PANEL: write(seed)
        self.assertEqual(MB.load_group(root, 'g', 5, manifest_sha256=h, weights=w)[0].shape, (2, 32))
        for kw in (dict(group='wrong'), dict(seed=999), dict(manifest_sha256='wrong'),
                   dict(weights=np.array([.5, .5])), dict(score_by_class=np.ones((2, 31))),
                   dict(score_by_class=np.full((2, 32), np.nan))):
            write(43, **kw)
            with self.assertRaises(ValueError): MB.load_group(root, 'g', 5, manifest_sha256=h, weights=w)
            write(43)

    def test_describe_and_summary(self):
        rng = np.random.default_rng(2)
        bx, by = rng.normal(size=10), rng.normal(size=10)
        r = MB.describe('g', 'k', 5, 'combined', bx, by, bx, by)
        self.assertAlmostEqual(r['truth_vs_none'], 1.0); self.assertAlmostEqual(r['truth_mean_over_none'], 1.0)
        rows = [MB.describe('cand', 'oracle', 5, 'combined', rng.normal(size=10), by, bx, by)]
        rows += [MB.describe(f'cand_random{k}', 'random', 5, 'combined', rng.normal(size=10), by, bx, by) for k in (1, 2, 3)]
        s = MB.summarise(rows, 'cand', [f'cand_random{k}' for k in range(1, 6)], 'combined')
        self.assertEqual(s['controls_found'], 3)
        lds = [x['lds'] for x in rows[1:]]
        self.assertAlmostEqual(s['control_lds_min'], min(lds)); self.assertAlmostEqual(s['control_lds_max'], max(lds))


if __name__ == '__main__':
    unittest.main(verbosity=1)
