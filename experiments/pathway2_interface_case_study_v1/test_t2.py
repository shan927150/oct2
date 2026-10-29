"""Local numerical tests for logit diagnostics and feature alignment. No GPU or OCT data.
These are optional developer checks, not a Slurm prerequisite.

    python -I test_t2.py <path to experiments/pathway2_interface_lds_v1 of the frozen checkout>
"""
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
IF_DIR = sys.argv.pop(1)
sys.path.insert(0, IF_DIR)
import logit_diag as LD            # noqa: E402
import case_features as CF         # noqa: E402


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ------------------------------------------------------------------ logit_diag
def synthetic_run(seed=0, N=60, nm=40, J=8):
    rng = np.random.default_rng(seed)
    R = N + nm
    y = rng.integers(0, 4, R)
    M = (rng.random((J, N)) < .5).astype(float)
    M[0], M[1] = 0, 0                              # every image is deleted in one of subsets 0, 1 and retained in the other
    M[0, :N // 2], M[1, N // 2:] = 1, 1
    z0 = rng.normal(size=(R, 4))
    z0[np.arange(R), y] += 12 + rng.normal(size=R) * 3
    dz = np.zeros((J, R, 4))
    dele = np.c_[M == 0, np.zeros((J, nm), bool)]
    c = rng.uniform(1, 3, size=(J, R))
    dz[dele, y[np.nonzero(dele)[1]]] = -c[dele]
    dz[:, :N][M == 1] += .05 * rng.normal(size=(int((M == 1).sum()), 4))
    dz[:, N:] += .2 * rng.normal(size=(J, nm, 4))
    full = softmax(z0)
    seeds, panel, gammas = [42, 43, 44], [42, 43], [.3, .03]
    subset = {s: softmax(z0 + dz + 1e-3 * rng.normal(size=dz.shape)) for s in seeds}
    truth = np.stack([subset[s] - full for s in seeds])
    K = M.sum(1)[:, None, None]
    solves = {}
    for s in panel:
        dz_if = np.stack([(K / N) * 1e-4 * dz, (K / N) * 1e-3 * dz], 1)             # matched IF = 1e-3 dz at gamma .03 (gamma .3: 1e-4)
        solves[s] = dict(arrays=dict(p0=full, dz_if=dz_if, dz_gall=np.zeros((2, R, 4)), dz_proxy=0 * dz,
                                     dz_proxy_gall=np.zeros((R, 4)), dz_true=.1 * dz, vbar=None, gammas=np.array(gammas)),
                         result=dict(gammas=[dict(gamma=g, qualified=True) for g in gammas]))
    vb = np.zeros((R, 4))
    vb[(y == 1) | (y == 2)] = rng.normal(size=(int(((y == 1) | (y == 2)).sum()), 4))
    for s in panel:
        solves[s]['arrays']['vbar'] = vb
    n_act = ((M == 0) & ((y[:N] == 1) | (y[:N] == 2))[None, :]).sum(1)
    v6 = {(j, s): [dict(cls=k, J10=.01 * n_act[j] + .02 * rng.normal(), J00=0.) for k in range(4)] for j in range(J) for s in panel}
    data = dict(frozen=dict(score_panel=panel), rows=dict(raw_index=np.arange(R), y=y), M=M, truth=truth,
                full={s: full for s in seeds}, subset=subset, solves=solves,
                vbar={s: vb * (1 + .03 * (s - 42)) for s in seeds},
                fr6=dict(omega={'1': .635, '2': .365}, attack_seeds=[7, 8]), v6_truth=v6)
    return data, dict(seeds=seeds, gammas=gammas, primary_gamma=.03, design_name='synthetic'), (dz, y, M, vb, n_act)


def subset_reader(data, c):
    lookup = {f'{c["design_name"]}_s{j:03d}_seed{s}': p
              for s, subset in data['subset'].items() for j, p in enumerate(subset)}
    def read(config, name, rows):
        np.testing.assert_array_equal(rows, data['rows']['raw_index'])
        return lookup[name]
    return read


class LogitDiag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.c, cls.plant = synthetic_run()
        with patch('analysis_if.v3_interface', side_effect=subset_reader(cls.data, cls.c)):
            cls.rungs, cls.own, cls.sweep, cls.panel, cls.primary = LD.compute(cls.data, cls.c)

    def row(self, group, pred, ref, decile='all'):
        r = self.rungs
        return r[(r.group == group) & (r.margin_decile == decile) & (r.pred == pred) & (r.ref == ref)].iloc[0]

    def test_relative_logit_is_invariant_and_exact(self):
        rng = np.random.default_rng(3)
        z, y = rng.normal(size=(5, 4)), np.array([0, 1, 2, 3, 1])
        u = LD.rel_logit(softmax(z), y)
        self.assertTrue(np.allclose(u, LD.rel_tangent(z, y)))
        self.assertTrue(np.allclose(LD.rel_logit(softmax(z + 7.), y), u))
        self.assertTrue(np.allclose(LD.drop_true(u, y).shape, (5, 3)) and (LD.drop_true(u, y) != 0).all())

    def test_ratios_recover_the_planted_scales(self):
        r = self.row('deleted', self.primary, 'truth')
        self.assertAlmostEqual(r.norm_ratio, 1e-3, delta=3e-5)            # IF = 1e-3 of the change
        self.assertAlmostEqual(r.ls_scale, 1e3, delta=30)                 # it has to be multiplied by 1000
        self.assertGreater(r.within_subset_pearson, .99)
        self.assertAlmostEqual(r.rel_error, .999, delta=3e-5)
        self.assertAlmostEqual(self.row('deleted', 'true dtheta (linear)', 'truth').norm_ratio, .1, delta=3e-3)
        self.assertAlmostEqual(self.row('deleted', self.primary, 'true dtheta (linear)').norm_ratio, 1e-2, delta=3e-4)
        g3 = self.rungs[(self.rungs.pred == 'IF (gamma=0.3)') & (self.rungs.group == 'deleted')].iloc[0]
        self.assertAlmostEqual(g3.norm_ratio, 1e-4, delta=3e-6)
        self.assertEqual(set(self.rungs.group), {'deleted', 'retained', 'nonmember'})

    def test_own_effect_in_margin_space(self):
        dz, y, M, vb, _ = self.plant
        N = M.shape[1]
        own = self.own
        c_del = np.array([np.where(M[:, i] == 0, -(dz[:, i, y[i]]), np.nan) for i in range(N)])
        planted = -np.nanmean(c_del, axis=1)                         # margin falls by the planted logit change
        self.assertGreater(np.corrcoef(own.own_true, planted)[0, 1], .99)
        self.assertAlmostEqual(float(np.median(own.own_if / own.own_true)), 1e-3, delta=2e-4)
        summ = LD.own_summary(own)
        self.assertEqual(int(summ.n.sum()), N)
        self.assertTrue(own.valid_contrast.all())
        np.testing.assert_array_equal(own.raw_index, self.data['rows']['raw_index'][:N])

    def test_bridge_rows_match_direct_computation(self):
        data, c = self.data, self.c
        import analysis_if as ai
        P_all, _, panel = ai.predictions(data, c)
        si = [c['seeds'].index(s) for s in panel]
        P, T = P_all[self.primary], data['truth'][si]
        vb, M = np.stack([data['vbar'][s] for s in panel]), data['M']
        J, N = M.shape
        R = P.shape[2]
        Ta = LD.attack_truth(data, panel)
        dele = np.c_[M == 0, np.zeros((J, R - N), bool)]
        score1 = np.array([np.mean([(vb[s] * P[s, j]).sum() for s in range(len(panel))]) for j in range(J)])
        mix = P.copy()
        mix[:, dele] = T[:, dele]
        score3 = np.array([np.mean([(vb[s] * mix[s, j]).sum() for s in range(len(panel))]) for j in range(J)])
        sw = self.sweep.set_index('label')
        self.assertAlmostEqual(sw.loc['row 1: all IF', 'lds'], LD.rho(score1, Ta), places=12)
        self.assertAlmostEqual(sw.loc['row 3: deleted rows <- retrained', 'lds'], LD.rho(score3, Ta), places=12)
        a1 = self.sweep[(self.sweep.label == 'IF deleted part x alpha') & (self.sweep.alpha == 1.)].lds.iloc[0]
        self.assertAlmostEqual(a1, LD.rho(score1, Ta), places=12)          # alpha = 1 is row 1

    def test_original_probabilities_survive_subtraction_cancellation(self):
        full = np.array([[.99, .003, .003, .004]])
        subset = np.array([[[1e-30, .2, .3, .5]]])
        data = dict(full={42: full}, subset={42: subset}, M=np.zeros((1, 1)),
                    rows=dict(raw_index=np.array([71]), y=np.array([0])))
        c = dict(design_name='tiny')
        self.assertEqual((full + (subset[0] - full))[0, 0], 0.)
        with patch('analysis_if.v3_interface', side_effect=subset_reader(data, c)):
            _, actual = LD.read_logit_truth(data, c, [42])
        expected = np.log(.2 / 1e-30) - np.log(.003 / .99)
        self.assertAlmostEqual(actual[0, 0, 1], expected, places=12)
        self.assertLess(actual.max(), 100.)

    def test_zero_probability_uses_common_missing_vectors(self):
        data, c, _ = synthetic_run()
        y = data['rows']['y'][0]
        raw = data['subset'][42][0, 0]
        raw[y] = 0.
        raw /= raw.sum()
        data['truth'][0, 0, 0] = raw - data['full'][42][0]
        with patch('analysis_if.v3_interface', side_effect=subset_reader(data, c)):
            rungs, own, _, _, _ = LD.compute(data, c)
        rows = rungs[(rungs.group == 'retained') & (rungs.margin_decile == 'all')]
        self.assertTrue((rows.n_missing_vectors == 1).all())
        self.assertTrue((rungs.n_vectors + rungs.n_missing_vectors == rungs.n_total_vectors).all())
        self.assertFalse(own.loc[0, 'valid_contrast'])
        self.assertEqual(int(LD.own_summary(own).n_missing.sum()), 1)

    def test_figure_and_summary_run(self):
        with tempfile.TemporaryDirectory() as t:
            LD.figures(self.rungs, self.own, self.sweep, self.primary, Path(t) / 'f.png')
            self.assertGreater((Path(t) / 'f.png').stat().st_size, 5000)


# ------------------------------------------------------------------ case_features
class CaseFeatures(unittest.TestCase):
    def test_features_align_by_image_and_keep_target_populations(self):
        rng = np.random.default_rng(5)
        n = 600
        margin = rng.uniform(5, 45, n)
        g = np.exp(-margin + rng.normal(size=n) * .3)
        cls = rng.integers(0, 4, n)
        active = (cls == 1) | (cls == 2)
        img = pd.DataFrame(dict(train_position=np.arange(n), raw_index=np.arange(n) + 10, cls=cls,
                                loss=g, margin=margin, attack_row_active=active,
                                a_if=active * g ** 2 * rng.choice([-1, 1], n),
                                a_true=active * rng.choice([-1, 1], n) *
                                np.exp(-margin / 25 + rng.normal(size=n)),
                                removal_score=g * rng.normal(size=n)))
        img['gap_share'] = img.a_true - img.a_if
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / 'gamma0.03_k100'
            (Path(t) / 'features').mkdir()
            d.mkdir()
            img.to_csv(d / 'images.csv', index=False)
            for seed in (42, 43, 44, 49):
                f = img[CF.KEYS].copy()
                f['grad_norm'] = g * (1 + .01 * seed)
                f['self_influence_g0_03'] = g ** 2
                f.sample(frac=1, random_state=seed).to_csv(
                    Path(t) / 'features' / f'features_seed{seed}.csv', index=False)
            full, cols, files = CF.load(d)
            self.assertEqual(files, [f'features_seed{s}.csv' for s in (42, 43, 44)])
            self.assertEqual(cols, ['grad_norm'])
            np.testing.assert_allclose(full.grad_norm, g * 1.43)
            s = CF.summary(full, cols).set_index(['population', 'target', 'feature'])
            self.assertGreater(s.loc[('attack active', 'a_if', 'grad_norm'), 'spearman_abs_score'], .99)
            self.assertLess(s.loc[('attack active', 'a_true', 'grad_norm'), 'spearman_abs_score'], .8)
            self.assertEqual(s.loc[('all training', 'removal_score', 'grad_norm'), 'n_images'], n)
            self.assertEqual(s.loc[('DME', 'gap_share', 'grad_norm'), 'n_images'], int((cls == 1).sum()))
            self.assertTrue(np.isnan(CF.partial_rho(margin, g, margin)))
            grp = CF.groups(full, cols, 100, 20)
            self.assertEqual(len(grp), 9)
            CF.figure(full, d / 'f.png')
            self.assertGreater((d / 'f.png').stat().st_size, 5000)


if __name__ == '__main__':
    unittest.main(verbosity=2)
