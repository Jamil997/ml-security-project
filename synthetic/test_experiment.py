import itertools
import unittest
import numpy as np
from experiment import (allocate, estimates, bounds, make_pool, attack_candidates,
                        search_attack, fit_surrogate, predict_labels, contrasts)

class SyntheticAuditTests(unittest.TestCase):
    def test_expected_budget_and_uniform_floor(self):
        q = np.array([1e-8, 0.01, 0.1, 1, 100.])
        for b, mix in itertools.product([.05, .2, .8, 1.], [0., .25, 1.]):
            p = allocate(q, b, mix)
            self.assertAlmostEqual(p.mean(), b, places=12)
            self.assertTrue(np.all(p > 0))
            self.assertTrue(np.all(p <= 1))
            self.assertTrue(np.all(p >= mix * b - 1e-12))

    def test_exhaustive_unbiasedness_and_exact_variance(self):
        # Enumerate ALL 2^4 audit subsets with their exact probabilities.
        y = np.array([[0,1,0],[1,1,0],[0,0,1],[1,0,1]], dtype=float)
        f = np.array([[.1,.7,.2],[.9,.8,.7],[.1,.5,.8],[.8,.3,.6]])
        p = np.array([.1,.3,.5,.9])
        outcomes, probabilities = [], []
        variance_estimates = []
        for bits in itertools.product([0,1], repeat=4):
            bits = np.array(bits)
            # 0 forces an audit, 1 forbids one for probabilities below 1.
            coins = (1-bits)[None,:].astype(float)
            est,var,_ = estimates(y,f,p,coins)
            outcomes.append(est[0]); variance_estimates.append(var[0])
            probabilities.append(np.prod(np.where(bits,p,1-p)))
        outcomes=np.array(outcomes); probabilities=np.array(probabilities)
        truth=contrasts(y).mean(axis=0)
        mean=probabilities @ outcomes
        np.testing.assert_allclose(mean,truth,atol=1e-12)
        exact_var=probabilities @ (outcomes-truth)**2
        formula=np.sum((1/p[:,None]-1)*(contrasts(y)-contrasts(f))**2,axis=0)/len(p)**2
        np.testing.assert_allclose(exact_var,formula,atol=1e-12)
        np.testing.assert_allclose(probabilities @ np.array(variance_estimates),formula,atol=1e-12)

    def test_complete_audit_exact_recovery(self):
        pool=make_pool(100,77)
        est,var,_=estimates(pool.labels,pool.scores,np.ones(100),np.zeros((5,100)))
        np.testing.assert_allclose(est,np.tile(contrasts(pool.labels).mean(axis=0),(5,1)),atol=1e-12)
        np.testing.assert_allclose(var,0)
        np.testing.assert_allclose(bounds(est,var,np.ones(100),'bernstein'),est)

    def test_attack_budget_and_truth_preservation(self):
        pool=make_pool(101,99)
        original_y=pool.labels.copy(); original_f=pool.scores.copy()
        attacks=attack_candidates(pool.scores,pool.priorities,pool.group,.1,99,60)
        for f,q,meta in attacks:
            changed=np.any(f!=pool.scores,axis=1)|(q!=pool.priorities)
            self.assertLessEqual(changed.sum(),10)
            np.testing.assert_array_equal(f[:,:2],pool.scores[:,:2])
            self.assertTrue(np.all((f>=0)&(f<=1)))
        np.testing.assert_array_equal(original_y,pool.labels)
        np.testing.assert_array_equal(original_f,pool.scores)

    def test_search_has_no_test_gold_dependency_and_is_reproducible(self):
        pool=make_pool(100,123)
        calibration=make_pool(300,123,1)
        surrogate=predict_labels(pool.scores,pool.group,fit_surrogate(calibration))
        attacks=attack_candidates(pool.scores,pool.priorities,pool.group,.1,123,12)
        coins=np.random.default_rng(4).random((32,100))
        a=search_attack(attacks,surrogate,.2,.25,coins,'harm')
        pool.labels[:]=1-pool.labels # evaluator-only truth cannot affect search
        b=search_attack(attacks,surrogate,.2,.25,coins,'harm')
        self.assertEqual(a[2],b[2])
        np.testing.assert_array_equal(a[0],b[0])

if __name__=='__main__': unittest.main()
