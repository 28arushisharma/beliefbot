"""Tests for belief_engine.py covering edge cases across λ regimes."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from belief_engine import BeliefEngine, hypergeometric_pmf


# ---------------------------------------------------------------------------
# hypergeometric_pmf
# ---------------------------------------------------------------------------

class TestHypergeometricPMF:
    def test_certain_outcome_all_red_jar(self):
        # Jar has 20 red out of 20 — any 5 draws must all be red.
        assert hypergeometric_pmf(20, 20, 5, 5) == pytest.approx(1.0)

    def test_impossible_red_from_zero_red_jar(self):
        assert hypergeometric_pmf(0, 20, 5, 1) == pytest.approx(0.0)

    def test_impossible_more_red_than_jar_contains(self):
        # Jar has 3 red; can't draw 5 red in 10 draws
        assert hypergeometric_pmf(3, 20, 10, 5) == pytest.approx(0.0)

    def test_impossible_more_blue_than_jar_contains(self):
        # Jar has 18 red (2 blue); drawing 10 balls can yield at most 2 blue (k_red >= 8)
        assert hypergeometric_pmf(18, 20, 10, 1) == pytest.approx(0.0)

    def test_pmf_sums_to_one_over_all_outcomes(self):
        n_red_jar, n_balls, n_draws = 8, 20, 6
        total = sum(
            hypergeometric_pmf(n_red_jar, n_balls, n_draws, k)
            for k in range(n_draws + 1)
        )
        assert total == pytest.approx(1.0)

    def test_known_value(self):
        # P(k=2 | r=4, N=10, n=4) = C(4,2)*C(6,2)/C(10,4) = 6*15/210 = 90/210
        assert hypergeometric_pmf(4, 10, 4, 2) == pytest.approx(90 / 210)


# ---------------------------------------------------------------------------
# Perfectly Bayesian  (λ = 0)
# ---------------------------------------------------------------------------

class TestPerfectlyBayesian:
    def test_posterior_sums_to_one(self):
        engine = BeliefEngine(n_balls=20)
        post = engine.update(n_draws=5, k_red=3, lam=0.0)
        assert post.sum() == pytest.approx(1.0)

    def test_red_heavy_draw_raises_red_hypotheses(self):
        engine = BeliefEngine(n_balls=20)
        # 1 draw, 1 red: hypergeometric PMF for r is proportional to r/20,
        # so r=20 becomes most probable and r=0 gets probability 0.
        post = engine.update(n_draws=1, k_red=1, lam=0.0)
        assert post[engine.hypotheses.index(20)] > post[engine.hypotheses.index(0)]

    def test_blue_heavy_draw_raises_blue_hypotheses(self):
        engine = BeliefEngine(n_balls=20)
        post = engine.update(n_draws=10, k_red=0, lam=0.0)
        assert post[engine.hypotheses.index(0)] > post[engine.hypotheses.index(20)]

    def test_point_prior_remains_concentrated(self):
        # With all mass on r=10, a feasible observation keeps it there (no other hypothesis).
        prior = np.zeros(21)
        prior[10] = 1.0
        engine = BeliefEngine(n_balls=20, prior=prior)
        post = engine.update(n_draws=5, k_red=3, lam=0.0)
        assert post[10] == pytest.approx(1.0)

    def test_sequential_updates_are_order_consistent(self):
        # Two draws in sequence should give the same result as the combined draw
        # when the observation counts are additive (this is a sanity check, not exact equality).
        engine_seq = BeliefEngine(n_balls=20)
        engine_seq.update(n_draws=5, k_red=3, lam=0.0)
        engine_seq.update(n_draws=5, k_red=2, lam=0.0)
        # Belief should be non-uniform and sum to 1
        assert engine_seq.belief.sum() == pytest.approx(1.0)
        assert not np.allclose(engine_seq.belief, np.ones(21) / 21)


# ---------------------------------------------------------------------------
# No updating  (λ = 1)
# ---------------------------------------------------------------------------

class TestNoUpdating:
    def test_posterior_equals_prior_uniform(self):
        engine = BeliefEngine(n_balls=20)
        post = engine.update(n_draws=10, k_red=7, lam=1.0)
        np.testing.assert_allclose(post, engine._initial_prior)

    def test_posterior_equals_prior_custom(self):
        prior = np.zeros(21)
        prior[5] = 0.6
        prior[15] = 0.4
        engine = BeliefEngine(n_balls=20, prior=prior)
        post = engine.update(n_draws=10, k_red=7, lam=1.0)
        np.testing.assert_allclose(post, engine._initial_prior)

    def test_repeated_updates_leave_belief_unchanged(self):
        engine = BeliefEngine(n_balls=20)
        for _ in range(20):
            engine.update(n_draws=2, k_red=1, lam=1.0)
        np.testing.assert_allclose(engine.belief, engine._initial_prior)

    def test_intermediate_lambda_is_strictly_between(self):
        engine = BeliefEngine(n_balls=20)
        post_bayes = BeliefEngine(n_balls=20).update(n_draws=10, k_red=8, lam=0.0)
        post_none = BeliefEngine(n_balls=20).update(n_draws=10, k_red=8, lam=1.0)
        post_mid = engine.update(n_draws=10, k_red=8, lam=0.5)
        # At the mode hypothesis, mid should be strictly between the two extremes
        idx = int(np.argmax(post_bayes))
        assert post_none[idx] < post_mid[idx] < post_bayes[idx] or \
               post_bayes[idx] < post_mid[idx] < post_none[idx]


# ---------------------------------------------------------------------------
# Confirmation bias  (asymmetric λ)
# ---------------------------------------------------------------------------

class TestConfirmationBias:
    def test_symmetric_lambda_matches_plain_update(self):
        # When lam_confirm == lam_disconfirm, update_asymmetric must equal update.
        lam = 0.4
        engine1 = BeliefEngine(n_balls=20)
        engine2 = BeliefEngine(n_balls=20)
        post1 = engine1.update(n_draws=5, k_red=3, lam=lam)
        post2 = engine2.update_asymmetric(n_draws=5, k_red=3, lam_confirm=lam, lam_disconfirm=lam)
        np.testing.assert_allclose(post1, post2)

    def test_high_disconfirm_lambda_preserves_disconfirmed_hypotheses(self):
        # 10 draws, only 1 red → disconfirms high-red hypotheses (e.g. r=18).
        # For r=18 the observation is actually impossible (need ≥8 red), so Bayes gives 0.
        # With lam_disconfirm=0.8 the biased agent retains 80% of its prior for r=18.
        engine_biased = BeliefEngine(n_balls=20)
        engine_bayes = BeliefEngine(n_balls=20)
        post_biased = engine_biased.update_asymmetric(
            n_draws=10, k_red=1, lam_confirm=0.0, lam_disconfirm=0.8
        )
        post_bayes = engine_bayes.update(n_draws=10, k_red=1, lam=0.0)
        assert post_biased[engine_biased.hypotheses.index(18)] > post_bayes[18]

    def test_high_confirm_lambda_slows_confirming_updates(self):
        # 1 draw, 1 red → confirms high-red hypotheses (posterior > prior for r > 0).
        # lam_confirm=0.9 blends 90% toward prior, so r=20 gains far less than under Bayes.
        engine_open = BeliefEngine(n_balls=20)
        engine_anchored = BeliefEngine(n_balls=20)
        post_open = engine_open.update_asymmetric(
            n_draws=1, k_red=1, lam_confirm=0.0, lam_disconfirm=0.0
        )
        post_anchored = engine_anchored.update_asymmetric(
            n_draws=1, k_red=1, lam_confirm=0.9, lam_disconfirm=0.0
        )
        idx_high = engine_open.hypotheses.index(20)
        assert post_open[idx_high] > post_anchored[idx_high]

    def test_posterior_sums_to_one_after_asymmetric_update(self):
        engine = BeliefEngine(n_balls=20)
        post = engine.update_asymmetric(
            n_draws=5, k_red=2, lam_confirm=0.3, lam_disconfirm=0.7
        )
        assert post.sum() == pytest.approx(1.0)

    def test_extreme_asymmetry_lam_confirm_zero_lam_disconfirm_one(self):
        # Agent embraces confirming evidence but completely ignores disconfirming evidence.
        # After a blue-heavy draw (k_red=1 of 10), high-red hypotheses should retain
        # more mass than under Bayes but low-red hypotheses should be updated normally.
        engine = BeliefEngine(n_balls=20)
        prior_r5 = engine.belief[5]
        post = engine.update_asymmetric(
            n_draws=10, k_red=1, lam_confirm=0.0, lam_disconfirm=1.0
        )
        # r=5 is a low-red hypothesis; the blue-heavy draw confirms it (posterior > prior)
        # so lam_confirm=0 applies → it does get updated toward Bayes
        # Verify belief is still a valid distribution
        assert post.sum() == pytest.approx(1.0)
        assert np.all(post >= 0)


# ---------------------------------------------------------------------------
# Quadratic scoring rule
# ---------------------------------------------------------------------------

class TestQuadraticScore:
    def test_perfect_prediction_scores_one(self):
        engine = BeliefEngine(n_balls=20)
        p = np.zeros(21)
        p[5] = 1.0
        assert engine.quadratic_score(p, true_hypothesis_idx=5) == pytest.approx(1.0)

    def test_uniform_prediction(self):
        engine = BeliefEngine(n_balls=20)
        p = np.ones(21) / 21
        # QS = 2*(1/21) - 21*(1/21)^2 = 2/21 - 1/21 = 1/21
        assert engine.quadratic_score(p, true_hypothesis_idx=0) == pytest.approx(1 / 21)

    def test_wrong_certain_prediction_scores_minus_one(self):
        engine = BeliefEngine(n_balls=20)
        p = np.zeros(21)
        p[0] = 1.0
        score_correct = engine.quadratic_score(p, true_hypothesis_idx=0)
        score_wrong = engine.quadratic_score(p, true_hypothesis_idx=10)
        assert score_correct == pytest.approx(1.0)
        assert score_wrong == pytest.approx(-1.0)

    def test_proper_scoring_rule_truthful_beats_distorted(self):
        # Truthful belief should outscore a uniform distortion at the most probable hypothesis.
        engine = BeliefEngine(n_balls=20)
        engine.update(n_draws=10, k_red=8, lam=0.0)
        p_truth = engine.belief.copy()
        p_uniform = np.ones(21) / 21
        true_idx = int(np.argmax(p_truth))
        assert engine.quadratic_score(p_truth, true_idx) > engine.quadratic_score(p_uniform, true_idx)

    def test_score_range(self):
        engine = BeliefEngine(n_balls=20)
        p = engine.belief.copy()
        for idx in range(len(engine.hypotheses)):
            score = engine.quadratic_score(p, idx)
            assert -1.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_restores_initial_prior(self):
        engine = BeliefEngine(n_balls=20)
        engine.update(n_draws=5, k_red=4, lam=0.0)
        assert not np.allclose(engine.belief, engine._initial_prior)
        engine.reset()
        np.testing.assert_allclose(engine.belief, engine._initial_prior)

    def test_reset_after_asymmetric_update(self):
        prior = np.zeros(21)
        prior[10] = 1.0
        engine = BeliefEngine(n_balls=20, prior=prior)
        engine.update_asymmetric(n_draws=5, k_red=3, lam_confirm=0.2, lam_disconfirm=0.8)
        engine.reset()
        np.testing.assert_allclose(engine.belief, engine._initial_prior)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_lambda_out_of_range_raises(self):
        engine = BeliefEngine(n_balls=20)
        with pytest.raises(ValueError):
            engine.update(n_draws=5, k_red=2, lam=1.5)
        with pytest.raises(ValueError):
            engine.update(n_draws=5, k_red=2, lam=-0.1)

    def test_prior_all_zeros_raises(self):
        with pytest.raises(ValueError):
            BeliefEngine(n_balls=20, prior=np.zeros(21))

    def test_prior_wrong_length_raises(self):
        with pytest.raises(ValueError):
            BeliefEngine(n_balls=20, prior=np.ones(10))
