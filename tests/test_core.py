"""Tests of the BAM model's mathematical properties.

These tests do not merely check that the code runs: they verify the
propositions the study rests on — Kosko stability, exact recall with orthogonal
patterns, the limiting cases of the least-squares rule, and the
prototype/majority-vote identity.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import hadamard

from bam.core import BAM, RULES, _least_squares_rule, sign


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


def bipolar(rng, *shape) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0]), size=shape)


# --- utilities --------------------------------------------------------------
def test_sign_breaks_ties_with_the_previous_state():
    local_field = np.array([3.0, -2.0, 0.0, 0.0])
    previous = np.array([1.0, 1.0, 1.0, -1.0])
    assert np.array_equal(sign(local_field, previous), [1.0, -1.0, 1.0, -1.0])


def test_fit_rejects_non_bipolar_input(rng):
    X = rng.normal(size=(8, 3))
    Y = bipolar(rng, 4, 3)
    with pytest.raises(ValueError, match="only -1 and"):
        BAM("hebb").fit(X, Y)


def test_fit_rejects_mismatched_pattern_counts(rng):
    with pytest.raises(ValueError, match="must match"):
        BAM("hebb").fit(bipolar(rng, 8, 3), bipolar(rng, 4, 5))


def test_unknown_rule():
    with pytest.raises(ValueError, match="unknown rule"):
        BAM("nonexistent")


# --- Kosko stability --------------------------------------------------------
@pytest.mark.parametrize("rule", RULES)
def test_energy_never_rises(rng, rule):
    """Kosko's theorem (1988): E is a Lyapunov function of the dynamics."""
    X, Y = bipolar(rng, 64, 6), bipolar(rng, 16, 6)
    model = BAM(rule, max_iter=25).fit(X, Y)
    for _ in range(20):
        traj = model.recall(x0=bipolar(rng, 64, 1)[:, 0])
        assert traj.energy_is_monotone(), f"{rule}: energy went up"


@pytest.mark.parametrize("rule", RULES)
def test_recall_always_converges(rng, rule):
    X, Y = bipolar(rng, 64, 5), bipolar(rng, 16, 5)
    model = BAM(rule, max_iter=50).fit(X, Y)
    for _ in range(10):
        assert model.recall(x0=bipolar(rng, 64, 1)[:, 0]).converged


# --- exactness with orthogonal patterns -------------------------------------
def test_hebb_recalls_orthogonal_patterns_exactly():
    """With no crosstalk, the Hebb rule is exact in both directions."""
    X = hadamard(32).astype(float)[:, 1:6]  # 5 orthogonal columns
    Y = hadamard(16).astype(float)[:, 1:6]
    model = BAM("hebb").fit(X, Y)
    assert np.array_equal(model.associate_forward(X), Y)
    assert np.array_equal(model.associate_backward(Y), X)
    assert model.stable_pairs().all()


def test_hadamard_codebook_cancels_backward_crosstalk(rng):
    """With YᵀY = p·I, `sgn(W y_j) = x_j` for any set of faces X."""
    Y = hadamard(64).astype(float)[1:21].T  # 20 orthogonal codes
    X = bipolar(rng, 500, 20)  # X arbitrary, correlated or not
    model = BAM("hebb").fit(X, Y)
    assert np.array_equal(model.associate_backward(Y), X)


# --- least-squares rule -----------------------------------------------------
def _proportional(A: np.ndarray, B: np.ndarray, tol: float = 1e-8) -> bool:
    """`A == α·B` for some α > 0.

    The least-squares rule solves the problem in the normalised variables
    `X/√n` and `Y/√p`, so `W` comes out up to a global factor `√(n/p)`. The
    scale is irrelevant to the network — recall only uses `sgn` — but the
    exactness tests have to account for it.
    """
    alpha = float(np.sum(A * B) / np.sum(B * B))
    return alpha > 0 and bool(np.allclose(A, alpha * B, atol=tol))


def test_least_squares_reduces_to_hebb_when_everything_is_orthogonal():
    """The sanity case documented in `_least_squares_rule`."""
    X = hadamard(64).astype(float)[:, 1:9]
    Y = hadamard(16).astype(float)[:, 1:9]
    W = _least_squares_rule(X, Y, lam=0.5)
    hebb = X @ Y.T
    scale = np.sum(W * hebb) / np.sum(hebb * hebb)
    assert np.allclose(W, scale * hebb, atol=1e-8)


def test_least_squares_with_lambda_one_is_exact_forward(rng):
    """λ = 1 is the classical pseudo-inverse: WᵀX ∝ Y exactly."""
    X, Y = bipolar(rng, 200, 12), bipolar(rng, 32, 12)
    W = _least_squares_rule(X, Y, lam=1.0)
    assert _proportional(W.T @ X, Y)
    assert np.array_equal(np.sign(W.T @ X), Y)


def test_least_squares_with_lambda_zero_is_exact_backward(rng):
    """λ = 0 mirrors the previous case: WY ∝ X exactly."""
    X, Y = bipolar(rng, 200, 12), bipolar(rng, 32, 12)
    W = _least_squares_rule(X, Y, lam=0.0)
    assert _proportional(W @ Y, X)
    assert np.array_equal(np.sign(W @ Y), X)


def test_least_squares_rejects_lambda_out_of_range(rng):
    with pytest.raises(ValueError, match="lam must lie"):
        _least_squares_rule(bipolar(rng, 20, 3), bipolar(rng, 8, 3), lam=1.5)


# --- multiple training ------------------------------------------------------
def test_multiple_training_stabilises_easy_patterns(rng):
    X, Y = bipolar(rng, 400, 4), bipolar(rng, 64, 4)
    model = BAM("multiple_training", max_mt_epochs=100).fit(X, Y)
    assert model.mt_converged_
    assert model.stable_pairs().all()
    assert (model.q_ >= 1).all()


def test_multiple_training_reports_non_convergence(rng):
    """Overloaded, the rule must admit failure instead of hiding it.

    This is exactly what happens with real faces in the experiments: recording
    the non-convergence is part of the result.
    """
    X, Y = bipolar(rng, 32, 40), bipolar(rng, 8, 40)  # P far above capacity
    model = BAM("multiple_training", max_mt_epochs=3).fit(X, Y)
    assert model.mt_converged_ is False
    assert not model.stable_pairs().all()


# --- dense rule -------------------------------------------------------------
def test_dense_stores_far_more_than_the_rank(rng):
    """The dense memory's capacity is not bounded by min(n, p)."""
    n, p, P = 64, 16, 120  # P far above min(n, p) = 16
    X, Y = bipolar(rng, n, P), bipolar(rng, p, P)
    model = BAM("dense", beta=30.0).fit(X, Y)
    assert np.array_equal(model.associate_forward(X), Y)
    assert np.array_equal(model.associate_backward(Y), X)


def test_dense_builds_no_weight_matrix(rng):
    model = BAM("dense").fit(bipolar(rng, 32, 4), bipolar(rng, 8, 4))
    assert model.W is None
    assert model.linear is False


# --- consistency between the two recall implementations ---------------------
@pytest.mark.parametrize("rule", RULES)
def test_batch_agrees_with_single_trajectory(rng, rule):
    X, Y = bipolar(rng, 48, 5), bipolar(rng, 16, 5)
    model = BAM(rule, max_iter=30).fit(X, Y)
    probes = bipolar(rng, 48, 7)
    Xf, Yf, _, _ = model.recall_batch(X0=probes)
    for k in range(probes.shape[1]):
        traj = model.recall(x0=probes[:, k])
        assert np.array_equal(traj.x_final, Xf[:, k])
        assert np.array_equal(traj.y_final, Yf[:, k])


def test_recall_requires_at_least_one_probe(rng):
    model = BAM("hebb").fit(bipolar(rng, 16, 2), bipolar(rng, 8, 2))
    with pytest.raises(ValueError, match="at least one"):
        model.recall()


def test_trajectory_records_one_frame_per_half_iteration(rng):
    X, Y = bipolar(rng, 32, 3), bipolar(rng, 8, 3)
    model = BAM("hebb", max_iter=4).fit(X, Y)
    traj = model.recall(x0=X[:, 0])
    # 1 probe frame + 2 per full iteration
    assert len(traj.xs) == 1 + 2 * traj.n_iter
    assert traj.steps[0] == "probe"
    assert set(traj.steps[1:]) <= {"X->Y", "Y->X"}


# --- the prototype = majority-vote identity ---------------------------------
def test_hebb_evokes_the_class_majority_vote(rng):
    """`sgn(W y_c) = sgn(Σ_{k∈c} x_k)` when the codebook is orthogonal.

    This is the result exploited in Experiment 5: with many faces per code, the
    Hebbian BAM synthesises the class prototype instead of returning one of the
    stored faces.
    """
    codes = hadamard(16).astype(float)[1:5].T  # 4 classes
    per_class = 5  # odd, so the vote never ties
    X = bipolar(rng, 300, 4 * per_class)
    labels = np.repeat(np.arange(4), per_class)
    Y = codes[:, labels]

    model = BAM("hebb").fit(X, Y)
    evoked = model.associate_backward(codes)
    expected = np.stack(
        [np.sign(X[:, labels == c].sum(axis=1)) for c in range(4)], axis=1
    )
    assert np.array_equal(evoked, expected)
