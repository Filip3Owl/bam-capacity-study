"""Tests of the supporting modules: codebooks, data, metrics, theory and text."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import hadamard

from bam import codebooks, datasets, metrics, ptbr, theory, viz
from bam.core import RULES


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(99)


# --- codebooks --------------------------------------------------------------
def test_hadamard_is_exactly_orthogonal():
    Y = codebooks.hadamard_codebook(20, 64)
    assert np.array_equal(Y.T @ Y, 64 * np.eye(20))
    assert codebooks.orthogonality(Y) == 0.0


def test_hadamard_refuses_more_codes_than_the_order_allows():
    with pytest.raises(ValueError, match="at most"):
        codebooks.hadamard_codebook(64, 64)


def test_hadamard_drops_the_constant_row():
    Y = codebooks.hadamard_codebook(3, 8)
    assert not np.all(Y[0] == Y, axis=0).any(), "degenerate code in the codebook"


def test_hadamard_order_is_a_power_of_two():
    assert codebooks.next_hadamard_order(40) == 64
    assert codebooks.next_hadamard_order(64) == 64
    assert codebooks.next_hadamard_order(65) == 128


def test_random_codebook_is_nearly_orthogonal():
    Y = codebooks.random_codebook(30, 512, seed=0)
    # The expected mean overlap is of order 1/sqrt(p).
    assert codebooks.orthogonality(Y) < 3.0 / np.sqrt(512)


@pytest.mark.parametrize("rho", [0.0, 0.2, 0.5, 0.8])
def test_correlated_codebook_hits_the_requested_overlap(rho):
    Y = codebooks.correlated_codebook(60, 2048, rho, seed=1)
    assert codebooks.orthogonality(Y) == pytest.approx(rho, abs=0.05)


def test_correlated_codebook_rejects_invalid_rho():
    with pytest.raises(ValueError, match="rho must lie"):
        codebooks.correlated_codebook(4, 16, 1.0)


def test_unknown_codebook_dispatch():
    with pytest.raises(ValueError, match="unknown codebook"):
        codebooks.build_codebook("none", 4, 16)


# --- data -------------------------------------------------------------------
def test_median_binarisation_balances_the_bits(rng):
    images = rng.random((7, 8, 8))
    X = datasets.binarize(images, method="median")
    assert set(np.unique(X)) <= {-1.0, 1.0}
    assert X.shape == (64, 7)
    # The median of 64 distinct values leaves exactly half on each side.
    assert np.all(X.sum(axis=0) == 0)


def test_mean_face_binarisation_lowers_the_correlation(rng):
    """The threshold choice is what controls crosstalk — worth testing."""
    base = rng.random((1, 12, 12)) * 0.6  # a strong shared "mean face"
    images = np.clip(base + 0.25 * rng.random((30, 12, 12)), 0, 1)
    by_median = datasets.binarize(images, method="median")
    by_mean_face = datasets.binarize(images, method="mean-face")
    assert (datasets.mean_correlation(by_mean_face)
            < datasets.mean_correlation(by_median))


def test_binarisation_rejects_unknown_method(rng):
    with pytest.raises(ValueError, match="unknown method"):
        datasets.binarize(rng.random((3, 4, 4)), method="otsu")


def test_train_test_split_is_disjoint_and_stratified():
    target = np.repeat(np.arange(40), 10)
    train, test = datasets.train_test_split(target, n_train=7, seed=0)
    assert len(train) == 280
    assert len(test) == 120
    assert not set(train) & set(test)
    assert np.all(np.bincount(target[train], minlength=40) == 7)
    assert np.all(np.bincount(target[test], minlength=40) == 3)


def test_one_image_per_subject():
    target = np.repeat(np.arange(40), 10)
    idx = datasets.one_index_per_subject(target)
    assert len(idx) == 40
    assert len(set(target[idx])) == 40


def test_mean_correlation_of_orthogonal_patterns_is_zero():
    assert datasets.mean_correlation(hadamard(32).astype(float)[:, 1:9]) == 0.0


# --- metrics ----------------------------------------------------------------
def test_hamming_and_overlap_are_consistent(rng):
    A = rng.choice(np.array([-1.0, 1.0]), size=(50, 4))
    B = rng.choice(np.array([-1.0, 1.0]), size=(50, 4))
    d = metrics.normalized_hamming(A, B)
    m = metrics.overlap(A, B)
    # For bipolar vectors: m = 1 - 2d.
    assert np.allclose(m, 1 - 2 * d)


def test_perfect_recall_is_stricter_than_bit_accuracy(rng):
    A = rng.choice(np.array([-1.0, 1.0]), size=(80, 10))
    B = A.copy()
    B[0, :5] *= -1  # one wrong bit in half the columns
    assert metrics.perfect_recall_rate(A, B) == 0.5
    assert metrics.bit_accuracy(A, B) > 0.99


def test_bit_noise_at_the_extremes(rng):
    X = rng.choice(np.array([-1.0, 1.0]), size=(60, 3))
    assert np.array_equal(metrics.bit_flip_noise(X, 0.0, rng), X)
    assert np.array_equal(metrics.bit_flip_noise(X, 1.0, rng), -X)


def test_bit_noise_hits_the_requested_rate(rng):
    X = np.ones((20000, 1))
    noisy = metrics.bit_flip_noise(X, 0.3, rng)
    assert float((noisy != X).mean()) == pytest.approx(0.3, abs=0.02)


def test_occlusion_zeroes_the_bottom_rows(rng):
    X = rng.choice(np.array([-1.0, 1.0]), size=(64, 2))
    Xo = metrics.occlude(X, 0.5, side=8)
    image = Xo[:, 0].reshape(8, 8)
    assert np.all(image[4:] == 0.0)
    assert np.all(image[:4] != 0.0)


def test_total_and_null_occlusion():
    X = np.ones((16, 1))
    assert np.all(metrics.occlude(X, 1.0, side=4) == 0.0)
    assert np.array_equal(metrics.occlude(X, 0.0, side=4), X)


def test_confidence_interval_shrinks_with_more_samples():
    rng = np.random.default_rng(7)
    _, ci_few = metrics.confidence_interval(rng.normal(size=10))
    _, ci_many = metrics.confidence_interval(rng.normal(size=1000))
    assert ci_many < ci_few


# --- theory -----------------------------------------------------------------
def test_snr_is_infinite_for_orthogonal_patterns():
    snr = theory.crosstalk_snr(hadamard(64).astype(float)[:, 1:11])
    assert np.all(np.isinf(snr))
    assert np.all(theory.predicted_bit_error(snr) == 0.0)


def test_snr_falls_as_the_pattern_count_grows(rng):
    X = rng.choice(np.array([-1.0, 1.0]), size=(512, 60))
    snrs = [float(np.mean(theory.crosstalk_snr(X[:, :P]))) for P in (5, 20, 60)]
    assert snrs[0] > snrs[1] > snrs[2]


def test_predicted_error_decreases_with_snr():
    errors = theory.predicted_bit_error(np.array([0.5, 1.0, 2.0, 4.0]))
    assert np.all(np.diff(errors) < 0)
    assert theory.predicted_bit_error(np.array([0.0]))[0] == pytest.approx(0.5)


def test_classical_theoretical_bounds():
    assert theory.kosko_bound(4096, 256) == 256
    # min(n,p)/(2 ln min(n,p)) is far below min(n,p)
    assert theory.asymptotic_bound(4096, 256) < 0.2 * 256


def test_predicted_capacity_is_monotone_in_the_threshold(rng):
    X = rng.choice(np.array([-1.0, 1.0]), size=(2048, 30))
    loose = theory.predicted_capacity(X, 64, threshold=0.5)
    tight = theory.predicted_capacity(X, 64, threshold=0.99)
    assert loose >= tight


# --- presentation layer -----------------------------------------------------
def test_number_formatting_switches_decimal_separator():
    assert viz.format_number(1234.5, 1) == f"1{viz.THIN_SPACE}234.5"
    assert viz.format_number(1234.5, 1, ptbr=True) == f"1{viz.THIN_SPACE}234,5"
    assert viz.format_percent(0.253, 1) == "25.3%"
    assert viz.format_percent(0.253, 1, ptbr=True) == "25,3%"


def test_end_labels_are_pushed_apart():
    """Series that saturate together must not stack their labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ends = [(1.0, 0.0, "a"), (1.0, 0.0, "b"), (1.0, 0.0, "c")]
    artists = viz.label_line_ends(ax, ends, separation=0.05)
    heights = sorted(a.xy[1] for a in artists)
    assert np.allclose(heights, [0.0, 0.05, 0.10])
    plt.close(fig)


def test_every_animation_has_portuguese_titles():
    """The GIF text is the one part of the project that must not be English."""
    expected = {"01_bidirectional_recall", "02_capacity", "03_robustness",
                "04_identification"}
    assert set(ptbr.TITLES) == expected
    assert set(ptbr.TAKEAWAYS) == expected, "every animation states its conclusion"
    for title, subtitle in ptbr.TITLES.values():
        assert title and subtitle


def test_portuguese_labels_cover_every_rule():
    assert set(ptbr.RULES) == set(RULES)
    assert set(viz.RULE_COLOR) == set(RULES)
    assert set(viz.RULE_MARKER) == set(RULES)


def test_rule_colours_are_distinct():
    assert len(set(viz.RULE_COLOR.values())) == len(RULES)
