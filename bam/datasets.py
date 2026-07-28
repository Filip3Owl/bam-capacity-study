"""Loading and binarisation of the Olivetti (AT&T/ORL) face database.

This is the study's consolidated dataset: 400 grayscale 64x64 images, ten
photographs of each of 40 subjects, distributed with scikit-learn as
`sklearn.datasets.fetch_olivetti_faces`. It is a classic face-recognition
benchmark, small enough for exhaustive capacity sweeps and large enough
(n = 4096 bits per pattern) that the crosstalk effects studied here are
statistically sharp.

Since a BAM operates on bipolar vectors, the images must be binarised. The
binarisation method is not a cosmetic detail: it directly controls the
**correlation between the stored patterns**, which is the quantity that governs
crosstalk (see `bam.theory`). Hence two schemes are offered and compared in
Experiment 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bam.config import DATA_DIR, configure_ssl

BINARIZATION_METHODS = ("median", "mean-face")

BINARIZATION_LABELS = {
    "median": "Threshold at each image's own median",
    "mean-face": "Threshold at the dataset's mean face",
}


@dataclass(frozen=True)
class FaceDataset:
    """A binarised face database ready for the BAM.

    Attributes:
        X: (n, P) matrix in {-1, +1}; each **column** is one flattened face.
        gray: (P, side, side) array with the original images in [0, 1].
        target: (P,) subject index (0..39) of each image.
        side: image width/height in pixels.
        method: binarisation scheme used.
    """

    X: np.ndarray
    gray: np.ndarray
    target: np.ndarray
    side: int
    method: str

    @property
    def n(self) -> int:
        """Dimension of the X layer (bits per face)."""
        return self.X.shape[0]

    @property
    def n_patterns(self) -> int:
        """Number of faces available in the database."""
        return self.X.shape[1]

    def as_image(self, column: np.ndarray) -> np.ndarray:
        """Reshape one bipolar X-layer vector back into a 2-D image."""
        return column.reshape(self.side, self.side)


def load_olivetti(*, side: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return `(images, targets)` from Olivetti, downloading it if necessary.

    Args:
        side: if given and smaller than 64, images are average-pooled down to
            `side` x `side`. Used only in tests.

    Returns:
        `images` (P, side, side) in [0, 1] and `targets` (P,) integers in 0..39.
    """
    configure_ssl()
    from sklearn.datasets import fetch_olivetti_faces

    data = fetch_olivetti_faces(data_home=str(DATA_DIR), shuffle=False)
    images = data.images.astype(np.float64)
    targets = data.target.astype(np.int64)

    if side is not None and side != images.shape[1]:
        images = _downsample(images, side)
    return images, targets


def _downsample(images: np.ndarray, side: int) -> np.ndarray:
    """Downsample square images by block averaging (integer factor)."""
    orig = images.shape[1]
    if orig % side:
        raise ValueError(f"{orig} is not divisible by {side}")
    f = orig // side
    p = images.shape[0]
    return images.reshape(p, side, f, side, f).mean(axis=(2, 4))


def binarize(images: np.ndarray, method: str = "median") -> np.ndarray:
    """Convert images in [0, 1] into bipolar column vectors.

    Args:
        images: (P, side, side) array.
        method:
            ``"median"`` — each image is thresholded at its own median. This
                yields exactly balanced vectors (equal counts of +1 and -1) but
                preserves the strong correlation between faces: all of them
                share the same global structure (dark background, bright
                forehead, dark eyes).
            ``"mean-face"`` — each pixel is compared against its mean value
                across the whole database. This amounts to centring the data
                before taking the sign, which removes the common "mean face"
                and **sharply reduces the correlation** between patterns.

    Returns:
        (n, P) matrix in {-1, +1}, one face per column.
    """
    if method not in BINARIZATION_METHODS:
        raise ValueError(f"unknown method: {method!r}; use {BINARIZATION_METHODS}")

    p = images.shape[0]
    flat = images.reshape(p, -1)

    if method == "median":
        threshold = np.median(flat, axis=1, keepdims=True)
    else:  # mean-face
        threshold = flat.mean(axis=0, keepdims=True)

    X = np.where(flat > threshold, 1.0, -1.0)
    return np.ascontiguousarray(X.T)


def load_dataset(*, method: str = "median", side: int | None = None) -> FaceDataset:
    """Convenience wrapper that loads and binarises in a single call."""
    images, targets = load_olivetti(side=side)
    X = binarize(images, method=method)
    return FaceDataset(X=X, gray=images, target=targets, side=images.shape[1],
                       method=method)


def one_index_per_subject(target: np.ndarray, *, pose: int = 0) -> np.ndarray:
    """Indices of a single image per subject (each subject's `pose`-th one)."""
    out = []
    for subject in np.unique(target):
        cols = np.flatnonzero(target == subject)
        out.append(cols[pose % len(cols)])
    return np.array(out, dtype=np.int64)


def train_test_split(
    target: np.ndarray, *, n_train: int = 7, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split, per subject, `n_train` poses for training and the rest for test.

    Used in Experiment 5, where the BAM must identify photographs it has
    **not** stored.
    """
    rng = np.random.default_rng(seed)
    train, test = [], []
    for subject in np.unique(target):
        cols = rng.permutation(np.flatnonzero(target == subject))
        train.extend(cols[:n_train])
        test.extend(cols[n_train:])
    return np.sort(np.array(train)), np.sort(np.array(test))


def mean_correlation(X: np.ndarray) -> float:
    """Mean absolute correlation between distinct columns of `X`.

    For bipolar patterns, `x_i · x_j / n` is the fraction of agreeing bits
    rescaled to [-1, 1]. This is the quantity that appears in the numerator of
    the crosstalk term (see `bam.theory.crosstalk_snr`).
    """
    n, p = X.shape
    if p < 2:
        return 0.0
    G = (X.T @ X) / n
    off = ~np.eye(p, dtype=bool)
    return float(np.abs(G[off]).mean())
