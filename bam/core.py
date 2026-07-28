"""The Bidirectional Associative Memory model and its learning rules.

Formalisation
-------------
A BAM stores `P` pairs of bipolar patterns `(x_k, y_k)`, with
`x_k ∈ {-1,+1}^n` in layer X and `y_k ∈ {-1,+1}^p` in layer Y. Recall is a
resonance between the layers:

    y(t+1) = sgn(Wᵀ x(t))          (X→Y direction)
    x(t+1) = sgn(W y(t+1))         (Y→X direction)

Kosko (1988) showed that, for any `W`, the Lyapunov function

    E(x, y) = -xᵀ W y

is non-increasing under these updates, so the network always converges to a
stable pair — the BAM is **unconditionally stable for any weight matrix**. The
scientific question, therefore, is never whether it converges: it is *where*
it converges, that is, whether the stored pairs are actually fixed points and
how large their basins of attraction are.

The crosstalk term
------------------
Under the Hebb rule `W = Σ_k x_k y_kᵀ`, probing with the stored pattern `x_j`
gives

    Wᵀ x_j = n·y_j + Σ_{k≠j} y_k (x_kᵀ x_j)
                     └──────── crosstalk ────────┘

and, symmetrically, the crosstalk in the reverse direction is
`Σ_{k≠j} x_k (y_kᵀ y_j)`. Note the decisive asymmetry: **X→Y** recall is
degraded by the correlation between *faces*, which we cannot choose; **Y→X**
recall is degraded by the correlation between *codes*, which we can. This
package explores exactly that asymmetry.

Implemented rules
-----------------
``hebb``               Kosko (1988). The historical baseline.
``multiple_training``  Wang, Cruz & Mulligan (1990): `W = Σ q_k x_k y_kᵀ` with
                       integer multiplicities `q_k` adjusted iteratively until
                       every stored pair is a fixed point.
``least_squares``      Exact solution of
                       `min_W λ‖W̃ᵀX̃ − Ỹ‖² + (1−λ)‖W̃Ỹ − X̃‖²` over normalised
                       patterns, obtained from the resulting Sylvester equation
                       by SVD diagonalisation (see `_least_squares_rule`). The
                       weight `λ` splits the fit between the two directions.
``dense``              Dense associative memory / modern Hopfield network
                       (Krotov & Hopfield 2016; Ramsauer et al. 2020) applied
                       to the concatenated pair. Not a weight matrix: the
                       exponential interaction between state and memories
                       replaces the outer-product sum, raising capacity
                       exponentially.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp, softmax

RULES = ("hebb", "multiple_training", "least_squares", "dense")

RULE_LABELS = {
    "hebb": "Hebb (Kosko, 1988)",
    "multiple_training": "Multiple training",
    "least_squares": "Least squares",
    "dense": "Dense (exponential)",
}

# Update directions, used as trajectory step labels.
X_TO_Y = "X->Y"
Y_TO_X = "Y->X"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def sign(local_field: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Sign with ties broken by the previous state.

    The standard convention in the BAM literature is to leave a unit unchanged
    when its local field is exactly zero — this is what preserves the
    monotonicity of the energy.
    """
    return np.where(local_field > 0.0, 1.0,
                    np.where(local_field < 0.0, -1.0, previous))


def _validate_bipolar(M: np.ndarray, name: str) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError(f"{name} must be 2-D (patterns in columns), got {M.ndim}-D")
    if not np.all(np.isin(M, (-1.0, 1.0))):
        raise ValueError(f"{name} must contain only -1 and +1")
    return np.ascontiguousarray(M)


# ---------------------------------------------------------------------------
# Learning rules
# ---------------------------------------------------------------------------
def _hebb_rule(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """`W = Σ_k x_k y_kᵀ = X Yᵀ` (Hebbian correlation, Kosko 1988)."""
    return X @ Y.T


def _multiple_training_rule(
    X: np.ndarray, Y: np.ndarray, *, max_epochs: int = 200
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Multiple-training learning (Wang, Cruz & Mulligan, 1990).

    Keeps the Hebbian form but presents each pair an integer number `q_k` of
    times. At every epoch, pairs that are not yet fixed points have their
    multiplicity incremented. If the algorithm terminates with all pairs
    stable, the training set is guaranteed to be stored.

    With strongly correlated patterns the procedure may fail to converge within
    the epoch budget — a situation we record explicitly rather than silence,
    because it is exactly what happens with real faces.

    Implementation note: the loop never materialises `W`. Since
    `W = X·diag(q)·Yᵀ`, the fields over the stored patterns are

        WᵀX = Y·diag(q)·(XᵀX)   and   WY = X·diag(q)·(YᵀY),

    so it suffices to precompute the two `P×P` Gram matrices once. This trades
    an `n×p · p×P` product per epoch (O(npP)) for an `n×P · P×P` one (O(nP²)) —
    with n = 4096 and p = 256 that is the difference between the capacity sweep
    taking minutes and taking hours.

    Returns:
        `(W, q, converged)` — the weight matrix, the multiplicities, and
        whether every pair became stable within the epoch budget.
    """
    Gx = X.T @ X
    Gy = Y.T @ Y
    q = np.ones(X.shape[1], dtype=np.float64)

    def unstable(q: np.ndarray) -> np.ndarray:
        forward = sign(Y @ (q[:, None] * Gx), Y) == Y  # (p, P)
        backward = sign(X @ (q[:, None] * Gy), X) == X  # (n, P)
        return ~(forward.all(axis=0) & backward.all(axis=0))

    converged = False
    for _ in range(max_epochs):
        bad = unstable(q)
        if not bad.any():
            converged = True
            break
        q[bad] += 1.0
    else:
        converged = not unstable(q).any()

    W = X @ (q[:, None] * Y.T)
    return W, q, converged


def _least_squares_rule(
    X: np.ndarray, Y: np.ndarray, *, lam: float = 0.5, tol: float = 1e-10
) -> np.ndarray:
    """Minimise the bidirectional reconstruction error weighted by `lam`.

    The patterns are first normalised to unit norm, `X̃ = X/√n` and `Ỹ = Y/√p`,
    and the objective is

        J(W̃) = λ·‖W̃ᵀX̃ − Ỹ‖²_F + (1−λ)·‖W̃Ỹ − X̃‖²_F,   λ ∈ [0, 1].

    The normalisation is essential and not cosmetic: without it the singular
    values of `X` (of order √(nP) ≈ 286 for faces) are tens of times larger
    than those of `Y` (√(pP) ≈ 36), `λ` stops being interpretable, and the X→Y
    direction dominates the fit for any `λ > 0`. After normalising, `λ = 0.5`
    genuinely balances the two directions, `λ → 1` recovers the classical
    pseudo-inverse (exact X→Y) and `λ → 0` recovers — for a Hadamard codebook —
    the Hebb rule itself up to scale.

    Setting the gradient to zero yields the Sylvester equation

        λ·X̃X̃ᵀW̃ + (1−λ)·W̃ỸỸᵀ = X̃Ỹᵀ.

    With the SVDs `X̃ = U_x S̃_x V_xᵀ` and `Ỹ = U_y S̃_y V_yᵀ`, the minimum-norm
    solution has the form `W̃ = U_x A U_yᵀ`, the equation **diagonalises**, and
    every entry of `A` follows in closed form:

        A_ij = σ̃_i C_ij τ̃_j / (λ σ̃_i² + (1−λ) τ̃_j²),   C = V_xᵀ V_y.

    The cost is that of two thin SVDs — O(nP²) — rather than solving a dense
    n×n Sylvester equation, which is what makes the rule tractable at n = 4096.
    The global scale of `W` is irrelevant, since recall only uses the sign of
    the field.

    Sanity check: if `XᵀX = n·I` and `YᵀY = p·I` (orthogonal patterns on both
    sides), then σ̃ = τ̃ = 1 and `C = I`, hence `A = I` and `W ∝ X Yᵀ` — the
    Hebb rule, as it should be, since without correlation Hebb is already
    optimal.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1]")
    n, p = X.shape[0], Y.shape[0]

    Ux, Sx, Vxt = np.linalg.svd(X, full_matrices=False)
    Uy, Sy, Vyt = np.linalg.svd(Y, full_matrices=False)
    Sx = Sx / np.sqrt(n)  # singular values of X̃
    Sy = Sy / np.sqrt(p)  # singular values of Ỹ

    rx = int((Sx > tol * Sx.max()).sum())
    ry = int((Sy > tol * Sy.max()).sum())
    Ux, Sx, Vxt = Ux[:, :rx], Sx[:rx], Vxt[:rx]
    Uy, Sy, Vyt = Uy[:, :ry], Sy[:ry], Vyt[:ry]

    C = Vxt @ Vyt.T  # (rx, ry)
    denom = lam * Sx[:, None] ** 2 + (1.0 - lam) * Sy[None, :] ** 2
    A = Sx[:, None] * C * Sy[None, :] / np.maximum(denom, tol)
    return Ux @ A @ Uy.T


# ---------------------------------------------------------------------------
# Recall trajectory
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    """Full history of one recall, for analysis and animation."""

    xs: list[np.ndarray] = field(default_factory=list)
    ys: list[np.ndarray] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    converged: bool = False
    n_iter: int = 0

    @property
    def x_final(self) -> np.ndarray:
        return self.xs[-1]

    @property
    def y_final(self) -> np.ndarray:
        return self.ys[-1]

    @property
    def n_frames(self) -> int:
        return len(self.xs)

    def energy_is_monotone(self, *, tol: float = 1e-9) -> bool:
        """Empirical check of Kosko's stability theorem."""
        e = np.asarray(self.energies)
        return bool(np.all(np.diff(e) <= tol))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class BAM:
    """Bidirectional Associative Memory with a selectable learning rule.

    Args:
        rule: one of `RULES`.
        max_iter: maximum number of full resonance iterations.
        beta: inverse temperature of the ``dense`` rule. Larger values make
            recall more selective (less mixing between nearby memories).
        lam: direction weight of the ``least_squares`` rule.
        max_mt_epochs: epoch budget of the ``multiple_training`` rule.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> X = rng.choice([-1.0, 1.0], size=(64, 4))
        >>> Y = rng.choice([-1.0, 1.0], size=(16, 4))
        >>> model = BAM("hebb").fit(X, Y)
        >>> traj = model.recall(x0=X[:, 0])
        >>> bool(np.array_equal(traj.y_final, Y[:, 0]))
        True
    """

    def __init__(
        self,
        rule: str = "hebb",
        *,
        max_iter: int = 30,
        beta: float = 12.0,
        lam: float = 0.5,
        max_mt_epochs: int = 200,
    ) -> None:
        if rule not in RULES:
            raise ValueError(f"unknown rule: {rule!r}; use {RULES}")
        self.rule = rule
        self.max_iter = max_iter
        self.beta = beta
        self.lam = lam
        self.max_mt_epochs = max_mt_epochs

        self.W: np.ndarray | None = None
        self.X_: np.ndarray | None = None
        self.Y_: np.ndarray | None = None
        self.q_: np.ndarray | None = None
        self.mt_converged_: bool | None = None

    # -- construction -------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray) -> BAM:
        """Store the pairs `(column_k(X), column_k(Y))`."""
        X = _validate_bipolar(X, "X")
        Y = _validate_bipolar(Y, "Y")
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} patterns and Y has {Y.shape[1]}; they must match"
            )
        self.X_, self.Y_ = X, Y

        if self.rule == "hebb":
            self.W = _hebb_rule(X, Y)
        elif self.rule == "multiple_training":
            self.W, self.q_, self.mt_converged_ = _multiple_training_rule(
                X, Y, max_epochs=self.max_mt_epochs
            )
        elif self.rule == "least_squares":
            self.W = _least_squares_rule(X, Y, lam=self.lam)
        else:  # dense — the interaction does not factor into a weight matrix
            self.W = None
        return self

    @property
    def n(self) -> int:
        return self.X_.shape[0]

    @property
    def p(self) -> int:
        return self.Y_.shape[0]

    @property
    def n_patterns(self) -> int:
        return self.X_.shape[1]

    @property
    def linear(self) -> bool:
        """`True` for the rules that reduce to a weight matrix."""
        return self.rule != "dense"

    # -- dynamics -----------------------------------------------------------
    def _dense_weights(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Per-probe `softmax(β·s_k)` attention weights of the dense rule.

        The similarity `s_k` adds the normalised overlaps in both layers, which
        is exactly the modern Hopfield energy applied to the concatenated
        vector `z = [x; y]`.
        """
        s = (self.X_.T @ X) / self.n + (self.Y_.T @ Y) / self.p  # (P, B)
        return softmax(self.beta * s, axis=0)

    def field_y(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Local field received by layer Y (pre-activation), batched."""
        if self.linear:
            return self.W.T @ X
        return self.Y_ @ self._dense_weights(X, Y)

    def field_x(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Local field received by layer X (pre-activation), batched."""
        if self.linear:
            return self.W @ Y
        return self.X_ @ self._dense_weights(X, Y)

    def energy(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Lyapunov energy of each state pair (batched).

        Linear rules: `E = -xᵀWy` (Kosko, 1988).
        Dense rule: `E = -(1/β)·log Σ_k exp(β s_k)`, the log-sum-exp energy of
        dense associative memories.
        """
        X = np.atleast_2d(X.T).T if X.ndim == 1 else X
        Y = np.atleast_2d(Y.T).T if Y.ndim == 1 else Y
        if self.linear:
            return -np.einsum("ib,ij,jb->b", X, self.W, Y)
        s = (self.X_.T @ X) / self.n + (self.Y_.T @ Y) / self.p
        return -logsumexp(self.beta * s, axis=0) / self.beta

    # -- single-pass association -------------------------------------------
    def associate_forward(self, X0: np.ndarray,
                          Y0: np.ndarray | None = None) -> np.ndarray:
        """A single X→Y propagation, with no feedback.

        This is "pure" heteroassociation: it measures the quality of the map
        `x ↦ sgn(Wᵀx)` in isolation. It contrasts with `recall_batch`, which
        lets the network resonate. The distinction is essential in this study
        because the two BAM directions have very different quality, and
        resonance propagates the error of the bad direction into the good one.
        """
        X0 = np.atleast_2d(np.asarray(X0, dtype=np.float64).T).T if X0.ndim == 1 else X0
        Y0 = np.zeros((self.p, X0.shape[1])) if Y0 is None else Y0
        return sign(self.field_y(X0, Y0), Y0)

    def associate_backward(self, Y0: np.ndarray,
                           X0: np.ndarray | None = None) -> np.ndarray:
        """A single Y→X propagation, with no feedback."""
        Y0 = np.atleast_2d(np.asarray(Y0, dtype=np.float64).T).T if Y0.ndim == 1 else Y0
        X0 = np.zeros((self.n, Y0.shape[1])) if X0 is None else X0
        return sign(self.field_x(X0, Y0), X0)

    # -- recall by resonance ------------------------------------------------
    def recall(
        self,
        *,
        x0: np.ndarray | None = None,
        y0: np.ndarray | None = None,
        max_iter: int | None = None,
        record: bool = True,
    ) -> Trajectory:
        """Recall a single pair from a probe, keeping the whole trajectory.

        Args:
            x0: probe in layer X. If `None`, recall starts from Y.
            y0: probe in layer Y. If `None`, it is initialised to zeros, which
                makes the first half-iteration driven by `x0` alone.
            max_iter: overrides `self.max_iter`.
            record: if `False`, only the final state is kept (faster).

        Returns:
            The `Trajectory` traversed. Each frame is **half** an iteration
            (one layer update), which is the granularity at which the energy is
            guaranteed to be non-increasing.
        """
        if x0 is None and y0 is None:
            raise ValueError("provide at least one of the probes x0 or y0")
        max_iter = self.max_iter if max_iter is None else max_iter

        x = np.zeros(self.n) if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        y = np.zeros(self.p) if y0 is None else np.asarray(y0, dtype=np.float64).copy()
        # Which half-iteration comes first depends on the probed layer.
        starts_at_x = x0 is not None

        traj = Trajectory()

        def note(step: str) -> None:
            if record or step == "final":
                traj.xs.append(x.copy())
                traj.ys.append(y.copy())
                traj.energies.append(float(self.energy(x[:, None], y[:, None])[0]))
                traj.steps.append(step)

        note("probe")
        for it in range(max_iter):
            x_prev, y_prev = x.copy(), y.copy()
            if starts_at_x:
                y = sign(self.field_y(x[:, None], y[:, None])[:, 0], y)
                note(X_TO_Y)
                x = sign(self.field_x(x[:, None], y[:, None])[:, 0], x)
                note(Y_TO_X)
            else:
                x = sign(self.field_x(x[:, None], y[:, None])[:, 0], x)
                note(Y_TO_X)
                y = sign(self.field_y(x[:, None], y[:, None])[:, 0], y)
                note(X_TO_Y)
            traj.n_iter = it + 1
            if np.array_equal(x, x_prev) and np.array_equal(y, y_prev):
                traj.converged = True
                break
        return traj

    def recall_batch(
        self,
        *,
        X0: np.ndarray | None = None,
        Y0: np.ndarray | None = None,
        max_iter: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised version of `recall` for statistical sweeps.

        Returns:
            `(X, Y, iterations, converged)` — final states of both layers, the
            number of iterations each probe took, and a convergence mask.
        """
        if X0 is None and Y0 is None:
            raise ValueError("provide at least one of the probes X0 or Y0")
        max_iter = self.max_iter if max_iter is None else max_iter
        b = X0.shape[1] if X0 is not None else Y0.shape[1]

        X = np.zeros((self.n, b)) if X0 is None else np.array(X0, dtype=np.float64)
        Y = np.zeros((self.p, b)) if Y0 is None else np.array(Y0, dtype=np.float64)
        starts_at_x = X0 is not None

        iterations = np.full(b, max_iter, dtype=np.int64)
        converged = np.zeros(b, dtype=bool)

        for it in range(max_iter):
            X_prev, Y_prev = X.copy(), Y.copy()
            if starts_at_x:
                Y = sign(self.field_y(X, Y), Y)
                X = sign(self.field_x(X, Y), X)
            else:
                X = sign(self.field_x(X, Y), X)
                Y = sign(self.field_y(X, Y), Y)
            stable = (X_prev == X).all(axis=0) & (Y_prev == Y).all(axis=0)
            newly = stable & ~converged
            iterations[newly] = it + 1
            converged |= stable
            if converged.all():
                break
        return X, Y, iterations, converged

    # -- diagnostics --------------------------------------------------------
    def stable_pairs(self) -> np.ndarray:
        """Mask of stored pairs that are fixed points of the network."""
        Xf, Yf, _, _ = self.recall_batch(X0=self.X_, Y0=self.Y_, max_iter=1)
        return (Xf == self.X_).all(axis=0) & (Yf == self.Y_).all(axis=0)

    def __repr__(self) -> str:  # pragma: no cover - interactive convenience
        if self.X_ is None:
            return f"BAM(rule={self.rule!r}, not fitted)"
        return (
            f"BAM(rule={self.rule!r}, n={self.n}, p={self.p}, "
            f"P={self.n_patterns})"
        )


def recall(
    model: BAM, x0: np.ndarray | None = None, y0: np.ndarray | None = None, **kw
) -> Trajectory:
    """Functional shortcut for `model.recall`."""
    return model.recall(x0=x0, y0=y0, **kw)
