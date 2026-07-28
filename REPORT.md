# Bidirectional Associative Memory on real faces: what stability guarantees, and what it does not

**Dataset:** Olivetti/ORL (AT&T Laboratories Cambridge) — 400 images of 64×64
pixels, ten photographs of each of 40 subjects, distributed with scikit-learn.
**Code:** `bam/` (library), `experiments/` (five experiments), `tests/`
(63 property tests).
**Reproduction:** `python experiments/run_all.py` — deterministic given
`bam.config.SEED = 20250728`.

---

## Abstract

Kosko's Bidirectional Associative Memory (1988) is *unconditionally stable*:
for any weight matrix, the energy `E = −xᵀWy` never rises and the network
always converges. We verified that guarantee over 800 trajectories, with zero
violations. And we show that on its own it is nearly useless.

Storing real faces, the Hebbian BAM converges — to the wrong places. With
`n = 4096` and `p = 256`, Kosko's capacity conjecture predicts 256 storable
pairs; we measure **zero useful capacity** in the face→code direction, with
failure already at five pairs. The cause is identified quantitatively:
crosstalk between correlated patterns. A signal-to-noise model with no fitted
parameter predicts the observed per-bit error with log-log correlation
`r = 0.984` and a median relative error of 4 %.

Three design results follow from the analysis:

1. **Orthogonal codes cancel one direction's crosstalk exactly.** With a
   Hadamard codebook, code→face recall is perfect in a single pass for all 200
   pairs tested — against 10 pairs with random codes, a 20× gain.
2. **Bidirectional resonance destroys that advantage.** The BAM feeds the bad
   direction's error back into the good one: the same configuration that
   recalls 100 % of the faces in a single pass recalls 0 % once the network is
   allowed to resonate. *A BAM is worth what its worse direction is worth.*
3. **Exactness is not robustness.** Exact recall via Hadamard codes has zero
   margin: 10 % of noise on the code drops accuracy from 100 % to 29 %.

A dense associative memory (exponential interaction, Krotov & Hopfield 2016;
Ramsauer et al. 2020) sidesteps all three limitations: it stores the 200 pairs,
tolerates 45 % bit noise and 80 % occlusion, and keeps its resonance intact.

---

## 1. The model and the four rules compared

A BAM stores `P` bipolar pairs `(x_k, y_k)`, `x_k ∈ {−1,+1}^n` in layer X
(faces) and `y_k ∈ {−1,+1}^p` in layer Y (codes). Recall is a resonance:

```
y(t+1) = sgn(Wᵀ x(t))        (X→Y)
x(t+1) = sgn(W y(t+1))       (Y→X)
```

We compare four learning rules:

| Rule | Definition | Origin |
|---|---|---|
| **Hebb** | `W = Σ x_k y_kᵀ` | Kosko (1988) |
| **Multiple training** | `W = Σ q_k x_k y_kᵀ`, with integer `q_k` adjusted until every stored pair is a fixed point | Wang, Cruz & Mulligan (1990) |
| **Least squares** | minimises `λ‖W̃ᵀX̃ − Ỹ‖² + (1−λ)‖W̃Ỹ − X̃‖²` over normalised patterns | derived in this work |
| **Dense** | exponential interaction over the concatenated pair, no weight matrix | Krotov & Hopfield (2016); Ramsauer et al. (2020) |

The least-squares rule follows from zeroing the objective's gradient, which
gives the Sylvester equation `λX̃X̃ᵀW̃ + (1−λ)W̃ỸỸᵀ = X̃Ỹᵀ`. Writing the SVDs
`X̃ = U_x S̃_x V_xᵀ` and `Ỹ = U_y S̃_y V_yᵀ` and seeking the minimum-norm
solution `W̃ = U_x A U_yᵀ`, the equation **diagonalises** and each entry of `A`
comes out in closed form:

```
A_ij = σ̃_i C_ij τ̃_j / (λ σ̃_i² + (1−λ) τ̃_j²),     C = V_xᵀ V_y.
```

The cost is that of two thin SVDs — O(nP²) — not that of solving a dense
4096×4096 Sylvester equation. Normalising the patterns beforehand
(`X̃ = X/√n`, `Ỹ = Y/√p`) is not cosmetic: without it, the singular values of
`X` are tens of times larger than those of `Y`, `λ` stops being interpretable
and the X→Y direction dominates the fit for any `λ > 0`. As a sanity check, if
`XᵀX = n·I` and `YᵀY = p·I` the formula returns `W ∝ XYᵀ` — the Hebb rule
itself, as it must, since without correlation Hebb is already optimal.

---

## 2. Experiment 1 — Stability is real, and insufficient

**Question:** does the network converge, and to the right pair?

800 trajectories (200 per rule), probes with 5 % to 45 % of bits flipped,
`P = 20`, Hadamard codes:

| Rule | Monotonicity violations | Convergence |
|---|---|---|
| Hebb | 0 / 200 | 100 % |
| Multiple training | 0 / 200 | 100 % |
| Least squares | 0 / 200 | 100 % |
| Dense | 0 / 200 | 100 % |

Kosko's theorem holds without exception, and the same property holds
empirically for the dense rule's log-sum-exp energy. But convergence says
nothing about the destination: under the same probe (25 % of bits flipped) the
Hebb rule converges in three iterations to a spurious state with **11.1 % of
the face bits and 28.1 % of the code bits wrong**, whereas the least-squares
and dense rules converge to the exact stored pair.

> **Figure 1** (`fig01_energy.png`) — all four energy curves descend; three stop
> far from zero error.
> **Animation 1** (`01_bidirectional_recall.gif`) — the capability itself: a
> corrupted face is restored together with its code, and then the code alone
> restores the face.

---

## 3. Experiment 2 — Capacity

**Question:** how many pairs does the BAM really hold?

`n = 4096`, `p = 256`, `P` from 5 to 200, 6 seeds, 432 configurations. Useful
capacity = largest `P` with ≥ 95 % **perfect** recall (not a single wrong bit —
in a content-addressable memory, an answer with 3 % of its bits flipped is not
partially correct, it is a spurious state).

| Codebook | Rule | X→Y (pass) | Y→X (pass) | Resonance |
|---|---|---:|---:|---:|
| Hadamard | Hebb | **0** | **200+** | 0 |
| Hadamard | Multiple training | 0 | **200+** | 0 |
| Hadamard | Least squares | 5 | 10 | 5 |
| Hadamard | Dense | 160 | 200+ | **200+** |
| Random | Hebb | 0 | 10 | 0 |
| Random | Multiple training | 0 | 10 | 0 |
| Random | Least squares | 5 | 10 | 10 |
| Random | Dense | 160 | 200+ | **200+** |

Theoretical references: Kosko's conjecture gives `min(n, p) = 256`; the
asymptotic analysis for i.i.d. patterns gives
`min(n,p)/(2·ln min(n,p)) ≈ 23`. **Both overestimate the classical rule's real
X→Y performance by a categorical margin**: Hebb's perfect-recall rate falls
from 83 % at `P = 5` to 1.7 % at `P = 20` and 0 % at `P = 60`.

Three readings:

**(a) The codebook decides the Y→X direction.** Hadamard: 200+ pairs. Random:
10. A factor of 20 obtained purely by choosing the labels well, without
touching the learning rule or the data.

**(b) Multiple training solves nothing here.** The algorithm is only guaranteed
when it converges; with correlated faces it exhausts its epoch budget without
stabilising every pair, and the result is indistinguishable from Hebb.

**(c) Resonance cancels the codebook's gain.** Look at the Hebb+Hadamard row:
Y→X perfect over 200 pairs, resonance **zero**. The network recalls the exact
face from the code, then uses that face to recompute the code, gets it wrong,
and the error contaminates the next half-iteration. Bidirectionality, the
model's selling point, is a liability here.

> **Figure 2** (`fig02_capacity.png`) · **Animation 2** (`02_capacity.gif`) —
> the same network and rule under two codebooks, as `P` grows.

---

## 4. Experiment 3 — Robustness

`P = 20`, Hadamard codes, 6 seeds. Maximum corruption tolerated with ≥ 90 %
perfect recall:

| Rule | Bit noise on face | Occlusion of face | Bit noise on code |
|---|---:|---:|---:|
| Hebb | — | — | 0 % |
| Multiple training | — | — | 0 % |
| Least squares | 35 % | 40 % | 0 % |
| Dense | **45 %** | **80 %** | **25 %** |

("—" means the rule does not reach 90 % even with no corruption at all.)

**Occlusion is far more tolerable than noise** — 80 % against 45 % for the
dense rule. The reason is structural: hidden pixels enter with zero local
field, i.e. the probe merely stays silent; flipped bits actively assert the
opposite of the truth, and each one pushes the field the wrong way.

**The most instructive finding is in the last column.** Hebb+Hadamard's Y→X
recall is *exact* — and completely brittle:

| Rule | No code noise | With 10 % code noise |
|---|---:|---:|
| Hebb | 100 % | **29 %** |
| Multiple training | 100 % | 33 % |
| Least squares | 90 % | 5 % |
| Dense | 100 % | **100 %** |

Hadamard orthogonality cancels crosstalk *exactly at the point* `y = y_j`. Away
from it, `ỹ` is not a valid code, the products `y_kᵀỹ` stop being zero and the
interference returns all at once. Exactness with no margin: the solution is
optimal and has no basin of attraction.

> **Figure 3** (`fig03_robustness.png`) · **Animation 3** (`03_robustness.gif`) —
> flipped pixels against erased pixels, damaged by the same fraction at once.

---

## 5. Experiment 4 — Crosstalk theory predicts the error

Probing with the stored pattern `x_j`, the field reaching unit `i` of layer Y is

```
h_i = n·y_{j,i} + Σ_{k≠j} y_{k,i} (x_kᵀ x_j)
                  └──────── crosstalk ────────┘
```

Multiplying by `y_{j,i}`, the bit is correct when `n + Σ_{k≠j} ± c_kj > 0`. The
sum has zero mean and variance `Σ_{k≠j} c_kj²`, which defines

```
SNR_j = n / √(Σ_{k≠j} c_kj²)        and        per-bit error ≈ Φ(−SNR_j).
```

**No parameter is fitted.** Comparing prediction against measurement over 72
configurations (9 memory sizes × 2 binarisations × 2 codebooks × 2 directions):

| | |
|---|---|
| Exact prediction of zero error (Hadamard, Y→X) | 18 points |
| Positive prediction and positive measurement | 50 points |
|  → Pearson correlation in log-log | **r = 0.984** |
|  → median relative error of the prediction | **4.0 %** |
| Positive prediction, zero measurement (theory pessimistic) | 4 points |
| Spearman correlation over all points | **ρ = 0.988** |

The model explains the failure quantitatively. The X→Y SNR on faces is
**1.31 at `P = 5`** and **0.22 at `P = 200`** (median thresholding): crosstalk
is of the same order as the signal with just five stored patterns. For
reference, an SNR of 2 corresponds to ~2 % error per bit — which, over 256
bits, still gives essentially zero perfect recall.

**Decorrelating the data helps, but does not save it.** Thresholding each face
at the dataset's mean face instead of its own median lowers the mean pattern
correlation from 0.32 to 0.19 and raises the SNR by ~55 % (from 1.31 to 2.04 at
`P = 5`). That is a real and cheap gain, and still insufficient: useful X→Y
capacity remains in the single digits.

### 5.1 The Pareto frontier between the two directions

No matrix `W` satisfies `WᵀX = Y` and `WY = X` at once, except in degenerate
cases. The least-squares rule's weight `λ` makes that compromise measurable:

| λ | X→Y (face→code) | Y→X (code→face) |
|---:|---:|---:|
| 0.00 | 0 % | 100 % |
| 0.30 | 60 % | 100 % |
| **0.50** | **85 %** | **85 %** |
| 0.70 | 95 % | 35 % |
| 1.00 | 100 % | 0 % |

The frontier is concave and its knee sits at `λ ≈ 0.5`, with 85 % in both
directions. Among the linear rules tested this is the best bidirectional
arrangement available at `P = 20` — and it is strictly worse, on both axes,
than the dense rule, which achieves 100 %/100 %.

> **Figures 4–5** (`fig04_crosstalk.png`, `fig05_tradeoff.png`).

---

## 6. Experiment 5 — Application: who it is, and what they look like

Protocol: 7 of each subject's 10 poses are stored (280 pairs), 3 are held out
(120 unseen photographs). All poses of a subject share the same Hadamard code,
which gives the two directions semantic roles: X→Y is *identification*, Y→X is
*evocation*.

| System | Accuracy on unseen photographs |
|---|---:|
| Hebb | 45.3 % ± 2.3 |
| Multiple training | 45.3 % ± 2.3 |
| Least squares | 86.8 % ± 1.9 |
| Dense | 86.8 % ± 1.5 |
| *Nearest neighbour (baseline, no BAM)* | *91.5 % ± 1.5* |
| Chance | 2.5 % |

**Honesty requires recording the negative result:** none of the BAMs beats a
nearest-neighbour classifier applied to the same binarised vectors. The BAM is
not competitive as a face classifier, and this work does not suggest otherwise.
What it does offer — and nearest neighbour does not — is the second direction.

### 6.1 The Hebbian BAM synthesises the class prototype

With a Hadamard codebook and the Hebb rule,

```
W y_c = Σ_k x_k (y_kᵀ y_c) = p · Σ_{k ∈ class c} x_k,
```

so `sgn(W y_c)` is **exactly the pixel-wise majority vote of that subject's
training photographs**. Numerical verification over 40 classes and 6 seeds:
**1.0000 matching bits**, i.e. perfect identity.

The Hebbian BAM does not retrieve a stored photograph — it synthesises a
prototype that was never presented. It is the only one of the four rules for
which this result is exact and provable in one line, and it is the strongest
argument in favour of the classical rule anywhere in this study.

> **Figure 6** (`fig06_identification.png`) · **Animation 4**
> (`04_identification.gif`).

---

## 7. Conclusions

1. **Unconditional stability is a weak guarantee.** The BAM always converges,
   including when it converges to garbage. Verified over 800 trajectories
   without a single violation of energy monotonicity — and with spurious states
   in nearly every configuration of the classical rule at `P ≥ 20`.

2. **Real capacity on correlated data is an order of magnitude below the
   classical predictions.** Kosko predicts 256 pairs; the i.i.d. analysis
   predicts 23; we measure zero useful capacity in the face→code direction. The
   gap is not experimental noise, it is the correlation between faces
   (0.19–0.32), absent from both predictions.

3. **The error is predictable.** The crosstalk model reproduces the per-bit
   error with no fitted parameter (`r = 0.984` in log-log, median relative
   error 4 %). Anyone designing a BAM can size it before training it, by
   computing `Φ(−n/√(Σc²))` from the patterns' Gram matrix.

4. **The label layer is the cheap design variable.** Swapping random codes for
   Hadamard codes multiplies code→face capacity by 20, at no cost. It is the
   best benefit-to-effort intervention in the whole study.

5. **But bidirectionality charges the price of the worse direction.** No
   one-directional gain survives resonance if the other direction is poor. In
   real applications that means a BAM should only resonate when both directions
   are trustworthy; otherwise use it as a pair of single-pass maps — which is,
   in fact, how it works well.

6. **The exponential interaction fixes everything the classical rule does
   not** — capacity, robustness and resonance — but at the cost of abandoning
   the weight matrix, and with it the local implementation that was the model's
   original biological motivation. The gain is real and so is the conceptual
   price.

---

## 8. Limitations

* **A single dataset.** Every number here refers to Olivetti/ORL, with aligned
  faces, controlled background and only 40 subjects. The qualitative
  conclusions (crosstalk governs capacity; orthogonality helps one direction;
  resonance propagates the worse direction's error) depend only on the
  correlation between patterns and should transfer; the numerical values will
  not.
* **Binarisation is mandatory.** The BAM is bipolar by construction. Binarising
  discards intensity information a real classifier would use — part of the gap
  to nearest neighbour may come from that.
* **Multiple training was capped at 25–30 epochs** in the sweeps, for cost. It
  does not converge on faces within that budget; we cannot rule out that it
  would with a much larger one, though `q_k` grows with no sign of settling.
* **The dense rule used a fixed `β = 12`**, untuned per configuration. A tuned
  `β` would probably improve its results, not worsen them.
* **The crosstalk model assumes independence across the destination layer's
  bits** when converting per-bit error into perfect recall. Errors are
  correlated through the same interfering memories, so the predicted
  perfect-recall curve acts as an optimistic upper bound — which is why the
  validation in §5 is done on the **per-bit** error, which does not rely on
  that assumption.

---

## References

- Kosko, B. (1988). Bidirectional associative memories. *IEEE Transactions on
  Systems, Man, and Cybernetics*, 18(1), 49–60.
- Haines, K. & Hecht-Nielsen, R. (1988). A BAM with increased information
  storage capacity. *IEEE International Conference on Neural Networks*.
- Wang, Y.-F., Cruz, J. B. & Mulligan, J. H. (1990). Two coding strategies for
  bidirectional associative memory. *IEEE Transactions on Neural Networks*,
  1(1), 81–92.
- McEliece, R., Posner, E., Rodemich, E. & Venkatesh, S. (1987). The capacity of
  the Hopfield associative memory. *IEEE Transactions on Information Theory*,
  33(4), 461–482.
- Personnaz, L., Guyon, I. & Dreyfus, G. (1986). Collective computational
  properties of neural networks: new learning mechanisms. *Physical Review A*,
  34(5), 4217–4228.
- Krotov, D. & Hopfield, J. J. (2016). Dense associative memory for pattern
  recognition. *Advances in Neural Information Processing Systems*, 29.
- Ramsauer, H. et al. (2020). Hopfield networks is all you need.
  *arXiv:2008.02217*.
- Samaria, F. & Harter, A. (1994). Parameterisation of a stochastic model for
  human face identification. *IEEE Workshop on Applications of Computer
  Vision*. (The ORL/Olivetti database.)
