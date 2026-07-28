# Bidirectional Associative Memory (BAM) on real faces

An experimental study of Kosko's 1988 model and three alternatives, on the
**Olivetti/ORL** face database (400 images, 40 subjects, 64×64 pixels).

The full report — methods, results and discussion — is in
**[`REPORT.md`](REPORT.md)**.

---

## The finding in one sentence

The BAM is unconditionally stable — it always converges, for any weight
matrix — and that guarantee does not stop it from converging to spurious states
on real faces from as few as five stored patterns. The reason is measurable,
predictable and, in part, avoidable.

---

## Install

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
```

The Olivetti database is downloaded automatically on first run and cached in
`data/`. No manual download is needed.

## Reproduce

```bash
.venv/bin/python experiments/run_all.py     # full study (~6 min)
.venv/bin/python -m pytest tests/           # 63 property tests
```

Each experiment also runs on its own:

```bash
.venv/bin/python experiments/exp2_capacity.py
```

Everything is deterministic given `bam.config.SEED`.

---

## The five experiments

| # | Question | Outputs |
|---|---|---|
| **1** | Does it converge? And to the right pair? | GIF 1, `fig01`, `tab01` |
| **2** | How many pairs does it really store? | GIF 2, `fig02`, `tab02` |
| **3** | How much corruption does the probe tolerate? | GIF 3, `fig03`, `tab03` |
| **4** | Is the error predictable from the correlations? | `fig04`, `fig05`, `tab04` |
| **5** | Is the BAM any good at identifying faces? | GIF 4, `fig06`, `tab05` |

Results land in `results/`: `gifs/` (square 1080×1080, black background,
Portuguese captions), `figures/` (PNG) and `tables/` (CSV with the raw sweep
data behind every claim).

### Animations

Four, one per question, each stating its own conclusion in its closing frames.

| File | Question it answers |
|---|---|
| `01_bidirectional_recall.gif` | What does a BAM actually do? A corrupted face goes in; the pair comes back. Then the face is erased and the code alone restores it. |
| `02_capacity.gif` | How many pairs fit? Same network, same rule — only the codebook changes, and capacity moves 20×. |
| `03_robustness.gif` | How much damage survives? Flipped pixels against erased pixels, corrupted by the same fraction at the same time. |
| `04_identification.gif` | What is it for? Unseen photograph → identity → the synthesised prototype of that person. |

Experiment 4 (crosstalk) produces no animation: its result is a scatter of
predictions against measurements, which is read rather than watched.

---

## Layout

```
bam/
  core.py        the model and the 4 rules (Hebb, multiple training,
                 least squares, dense); energy and recall
  datasets.py    Olivetti loading and the two binarisations
  codebooks.py   Hadamard, random and correlated codebooks
  theory.py      crosstalk SNR and the classical capacity bounds
  metrics.py     recall metrics, noise and occlusion
  viz.py         dark theme and plotting primitives (palette validated on black)
  ptbr.py        the Portuguese strings rendered inside the animations
  animation.py   GIF assembly
  config.py      paths, seed and constants

experiments/     exp1..exp5 + run_all.py
tests/           63 tests of the model's mathematical properties
results/         figures/ · gifs/ · tables/
```

## The four rules

| Rule | Form | Reference |
|---|---|---|
| Hebb | `W = Σ x_k y_kᵀ` | Kosko (1988) |
| Multiple training | `W = Σ q_k x_k y_kᵀ`, `q_k` adjusted iteratively | Wang, Cruz & Mulligan (1990) |
| Least squares | minimises the λ-weighted bidirectional error; Sylvester equation solved in closed form via SVD | derived in this work |
| Dense | exponential interaction, no weight matrix | Krotov & Hopfield (2016) |

```python
from bam.core import BAM
from bam.codebooks import build_codebook
from bam.datasets import load_dataset

data = load_dataset(method="mean-face")
X = data.X[:, :20]
Y = build_codebook("hadamard", 20, 64)

model = BAM("dense").fit(X, Y)
traj = model.recall(x0=noisy_probe)        # full resonance
code = model.associate_forward(X)          # single-pass X→Y
faces = model.associate_backward(Y)        # single-pass Y→X
```

---

## Headline results

- **Useful capacity (≥ 95 % perfect recall, `n = 4096`, `p = 256`):** Kosko
  predicts 256 pairs; we measure **0** in the face→code direction with the
  classical rule.
- **Hadamard codes multiply code→face capacity by 20** (10 → 200+ pairs) by
  cancelling crosstalk exactly.
- **Bidirectional resonance destroys that gain:** 100 % in a single pass, 0 %
  once the network resonates.
- **Crosstalk theory predicts the per-bit error with no fitted parameter:**
  `r = 0.984` in log-log, median relative error 4 %.
- **Exactness ≠ robustness:** 10 % of code noise drops Hebb+Hadamard from 100 %
  to 29 %.
- **The Hebbian BAM evokes the class prototype**, bit-for-bit identical to the
  majority vote of the training poses (verified over 40 classes × 6 seeds).
- **As a classifier the BAM loses to nearest neighbour** (86.8 % vs 91.5 %) — a
  negative result recorded, not hidden.

## Data licence

The Olivetti/ORL database belongs to AT&T Laboratories Cambridge and is
distributed for research purposes. This repository does not redistribute the
images: scikit-learn downloads them on first run.
