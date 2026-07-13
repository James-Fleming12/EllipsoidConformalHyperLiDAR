# Adaptive Ellipsoidal Confidence Sets for Hyperdimensional Test-Time Adaptation

## 1. Problem Setting

Test-time adaptation (TTA) updates a pre-trained model on an unlabeled target stream,
without retraining and, in our case, without backpropagation. Our model is a
Hyperdimensional Computing (HDC) classifier: a frozen feature extractor produces
features, a random projection lifts them into a high-dimensional space, and each class
is represented by one or more hypervectors. Inference is a similarity comparison;
adaptation is a cheap vector update. No gradients, no optimizer, no batch-norm
statistics to corrupt.

The hard part of unsupervised TTA is not the update rule — it is **deciding which
predictions to trust**. Every update is driven by pseudo-labels, and under domain shift
pseudo-labels become unreliable in a structured, uneven way. Update on everything and
confidently-wrong samples poison the class representations. Gate too aggressively and
you starve the model of the distorted-but-legitimate samples it needs in order to adapt
at all.

So the question that actually matters is:

> **Given a test sample, is it inside the region of representation space where the
> source model's predictions are trustworthy?**

That is a **confidence set** question, not a classification question. This paper treats
it as one.

## 2. The Empirical Observation (CIFAR10 → CIFAR10-C)

Our starting point is a property of HDC representations that we measure directly on
CIFAR10 → CIFAR10-C:

> **HDC representations degrade more gracefully under domain shift than the feature
> space they are built from — inter-class separation is better preserved — but the HDC
> class representations must *move further* to track the shifted domain than their
> feature-space counterparts do.**

Two consequences follow, and they set up the two halves of the contribution:

1. **HDC is a good place to do TTA.** Because inter-class separation survives the
   shift, the geometry remains informative: there is still a meaningful notion of "this
   sample lies near the class manifold." A gate built in HDC space has better raw
   material to work with than one built in feature space.

2. **But a *static* confidence set will not survive.** The very property that makes HDC
   attractive — class representations travelling further under shift — means any region
   calibrated on the source distribution is, by construction, in the wrong place on the
   target. A frozen confidence set either shuts (admitting nothing, starving adaptation)
   or falls out of calibration (admitting the wrong things).

So we need **(a)** a better-shaped confidence set, and **(b)** a way to move it as the
domain moves. Those are the two contributions.

## 3. Contribution 1: Ellipsoidal Confidence Sets

### Why the usual gates are the wrong shape

The standard HDC gates are all implicitly **isotropic**:

- **Prototype gating.** One vector per class; admit if cosine similarity exceeds a
  threshold. The implied confidence set is a spherical cap — a ball.
- **Multi-mode (subcluster) gating.** `K` mode vectors per class; admit if the max
  similarity to any mode exceeds a threshold. The implied confidence set is a **union of
  K balls**.

Both assume the trustworthy region is (a union of) balls. There is no reason a class
manifold in HDC space should be isotropic — variance is typically concentrated in a few
nuisance directions while the discriminative directions are comparatively tight.
Covering an elongated, anisotropic manifold with a **ball** forces a bad tradeoff: the
ball must be large enough to span the elongated direction, which makes it far too
permissive in every other direction. It admits exactly the corrupted samples it was
meant to reject.

**Hypothesis: the failure of similarity-threshold gating is a failure of _shape_, not of
granularity.**

### The fix

We adopt the confidence-set machinery of Gao, Shan, Srinivas & Vijayaraghavan
(*Computing High-dimensional Confidence Sets for Arbitrary Distributions*,
arXiv:2504.02723), whose central result is exactly the one we need.

Finding the minimum-volume **ball** covering a `δ` fraction of an arbitrary distribution
is NP-hard to approximate well (their Thm 1.3 — *proper* learning is intractable). But
allowing the output to be an **ellipsoid** — *improper* learning — does dramatically
better in polynomial time: `exp(Õ(d^{1/2}))`-competitive in volume, versus
`exp(Õ(d / log d))` for the best ball-based (coreset) methods (their Thm 1.1).

The mechanism is precisely the anisotropy fix we want:

1. Estimate a preconditioner `M^{-1/2}` that **shrinks the high-variance directions** of
   the class manifold and leaves the low-variance directions alone.
2. In the transformed (approximately isotropic) space, a **ball centered at the mean** is
   a good confidence set — their Lemma 2.1 shows that once variance is controlled, the
   mean is a good proxy for the true optimal center.
3. Map back through `M^{1/2}`. The result is an **ellipsoid**: tight along the
   discriminative directions, appropriately elongated along the nuisance directions.

The gate becomes a Mahalanobis-style score with their specific preconditioner:

```
score(h) = || M_c^{-1/2} (h - mu_c) ||_2        admit iff  score(h) <= R_c
```

with `R_c` calibrated so the set covers a `δ` fraction of source-domain representations
of class `c`.

**Why volume, not just coverage.** A region that achieves `δ` coverage but sprawls
across the space admits everything and gives no selectivity. Coverage without
volume-optimality is a useless gate. The volume-competitive guarantee is what makes this
a *discriminative* confidence set rather than a tautological one — and it is exactly what
ball-based gates provably cannot achieve in high dimensions.

**Multi-modality, done right.** Their Thm 1.2 and greedy algorithm extend the result to
**unions of `k` ellipsoids**. A class genuinely can be multi-modal; the mistake is
forcing each mode into a ball. Prediction: gate quality should **improve** with `k` for
unions of ellipsoids — in contrast to unions of balls, where added modes buy little and
can actively hurt.

**Conformal guarantee.** The construction conformalizes via nested sets (their §7),
giving distribution-free coverage under exchangeability while remaining approximately
volume-optimal under i.i.d. sampling. The gate is therefore not a heuristic threshold —
it is a conformal predictor with a coverage guarantee.

## 4. Contribution 2: Adapting the Confidence Sets

A well-shaped but **static** set is calibrated to the source distribution. Per §2, HDC
representations move substantially under shift — so the set must move too.

Three update rules of increasing expressiveness, to be ablated:

- **(a) Center tracking.** EMA the ellipsoid center `mu_c` toward admitted target
  samples. Cheapest; handles pure translation of the manifold.

- **(b) Center + conformal radius recalibration.** Additionally adjust `R_c` online so
  the set maintains its target coverage `δ` on the observed target stream. This is the
  most principled option: coverage is exactly what conformal prediction guarantees, so
  *maintaining coverage under shift* is the natural adaptation objective. It also
  structurally prevents both failure modes — the gate cannot slam shut (coverage would
  collapse) nor admit everything (coverage would overshoot).

- **(c) Shape tracking.** Additionally EMA the covariance and periodically recompute the
  preconditioner `M_c`. Most expressive, most drift-prone; needs an anchor back to the
  source shape.

**The story:** we do not merely build a better-shaped confidence set — we keep it
*calibrated* as the domain moves. The extra compute buys geometry (ellipsoids over balls)
and calibration (online coverage maintenance), both motivated by a measured property of
HDC under shift rather than bolted on.

## 5. Evaluation Plan

**Benchmark.** CIFAR10 → CIFAR10-C (primary; where the motivating observation was
measured). Extension to further corruption / cross-domain benchmarks once the method is
established.

**Primary test — gate quality, independent of adaptation.** AUROC and precision–coverage
curves for predicting pseudo-label correctness, comparing:

| Gate | Implied confidence set |
|---|---|
| Prototype cosine similarity | spherical cap (ball) |
| Union-of-K max-similarity | union of K balls |
| **Single ellipsoid** | ellipsoid |
| **Union-of-k ellipsoids** | union of k ellipsoids |
| Shrinkage Mahalanobis (`Σ + λI`)⁻¹ | ellipsoid, no eigenvalue binning |

That last row is the key ablation: it isolates whether the paper's *specific*
preconditioner matters, or whether *any* anisotropic metric suffices.

This test needs no adaptation loop and is immune to every confound in the TTA pipeline.
**Run it first** — if the ellipsoid does not beat the ball on AUROC, the rest of the
method has no foundation.

**Secondary test — adaptation.** Target accuracy under TTA, comparing static vs. adaptive
confidence sets (rules a/b/c), against a frozen baseline.

**Reporting discipline.** Adaptation gains must be measured against a frozen model on the
*same* samples. Any metric that accumulates over a stream conflates adaptation with drift
in the stream's intrinsic difficulty, and must not be used as a headline number.