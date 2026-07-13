# Adaptive KNN-Based Confidence Sets for Hyperdimensional Test-Time Adaptation

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

## 3. Contribution 1: Non-Parametric KNN Confidence Sets

### Why parametric gates fail in HD space

Standard confidence sets attempt to model the density of each class parametrically:
- **Balls (Prototypes/Centroids):** Assume the trustworthy region is isotropic.
- **Unions of Balls (Subclusters):** Try to capture multi-modality but remain isotropic locally.
- **Ellipsoids (Covariance/Mahalanobis):** Try to capture anisotropy by squashing high-variance directions.

In the high-dimensional, low-sample regime ($n \ll d$) typical of HDC, second-order estimators like covariance ellipsoids are completely noise-dominated. More importantly, attempting to minimize the *volume* of the confidence set (by shrinking principal axes) actively destroys the correctness signal, because domain shift corruptions displace samples precisely along a class's own principal axes. 

Furthermore, parametric metrics ask: *"How close is this point to its own class?"* But pseudo-label correctness is comparative: a label is wrong exactly when another class fits better.

**Hypothesis: The failure of parametric gating is a failure of _estimation_ and _contrastiveness_, not granularity.**

### The fix: Contrastive k-Nearest Neighbors

Instead of fitting parametric models, we use a non-parametric, contrastive k-NN ratio based on a frozen "bank" of source samples. For any target sample, we compute two distances based on cosine similarity:

1.  **$d_{in}$**: The mean distance to its $k$ nearest neighbors *within its predicted class*. This non-parametrically captures local, non-convex density.
2.  **$d_{out}$**: The mean distance to its $k$ nearest neighbors *across all other classes combined*. This represents the nearest competing hypothesis.

**The Output Score (`knn_ratio`)**: We rank confidence using the contrastive ratio **$d_{in} / d_{out}$**. 

```
score(h) = d_in / d_out        admit iff  score(h) <= τ_c
```

with `τ_c` calibrated so the set covers a `δ` fraction of source-domain representations of class `c`.

This bypasses the dimensionality curse entirely because it estimates no covariance matrices. It directly isolates the boundary samples where pseudo-labels are likely to fail, yielding massive AUROC gains over distance-to-centroid and prototype-margin baselines.

## 4. Contribution 2: Adapting the Confidence Sets

A well-shaped but **static** set is calibrated to the source distribution. Per §2, HDC
representations move substantially under shift — so the set must move too.

Two adaptation mechanisms to maintain the k-NN confidence sets:

- **(a) Bank Updating (Tracking the Manifold).** As target samples are admitted with high confidence, they are added to the k-NN bank (e.g., via a FIFO queue replacing older source samples). This allows the non-parametric manifold to seamlessly drift and deform alongside the target domain shift.

- **(b) Conformal Threshold Recalibration.** Adjust the threshold `τ_c` online so
  the set maintains its target coverage `δ` on the observed target stream. This is the
  most principled option: coverage is exactly what conformal prediction guarantees, so
  *maintaining coverage under shift* is the natural adaptation objective. It also
  structurally prevents both failure modes — the gate cannot slam shut (coverage would
  collapse) nor admit everything (coverage would overshoot).

## 5. Evaluation Plan

**Benchmark.** SemanticKITTI -> SemanticKITTI-C and NuScenes -> NuScenes-C

**Primary test — gate quality, independent of adaptation.** AUROC and precision–coverage
curves for predicting pseudo-label correctness, comparing:

| Gate | Implied confidence set / Mechanism |
|---|---|
| **Prototype Cosine Similarity** | Spherical cap (learned direction) |
| **Unsupervised Centroid (Ball)** | First-order, own-class only |
| **Margin (top1 - top2)** | Cheap prototype contrastive |
| **k-NN In-Class Only** | Local density only |
| **k-NN Ratio (Proposal A)** | Contrastive + Non-parametric |

This test needs no adaptation loop and is immune to every confound in the TTA pipeline.
**Run it first** — if the k-NN ratio does not decisively beat the ball on AUROC, the rest of the
method has no foundation.

**Secondary test — adaptation.** Target accuracy under TTA, comparing static vs. adaptive
confidence sets (bank updating + conformal recalibration), against a frozen baseline.

**Reporting discipline.** Adaptation gains must be measured against a frozen model on the
*same* samples. Any metric that accumulates over a stream conflates adaptation with drift
in the stream's intrinsic difficulty, and must not be used as a headline number.