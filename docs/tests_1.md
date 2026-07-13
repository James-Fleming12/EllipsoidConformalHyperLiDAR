# Implementation Details

Reference: Gao, Shan, Srinivas & Vijayaraghavan, *Computing High-dimensional Confidence
Sets for Arbitrary Distributions* (arXiv:2504.02723). Section numbers below refer to that
paper.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| `d` | HDC dimension (hypervector length) |
| `h ∈ R^d` | a sample's hypervector (post random-projection) |
| `Y_c` | source-domain hypervectors of class `c` |
| `mu_c` | center of class `c`'s confidence set |
| `Σ_c` | covariance of `Y_c` |
| `M_c` | ellipsoidal shape matrix (the preconditioner) |
| `R_c` | radius of the ball in the *transformed* space |
| `δ` | target coverage (e.g. 0.90) |
| `k` | number of ellipsoids in a union |

---

## 1. Building a Single Ellipsoid (offline, per class, on source data)

This is Algorithm **Dense Ellipsoid** (their Fig. 5), specialized to the case where we
already know which points belong to the class (we have source labels), so we can skip
their coarse-ball search over `O(n²)` candidate balls — that search exists to *find* the
dense region in an unlabeled point set. We already know it. **This is a significant
simplification and worth stating explicitly in the paper.**

```
Input:  Y_c  (n × d),  coverage δ,  parameter tau_hat
Output: mu_c,  M_c,  R_c

1.  mu_c  = mean(Y_c)
2.  Sigma = Cov(Y_c)                       # d × d
3.  R     = max_i || y_i - mu_c ||         # scale reference (radius of enclosing ball)
4.  Eigendecompose:  Sigma = V diag(lambda) V^T
5.  Build the shape matrix by BINNING eigenvalues (their §4, step 2c):
        a_j^2 = d   if lambda_j >= tau_hat^2 * R^2 / d      # "high variance" direction
        a_j^2 = 1   otherwise                               # "low variance" direction
        M_c   = sum_j  a_j^2  v_j v_j^T
    with tau_hat = d^{1/4}   (their choice, trading off (ii) vs (iii) in §2.2)
6.  Transform:  Z = M_c^{-1/2} (Y_c - mu_c)
7.  R_c = the δ-quantile of  || z_i ||     # smallest ball at the mean covering δ of Z
```

**Why the binning, rather than plain `Σ^{-1/2}` whitening?** Two reasons, both from §2.2:

- Property (i): with eigenvalues in `{1, d}`, `M_c ⪰ I`, so `M_c^{-1/2}` is *non-expanding*
  in every direction. This guarantees the transformed ball is no larger than the original.
  Full whitening does not give you this.
- Property (iii): at most `d / tau_hat²` eigenvalues get set to `d`, which **bounds the
  volume distortion** of the inverse transform: `vol(S) / vol(M^{-1/2} S) ≤ d^{d/tau_hat²}`.
  Full whitening can blow this up arbitrarily when `Σ` has tiny eigenvalues.

The binning is what makes the volume guarantee go through. It is also numerically much
better behaved than `Σ^{-1/2}` — see §4.

**Store per class:** `mu_c` (d), `V_c` and the binary high/low mask (or directly the
factor `M_c^{-1/2} = sum_j a_j^{-1} v_j v_j^T`), and the scalar `R_c`.

---

## 2. The Gate at Test Time

```
score(h, c) = || M_c^{-1/2} (h - mu_c) ||_2
admit iff  score(h, c) <= R_c
```

Cost: one `d × d` matvec per class (or `d × r` if using the low-rank form of §4). This is
the drop-in replacement for `max_subcluster_similarity(h, c) > threshold`.

**Soft version (recommended for the update weighting):** rather than a hard admit/reject,
use a continuous weight that decays with the score:

```
w(h, c) = clamp( 1 - score(h, c) / R_c , 0, 1 )      # linear ramp, 1 at center, 0 at boundary
```

This keeps the "purify, don't filter" property — nothing is discarded abruptly, samples
just contribute in proportion to how deep inside the confidence set they lie.

---

## 3. Union of `k` Ellipsoids

Their Theorem 1.2 / Algorithm **Greedy Density** (Fig. 6). Applied per class, on that
class's source points.

```
Input:  Y_c,  coverage δ,  number of sets k
Output: list of (mu_j, M_j, R_j) for j = 1..k'   (k' = O(δk/γ))

Remaining = Y_c
Sets = []
while coverage(Sets, Y_c) < δ:
    # find the highest-DENSITY ellipsoid on the remaining points,
    # i.e. the one maximizing  |points covered| / volume
    best = argmax over candidate coverage levels of  (n_covered / vol(E))
           where each candidate E is built by §1 on `Remaining` at that coverage level
    Sets.append(best)
    Remaining = Remaining \ points(best)
```

The greedy step maximizes **points-covered per unit volume**, not just points covered —
this is what keeps the union tight rather than letting it sprawl. `vol(E)` for an
ellipsoid with shape `M` and radius `R` is proportional to `R^d * det(M)^{1/2}`; in
practice work with `log vol = d log R + 0.5 * log det(M)` to avoid overflow at large `d`.

**Gate for a union:** `score(h, c) = min_j score_j(h)` — the sample is admitted if it lies
inside *any* of the class's ellipsoids. (Note this is the exact analogue of
max-over-subclusters, but with the right shape per mode.)

**Key experiment:** sweep `k = 1, 2, 4, 8`. If AUROC improves with `k` for ellipsoids
where it did not for balls, that isolates *shape* as the operative variable and rescues
multi-modality as a legitimate mechanism.

---

## 4. Numerical Concerns (these will bite)

**`Σ_c` is rank-deficient.** With `n_c` samples in dimension `d` and typically `n_c < d`
for HDC, `Σ_c` has rank at most `n_c - 1`. Options, in order of preference:

1. **Low-rank + isotropic tail (recommended, and it is what the binning already does).**
   Keep only the top-`r` eigenvectors (those above the `tau_hat² R²/d` threshold — there
   are at most `d / tau_hat² = d^{1/2}` of them by construction). Everything else gets
   `a_j² = 1`, i.e. identity. So you never need the full eigendecomposition:
   ```
   M_c^{-1/2} = I + sum_{j in high} (d^{-1/2} - 1) v_j v_j^T
   score = || (h - mu_c) + sum_{j in high} (d^{-1/2} - 1) <h - mu_c, v_j> v_j ||
   ```
   This is `O(r·d)` per score, with `r ≈ d^{1/2}`, and requires only a truncated SVD of
   the centered data matrix — no `d × d` covariance ever formed.

2. **Shrinkage:** `Σ_c + λI` with `λ` chosen by Ledoit–Wolf or cross-validation. Use this
   as the *baseline ablation* (see §6), not the main method.

**Do not form `Σ_c` explicitly if `d` is large.** Use a truncated SVD of the centered
`n_c × d` data matrix directly.

**`R` in step 3.** Use a robust radius (e.g. the 99th percentile of `||y - mu||`, not the
max) — a single outlier otherwise inflates `R` and shifts every eigenvalue bin.

**Normalization.** Decide once whether hypervectors are L2-normalized before the ellipsoid
is fit, and be consistent between fit time and test time. If they are normalized, the data
lie on a sphere and the ellipsoid is being fit to a spherical manifold — this is fine
(the tangent-space geometry is still anisotropic) but it means `mu_c` will have norm < 1
and should *not* be re-normalized.

---

## 5. Adapting the Confidence Sets

All three rules operate on the admitted (or softly-weighted) target samples of class `c`
in the current batch. All should be run with the HDC prototype update as usual — the
confidence set governs *which* samples drive the prototype update, and separately adapts
itself.

### (a) Center tracking

```
mu_c <- normalize_if_applicable( (1 - beta) * mu_c + beta * weighted_mean(admitted_c) )
```
Cheapest. Handles pure translation of the class manifold. Start with `beta ≈ 0.01`.

### (b) Center + conformal radius recalibration  ← *the principled one*

Maintain a running estimate of the **empirical coverage** on the target stream:

```
observed_coverage_c  =  (# target samples of class c with score <= R_c) / (# target samples of class c)
```

Then drive `R_c` to hold the target coverage `δ`:

```
R_c <- R_c * (1 + eta * (delta - observed_coverage_c))
```

- If the domain shifts away and the gate starts admitting too little
  (`observed_coverage < δ`), `R_c` **grows** — the gate cannot slam shut.
- If the gate starts admitting too much (`observed_coverage > δ`), `R_c` **shrinks** — the
  gate cannot degrade into "admit everything."

This is an online conformal recalibration: it is the direct analogue of adaptive conformal
inference, and it makes "maintain `δ` coverage under shift" the explicit adaptation
objective. It is also self-stabilizing in exactly the way that fixed thresholds are not.

**Implementation note:** estimate `observed_coverage_c` over a sliding window (last `N`
samples of that class), not cumulatively — a cumulative estimate becomes insensitive to
recent shift.

### (c) Shape tracking

```
Sigma_c <- (1 - beta_s) * Sigma_c + beta_s * Cov(admitted_c)      # low-rank / streaming update
recompute M_c from Sigma_c every T batches (T ~ 100)
```

Most expressive, most drift-prone. **Anchor it:** keep the source shape `M_c^{src}` and
blend `M_c <- (1-a) * M_c + a * M_c^{src}` so it cannot wander arbitrarily far. Recomputing
the eigendecomposition every batch is too expensive; amortize over `T`.

---

## 6. AB Test Matrix

### Phase 1 — gate quality only (no adaptation, no TTA loop)

Metric: **AUROC** for predicting whether the pseudo-label is correct, plus
**precision–coverage curves**. Per corruption, per severity, on CIFAR10-C.

| Arm | What it isolates |
|---|---|
| Prototype cosine | the current baseline gate (ball) |
| Union-of-K max-similarity, `K ∈ {1,2,4,8,16}` | does granularity help *with balls*? |
| **Single ellipsoid (§1)** | does *shape* help? ← **the core hypothesis** |
| **Union-of-k ellipsoids (§3), `k ∈ {1,2,4,8}`** | does granularity help *once the shape is right*? |
| Shrinkage Mahalanobis `(Σ + λI)^{-1}` | does the paper's *specific* preconditioner matter, or would any anisotropic metric do? |

The last row is the ablation a reviewer will ask for. If shrinkage-Mahalanobis matches the
binned-ellipsoid, the contribution is "anisotropy matters," not "this algorithm matters" —
still publishable, but a different claim, and better to know now.

**Also report, per arm:** the **volume** of the resulting confidence set (`log vol`) at
matched coverage. This is the paper's actual theoretical quantity, and showing that
ellipsoids achieve equal coverage at dramatically smaller volume is the cleanest possible
empirical validation of the theory.

### Phase 2 — adaptation (only if Phase 1 succeeds)

Fixed: the best gate from Phase 1. Varying: the update rule.

| Arm | What it isolates |
|---|---|
| Frozen (no adaptation) | the floor |
| Static ellipsoid + prototype update | is a *fixed* confidence set enough? |
| + (a) center tracking | does moving the set help? |
| + (b) center + conformal radius | does maintaining coverage help beyond moving? |
| + (c) full shape tracking | is shape drift worth the cost/risk? |

**Success criterion, set in advance:** (b) ≥ (a) ≥ static on mean target accuracy, with no
individual corruption worse than frozen.

### Phase 3 — the motivating claim, made explicit

Reproduce the CIFAR10 → CIFAR10-C observation as a *figure*, since it is the paper's
premise:

- inter-class separation (e.g. mean inter-class / intra-class distance ratio) in feature
  space vs. HDC space, clean → corrupted;
- distance travelled by class representations (feature-space centroid vs. HDC prototype),
  clean → corrupted, normalized by each space's intra-class scale.

This is the figure that justifies why the confidence sets must be adaptive at all.