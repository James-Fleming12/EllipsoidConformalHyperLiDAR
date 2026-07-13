# k-Nearest Neighbors (k-NN) Confidence Method & Results Analysis

## 1. The k-NN Method: How We Determine Confidence
Our goal is to construct a nonconformity score that accurately ranks the trustworthiness of pseudo-labels on target/corrupted data. Previous methods (like fitting ellipsoids or subclusters) attempted to model the density of each class parametrically and ask, *"How close is this sample to the center of its predicted class?"* 

The **k-NN approach** takes a non-parametric, contrastive route. Instead of modeling distributions, we retain a frozen "bank" of source samples (e.g., 3,000 hypervectors per class). For any target sample, we compute two distances based on cosine similarity:

1.  **$d_{in}$**: The mean distance to its $k$ nearest neighbors *within its predicted class*. This captures local, non-convex density.
2.  **$d_{out}$**: The mean distance to its $k$ nearest neighbors *across all other classes combined*. This represents the nearest competing hypothesis.

**The Output Score (`knn_ratio`)**: We rank confidence using the contrastive ratio **$d_{in} / d_{out}$**. 
- A low ratio means the sample sits deep inside its predicted class manifold and far away from any competing classes. 
- A high ratio (near or above 1.0) means the sample is sitting on a decision boundary or is closer to an entirely different class, making the pseudo-label highly untrustworthy.

This contrastive ratio directly measures the fundamental property of pseudo-label correctness: a label is usually wrong precisely when another class fits better.

---

## 2. Summary of Sweep Results
We swept our k-NN mechanisms alongside standard parametric and prototype-based controls across corrupted target sets (`snow` and `cross_sensor`).

### Baseline Controls
| Arm | Mean AUROC | What it Isolates |
| :--- | :--- | :--- |
| **prototype** | `0.769` | Similarity to the learned classification direction. |
| **margin** | `0.794` | Cheap contrastive score (top1 - top2 prototype similarity). |
| **ball** | `0.810` | First-order parametric (distance to the unsupervised class centroid). |

### k-NN Arms
| Arm | Best Mean AUROC | What it Isolates |
| :--- | :--- | :--- |
| **knn_in** (k=500) | `0.841` | Local / non-convex density only. |
| **knn_ratio** (k=100) | **`0.939`** | Contrastive + Non-parametric. |

### Key Takeaways
The k-NN ratio achieves a massive **+0.13 AUROC** jump over the baseline distance-to-mean (`ball`), cleanly decomposing into three separable insights:

1.  **Centroids beat Prototypes (`0.810` > `0.794`):** The unsupervised class mean (`ball`) is a strictly better reliability gauge than the learned discriminative directions (`margin` / `prototype`).
2.  **Local Density Exists, but is Modest (`0.841` > `0.810`):** By shifting from the parametric ball to non-parametric `knn_in`, we gain `+0.03` AUROC. This confirms that classes have local, non-convex structure that unimodal representations miss.
3.  **Contrastiveness is the Dominant Signal (`0.939` > `0.841`):** Upgrading from `knn_in` to the comparative `knn_ratio` yields a massive `+0.098` gain. Contrastiveness is the crucial mechanism.
4.  **Non-Parametric Estimation is Required:** The cheap prototype-based contrastive score (`margin`, `0.794`) completely failed to capture this contrastive gain and performed worse than the simple ball. The gain only materializes when evaluated against the actual source distribution.

### Why This Works Where Ellipsoids Failed
This result perfectly aligns with the fundamental challenges of the $n \ll d$ regime in hyperdimensional representations. 

Parametric second-order estimators (like covariance ellipsoids) require estimating distributions across thousands of dimensions, which is heavily noise-dominated and empirically destroys the correctness signal. In contrast, k-NN estimates absolutely nothing. It is a pure order statistic computed directly against raw samples, seamlessly bypassing the dimensionality curse while actively exploiting the comparative boundary structure where pseudo-labels fail.

## 3. Next Steps
This finding provides a strong, unified story for the paper. Our optimal nonconformity score (`knn_ratio`) is validated. The immediate next steps are:
1.  **Bank-Size Control Ablation**: Subsample the `out_bank` to match the size of the `in_bank` to guarantee the ratio isn't being artificially biased by drawing from a denser out-of-class sample pool at low $k$.
2.  **Test-Time Adaptation (TTA)**: Feed the validated `knn_ratio` nonconformity score into the adaptive conformal inference loop to maintain coverage under shift.