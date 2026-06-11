Augments SAM3's deep memory bank with RAFT optical flow as an explicit
spatial prior. Works by injecting RAFT-warped mask prompts into SAM3's
per_frame_geometric_prompt slots, so the tracker sees BOTH its own
temporal memory AND our geometric motion prediction each frame.

```python

# ===========================================================================
#            QUANTITATIVE TRACKING STABILITY REPORT
# ===========================================================================
# Metric                              | Vanilla SAM3    | Hybrid SAM3+RAFT
# ---------------------------------------------------------------------------
# Mean Warping IoU (higher=best)      | 0.993838          | 0.993795
# Std Warping IoU (lower=best)        | 0.003816          | 0.003412
# Mean Consecutive IoU (higher=best)  | 0.995593          | 0.995479
# Std Consecutive IoU (lower=best)    | 0.003007          | 0.002649
# Mean Perimeter Jitter (lower=best)  | 7.99           | 4.14
# Std Perimeter Jitter (lower=best)   | 12.00           | 4.88
# ---------------------------------------------------------------------------
# Mean Soft Warping IoU (higher=best) | 0.991230          | 0.991679
# Std Soft Warping IoU (lower=best)   | 0.003379          | 0.003716
# Mean Soft Consecutive IoU (hi=best) | 0.995593          | 0.994691
# Std Soft Consecutive IoU (lo=best)  | 0.003007          | 0.003739
# ===========================================================================
# [DEBUG] Total binarized mask pixels: 634388480
# [DEBUG] Differing binarized mask pixels: 696522 (0.109794%)
# ===========================================================================
```
