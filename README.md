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
# Mean Warping IoU (higher=best)      | 0.996138          | 0.996198
# Std Warping IoU (lower=best)        | 0.001417          | 0.001358
# Mean Consecutive IoU (higher=best)  | 0.997778          | 0.997840
# Std Consecutive IoU (lower=best)    | 0.001145          | 0.001085
# Mean Perimeter Jitter (lower=best)  | 2.89           | 11.41 *
# Std Perimeter Jitter (lower=best)   | 3.17           | 18.24 *

# ---------------------------------------------------------------------------
# Mean Soft Warping IoU (higher=best) | 0.992781          | 0.993976
# Std Soft Warping IoU (lower=best)   | 0.001172          | 0.001645
# Mean Soft Consecutive IoU (hi=best) | 0.997778          | 0.997697
# Std Soft Consecutive IoU (lo=best)  | 0.001145          | 0.001956
# ===========================================================================
# [DEBUG] Total binarized mask pixels: 89128960
# [DEBUG] Differing binarized mask pixels: 49015 (0.054993%)
# ===========================================================================

# * Helps smooth the boil while increasing detail of things like hair.
```
