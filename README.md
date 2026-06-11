Uses SAM3 and RAFT as a motion compensated geometric priori (motion guided prompting)




```python
===========================================================================
           QUANTITATIVE TRACKING STABILITY REPORT
===========================================================================
Metric                              | Vanilla SAM3    | Hybrid SAM3+RAFT
---------------------------------------------------------------------------
Mean Warping IoU (higher=best)      | 0.984607          | 0.985049
Std Warping IoU (lower=best)        | 0.028522          | 0.028647
Mean Consecutive IoU (higher=best)  | 0.988062          | 0.988505
Std Consecutive IoU (lower=best)    | 0.027898          | 0.028009
Mean Perimeter Jitter (lower=best)  | 26.85           | 14.66
Std Perimeter Jitter (lower=best)   | 138.69           | 91.53
---------------------------------------------------------------------------
Mean Soft Warping IoU (higher=best) | 0.981806          | 0.983376
Std Soft Warping IoU (lower=best)   | 0.028442          | 0.028692
Mean Soft Consecutive IoU (hi=best) | 0.988062          | 0.987669
Std Soft Consecutive IoU (lo=best)  | 0.027898          | 0.028257
===========================================================================
[DEBUG] Total binarized mask pixels: 509048064
[DEBUG] Differing binarized mask pixels: 488665 (0.095996%)
===========================================================================

```
