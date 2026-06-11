Uses SAM3 and RAFT as a motion compensated geometric priori (motion guided prompting)




```python
===========================================================================
           QUANTITATIVE TRACKING STABILITY REPORT
===========================================================================
Metric                              | Vanilla SAM3    | Hybrid SAM3+RAFT
---------------------------------------------------------------------------
Mean Warping IoU (higher=best)      | 0.983689          | 0.986012
Std Warping IoU (lower=best)        | 0.045097          | 0.009321
Mean Consecutive IoU (higher=best)  | 0.987217          | 0.989549
Std Consecutive IoU (lower=best)    | 0.044763          | 0.006634
Mean Perimeter Jitter (lower=best)  | 17.65           | 8.54
Std Perimeter Jitter (lower=best)   | 115.38           | 22.16
---------------------------------------------------------------------------
Mean Soft Warping IoU (higher=best) | 0.980920          | 0.984359
Std Soft Warping IoU (lower=best)   | 0.045050          | 0.009337
Mean Soft Consecutive IoU (hi=best) | 0.987217          | 0.988708
Std Soft Consecutive IoU (lo=best)  | 0.044763          | 0.007130
===========================================================================
[DEBUG] Total binarized mask pixels: 509048064
[DEBUG] Differing binarized mask pixels: 538757 (0.105836%)
===========================================================================

```
