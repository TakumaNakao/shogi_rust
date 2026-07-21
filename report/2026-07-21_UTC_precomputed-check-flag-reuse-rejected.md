---
status: rejected
revision: 0368306 (uncommitted candidate measured on top)
inputs:
  - policy_weights_halfkp64_kpp_distilled_v2.5.0.binary
  - taya36.sfen
conclusion: Reusing move-ordering check flags made the median 0.43% slower, so the candidate was reverted.
superseded_by: null
---

# Precomputed check flag reuse

## Hypothesis

Alpha-beta and root move ordering call `Position::is_check_move` to award the check bonus, then
`Position::do_move` calculates the same value again for a searched move. Carrying the boolean in the
scored move tuple and passing it to move application could remove the duplicate calculation.

## Semantics gate

The candidate added a debug assertion comparing the supplied value with `is_check_move` and a test
covering quiet moves, capture/promotion, and a checking drop. HalfKP-64 library tests and the fixed
search fingerprint passed exactly. Nodes and every qsearch/search counter matched the control.

## Measurement

Both revisions were separately built with release ThinLTO and run alternately:

```text
search_profile --halfkp-weights policy_weights_halfkp64_kpp_distilled_v2.5.0.binary \
  --positions taya36.sfen --samples 16 --depth 5 --seed 9501 --threads 1
```

| revision | sorted elapsed ms | median ms |
|---|---|---:|
| control `0368306` | 6350.93, 6432.72, 6503.83, 6507.39, 6545.13, 6547.60, 6549.24 | 6507.39 |
| candidate | 6415.62, 6509.41, 6511.02, 6535.22, 6537.66, 6539.91, 6591.44 | 6535.22 |

Candidate delta was `+0.428%`. The extra boolean widened every scored-move tuple and did not produce a
measurable benefit large enough to justify the API and representation cost. The implementation and its
test were reverted; this report is the retained artifact.
