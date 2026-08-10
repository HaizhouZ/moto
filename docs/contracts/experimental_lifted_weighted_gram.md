# Experimental Lifted Weighted-Gram Contract

Status: isolated experiment awaiting production approval.

## Objective

Determine whether replacing the existing lifted Hessian contraction schedule
with one symmetric weighted-Gram kernel materially reduces hot execution time.
This experiment does not authorize a production-path change.

## Fixed Inputs

- `Q_ll` has the Go2 lifted-acceleration layout: 18 acceleration diagonal
  entries and four adjacent 3-entry contact-force diagonal panels.
- `Z_l` is dense `30 x 12`.
- `l_y` is dense `30 x 36`, physically split at the lifted-symbol row
  boundaries.
- Both candidates consume identical numerical values, alignment, dimensions,
  output blocks, threading, compiler flags, and cache traversal order.

## Candidates

Existing schedule:

1. form `T = Q_ll R_l`;
2. form the top block `Z_l^T T`;
3. form the state block `l_y^T T_x`.

Experimental schedule:

1. form the upper triangle of `R_l^T Q_ll R_l` directly;
2. expose its input-input, input-state, and state-state blocks;
3. mirror the state-state triangle because current `V_xx` storage is full.

## Acceptance

- Maximum absolute output difference must be at most `1e-11`.
- Compare median hot runtime over repeated traversals of 100 stage buffers.
- Report both candidates and speedup; do not adopt the experiment merely
  because it is faster.
- Production integration requires separate user approval after the result is
  known.
