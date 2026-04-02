# Normal vs Restoration: Computation Path Comparison

This document compares the **currently implemented** `normal` and
`restoration` phases in `moto`, using the code as the source of truth.

It is meant to answer one narrow question:

- for the same top-level SQP skeleton, what does each phase treat as
  its original problem, its internal linear solve, its iterative-refinement
  residual, its public KKT summary, and its globalization logic?

The main files checked while writing this note are:

- [src/solver/sqp_impl/ns_sqp_impl.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp)
- [src/solver/sqp_impl/restoration.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/restoration.cpp)
- [src/solver/sqp_impl/iterative_refinement.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/iterative_refinement.cpp)
- [src/solver/sqp_impl/line_search.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/line_search.cpp)
- [src/solver/nsp_impl/rollout.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/rollout.cpp)
- [src/solver/restoration/resto_runtime.cpp](/home/harper/Documents/moto/src/solver/restoration/resto_runtime.cpp)
- [restoration.md](/home/harper/Documents/moto/restoration.md)
- [normal_iteration.md](/home/harper/Documents/moto/normal_iteration.md)

## 1. Shared Outer Skeleton

Both phases reuse the same broad SQP skeleton:

1. assemble stage data
2. factorize / run Riccati
3. recover primal and dual Newton steps
4. optionally run iterative refinement
5. finalize dual step
6. backtracking line search with trial evaluation
7. accept, reject, or fail

The difference is not the outer control flow.
The difference is what each phase places into that flow.

## 2. Original Problem Layer

### 2.1 Normal

The original `normal` phase problem is the barrierized user NLP:

- original stage cost
- hard constraints
- inequalities handled through the normal IPM layer
- soft equalities handled through the normal PMM layer

Public quantities such as `compute_kkt_info()` and `print_stats()` are meant
to summarize this original normal-phase problem, not the Riccati subproblem.

### 2.2 Restoration

The original `restoration` phase problem is the restoration NLP described in
[restoration.md](/home/harper/Documents/moto/restoration.md):

- hard dynamics remain hard
- equalities become elastic through `(p_c, n_c)`
- inequalities become restoration-owned elastic constraints through
  `(t, p_d, n_d)`
- the phase objective is the restoration objective:
  - proximal `u/y` cost
  - exact `L1` elastic penalty
  - restoration barrier terms

Public quantities in restoration should summarize this restoration problem,
not the Riccati subproblem.

## 3. Stage Assembly

### 3.1 Normal

`normal` stage assembly comes from
[node_data.cpp](/home/harper/Documents/moto/src/ocp/node_data.cpp)
plus normal solver corrections:

- base cost / Jacobian / Hessian
- hard constraint residuals and Jacobians
- IPM / PMM first-order terms into `lag_jac_corr_`
- IPM / PMM second-order terms into `hessian_modification_`

The key storage split is:

- base Hessian: `lag_hess_`
- solver correction Hessian: `hessian_modification_`

### 3.2 Restoration

`restoration` stage assembly is performed by
[assemble_restoration_problem()](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp)
and
[assemble_resto_base_problem()](/home/harper/Documents/moto/src/solver/restoration/resto_runtime.cpp).

The implemented split is:

- raw function evaluation still comes from `update_approximation(...)`
  with normal IPM corrections disabled
- restoration prox on `u/y` is assembled as base restoration cost:
  - value into `cost_`
  - gradient into `cost_jac_` and `lag_jac_`
  - diagonal Hessian into `restoration_prox_hess_diag_`
- restoration elastic equality and inequality blocks are condensed locally and
  contribute:
  - first-order terms into `lag_jac_corr_`
  - second-order terms into `hessian_modification_`

So restoration already mirrors normal's base-vs-correction split:

- base object: restoration cost and hard-constraint Lagrangian terms
- correction object: condensed elastic local reductions

## 4. Internal Linear Solve

### 4.1 Normal

`normal` solves the nullspace / Riccati-reduced stagewise QP:

- projected dynamics
- hard equality elimination
- reduced control-space solve in `Q_zz`

This path is implemented by:

- [presolve.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/presolve.cpp)
- [backward.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/backward.cpp)
- [rollout.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/rollout.cpp)

### 4.2 Restoration

`restoration` reuses the same Riccati machinery.

What changes is the assembled stage system:

- hard dynamics still define the hard part of the linear solve
- restoration base cost replaces the normal barrier objective in the phase solve
- restoration elastic blocks are already condensed into
  `lag_jac_corr_ + hessian_modification_`

So restoration does not currently use a second Riccati implementation.
It uses the same linear solver on a different assembled phase system.

## 5. Iterative Refinement Target

### 5.1 Normal

`normal` iterative refinement does **not** reduce the Riccati-internal `z`
residual.

From
[rollout.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/rollout.cpp)
and
[iterative_refinement.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/iterative_refinement.cpp),
the current implementation does:

- build `kkt_stat_err_[x/u/y]` from
  - `base_lag_grad_backup`
  - Hessian action on the recovered Newton step
  - `J^T * d_lambda`
- then fold `next.x` onto `cur.y`

So the `normal` IR target is:

- the recovered original-phase Lagrangian stationarity residual
- expressed in stage `x/u/y` coordinates
- not the Riccati-internal reduced coordinate residual

### 5.2 Restoration

`restoration` currently reuses the same IR machinery, but with a phase-specific
RHS loader in
[resto_runtime.cpp](/home/harper/Documents/moto/src/solver/restoration/resto_runtime.cpp).

Today the restoration IR path behaves as follows:

- `kkt_stat_err_[x/u/y]` is still built by the generic rollout residual path
- restoration correction RHS is loaded from `kkt_stat_err_[u/y]`
- local elastic residuals are checked separately through
  `refinement_local_residuals(...)`

This means restoration is **not yet fully aligned** with normal:

- normal IR targets a recovered original-phase stationarity residual
- restoration currently mixes:
  - recovered `w=(x,u,y)` residual pieces
  - separately monitored local elastic residuals

That is the current main mathematical gap.

## 6. Public KKT Summary

### 6.1 Normal

In `normal`, [compute_kkt_info()](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp)
reports a public summary of the original barrierized phase problem:

- `objective = cost - mu * log_slack_sum`
- primal violation summary
- stationarity summary
- complementarity summary

It is not the IR residual itself.

### 6.2 Restoration

In `restoration`, `compute_kkt_info()` now follows the same public-summary rule:

- `objective` is the restoration phase objective
- primal residual includes hard dynamics and elastic primal residual summaries
- public stationarity summary is computed from the current iterate's active
  gradients, with the same `next.x -> cur.y` folding pattern as normal
- local elastic stat/comp summaries are included as restoration-problem data,
  not treated as Riccati residuals

This is one place where restoration is now intentionally aligned with normal.

## 7. Globalization

### 7.1 Normal

`normal` uses the usual filter globalization in
[line_search.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/line_search.cpp):

- public objective
- public primal violation
- filter / Armijo switching logic

`inf_dual_res` is a public statistic, but it is not the main filter driver.

### 7.2 Restoration

`restoration` uses a restoration-specific acceptor in the same line-search file.

Its current trial logic uses:

- `inf_prim_res`
- `objective`
- `max(inf_dual_res, inf_comp_res)` as `resto_res`

and then separately requires successful return to the outer filter in
[restoration.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/restoration.cpp).

So restoration currently has two globalization layers:

- inner restoration acceptor
- outer return-to-normal acceptor

## 8. Entry And Exit

### 8.1 Normal

`normal` is the default phase.

If line search returns a tiny-step failure while primal infeasibility is still
large, the outer loop may enter restoration.

### 8.2 Restoration

At entry, restoration:

- snapshots outer duals
- snapshots prox references
- initializes `mu_bar`
- initializes elastic equality and inequality local state
- reassembles the restoration phase problem

At exit, restoration:

- either accepts and returns to normal
- or fails and cleans up

Cleanup currently includes:

- restore outer dual semantics
- copy back restoration-owned bound state on success
- reset equality multipliers according to the restoration cleanup policy
- recompute normal derivatives after leaving restoration

## 9. Current Alignment Status

The current state is:

### Already Aligned

- both phases share the same outer SQP skeleton
- both phases distinguish public phase summary from internal linear solve
- both phases use the same Riccati machinery on phase-specific assembled systems
- restoration public `compute_kkt_info()` is now much closer to normal semantics

### Not Yet Fully Aligned

- `normal` IR target is the recovered original-phase Lagrangian stationarity residual
- `restoration` IR currently still mixes:
  - recovered `w` residuals used for correction
  - separate local elastic residual checks
- so restoration still does not have a single completely unified
  "solve this system, refine this same residual" contract

That unresolved point is the main remaining difference between the two paths.

## 10. Practical Next Step

If restoration is to fully match normal's computation-path contract, the next
step should be:

1. define a single restoration IR target object with the same role that
   normal's recovered stationarity residual plays in normal mode
2. derive the restoration correction RHS from that same object
3. use that same object for restoration IR stop checks
4. keep `compute_kkt_info()` and `print_stats()` as public restoration-problem
   summaries, not as aliases of the internal IR residual

That is the cleanest way to finish the alignment.
