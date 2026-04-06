# Normal vs Restoration: Computation Path Comparison

This document compares the **currently implemented** `normal` and
`restoration` phases in `moto`, using the current code as the source of truth.

It focuses on one question:

- for the same SQP shell, what differs between normal and restoration in
  problem definition, assembly, refinement, public summaries, and globalization?

Checked files:

- [src/solver/sqp_impl/ns_sqp_impl.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp)
- [src/solver/sqp_impl/restoration.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/restoration.cpp)
- [src/solver/sqp_impl/iterative_refinement.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/iterative_refinement.cpp)
- [src/solver/sqp_impl/line_search.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/line_search.cpp)
- [src/solver/sqp_impl/print_stat.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/print_stat.cpp)
- [src/solver/restoration/resto_overlay.cpp](/home/harper/Documents/moto/src/solver/restoration/resto_overlay.cpp)
- [normal_iteration.md](/home/harper/Documents/moto/normal_iteration.md)
- [restoration.md](/home/harper/Documents/moto/restoration.md)

## 1. Shared Outer Skeleton

Both phases reuse the same outer SQP shell:

1. assemble stage data
2. solve the stagewise NSP / Riccati subproblem
3. recover primal and dual Newton steps
4. optionally run iterative refinement
5. run backtracking line search
6. accept, reject, or fail

So the implementation difference is not the shell.
It is the active problem definition and the phase-specific globalization rules.

## 2. Original Problem Layer

### 2.1 Normal

Normal uses the standard barrierized user NLP:

- original stage cost
- hard constraints
- normal inequalities under the normal IPM layer
- normal soft equalities under the normal PMM layer

`compute_kkt_info()` and `print_stats()` report public summaries of this normal
phase problem.

### 2.2 Restoration

Restoration now runs on a separate **overlay graph**.
Its original problem is:

- hard dynamics remain hard
- original non-dynamics constraints are replaced by elastic wrappers
- cost becomes `proximal + exact elastic penalty`

So restoration differs from normal only by:

1. active cost
2. active non-dynamics constraints

The solver shell is otherwise the same.

## 3. Graph / Storage Ownership

### 3.1 Normal

Normal uses the main realized solver graph stored in the active `graph_model_state`.

### 3.2 Restoration

Restoration does **not** mutate the normal graph in place.

It creates a separate graph from composed finalized interval `edge_ocp`s and
then builds restoration overlays on top of them.

Therefore normal and restoration do **not** share:

- `node_data`
- `lag_data`
- KKT summaries
- line-search trial storage

They synchronize only selected primal and dual state at restoration entry/exit.

## 4. Stage Assembly

### 4.1 Normal

Normal stage assembly is the usual one:

- base cost into `cost_ / cost_jac_ / lag_hess_`
- constraint residuals into `approx_[cf]`
- solver-owned first-order corrections into `lag_jac_corr_`
- solver-owned second-order corrections into `hessian_modification_`

### 4.2 Restoration

Restoration stage assembly now also goes entirely through standard `cost/constr`
objects:

- `resto_prox_cost` is a real `__cost`
- `resto_eq_elastic_constr` is a real soft-constraint wrapper
- `resto_ineq_elastic_ipm_constr` is a real inequality wrapper

So restoration no longer injects framework-external stage algebra.

The split is:

- prox Hessian -> base `lag_hess_`
- elastic condensed first-order terms -> `lag_jac_corr_`
- elastic condensed second-order terms -> `hessian_modification_`

That is the same structural split as normal.

## 5. Internal Linear Solve

### 5.1 Normal

Normal solves the stagewise NSP / Riccati reduced system built from the
assembled stage QP.

### 5.2 Restoration

Restoration uses the **same** NSP / Riccati machinery.

It does not use a second Riccati implementation.
The only difference is that the assembled stage problem comes from the
restoration overlay graph rather than the normal graph.

## 6. Iterative Refinement

### 6.1 Normal

Normal iterative refinement reduces the recovered original-phase Lagrangian
stationarity residual in stage `x/u/y` coordinates.

That is the contract documented in
[normal_iteration.md](/home/harper/Documents/moto/normal_iteration.md).

### 6.2 Restoration

Restoration reuses the same iterative-refinement shell on the overlay graph.

Important current behavior:

- there is no longer any framework-external `resto_runtime` residual path
- restoration local residual summaries, when used, are aggregated from the
  active overlay wrappers themselves

So restoration IR is now structurally aligned with normal:

- same solver shell
- same correction infrastructure
- different active phase problem

## 7. Public Objective Summary

### 7.1 Normal

Normal now distinguishes:

- `objective`
- `penalized_obj`

The printed `obj` column is `objective`.
The line-search logic uses `penalized_obj`.

### 7.2 Restoration

Restoration uses the same split:

- `objective`
  - restoration original objective
  - currently `prox + exact penalty`
- `penalized_obj`
  - restoration search objective used by line search
  - `objective - search_barrier_value`

This distinction matters because restoration logs can legitimately show:

- moderate `objective`
- much larger `search_obj`

without that implying a mismatch in stage re-evaluation.

## 8. Globalization

### 8.1 Normal

Normal filter globalization uses:

- normal primal metric
- normal search objective
- standard filter / Armijo switching logic

### 8.2 Restoration

Restoration uses the same backtracking shell, but a restoration phase adapter:

- primal metric: `inf_prim_res`
- dual metric: `max(inf_dual_res, inf_comp_res)`
- objective metric: `penalized_obj`

Restoration also has a second outer return check:

- a restoration candidate must additionally be acceptable to the outer normal
  filter before restoration returns successfully

## 9. `alpha_min`

### 9.1 Normal

Normal uses the usual line-search `alpha_min`.

### 9.2 Restoration

Restoration does not currently use an independent absolute minimum step size.
Its effective lower bound is:

$$
\alpha_{\min}
=
\max\!\Bigl(
\texttt{settings.ls.primal.alpha\_min},
\texttt{settings.restoration.alpha\_min\_factor}\cdot \alpha_{\mathrm{init}}
\Bigr).
$$

So if the restoration initial step is already tiny, the restoration
`alpha_min` can still be tiny.

This is visible in the current `arm` logs.

## 10. Logging / Iteration Display

### 10.1 Normal

Normal prints one line per accepted SQP iteration through `print_stats(...)`.

### 10.2 Restoration

Restoration now also uses `print_stats(...)` for each restoration iteration.

Differences in the log:

- restoration entry prints `=== enter restoration ===`
- restoration iteration numbers have an `r` suffix such as `25r`
- restoration prints an extra line with:
  - `objective`
  - `search_obj`
  - `barrier`
- restoration exit prints `=== leave restoration: ... ===`

So the logs now make phase transitions explicit.

## 11. Current Alignment Status

The current implementation is now aligned in these ways:

- same SQP shell
- same `cost/constr` framework
- same `lag_jac / lag_jac_corr / hessian_modification_` lifecycle
- same NSP / Riccati machinery
- same iterative-refinement shell
- same top-level `print_stats(...)` formatting path

The current remaining issue is no longer dataflow ambiguity.
It is numerical:

- in the current `arm` case, restoration reaches the overlay graph correctly
- restoration objective and search objective are re-evaluated correctly
- but restoration still ends in tiny-step failure because useful primal progress
  only appears at extremely small accepted steps
