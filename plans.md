# Derivative Propagation Redesign Plan

## Status

- This plan supersedes the old box-migration notes.
- The old `plans.md` content is intentionally cancelled.
- The new goal is not "add more boxed special cases".
- The new goal is to remove solver-local Jacobian algebra wrappers and move derivative propagation back to the stage data flow.
- Completed:
  - restoration dual-step commit and bound-multiplier reset now route through `ineq_constr` interfaces instead of direct driver-side IPM state mutation
  - PMM now publishes row/diag propagation coefficients through `soft_constr::approx_data`, and `node_data::update_approximation()` applies those coefficients centrally
  - restoration overlays now publish row/diag propagation coefficients for their normal `jacobian_impl/hessian_impl` relinearization path, and `node_data::update_approximation()` applies those coefficients centrally
  - a shared immediate-propagation entry now exists in `solver::ineq_soft::apply_immediate_jacobian_precorrection(...)`, and restoration/IPM corrector paths use it for Jacobian correction updates that must happen outside the main `update_approximation()` pass
  - IPM now publishes row/diag propagation coefficients for its normal `jacobian_impl()` path, and uses the same immediate propagation entry for corrector-time Jacobian updates
  - primal residual summary overrides are now published through soft-constraint approx-data, `generic_constr::primal_residual_inf/l1` were removed, and `node_data` again owns primal summary aggregation
  - restoration overlay now routes its precorrection publishing through the shared immediate-propagation / published-term path instead of carrying its own direct Jacobian/Hessian accumulation loops
  - `accumulate_jtj_identity` has been removed from the shared helper layer; the remaining helper layer is now smaller and limited to paths not yet moved off `matrix_ref`-level Jacobian access
  - `compute_jacobian_step` has been removed from the shared helper layer; IPM, PMM, and restoration now use local minimal panel-step logic at their actual Newton-step use sites
  - `solver/detail/jac_sparsity_ops.hpp` has been removed entirely; the last shared `row_times_jacobian` and weighted `J^T D J` logic now live inside the single published-propagation entry in `ineq_soft_impl.cpp`
  - the remaining `matrix_ref`-level Jacobian panel operations have been moved onto `func_approx_data` itself, so IPM/PMM/restoration and the shared propagation entry now call data-layer methods instead of carrying their own local panel helpers
  - Gauss-Newton identity `J^T J` accumulation now also routes through `func_approx_data`, so `cost.cpp` no longer carries its own duplicate Jacobian panel view logic
  - remaining solver-side Jacobian block-availability checks are now routed through `func_approx_data::has_jacobian_block(...)`, so call sites no longer inspect `jac_` storage shape directly
  - PMM/restoration Newton-step paths now also skip absent Jacobian blocks through `func_approx_data::has_jacobian_block(...)`, and restoration prox cost no longer directly indexes `d.jac_[arg_idx]`
- Still intentionally not migrated:
  - moving weighted `J^T D J` algebra fully under a `sparse_mat` / `spmm` centered API instead of the current helper layer
- Confirmed blocker:
  - current solver-level `func_approx_data::jac_` access is `matrix_ref` panel storage, not `sparse_mat`
  - therefore a direct mechanical replacement of solver-side helper calls with `sparse_mat::times/right_T_times` at current call sites is the wrong move
  - the next real step must move propagation upward in the data flow, not keep rewriting solver-local helper call sites

## Problem Statement

The current branch has grown a second abstraction layer for Jacobian propagation:

- `solver/detail/jac_sparsity_ops.hpp`
- helper functions such as:
  - `compute_jacobian_step`
  - `accumulate_row_times_jacobian`
  - `accumulate_weighted_jtj`
  - `accumulate_jtj_identity`

This is the wrong design center.

We already have:

- `lag_data::approx_[field].jac_[pf]` as the canonical Jacobian storage
- `sparse_mat` as the canonical sparse/dense/diag/eye panel representation

So Jacobian/Hessian propagation should be designed around:

- `node_data`
- `lag_data`
- `sparse_mat`

and not around a new solver-only helper layer.

## Target Design

### 1. Single Jacobian source of truth

The only Jacobian representation should remain:

- `lag_data::approx_[field].jac_[pf]`

which is a `sparse_mat`.

No solver module should maintain or require a parallel "Jacobian algebra helper API" just to multiply:

- `J * dx`
- `J^T * r`
- `J^T * D * J`

### 2. Propagation moves to `node_data`

Constraint derivative propagation should be orchestrated by `node_data::update_approximation()` or a closely-related stage-level pass.

Constraint implementations should no longer directly write:

- `lag_jac_corr_`
- `hessian_modification_`

through ad hoc Jacobian algebra helpers.

Instead, each constraint should publish only the propagation coefficients it needs, and `node_data` should apply them uniformly through the stored `sparse_mat` Jacobians.

### 3. Constraint-owned propagation coefficients

Soft/IPM/restoration constraints may still compute local condensed quantities, but they should only expose coefficient vectors such as:

- row/Jacobian scaling residual
- Hessian diagonal scaling

Examples:

- IPM:
  - Jacobian correction coefficient similar to current `scaled_res`
  - Hessian diagonal coefficient similar to current `diag_scaling_sum`
- PMM:
  - Jacobian correction coefficient proportional to `g / rho`
  - Hessian diagonal coefficient proportional to `1 / rho`
- restoration overlays:
  - Jacobian correction coefficient similar to local Schur RHS
  - Hessian diagonal coefficient similar to local Schur inverse diagonal

These coefficients should live in constraint approx-data, but propagation should be centralized.

### 4. `sparse_mat`-centric operations

If additional primitives are needed, they should belong to `sparse_mat` / `spmm`, not to `solver/detail/jac_sparsity_ops.hpp`.

Acceptable directions:

- add a proper `J * x` helper around `sparse_mat::times`
- add a proper `r^T * J` helper around `sparse_mat::right_T_times`
- add a proper weighted panel traversal for `J^T D J`
- add identity-weighted `J^T J` support as a `sparse_mat` / `spmm` facility if truly needed

Unacceptable direction:

- growing more solver-local wrappers that switch on `jac_sparsity()` and manually reconstruct sparse algebra outside `sparse_mat`

## Structural Changes

### 5. Node-owned primal residual summaries

Done:

- `generic_constr::primal_residual_inf`
- `generic_constr::primal_residual_l1`

have been removed.

Current design:

- `node_data` owns primal residual summaries again
- soft/IPM constraints may publish residual overrides through approx-data
- SQP summary code reads cached stage summaries again, like the old flow

### 6. Add propagation views to soft constraint approx-data

Add optional stage-level propagation views to `soft_constr::approx_data` and/or `ineq_constr::approx_data`.

Proposed minimal shape:

- `const vector *jacobian_row_scaling_ = nullptr`
- `const vector *hessian_diag_scaling_ = nullptr`

or mutable owned buffers with pointer-style enable/disable semantics.

Constraints set these when they have active corrections for the current pass.

### 7. Centralize stage propagation

Add a node-level propagation pass after function evaluation:

- propagate all pending Jacobian corrections from constraint coeffs
- propagate all pending Hessian corrections from constraint coeffs

Possible shape inside `node_data::update_approximation()`:

1. evaluate values/Jacobians/Hessians into sparse approx-data
2. assemble base `lag_jac_` from `J^T lambda`
3. if derivative mode is active:
   - walk soft/ineq constraints
   - read their propagation coefficient views
   - apply `J^T r`
   - apply `J^T D J`

This restores a single stage-centric assembly path.

### 8. Move restoration reset/sync behind interfaces

`restoration.cpp` should not manually mutate IPM internals such as:

- side multipliers
- multiplier backups
- box side state

Instead add explicit interfaces on the relevant constraint layer, probably `ineq_constr` / `soft_constr`, for operations like:

- sync restoration state back to outer inequality state
- reset bound multipliers after restoration exit
- commit accepted post-restoration bound state

The restoration driver should only call those interfaces.

## Concrete Refactor Steps

### Phase A. Remove extra SQP bookkeeping

1. Keep SQP primal/step summary code aligned with the old shape.
2. Remove any fallback logic that exists only because of the new helper layer.
3. Keep only the minimum additional logic required for boxed constraints.

Done when:

- `accumulate_trial_direction_terms` matches old flow as closely as possible
- `finalize_step_info` no longer carries redundant derived/fallback bookkeeping
- SQP summaries are again driven by node-level cached quantities whenever possible

### Phase B. Delete `jac_sparsity_ops.hpp`

Done:

- `solver/detail/jac_sparsity_ops.hpp` has been removed
- solver call sites no longer include it

Current follow-up:

- the only remaining `matrix_ref`-level sparsity switching now lives in `func_approx_data`
- the next cleanup step is to decide whether these methods should stay as `func_approx_data` data-layer utilities, or move further down into lower-level panel/SPMM facilities

### Phase C. Introduce stage-owned propagation coefficients

1. Extend constraint approx-data with propagation coefficient buffers/views
2. Make:
   - IPM
   - PMM
   - restoration overlay
   fill those buffers instead of directly touching `lag_jac_corr_` / Hessian mods

Done when:

- solver constraints compute local condensed coefficients only
- propagation itself is no longer implemented independently in each solver module

### Phase D. Centralize derivative propagation in `node_data`

1. Add stage-level propagation routines operating on:
   - stored `sparse_mat` Jacobians
   - constraint-provided scaling vectors
2. Route all Jacobian/Hessian correction assembly through that stage pass

Done when:

- `publish_jacobian_precorrection()` and `publish_hessian_precorrection()` in solver constraints are empty or trivial metadata publishers
- `node_data` is the single place that applies `J^T r` and `J^T D J`

### Phase E. Move restoration outer-IPM reset/sync behind interfaces

1. Add proper interfaces for:
   - restoration-to-outer state sync
   - post-restoration multiplier reset
   - accepted-state commit
2. Remove all direct mutation of IPM side internals from restoration driver code

Done when:

- `restoration.cpp` no longer reaches into boxed IPM internals directly
- outer-ineq state transitions are invoked through constraint-owned interfaces

## File Landing Zones

### Must-change files

- `include/moto/ocp/soft_constr.hpp`
- `include/moto/ocp/ineq_constr.hpp`
- `include/moto/ocp/impl/lag_data.hpp`
- `include/moto/ocp/impl/node_data.hpp`
- `src/ocp/node_data.cpp`
- `src/solver/ipm_impl/ipm_constr.cpp`
- `src/solver/soft_impl/pmm_constr.cpp`
- `src/solver/restoration/resto_overlay_runtime.cpp`
- `src/solver/sqp_impl/restoration.cpp`

### Files expected to shrink or disappear

- `include/moto/solver/detail/jac_sparsity_ops.hpp`

### Possible `spmm` / `sparse_mat` extension points

- `include/moto/spmm/sparse_mat.hpp`
- `src/spmm/*`

## Validation Plan

### Required tests

- `cmake --build build -j8 --target restoration_test equality_multiplier_init_test ineq_jac_sparsity_test`
- `./build/unittests/restoration_test`
- `./build/unittests/equality_multiplier_init_test`
- `./build/unittests/ineq_jac_sparsity_test`
- `ctest --output-on-failure`

### Required runtime checks

- isolated arm example regression still matches baseline behavior
- boxed-vs-stacked restoration equivalence tests still pass
- sparse Jacobian detection tests still pass after `jac_sparsity_ops` removal

## Done Criteria

The redesign is complete only when all of the following are true:

- `jac_sparsity_ops.hpp` is gone
- solver modules no longer implement their own Jacobian algebra helper layer
- stage derivative propagation is centralized in `node_data`
- `sparse_mat` is the only Jacobian algebra abstraction
- restoration no longer manually mutates outer IPM internals
- `generic_constr::primal_residual_inf/l1` are removed unless a proven blocker remains
- SQP bookkeeping is as close to the old version as possible, with only the minimum boxed-specific deviations left

## Non-goals

- introducing a third Jacobian abstraction
- keeping helper wrappers just because they are already written
- preserving recent bookkeeping changes if they are not strictly required for boxed correctness
