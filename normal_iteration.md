# Normal Iteration: Math And Runtime Contract

This document summarizes what the current `normal` SQP iteration in `moto` actually does, in terms that are precise enough to serve as the alignment target for `restoration`.

It is intentionally narrower than [AGENTS.md](/home/harper/Documents/moto/AGENTS.md):
- this file focuses on the active normal iteration contract
- especially what system is solved, what residual iterative refinement reduces, and what `print_stats` reports

## 1. Three Different Problems Must Be Distinguished

The current normal iteration involves three different mathematical objects.
They are related, but they are not interchangeable.

### 1.1 Original Problem

The original nonlinear OCP is the user-authored problem:

- original cost terms
- hard constraints
- raw inequalities
- soft equalities

This is the problem represented by the expression graph and evaluated by
[node_data.cpp](/home/harper/Documents/moto/src/ocp/node_data.cpp).

At this level:

- inequalities are still `g(x,u,y) <= 0`
- soft equalities are still user/model constraints, not yet condensed
- there is no Riccati reduction yet

### 1.2 Soft / Inequality Reduced Local Problems

`IPM` and `PMM` do not pass their local auxiliary variables directly into Riccati.
Instead, each such constraint builds a reduced local contribution that is pushed into the stage QP.

The crucial binding rule lives in
[func_data.cpp](/home/harper/Documents/moto/src/ocp/func_data.cpp):

- for `__cost`, function-local `lag_hess_` is bound to stage `lag_hess_`
- for non-cost functions, function-local `lag_hess_` is bound to stage `hessian_modification_`

So reading `d.lag_hess_` inside an IPM / PMM implementation does not mean it is writing into the base cost Hessian storage.

For IPM inequalities in
[ipm_constr.cpp](/home/harper/Documents/moto/src/solver/ipm_impl/ipm_constr.cpp):

- primal slack `s`
- inequality multiplier `z`
- local complementarity/barrier algebra

are condensed into:

- first-order correction in `lag_jac_corr_`
- second-order contribution through the function-local `lag_hess_` view, which for non-cost functions is bound to stage `hessian_modification_`

For PMM soft equalities in
[pmm_constr.cpp](/home/harper/Documents/moto/src/solver/soft_impl/pmm_constr.cpp):

- local multiplier elimination is condensed into:
  - first-order correction in `lag_jac_corr_`
  - second-order contribution through the function-local `lag_hess_` view, which for non-cost functions is bound to stage `hessian_modification_`

So these soft/ineq reduced problems are local Schur-complement style subproblems.
They are not the original problem, and they are not yet the Riccati-reduced global subproblem.

### 1.3 Stagewise Riccati Subproblem

After local soft/ineq reductions have already been folded into the stage derivatives,
the solver builds the stagewise reduced QP used by nullspace / Riccati:

- projected dynamics
- hard equality elimination
- reduced control-space Hessian `Q_zz`
- backward value recursion through `V_xx`, `V_yy`

This is the object actually solved by:

- [presolve.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/presolve.cpp)
- [backward.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/backward.cpp)
- [rollout.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/rollout.cpp)

This is the most important distinction:

- the original problem is the modeling/NLP object
- IPM/PMM reduced problems are local eliminated subproblems
- the Riccati subproblem is the final stagewise reduced linear system after those local reductions have already been absorbed

## 2. How The Stage QP Is Assembled Before Riccati

Per SQP iteration, each stage first assembles the dense base approximation in
[node_data.cpp](/home/harper/Documents/moto/src/ocp/node_data.cpp):

- `cost_`
- `cost_jac_[pf]`
- `lag_jac_[pf]`
- `lag_hess_[a][b]`
- hard/ineq residuals and Jacobians in `approx_[cf]`

Then solver-owned first/second-order corrections are added through:

- `lag_jac_corr_[pf]`
- `hessian_modification_[a][b]`

Important qualification:

- in `func_approx_data`, the function-local `lag_hess_` view is not always bound to stage `lag_hess_`
- for `__cost`, it binds to stage `lag_hess_`
- for non-cost functions, it binds to stage `hessian_modification_`

So normal IPM / PMM second-order reduced terms do not pollute the base cost Hessian blocks.

The active stage system seen by Riccati is therefore:

- gradient:
  `Q_pf := lag_jac_[pf] + lag_jac_corr_[pf]`
- Hessian blocks:
  `lag_hess_[a][b] + hessian_modification_[a][b]`

This activation happens in
[data_base.cpp](/home/harper/Documents/moto/src/solver/data_base.cpp)
via `activate_lag_jac_corr()`, and is called from
[presolve.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/presolve.cpp).

That is the stage-local QP representation before nullspace / Riccati reduction.

Important distinction:

- this assembled stage QP is already no longer the raw original problem
- IPM / PMM local reduced effects have already been folded into it
- base cost curvature and non-cost solver corrections remain separated in storage
- but it is still not yet the final Riccati-reduced subproblem

## 3. What Riccati Actually Solves

The solve path is:

1. `ns_factorization(...)`
2. `riccati_recursion(...)`
3. `compute_primal_sensitivity(...)`
4. `fwd_linear_rollout(...)`
5. `finalize_primal_step(...)`
6. `finalize_dual_newton_step(...)`

The important point is that normal mode does not solve the full KKT system explicitly.
It solves a reduced system after:

- projecting dynamics with `F_x`, `F_u`, `F_0`
- stacking hard equalities into `s_c_stacked`
- eliminating constrained control directions through LU / nullspace machinery
- solving the reduced LLT system in `Q_zz`

Relevant files:

- [presolve.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/presolve.cpp)
- [backward.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/backward.cpp)
- [rollout.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/rollout.cpp)

So the Riccati solver contract is:

- start from the assembled stage QP
- eliminate hard-equality structure and project through dynamics
- solve the resulting Riccati / nullspace reduced subproblem
- recover primal and dual steps from that reduced subproblem

## 4. What Iterative Refinement Reduces In Normal Mode

Normal iterative refinement is not based on the raw `compute_kkt_info()` summary.
It uses the solver-side correction residual built after the Newton step is recovered.

The core residual builder is
[rollout.cpp](/home/harper/Documents/moto/src/solver/nsp_impl/rollout.cpp)
in `generic_solver::compute_kkt_residual(...)`.

Per stage it forms:

- `kkt_stat_err_[__u]`
- `kkt_stat_err_[__y]`
- `kkt_stat_err_[__x]`

from:

- `base_lag_grad_backup`
- `+` the stage-local reduced self-block terms already exposed in solver coordinates
  - `Q_uu * delta u`
  - `Q_yy * delta y`
  - `Q_xx * delta x`
- `+ J^T * trial_dual_step`

Then, in
[iterative_refinement.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/iterative_refinement.cpp),
the cross-stage coupling is folded in:

- `next.kkt_stat_err_[__x]` is permuted onto `cur.kkt_stat_err_[__y]`

So the normal refinement stopping metric is effectively:

- `u` stationarity residual at the current stage
- `y` stationarity residual after adding the downstream `x` contribution

That is the practical solver-side reduced stationarity residual of the solved normal system.

It is important not to overstate this:

- this is not assembled as a textbook full-space KKT residual
- it is the residual of the solver's reduced first-order system in the coordinates used by correction solves

This means the IR target is closest to the Riccati subproblem layer, not the raw original NLP layer.

This is the key alignment point for restoration:

- normal IR reduces the residual of the same reduced first-order correction model used by the solver
- it does not reduce some separate diagnostic residual

## 5. How Correction Solves Work In Normal Mode

Normal correction solves use the same reduced solver pipeline.
More precisely, they reuse the already-built normal factorization and solve a first-order correction system on top of it.

The correction path is:

1. write the correction RHS into `lag_jac_corr_`
2. `swap_active_and_lag_jac_corr()`
3. run correction recursion / correction rollout using the existing factorization data
4. add the correction to `trial_prim_step`
5. restore the base active gradient

The relevant lifecycle is in
[data_base.hpp](/home/harper/Documents/moto/include/moto/solver/data_base.hpp)
and
[ns_sqp_impl.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp).

This is the normal phase's correction contract:

- the correction RHS is expressed in the same solver coordinates as the reduced first-order correction system
- in practice the normal RHS is loaded only on `__u` and `__y`; `__x` is carried through the rollout coupling rather than loaded directly
- the correction solve reuses the same Riccati/nullspace data from the main solve rather than rebuilding a second factorization
- the correction is not a separate algorithm

## 6. What Globalization Evaluates In Normal Mode

The trial evaluation path in normal mode is:

1. restore backed-up primal/dual state
2. apply affine primal and dual step
3. evaluate values
4. if needed, evaluate full derivatives
5. compute `kkt_info`

Relevant file:

- [ns_sqp_impl.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp)

For line search:

- filter mode may evaluate only value-level quantities on intermediate rejected trials
- accepted trials refresh derivatives and recompute full `kkt_info`

The normal acceptance logic in
[line_search.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/line_search.cpp)
uses:

- objective:
  `cost - mu * log_slack_sum`
- primal violation:
  `prim_res_l1`

In filter mode, the actual accept/reject test is driven by:

- primal violation
- objective

not by `inf_dual_res`.

`inf_dual_res` is still computed for accepted/full evaluations and printed publicly, but it is not part of the normal filter dominance test.

This is much closer to the original barrierized phase objective than to the internal Riccati reduced subproblem.

So in normal mode:

- solve system: Riccati reduced subproblem
- refinement residual: reduced stationarity residual of the solver correction model
- line search objective: public barrierized phase objective / feasibility summary

These pieces are consistent enough to be treated as one phase contract, with one caveat:

- filter globalization is objective/primal-driven
- dual residual is primarily a convergence/statistics quantity, not the filter acceptance metric

## 7. What `compute_kkt_info()` Means In Normal Mode

In normal mode, [compute_kkt_info()](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp#L753)
reports:

- `cost`:
  pure running cost
- `objective`:
  `cost - settings.ipm.mu * log_slack_sum`
- `inf_prim_res`:
  stagewise max primal infeasibility
- `prim_res_l1`:
  summed primal infeasibility used by the filter
- `inf_dual_res`:
  stationarity summary, including the folded `next.x -> cur.y` contribution
- `inf_comp_res`:
  IPM complementarity residual

This is not the same object as:

- the raw original NLP residual vector
- or the internal IR correction residual vector

It is the public phase summary used for convergence checks and logging.

In particular:

- `inf_dual_res` is the normal phase's public stationarity summary
- it is the quantity `print_stats` exposes
- it is not the quantity the normal filter directly optimizes on rejected intermediate trials

## 8. What `print_stats` Means In Normal Mode

Current `print_stats(...)` in
[print_stat.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/print_stat.cpp)
should be interpreted in normal mode as:

- `obj`:
  current phase objective, which in normal mode is the barrier objective
- `r(prim)`:
  `inf_prim_res`
- `r(dual)`:
  `inf_dual_res`
- `r(comp)`:
  `inf_comp_res`
- `||p||`:
  infinity norm of the primal Newton step
- `||d_eq||`, `||d_iq||`:
  equality and inequality dual-step magnitudes
- `mu(max)`:
  normal barrier parameter `settings.ipm.mu`

Important note:

- these are public summary scalars
- they are not the literal full KKT vectors used internally for correction

Still, in normal mode they are phase-consistent and good diagnostics.

## 9. The Precise Normal Contract Restoration Should Match

If restoration is to align mathematically with normal mode, it must satisfy the same pattern:

1. distinguish the original phase problem from local reduced subproblems and from the Riccati subproblem
2. define one unique reduced system for the phase after local reductions
3. solve that reduced system with Riccati
4. compute correction RHS from that same reduced system's residual
5. run IR against that same residual
6. print and accept using clearly-labeled public summaries of that phase

The most important normal invariant is:

- **the solved Riccati subproblem and the IR correction residual refer to the same internal phase object**

A second, equally important invariant is:

- **the public `compute_kkt_info()` / `print_stats()` summaries are derived from the same phase, but they are not literally the same internal vector object as the IR residual**

For normal, that object is the reduced barrier-QP system.

For restoration, the target should be:

- the reduced KKT system of the restoration Lagrangian after local elastic condensation

## 10. Immediate Implication For Restoration Work

Before restoration IR is re-enabled safely, it should match normal in this sense:

- its correction RHS must come from the reduced residual of the restoration system
- its `compute_kkt_info()` must summarize that same reduced residual
- its line search `resto_res` should use that same quantity
- its `print_stats` output should describe that same phase object

Until those four align, restoration may still behave numerically, but it is not yet mathematically aligned with normal mode.
