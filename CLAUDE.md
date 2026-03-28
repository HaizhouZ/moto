# moto — Claude Code Notes

## Project overview
Nonsmooth SQP (Sequential Quadratic Programming) solver for trajectory optimization, with Python bindings via nanobind. Core solver is C++, exposed to Python via `bindings/`.

## Key files
- `include/moto/solver/ns_sqp.hpp` — top-level solver class, settings structs, `data` type
- `src/solver/sqp_impl/ns_sqp_impl.cpp` — main SQP update loop, diagnostics
- `src/solver/sqp_impl/line_search.cpp` — filter line search implementation (IPOPT-style)
- `src/solver/sqp_impl/restoration.cpp` — `ns_sqp::restoration_update()`: restoration phase loop
- `src/solver/nsp_impl/presolve.cpp` — `ns_factorization`: builds `s_c_stacked`, `s_c_stacked_0_K`, LU, nullspace; pass `gauss_newton=true` for restoration mode
- `src/solver/nsp_impl/backward.cpp` — Riccati recursion (backward pass)
- `src/solver/nsp_impl/rollout.cpp` — forward linear rollout
- `include/moto/solver/ns_riccati/ns_riccati_data.hpp` — per-node solver data, `nullspace_data` struct
- `include/moto/solver/ns_riccati/generic_solver.hpp` — solver interface (virtual methods)
- `include/moto/solver/ipm/ipm_constr.hpp` — IPM inequality constraint implementation
- `include/moto/solver/ipm/ipm_config.hpp` — IPM barrier parameter config (Mehrotra predictor-corrector)
- `include/moto/core/fields.hpp` — field enum (`__x`, `__u`, `__y`, `__dyn`, `__eq_x`, `__eq_xu`, `__ineq_x`, `__ineq_xu`, ...)
- `bindings/definition/ns_sqp.cpp` — nanobind Python bindings for solver settings
- `example/quadruped/` — quadruped locomotion example (often used for manual testing/experiments)

---

## Data hierarchy

### `ns_sqp` (top-level class, `ns_sqp.hpp`)
- `settings` (`settings_t`): inherits `linesearch_setting` and `ipm_config` via `workspace_data_collection`
  - `settings.ls` — line search params (`s_phi`, `s_theta`, `armijo_dec_frac`, `enable_soc`, ...)
  - `settings.ipm` — barrier params (`mu`, `mu0`, `mu_method`, `warm_start`, ...)
  - `settings.rf` — iterative refinement (`enabled`, `max_iters`, tolerances)
  - `settings.verbose`, `settings.prim_tol`, `settings.dual_tol`, `settings.comp_tol`
- `graph_` (`directed_graph<shooting_node<data>>`): owns all stage nodes; supports `apply_forward`, `apply_backward`, `for_each_parallel`
- `riccati_solver_` (`unique_ptr<generic_solver>`): per-stage solver operations (virtual interface)
- `mem_` (`data_mgr`): memory pool for node data

### `ns_sqp::data` (per-stage node, `ns_sqp.hpp:79`)
Multiple-inherits from:
- **`node_data`** (`include/moto/ocp/impl/node_data.hpp`): holds the OCP problem formulation, sparse approximation data (`sparse_[field]`), dense merged data (`dense_`); provides `for_each(field, cb)` and `update_approximation()`
- **`ns_riccati_data`** (`include/moto/solver/ns_riccati/ns_riccati_data.hpp`): holds Riccati/nullspace solver state

### `ns_riccati_data` fields
- `ns`, `nc`, `ncstr` — number of `__eq_x`, `__eq_xu`, and total equality constraints
- `nis`, `nic` — active inequality counts (currently unused/zero in standard flow)
- `F_x`, `F_u` — projected dynamics Jacobians (`sparse_mat&`), w.r.t. x and u
- `s_y`, `s_x` — equality `__eq_x` constraint Jacobian w.r.t. `__y` and `__x` (`sparse_mat&`)
- `c_x`, `c_u` — equality `__eq_xu` constraint Jacobians w.r.t. `__x` and `__u` (`sparse_mat&`)
- `rank_status_` — `unconstrained` / `constrained` / `fully_constrained`
- `d_u`, `d_y` — primal step sensitivities (`sensitivity{k, K}`: feedforward and feedback)
- `nsp_` (`nullspace_data`) — see below

### `nullspace_data` (inside `ns_riccati_data::nsp_`)
Built by `ns_factorization` (Jacobian-based, once per SQP iteration) and updated by `ns_factorization_correction` (residual-based, also called for SOC/iterative refinement):
- `s_c_stacked` (ncstr × nu): stacked equality constraint u-Jacobians = `[s_y*F_u ; c_u]`
- `s_c_stacked_0_K` (ncstr × nx): stacked equality constraint x-Jacobians = `[s_x + s_y*F_x ; c_x]`
- `s_c_stacked_0_k` (ncstr): stacked equality constraint residuals = `[s_y*F_0 ; eq_xu.v]` — built by `ns_factorization_correction`
- `lu_eq_` (`Eigen::FullPivLU`): LU of `s_c_stacked`; used to solve for `u_y_K`/`u_y_k`, compute rank/kernel
- `rank`: rank of `s_c_stacked` (0 = unconstrained, ncstr = fully_constrained)
- `Z_u` (nu × nz): null space basis of `s_c_stacked` (kernel of equality u-Jacobian)
- `Z_y` (ny × nz): `F_u * Z_u` — null space mapped to next-state coordinates
- `Q_zz` (nz × nz): projected Hessian in null-space coordinates
- `u_y_K` (nu × nx): `lu_eq_.solve(s_c_stacked_0_K)` — u sensitivity to x (Jacobian-based)
- `u_y_k` (nu): `lu_eq_.solve(s_c_stacked_0_k)` — u feedforward from residual (built by `ns_factorization_correction`)
- `y_y_K` (ny × nx): `F_x + F_u * u_y_K` — y sensitivity to x (closed-loop dynamics in null-space)
- `y_y_k` (ny): `F_0 - F_u * u_y_k` — y feedforward from residual (built by `ns_factorization_correction`)

---

## Field system (`include/moto/core/fields.hpp`)
Fields label the role of each variable/function in the OCP:

| Field | Meaning |
|---|---|
| `__x` | current state |
| `__u` | control input |
| `__y` | next state (= x_{k+1}) |
| `__p` | non-decision parameters |
| `__dyn` | dynamics constraint (f(x,u) - y = 0) |
| `__eq_x` | state-only equality constraint g(x)=0; **x-arg substituted to `__y`** in `finalize_impl` |
| `__eq_xu` | state-input equality constraint g(x,u)=0 |
| `__ineq_x` | state-only inequality constraint g(x)≤0; **x-arg substituted to `__y`** in `finalize_impl` |
| `__ineq_xu` | state-input inequality constraint g(x,u)≤0 |
| `__eq_x_soft` / `__eq_xu_soft` | soft equality constraints |
| `__cost` | running cost |

**Important**: `__eq_x` and `__ineq_x` have their `__x` argument substituted to `__y` during `finalize_impl` (see `src/ocp/constr.cpp`). Their Jacobians are therefore w.r.t. `y_k = x_{k+1}`, not `x_k`. To get x_k / u_k sensitivities, project through dynamics:
- u-column: `J_y * F_u`
- x-column: `J_y * F_x`

This is exactly how `ns_factorization` builds `s_c_stacked` and `s_c_stacked_0_K` from `s_y`.

### Field groupings
```cpp
hard_constr_fields      = {__dyn, __eq_x, __eq_xu}
ineq_constr_fields      = {__ineq_x, __ineq_xu}
soft_constr_fields      = {__eq_x_soft, __eq_xu_soft}
ineq_soft_constr_fields = ineq_constr_fields + soft_constr_fields
constr_fields           = hard_constr_fields + ineq_soft_constr_fields
```

---

## Constraint class hierarchy
```
generic_func
└── generic_constr          (base; holds multiplier_, jac_[], v_)
    └── soft_constr         (adds jac_modification_[], d_multiplier_; data_map_t = approx_data)
        └── ineq_constr     (adds comp_ complementarity residual)
            └── ipm_constr  (adds slack_, g_, diag_scaling, active_, r_s_, d_slack_, d_multiplier_)
```
- `ipm_constr::approx_data` (= `ipm_data`): the main inequality constraint data type
  - `g_[i]` — raw constraint value
  - `slack_[i]` — slack variable (kept > 0 by IPM); small → approximately active
  - `active_[i]` — always 1.0 in current IPM (not a hard active-set mask)
  - `diag_scaling[i]` — Nesterov-Todd scaling `λ_i / (s_i + ε·λ_i)`
  - `jac_[arg_idx]` — Jacobian rows (matrix_ref, rows = constraint dim, cols = arg dim)

To iterate over IPM constraints from a `node_data*` or `data*`:
```cpp
d->for_each(__ineq_xu, [](const soft_constr &sf, soft_constr::approx_data &sd) {
    auto *id = dynamic_cast<const ipm_constr::approx_data *>(&sd); // nullptr if not ipm
    ...
});
```

---

## Jacobian scaling (`src/solver/sqp_impl/scaling.cpp`)
Applied in-place to `dense().approx_[cf]` before factorization; reversed after the QP solve via `unscale_duals`.

**`scaling_settings` fields** (in `ns_sqp::scaling_settings`):
- `mode`: `none` / `gradient` / `equilibrium` (default: `gradient`; `gradient` preferred — `equilibrium` is ~2× more expensive per scale-compute step)
- `equilibrium_iters`: Ruiz iterations (default: 5)
- `min_scale`: clamp floor to avoid division by zero (default: 1e-6)
- `update_ratio_threshold`: recompute scales when `inf_prim_step ≥ 1/threshold` (large primal step → Jacobians changed); cached otherwise (default: 10 → recompute when step ≥ 0.1)

**Scale vectors** (cached in `data::scale_c_[cf]` and `data::scale_p_[pi]`):
- `scale_c_[cf]`: per-row scale for constraint field `cf`; empty means not yet computed (first call or after `reset_scaling`)
- `scale_p_[pi]`: per-primal-field scale for cost gradient (1 scalar per primal field)
- `scaling_applied_`: flag; true between `compute_and_apply_scaling` and `unscale_duals`

**`compute_and_apply_scaling(kkt)`**:
1. Decides whether to recompute scales: recomputes if `inf_prim_step ≥ 1/update_ratio_threshold` (large step → Jacobians changed) OR `scale_c_` is empty (first call). Near convergence the cached scales are reused.
2. Only scales `hard_constr_fields_non_dyn` = `{__eq_x, __eq_xu}`. `__dyn` excluded because `approx_[__dyn].jac_[__y]` aliases `f_y_` used by `dense_dynamics::compute_project_jacobians` LU.
3. `gradient` mode: `s[i] = 1 / max(min_scale, max(row_infnorm(J_i), |v_i|))` — row normalises Jacobian and residual.
4. `equilibrium` mode: iterative Ruiz; each iter computes row inf-norm of `diag(s) * J`, then `s[i] /= row_norm[i]`.
5. Applies `s` in-place: `v_[i] *= s[i]`, `J_[row i] *= s[i]`.
6. `scale_p_` is always 1 (cost gradient scaling disabled — `Q_y` propagates across stages via backward Riccati and per-stage scaling would cause compounding errors).
7. Must be called **after** `update_approximation` and **before** `ns_factorization`.

**`unscale_duals()`** — reverses in-place scaling for `hard_constr_fields_non_dyn` only:
- `approx.v_[i] /= s[i]` — reverse residual scaling
- `approx.jac_[pf]` rows divided by `s` (i.e., multiplied by `s.cwiseInverse()`)
- `trial_dual_step[cf][i] *= s[i]` — unscale the dual **step** computed in scaled space

**Key math**: scaling applies `s·J` to constraint Jacobians. Scaled KKT: `(s·J)^T Δλ_scaled = -∇f`. Original: `J^T Δλ_orig = -∇f`. Therefore `Δλ_orig = s · Δλ_scaled`. Only `trial_dual_step` is unscaled — `dual_[cf]` is the accumulated λ from previous iterations, already in original units, and must NOT be touched.

---

## Restoration mode (`src/solver/sqp_impl/restoration.cpp`)

Triggered automatically when the outer filter line search produces a `stop` action and primal infeasibility does not improve for `settings.restoration.trigger_on_failure_count` consecutive SQP iterations.

**Goal**: reduce total hard-constraint infeasibility (dynamics + `__eq_x` + `__eq_xu`) by solving a Gauss-Newton sub-problem with proximal regularization. All equality constraints are treated as objectives (not hard constraints) for the duration of the restoration Riccati solve.

**Restoration objective** at each node:
```
J_rest = ½‖F_0‖²  +  ½‖s_c_stacked_0_k‖²
       + ½ρ_u ‖diag(σ_u)(u - u_ref)‖²  +  ½ρ_y ‖diag(σ_y)(y - y_ref)‖²
```
where `u_ref`, `y_ref`, `rho_u`, `rho_y`, `sigma_u`, `sigma_y` are stored in `ns_riccati_data::restoration_aux_data` (via `aux_` pointer), snapshotted at restoration entry.
`sigma_u[i] = 1/max(|u_ref[i]|, 1)` — per-component primal scaling so the proximal cost is on a percentage (relative) level.

**GN mode in `ns_factorization(cur, gauss_newton=true)`** (`src/solver/nsp_impl/presolve.cpp`):
- Runs the standard factorization preamble: `update_projected_dynamics`, `merge_jacobian_modification`, builds `s_c_stacked`, `lu_eq_`, `s_c_stacked_0_K` (always built unconditionally before the rank branch).
- **PMM (penalty method) treatment of `__eq_x`/`__eq_xu`**: original running cost Hessians (`Q_uu`, `Q_ux`, `Q_yx`, `Q_xx`, `Q_yy`) and gradients (`Q_u`, `Q_y`) are **kept intact**; equality constraints are removed from the hard-constraint path and instead penalized. IPM/soft-constraint `_mod` terms also retained.
- Proximal gradient/Hessian are written by `restoration_update` directly into `dense().jac_modification_[__u/y]` and `dense().primal_prox_hess_diagonal_[__u/y].diagonal()` before each `sqp_iter`. `merge_jacobian_modification` folds them into `Q_u`/`Q_y`; the diagonal Hessian lands in `Q_uu_mod`/`Q_yy_mod` via the pre-allocated diagonal sparsity in `hessian_[__u][__u]` (see `merit_data` constructor). GN mode sees proximal already in `Q_u`/`Q_uu_mod`.
- Builds `s_c_stacked_0_k` explicitly after `update_projected_dynamics_residual()` for the PMM gradient.
- **Gradients** (PMM Schur complement, mirrors `pmm_constr::propagate_jacobian`):
  - `Q_u += (h/rho_eq)^T * s_c_stacked`  where `h = s_c_stacked_0_k`
  - `Q_x += (h/rho_eq)^T * s_c_stacked_0_K`
- Calls `unconstrain_setup()` (sets `rank_status_ = unconstrained`, copies `Q_uu+Q_uu_mod` into `Q_zz`, etc.), then adds **PMM Hessian terms**:
  - `Q_zz += (1/rho_eq) * s_c^T * s_c`
  - `u_0_p_K / z_0_K += (1/rho_eq) * s_c^T * s_c_0_K`
  - `V_xx += (1/rho_eq) * s_c_0_K^T * s_c_0_K`
- `rank_status_ = unconstrained` (set by `unconstrain_setup`), so `ns_factorization_correction` takes the early return path (`z_0_k = Q_u^T`, `y_y_k = F_0`).
- `ns/nc/ncstr` are left intact after GN factorization for `finalize_dual_newton_step`.
- `lu_eq_` preserved for dual recovery: `dlam = (s_c * du + h) / rho_eq` in `finalize_dual_newton_step` / `rollout.cpp`.

**`restoration_update(kkt_before, ls)`** (`src/solver/sqp_impl/restoration.cpp`):
- Sets `settings.in_restoration = true` on entry, `false` on exit.
- Iterates over **all** nodes via `graph_.flatten_nodes()` (includes key nodes AND intermediate edge nodes). Using `graph_.nodes()` instead would miss intermediate nodes, leaving their `aux_` as `nullptr` and causing a segfault in `ns_factorization`.
- Snapshots proximal anchors (`u_ref`, `y_ref`) and per-component scaling `sigma_u_sq[i] = 1/max(|u_ref[i]|,1)²` into `prox_data` for each node. Also allocates `restoration_aux_data` on each node's `aux_` (carries `rho_eq` for the GN dual regularization).
- Each iteration: writes proximal gradient `rho*(sigma_sq*(u-u_ref))^T` directly into `dense().jac_modification_[__u/y]`, and `rho*sigma_sq` into `dense().primal_prox_hess_diagonal_[__u/y].diagonal()`. `merge_jacobian_modification` (inside `ns_factorization`) folds the gradient into `Q_u/Q_y`; the diagonal Hessian lands in `Q_uu_mod`/`Q_yy_mod` (already aliased from `hessian_[__u][__u]` diagonal sparsity pattern).
- No scaling; IPM predictor/corrector and iterative refinement active.
- Uses `filter_linesearch_data rls = ls` — **copies the outer filter** (not a fresh one). This lets the restoration filter see prior accepted points, preventing restoration from accepting steps that would be rejected by the outer loop.
- **Exit criteria** (any one breaks the loop):
  1. `inf_dual_res < dual_tol` AND NOT fully feasible → sets `result = infeasible_stationary` (converged to an infeasible stationary point, e.g. due to LICQ failure).
  2. `rest_action == accept` AND `inf_prim_res < restoration_improvement_frac * kkt_before.inf_prim_res` → sets `result` as successful.
  3. Loop exhausted → sets `result = restoration_failed`.
- Does **not** call `ls.update_filter` — the outer filter is not updated after restoration.
- Clears all `aux_` pointers after the loop, returns updated `kkt_info` with `result` set.

**`settings.restoration`** (`restoration_settings`):
- `enabled` (default `true`)
- `max_iter` (default 50)
- `trigger_on_failure_count` (default 3)
- `rho_u`, `rho_y` (default `1e-4`) — proximal regularization weights for u and y
- `rho_eq` (default `1e-6`) — PMM penalty weight for equality constraints in GN mode: `Hess += (1/rho_eq)*J^T*J`, `dlam = (J*du+h)/rho_eq`. Smaller → tighter constraint satisfaction per step.
- `restoration_improvement_frac` (default 0.9) — required fraction of entry infeasibility for the accept+improve exit criterion

---

## Per-iteration SQP loop (`ns_sqp_impl.cpp`)
Both `update()` and `restoration_update()` share a common inner loop via `sqp_iter(ls, kkt, do_scaling, do_refinement, gauss_newton=false)`:
1. `compute_and_apply_scaling(kkt)` (if `do_scaling`) — row-scale Jacobians and residuals in-place
2. `ns_factorization(gauss_newton)` (parallel) — normal: build `s_c_stacked`, LU, Z_u/Q_zz/`u_y_K`/`y_y_K`; GN: PMM treatment of equalities, proximal already in `Q_u/Q_y/Q_uu_mod/Q_yy_mod` via `jac_modification_`/`primal_prox_hess_diagonal_`
3. `riccati_recursion` (backward) — backward pass, updates V_xx
4. `compute_primal_sensitivity` (parallel) — compute d_u.K, d_y.K
5. `fwd_linear_rollout` (forward) — compute Newton step
6. `ineq_constr_prediction` — start Mehrotra predictor step (if `has_ineq_soft`)
7. `finalize_primal_step` + `update_ls_bounds` (parallel)
8. `ineq_constr_correction` — Mehrotra corrector step + re-solve (if `has_ineq_soft`)
9. Iterative refinement (if `do_refinement && rf.enabled && rf.max_iters > 0 && has_ineq_soft`)
10. `unscale_duals()` — reverse all in-place scaling on duals, Jacobians, residuals
11. `finalize_dual_newton_step` + `backup_trial_state` (once, before line search)
12. Filter line search loop: `apply_affine_step` → `compute_kkt_info` → `filter_linesearch` → `line_search_action`
    - `accept`: eval derivatives, update filter, continue
    - `backtrack`: restore trial state, reduce alpha, goto step 12
    - `retry_second_order_correction`: `second_order_correction()`, goto step 12 (currently commented out)
    - `stop`: use backup strategy (min_step or best_trial), eval derivatives

`update()` calls `sqp_iter` with `do_scaling=true, do_refinement=true, gauss_newton=false`, then handles the restoration trigger, convergence check, stats printing, and monotone mu update.
`restoration_update()` calls `sqp_iter` with `do_scaling=false, do_refinement=true, gauss_newton=true`.
- `print_licq_info()` and `print_scaling_info()` are available as diagnostics but currently invoked under comments — call manually when needed

---

## Line search / filter (IPOPT-style)
`filter_linesearch` returns a `line_search_action` enum; the main loop dispatches on it. `try_step()` implements:
1. **Filter rejection**: trial point dominated by any filter point → reject.
2. **Switching condition** (IPOPT §3.3): `obj_fullstep_dec < 0` AND `(-obj_fullstep_dec)^s_phi ≥ (inf_prim_res)^s_theta` AND `inf_prim_res_k ≤ constr_vio_min`:
   - Yes → **Armijo mode**: accept if `obj_trial ≤ obj_k + armijo_dec_frac * alpha * obj_fullstep_dec`. Sets `last_step_was_armijo = true`.
   - No → **Filter mode**: accept if trial point is not dominated by the current iterate.
3. `update_filter` adds the accepted point to the filter only when `!last_step_was_armijo`.
4. `constr_vio_min` is set once at the start of `update()` as `kkt_last.inf_prim_res * constr_vio_min_frac` — fixed for the duration of the call.
- `filter_linesearch_data::point` holds `{prim_res, dual_res, objective}` — objective is now part of the filter.
- **Switching condition guard**: `obj_fullstep_dec < 0.0` must come before the `std::pow` call — negative base with non-integer exponent is NaN in C++.
- `s_phi` (default 2.3), `s_theta` (default 1.1), `armijo_dec_frac` (default 1e-4), `constr_vio_min_frac` (default 1e-4) are configurable via `linesearch_setting` and exposed to Python.
- Backtracking uses a **linear sweep**: alpha decreases by `initial_alpha / max_steps` per step (not geometric).
- `failure_strategy` (renamed from `failure_backup`): `min_step` or `best_trial` (default).
- SOC (`second_order_correction`) is implemented but its call site in `filter_linesearch` is commented out — do not remove.

---

## IPM (interior point method for inequalities)
- Implemented in `ipm_constr` (`src/solver/ipm_impl/ipm_constr.cpp`)
- Barrier parameter `mu` controlled by `ipm_config`; default method: **Mehrotra predictor-corrector**
- Per iteration: predictor step (`ineq_constr_prediction`) → solve → corrector step (`ineq_constr_correction`) → adaptive `mu` update
- `active_[i]` is always 1.0 in `value_impl`; activeness is implicit via complementarity `slack[i] * lambda[i] → mu`
- `slack[i]` small (< threshold) → constraint `i` is approximately active at current point

---

## LICQ diagnostic (`print_licq_info`)
- Defined in `ns_sqp_impl.cpp`. Call manually (currently under a comment) after `ns_factorization` has run.
- Implements **global LICQ via forward nullspace propagation** (DMS staircase structure).
- `Z_x` (nx × nz_x): null space of all prior-stage constraints expressed in x_k coords; starts empty (x_0 fixed).
- Per stage builds augmented Jacobian `A_k = [eq_rows ; active_ineq_rows]` in `(Z_x, u)` coordinates and checks rank.
- **Equality rows** reuse `nsp_.s_c_stacked` (u-cols) and `nsp_.s_c_stacked_0_K` (x-cols) from `ns_factorization` — no recomputation.
- **Active inequality rows** (`slack[i] < 1e-3`):
  - `__ineq_xu`: direct `__x`/`__u` Jacobians from `ipm_constr::approx_data`.
  - `__ineq_x`: x-arg is substituted to `__y` in `finalize_impl`, so Jacobian is `dg/dy_k`; mapped back via dynamics: u-col = `J_y * F_u`, x-col = `J_y * F_x` (mirrors how `s_c_stacked` is built in `ns_factorization`).
- Fast-path for `nz_x == 0 && n_active == 0`: reuses `nsp_.rank` / `nsp_.Z_u` directly.
- Null space propagated: `Z_{x,k+1} = [F_x * Z_x | F_u] * null(A_k)`.
- Requires `#include <moto/solver/ipm/ipm_constr.hpp>` in `ns_sqp_impl.cpp`.

---

## Conventions
- Settings structs live in the header; adding a new field requires updating the header, the `.cpp` that uses it, and the bindings file.
- Verbose logging is always gated on `settings.verbose` — never unconditional `fmt::print` in hot paths.
- IPOPT paper naming conventions are used for line search parameters (e.g., `s_phi`, `s_theta`, `armijo_dec_frac`).
- `__eq_x` / `__ineq_x` Jacobians are always w.r.t. `__y` after finalization — never assume `__x` is present.
- `for_each<field>(cb)` deduces `func_type` and `approx_type` from the callback signature via static cast through `shared<expr>`; use `dynamic_cast` inside when the concrete approx type may vary.
- Use `solver_call(f)` (not `bind`) to wrap `generic_solver` member function pointers for `graph_` traversal methods.
- `mu_method` is a **value type** (`adaptive_mu_t`), not a pointer — default is `mehrotra_predictor_corrector`. Do not check for `nullptr`.
- `settings.ls` and `settings.ipm` are references into `settings_t` (which multiple-inherits them via `workspace_data_collection`). Do not copy `settings_t` by value after construction — the references would alias the old object.

---

## example/quadruped/run.py
This file is frequently modified for manual experiments (gait, `cfg`, `full` flag, commented constraints). **Do not treat its current state as canonical** — always check git diff before reviewing or merging.
