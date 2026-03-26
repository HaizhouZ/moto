# moto — Claude Code Notes

## Project overview
Nonsmooth SQP (Sequential Quadratic Programming) solver for trajectory optimization, with Python bindings via nanobind. Core solver is C++, exposed to Python via `bindings/`.

## Key files
- `include/moto/solver/ns_sqp.hpp` — top-level solver class, settings structs, `data` type
- `src/solver/sqp_impl/ns_sqp_impl.cpp` — main SQP update loop, diagnostics
- `src/solver/sqp_impl/line_search.cpp` — filter line search implementation (IPOPT-style)
- `src/solver/nsp_impl/presolve.cpp` — `ns_factorization`: builds `s_c_stacked`, `s_c_stacked_0_K`, LU, nullspace
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
Built by `ns_factorization`, consumed by Riccati recursion and diagnostics:
- `s_c_stacked` (ncstr × nu): stacked equality constraint u-Jacobians = `[s_y*F_u ; c_u]`
- `s_c_stacked_0_K` (ncstr × nx): stacked equality constraint x-Jacobians = `[s_x + s_y*F_x ; c_x]`
- `lu_eq_` (`Eigen::FullPivLU`): LU of `s_c_stacked`; used to solve for `u_y_K`, compute rank/kernel
- `rank`: rank of `s_c_stacked` (0 = unconstrained, ncstr = fully_constrained)
- `Z_u` (nu × nz): null space basis of `s_c_stacked` (kernel of equality u-Jacobian)
- `Z_y` (ny × nz): `F_u * Z_u` — null space mapped to next-state coordinates
- `Q_zz` (nz × nz): projected Hessian in null-space coordinates
- `u_y_K` (nu × nx): `lu_eq_.solve(s_c_stacked_0_K)` — u sensitivity to x
- `y_y_K` (ny × nx): `F_x + F_u * u_y_K` — y sensitivity to x (closed-loop dynamics in null-space)

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

## Per-iteration SQP loop (`ns_sqp_impl.cpp`)
Each call to `update(n_iter, verbose)`:
1. `initialize()` — eval values + derivatives, print header
2. For each iteration:
   1. `ns_factorization` (parallel) — build `s_c_stacked`, LU, Z_u, Q_zz per node
   2. `print_licq_info()` (if verbose) — global LICQ diagnostic (see below)
   3. `riccati_recursion` (backward) — backward pass, updates V_xx
   4. `compute_primal_sensitivity` (parallel) — compute d_u.K, d_y.K
   5. `fwd_linear_rollout` (forward) — compute Newton step
   6. `ineq_constr_prediction` — start Mehrotra predictor step (if IPM + adaptive mu)
   7. `finalize_primal_step` + `update_ls_bounds` (parallel)
   8. `ineq_constr_correction` — Mehrotra corrector step (if IPM + adaptive mu)
   9. Iterative refinement (if enabled and has inequality constraints)
   10. `finalize_dual_newton_step`, `backup_trial_state`
   11. Filter line search loop (`filter_linesearch` → `try_step`)
   12. `update_approximation` (derivatives) on accept
   13. Print stats; check convergence; update mu (monotone mode)

---

## Line search / filter (IPOPT-style)
- `try_step()` in `line_search.cpp`: checks filter dominance, then switching condition, then Armijo or filter acceptance.
- Switching condition guard: **`obj_fullstep_dec < 0.0` must come before the `std::pow` call** — negative base with non-integer exponent is NaN in C++.
- `s_phi` (default 2.3) and `s_theta` (default 1.1) are configurable via `linesearch_setting` and exposed to Python.
- SOC (second-order correction) block is currently commented out — do not remove, it may be re-enabled.

---

## IPM (interior point method for inequalities)
- Implemented in `ipm_constr` (`src/solver/ipm_impl/ipm_constr.cpp`)
- Barrier parameter `mu` controlled by `ipm_config`; default method: **Mehrotra predictor-corrector**
- Per iteration: predictor step (`ineq_constr_prediction`) → solve → corrector step (`ineq_constr_correction`) → adaptive `mu` update
- `active_[i]` is always 1.0 in `value_impl`; activeness is implicit via complementarity `slack[i] * lambda[i] → mu`
- `slack[i]` small (< threshold) → constraint `i` is approximately active at current point

---

## LICQ diagnostic (`print_licq_info`)
- Defined in `ns_sqp_impl.cpp`, called every iteration inside the `settings.verbose` block (after `ns_factorization`).
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

---

## example/quadruped/run.py
This file is frequently modified for manual experiments (gait, `cfg`, `full` flag, commented constraints). **Do not treat its current state as canonical** — always check git diff before reviewing or merging.
