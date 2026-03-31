# moto Agent Guide

## Purpose

This file is a maintainer-oriented map of the `moto` codebase. It is meant to help an agent or human contributor answer:

- where core data lives
- how an OCP stage is represented
- how raw function evaluations become a QP
- how the SQP / Riccati / line-search pipeline moves data each iteration
- which conventions are easy to violate when editing solver code

This guide was derived from the code itself, especially:

- `CLAUDE.md`
- `include/moto/model/graph_model.hpp`
- `include/moto/solver/ns_sqp.hpp`
- `src/solver/sqp_impl/*.cpp`
- `src/solver/nsp_impl/*.cpp`
- `include/moto/ocp/impl/*.hpp`
- `src/ocp/*.cpp`
- `include/moto/solver/ipm/*.hpp`
- `include/moto/solver/soft_constr/*.hpp`

## High-Level Architecture

`moto` is a C++20 trajectory optimization library with:

- symbolic OCP stage definitions
- sparse-to-dense approximation storage
- a nonsmooth SQP solver
- a nullspace / Riccati-based stagewise QP solve
- optional IPM treatment for inequalities
- optional PMM treatment for soft equalities
- nanobind Python bindings

The main solver entry point is:

- [`include/moto/solver/ns_sqp.hpp`](/home/harper/Documents/moto/include/moto/solver/ns_sqp.hpp)

The main SQP iteration loop lives in:

- [`src/solver/sqp_impl/ns_sqp_impl.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp)

## Repo Map

- [`include/moto/core/fields.hpp`](/home/harper/Documents/moto/include/moto/core/fields.hpp): field taxonomy like `__x`, `__u`, `__y`, `__dyn`, `__eq_x`, `__ineq_xu`
- [`include/moto/ocp/problem.hpp`](/home/harper/Documents/moto/include/moto/ocp/problem.hpp): stage formulation container
- [`include/moto/ocp/impl/func.hpp`](/home/harper/Documents/moto/include/moto/ocp/impl/func.hpp): generic function abstraction
- [`include/moto/ocp/impl/func_data.hpp`](/home/harper/Documents/moto/include/moto/ocp/impl/func_data.hpp): sparse maps from symbolic args to dense storage
- [`include/moto/ocp/impl/node_data.hpp`](/home/harper/Documents/moto/include/moto/ocp/impl/node_data.hpp): per-stage runtime storage
- [`include/moto/ocp/impl/lag_data.hpp`](/home/harper/Documents/moto/include/moto/ocp/impl/lag_data.hpp): dense merged cost/constraint derivatives
- [`include/moto/solver/data_base.hpp`](/home/harper/Documents/moto/include/moto/solver/data_base.hpp): solver-facing aliases and Newton-step storage
- [`include/moto/solver/ns_riccati/ns_riccati_data.hpp`](/home/harper/Documents/moto/include/moto/solver/ns_riccati/ns_riccati_data.hpp): nullspace and Riccati state
- [`include/moto/solver/ns_riccati/generic_solver.hpp`](/home/harper/Documents/moto/include/moto/solver/ns_riccati/generic_solver.hpp): stage solver interface
- [`src/solver/nsp_impl/presolve.cpp`](/home/harper/Documents/moto/src/solver/nsp_impl/presolve.cpp): nullspace factorization setup
- [`src/solver/nsp_impl/backward.cpp`](/home/harper/Documents/moto/src/solver/nsp_impl/backward.cpp): backward Riccati recursion
- [`src/solver/nsp_impl/rollout.cpp`](/home/harper/Documents/moto/src/solver/nsp_impl/rollout.cpp): forward rollout and dual-step recovery
- [`src/solver/sqp_impl/line_search.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/line_search.cpp): filter and merit backtracking
- [`src/solver/sqp_impl/scaling.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/scaling.cpp): Jacobian scaling
- [`src/solver/sqp_impl/restoration.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/restoration.cpp): restoration mode
- [`src/solver/sqp_impl/iterative_refinement.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/iterative_refinement.cpp): residual correction
- [`include/moto/solver/ipm/ipm_constr.hpp`](/home/harper/Documents/moto/include/moto/solver/ipm/ipm_constr.hpp): IPM inequality implementation
- [`include/moto/solver/soft_constr/pmm_constr.hpp`](/home/harper/Documents/moto/include/moto/solver/soft_constr/pmm_constr.hpp): PMM soft equality implementation
- [`bindings/`](/home/harper/Documents/moto/bindings): Python bindings
- [`example/`](/home/harper/Documents/moto/example): manual examples
- [`unittests/`](/home/harper/Documents/moto/unittests): Catch2 tests
- [`include/moto/model/graph_model.hpp`](/home/harper/Documents/moto/include/moto/model/graph_model.hpp): graph-first modeling layer
- [`include/moto/core/directed_graph.hpp`](/home/harper/Documents/moto/include/moto/core/directed_graph.hpp): internal expanded solver graph

## Build And Validation

Top-level build uses CMake:

```bash
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
cmake --build . -j8
ctest --output-on-failure
```

Important build facts:

- project uses C++20
- dependencies include Eigen, BLASFEO, OpenMP, fmt, magic_enum, re2, OpenSSL, CasADi, nlohmann_json, nanobind
- `WITH_NATIVE_OPT=ON` enables `-march=native`
- Python bindings are built from [`bindings/CMakeLists.txt`](/home/harper/Documents/moto/bindings/CMakeLists.txt)
- unit tests are defined in [`unittests/CMakeLists.txt`](/home/harper/Documents/moto/unittests/CMakeLists.txt)
- only `sym_test` is currently enabled by default in CMake

Manual runs from the repo docs:

```bash
python example/arm/run.py
python example/quadruped/run.py
python example/quadruped/mpc.py
```

Useful current validation commands:

```bash
python -m py_compile example/quadruped/run.py
find gen -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
./build/unittests/graph_model_compose_test
MOTO_SQP_MAX_ITER=50 python example/quadruped/run.py
```

## Field System

Fields encode the semantic role of symbols and functions.

Primary primal fields:

- `__x`: current state
- `__u`: control
- `__y`: next state
- `__p`: parameters

Core function / constraint fields:

- `__dyn`: dynamics residual
- `__eq_x`: state-only equality
- `__eq_xu`: state-input equality
- `__ineq_x`: state-only inequality
- `__ineq_xu`: state-input inequality
- `__eq_x_soft`
- `__eq_xu_soft`
- `__cost`

Important convention:

- pure state-only path terms are modeled by the user on `x`
- terminal pure-`x` terms should be added with `prob.add_terminal(...)`
- `node_ocp` is now an `x/u`-only prototype layer
- `edge_ocp` is the only place where `y`-dependent terms and `__dyn` belong
- any required lowering from node semantics to solver storage should happen during graph-aware compose, not during generic expression finalization

Why this is easy to trip over:

- a user may write a stage-local constraint `g(x_k)` and expect it to remain attached to the current `x`
- any hidden remap changes formulation, so it must be explicit in logs and scoped by graph topology
- the intended semantics are edge-centric:
  - if a node has no predecessor edge, its pure state equality terms stay on that node
  - if a node has a predecessor edge, selected pure state equality terms may be lowered onto that predecessor edge as `y_{k-1}`-anchored solver storage
- this decision cannot be made correctly by a local `finalize()` on a standalone expression or a standalone problem

Useful field groups used throughout the solver:

- `primal_fields = {__x, __u, __y}`
- hard constraints = `{__dyn, __eq_x, __eq_xu}`
- inequalities = `{__ineq_x, __ineq_xu}`
- soft equalities = `{__eq_x_soft, __eq_xu_soft}`
- `ineq_soft_constr_fields = inequalities + soft equalities`

## Problem Representation

An OCP stage is represented by [`ocp`](/home/harper/Documents/moto/include/moto/ocp/problem.hpp).

What `ocp` stores:

- expressions grouped by field
- enabled, disabled, and pruned expressions
- flattened indexing for each expression into dense field vectors
- dimensions per field
- tangent dimensions per primal field
- sub-problems

Important `ocp` behaviors:

- `add(expr)` registers symbols/functions in the stage
- `finalize()` computes field dimensions, flattened indices, ordering, and consistency
- `extract(...)` and `extract_tangent(...)` provide views into serialized field vectors
- `maintain_order()` preserves primal ordering required by dynamics-related block computations
- `wait_until_ready()` blocks until code-generated functions are available

`ocp` is the static formulation. Runtime values and derivatives live elsewhere.

### Problem Types

There are now three related problem containers that matter in practice:

- `ocp`
  - generic base/container used by solver internals and legacy paths
- `node_ocp`
  - node-local modeling prototype
  - only `x/u/p`-style terms are accepted
  - rejects `y`-dependent terms
  - rejects `__dyn`
- `edge_ocp`
  - interval/stage problem consumed by the solver
  - may contain `x/u/y`
  - dynamics must live here

The key invariant is:

- users model node semantics on `node_ocp`
- graph compose produces `edge_ocp`
- the SQP solver ultimately consumes composed interval problems

## Graph Modeling

The current recommended API is graph-first.

Core types:

- [`graph_model`](/home/harper/Documents/moto/include/moto/model/graph_model.hpp)
- [`model_node`](/home/harper/Documents/moto/include/moto/model/graph_model.hpp)
- [`model_edge`](/home/harper/Documents/moto/include/moto/model/graph_model.hpp)

Intended semantics:

- `model_node` holds node-local prototype terms
- `model_edge` holds edge-local terms such as dynamics
- pure `x` node terms may be lowered during compose onto edge `y` storage if the solver backend wants that
- explicit terminal terms added with `add_terminal(...)` must stay explicit and not be silently materialized onto regular edges

Important compose rules that are now covered by unit tests:

- intermediate node `__eq_x` lowers onto the predecessor edge `y`
- sink non-terminal pure-`x` costs may materialize onto the incoming edge `y`
- explicit terminal costs stay on the terminal node / terminal tail
- codegen finalization of lowered/materialized clones must be serialized or uniquely named to avoid `.so` races

## SQP Graph Ownership

`ns_sqp` still owns an internal [`directed_graph`](/home/harper/Documents/moto/include/moto/core/directed_graph.hpp), but user-facing modeling should go through [`graph_model`](/home/harper/Documents/moto/include/moto/model/graph_model.hpp), not through solver graph helpers.

Current recommended Python flow:

```python
sqp = moto.sqp(n_job=10)
modeled = sqp.create_graph()

stage_node = modeled.create_node(stage_node_proto)
terminal_node = modeled.create_node(terminal_node_proto)

for edge in modeled.add_path(stage_node, terminal_node, N):
    edge.add(model.dyn)

flat_nodes = modeled.flatten_nodes()
```

Current division of responsibility:

- user edits only the `graph_model`
- `sqp.create_graph()` returns a solver-aware graph model that reuses `graph_model`'s API and only adds solver realization via `flatten_nodes()`
- the raw `sqp.graph` object still exists as solver storage / traversal infrastructure, but it is not the recommended modeling surface

Useful modeling entry points:

- `create_graph()`
- `graph_model.create_node(...)`
- `graph_model.connect(...)`
- `graph_model.add_path(...)`
- `graph_model.flatten_nodes()`

The design direction is:

- keep `directed_graph` as internal solver storage / traversal machinery
- let `graph_model` / `sqp.create_graph()` be the public modeling surface
- avoid re-exposing graph-building APIs directly on `ns_sqp`

## X-U-Y Triplet Formulation

The current stage model is built around three primal blocks:

- `x_k`: current state entering stage `k`
- `u_k`: control applied at stage `k`
- `y_k`: state leaving stage `k`

This is a valid solver formulation, but it mixes two different concerns:

- modeling semantics: "what variables does the user think this stage owns?"
- solver algebra: "which state copy is most convenient for the nullspace / Riccati factorization?"

Today the solver still stores some path-state algebra on `y`, but the modeling interface should stay simpler:

- users write `constr.create(...)` and `cost.create(...)` normally
- if an expression is terminal, the user writes `prob.add_terminal(...)`
- if an expression is a path-state equality that the solver wants on predecessor storage, that should be decided during graph compose, not by hidden mutation of the authored expression

That makes stage-local modeling harder than it needs to be.

Recommended usage:

- define node-local expressions on `node_ocp`
- create graph nodes with `graph_model.create_node(...)`
- connect or expand paths with `graph_model.connect(...)` / `graph_model.add_path(...)`
- add terminal terms with `prob.add_terminal(...)`
- put `__dyn` and any `y`-dependent terms only on `edge_ocp` / `model_edge`
- build solver paths through `sqp.create_graph()`

### Best Internal Mental Model

If the solver keeps the triplet, the cleanest interpretation is:

- `x`: state owned by the node
- `u`: action owned by the outgoing edge
- `y`: predicted outgoing state copy used only by the solver

That is better than presenting all three as peer modeling variables.

## Model Graph And Directed Graph

Recent refactor work exposed an important design constraint:

- `graph_model` is the modeling-side graph
- `directed_graph` is the solver/runtime graph
- graph-building APIs should live on `graph_model`
- `ns_sqp` should consume / realize a graph model rather than mirror its topology API

This matters for lowering:

- lowering is not fundamentally "move a term from one node to another node"
- it is "assign a node-authored term to the correct solver edge storage"
- in particular, the intended legacy-compatible interpretation is:
  - authored `g(x_k)` may lower to predecessor-edge `y_{k-1}` storage
  - not to `x_{k-1}`
  - and not merely to the current edge's local `y_k` by a blind argument substitution

Because of that, graph-aware compose should be centered on edges:

- node remains the authoring surface for local costs and constraints
- edge is the unit that receives solver-local dynamics and lowered path equalities
- "first point does not lower, subsequent repeated points do lower" must be decided from actual predecessor-edge structure

Practical implication for future work:

- do not keep adding semantic policy inside `edge_ocp::compose()` alone
- do not reintroduce finalize-time silent substitution
- prefer a graph-level lowering/composition pass that:
  - sees the whole modeled graph
  - assigns eligible node-local `__eq_x` terms onto predecessor edges
  - leaves costs, inequalities, and terminal terms untouched unless explicitly requested
  - then materializes solver problems in a form compatible with `directed_graph`

Current status of the refactor:

- Python `model_node` / `model_edge` now reuse `node_ocp` / `edge_ocp` APIs through inheritance
- compose-time logging exists for lowering and should remain explicit because it changes formulation
- quadruped has been partially migrated to modeled composition, but full formulation parity with the pre-refactor version has not yet been restored
- the remaining gap is likely because the final compose logic still needs to become truly graph-level and edge-centric rather than per-edge local patching

In short:

- user-facing model: `x_k`, `u_k`, `x_N`
- internal solver model: `x/u/y`
- bridge between them: explicit lowering, not implicit substitution

## Expression And Function Model

Most solver-facing functions derive from [`generic_func`](/home/harper/Documents/moto/include/moto/ocp/impl/func.hpp).

What `generic_func` provides:

- expression identity and field
- input argument list `in_args_`
- approximation order: `zero`, `first`, `second`
- callbacks `value`, `jacobian`, `hessian`
- CasADi-backed codegen support
- enable/disable rules like `enable_if_all(...)`, `enable_if_any(...)`, `disable_if_any(...)`
- active-argument queries per OCP

How evaluation works:

- `compute_approx(data, eval_val, eval_jac, eval_hess)` dispatches to `value_impl`, `jacobian_impl`, `hessian_impl`
- if a function was created from CasADi, finalize/codegen loads compiled callbacks
- function arguments are mapped into dense primal storage through `func_arg_map`

## Constraint Hierarchy

Constraint classes layer solver-specific state on top of `generic_func`.

Hierarchy:

- `generic_constr`
- `soft_constr`
- `ineq_constr`
- `solver::ipm_constr`
- `solver::pmm_constr`

Responsibilities:

- `generic_constr`: multiplier mapping and constraint-field finalization
- `soft_constr`: Newton-step split state, Jacobian modifications, dual-step storage
- `ineq_constr`: complementarity residual storage
- `ipm_constr`: slack, NT scaling, barrier residuals, IPM predictor/corrector logic
- `pmm_constr`: soft-equality PMM Schur-complement terms

Registry-based conversion:

- `generic_constr::cast_ineq("ipm")`
- `generic_constr::cast_soft("pmm_constr")`

is backed by [`src/solver/ineq_soft_reflect.cpp`](/home/harper/Documents/moto/src/solver/ineq_soft_reflect.cpp).

## Per-Stage Runtime Storage

### `node_data`

[`node_data`](/home/harper/Documents/moto/include/moto/ocp/impl/node_data.hpp) is the runtime view of one stage. It owns:

- `prob_`: the stage `ocp`
- `sym_`: serialized symbol values
- `dense_`: dense merged derivative storage
- `shared_`: shared auxiliary data for custom functions
- `sparse_`: per-function sparse approximation objects

Each function in the problem gets one sparse approx object created during `node_data` construction.

### `sym_data`

`sym_data` holds the current primal values for each symbolic field.

Important behavior:

- primal vectors are initialized with symbol default values when present
- `integrate(field, dx, alpha)` applies tangent-space updates symbol-by-symbol
- `get(sym)` returns a view into the appropriate serialized field vector

### `shared_data`

`shared_data` is a UID-keyed store for precompute or user-defined custom function state shared within a stage.

### `func_arg_map`

`func_arg_map` is a sparse view from a function’s argument list into the stage’s serialized primal values.

It stores:

- references to input arguments
- a UID-to-argument-index map
- a backreference to the problem and shared-data store

### `func_approx_data`

`func_approx_data` adds derivative mappings on top of `func_arg_map`.

It exposes:

- `v_`: function value view
- `jac_`: per-argument Jacobian views
- `lag_jac_`: views into the stage’s dense cost/Lagrangian gradient blocks
- `lag_hess_`: views into the stage’s dense Hessian blocks

This is the bridge from function-local derivatives to the global per-stage QP data structures.

## Dense Merged Derivative Storage

[`lag_data`](/home/harper/Documents/moto/include/moto/ocp/impl/lag_data.hpp) is the central dense store for one stage.

It contains:

- `approx_[cf].v_`: dense residual vector for each constraint field
- `approx_[cf].jac_[pf]`: dense/sparse Jacobian blocks by constraint field and primal field
- `dual_[cf]`: current dual variables
- `comp_[cf]`: complementarity residuals for inequalities
- `cost_`: pure stage cost
- `lag_`: cost plus dual-weighted constraint residual terms
- `cost_jac_[pf]`: pure cost gradient
- `lag_jac_[pf]`: base stage Lagrangian gradient
- `lag_jac_corr_[pf]`: pending additive gradient correction for the next linear solve
- `lag_hess_[a][b]`: main upper-triangular Hessian blocks
- `hessian_modification_[a][b]`: pending Hessian correction terms
- projected dynamics buffers `proj_f_x_`, `proj_f_u_`, `proj_f_res_`

Important gradient distinction:

- `cost_jac_` is pure cost only
- `lag_jac_` is the persistent base stage gradient `cost_jac_ + J^T lambda`
- `lag_jac_corr_` is solver-owned scratch for pending corrections from IPM, PMM, restoration, or refinement
- line search uses `cost_jac_` for `obj_fullstep_dec`
- dual residual / stationarity checks use `lag_jac_`

## `update_approximation()` Data Flow

The core stage assembly routine is [`node_data::update_approximation()`](/home/harper/Documents/moto/src/ocp/node_data.cpp).

Its flow is:

1. Zero cost/lagrangian value if value evaluation is requested.
2. Zero derivative accumulators and pending gradient/Hessian correction buffers if Jacobians/Hessians are requested.
3. Run `__pre_comp` custom functions.
4. Call `compute_approx(...)` on every function in every function field.
5. Run `__post_comp` custom functions.
6. Snapshot `cost_jac_ = lag_jac_` before constraint dual contributions are added.
7. For each stored constraint field:
   constraint residual contributes to `lag_`
   Jacobian-transpose times dual contributes to `lag_jac_`
8. If value evaluation is active:
   compute `inf_prim_res_`, `prim_res_l1_`, `inf_comp_res_`
   add `cost_` into `lag_`

Mental model:

- every function writes into local sparse refs
- those refs are aliases into `lag_data`
- after all functions run, `lag_data` contains the entire per-stage dense QP approximation

## Solver Data Layer

### `data_base`

[`data_base`](/home/harper/Documents/moto/include/moto/solver/data_base.hpp) wraps `lag_data` with solver-facing aliases and step storage.

Important aliases:

- `Q_x`, `Q_u`, `Q_y` alias the active stage gradient in `lag_jac_`
- `Q_xx`, `Q_ux`, `Q_uu`, `Q_yx`, `Q_yy` alias `lag_hess_`
- `_mod` variants alias `hessian_modification_`

Additional solver state:

- `base_lag_grad_backup[pf]`: snapshot of the base stage gradient before a correction solve activates a pending modification
- `kkt_stat_err_[pf]`: solver-owned KKT stationarity error used by iterative refinement
- `V_xx`, `V_yy`: value-function Hessian terms accumulated by Riccati recursion
- `trial_prim_step[pf]`: current Newton step for each primal field
- `prim_corr[pf]`: correction step for iterative refinement / corrector steps
- `trial_prim_state_bak[pf]`: line-search rollback state
- `trial_dual_step[cf]`: dual Newton step for each constraint field
- `trial_dual_state_bak[cf]`: line-search rollback dual state

Key helper methods:

- `activate_lag_jac_corr()`: backs up `Q_x/Q_u/Q_y`, then adds `lag_jac_corr_` into the active stage gradient
- `swap_active_and_lag_jac_corr()`: swaps `lag_jac_` and `lag_jac_corr_` for correction solves
- `backup_trial_state()` / `restore_trial_state()`: line-search checkpointing
- `first_order_correction_start/end()`: prepare and restore correction-mode gradient corrections

### `ns_riccati_data`

[`ns_riccati_data`](/home/harper/Documents/moto/include/moto/solver/ns_riccati/ns_riccati_data.hpp) extends `data_base` with nullspace/Riccati-specific objects.

Dimensions and matrix aliases:

- `nx`, `nu`, `ny` from `data_base`
- `ns`, `nc`, `ncstr` for equality counts
- `nis`, `nic` for active inequality counts
- `F_x`, `F_u`, `F_0` for projected dynamics
- `s_y`, `s_x`, `c_x`, `c_u` for equality Jacobian blocks

Step sensitivities:

- `d_u.k`, `d_u.K`
- `d_y.k`, `d_y.K`

Multiplier-related state:

- `d_lbd_f`
- `d_lbd_s_c_pre_solve`
- `d_lbd_s_c`

Auxiliary mode hook:

- `aux_` can hold mode-specific state
- restoration uses `restoration_aux_data` with `rho_eq`

### `nullspace_data`

Nested inside `ns_riccati_data`, this stores the factorization products used by the stage solve.

Key members:

- `s_c_stacked`: stacked equality Jacobian w.r.t. `u`
- `s_c_stacked_0_K`: stacked equality Jacobian w.r.t. `x`
- `s_c_stacked_0_k`: stacked equality residual
- `lu_eq_`: LU of equality `u` Jacobian
- `rank`: rank of `s_c_stacked`
- `Z_u`: nullspace basis in control space
- `Z_y`: nullspace basis mapped through dynamics
- `Q_zz`: projected Hessian in nullspace coordinates
- `u_y_K`, `u_y_k`: particular solution components for equality satisfaction
- `y_y_K`, `y_y_k`: induced closed-loop dynamics under equality elimination
- `z_0_K`, `z_K`, `z_0_k`, `z_k`: nullspace reduced coordinates and solves

## Top-Level Solver Object

[`ns_sqp`](/home/harper/Documents/moto/include/moto/solver/ns_sqp.hpp) owns:

- `graph_`: ordered stage graph of `shooting_node<data>`
- `mem_`: node-data memory pool
- `riccati_solver_`: `generic_solver`
- `settings`
- `kkt_last`

`ns_sqp::data` combines:

- `node_data`
- `ns_riccati_data`
- scaling caches:
  - `scale_c_`
  - `scale_p_`
  - `scaling_applied_`

## Settings Layout

Main settings live in `ns_sqp::settings_t` and are defined in the header.

Important sub-groups:

- `settings.ls`: line search parameters
- `settings.ipm`: barrier and predictor-corrector settings
- `settings.rf`: iterative refinement settings
- `settings.scaling`: Jacobian scaling settings
- `settings.restoration`: restoration settings

Important invariants:

- `settings.ls` and `settings.ipm` are references into `settings_t`
- do not copy `settings_t` by value after construction
- if adding a setting:
  update the header
  update the implementation
  update the bindings

## Initialization Flow

[`ns_sqp::initialize()`](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp) performs:

1. reset `mu` if not warm-starting
2. for every node:
   call `setup_workspace_data(...)` on each constraint
   evaluate values with `update_approximation(eval_val)`
   initialize inequality / soft constraints via `solver::ineq_soft::initialize`
3. for every node:
   evaluate derivatives with `update_approximation(eval_derivatives)`
4. compute initial `kkt_info`
5. reset scaling caches
6. print stats header and iteration-0 stats if verbose

`solver::ineq_soft::initialize(...)` wires each soft constraint’s `prim_step_` views into `trial_prim_step[...]`, binds its `d_multiplier_` into `trial_dual_step[...]`, and calls the specific soft-constraint initializer.

## SQP Iteration Flow

The core iteration routine is `ns_sqp::sqp_iter(...)`.

Its runtime sequence is:

1. Reset line-search worker state.
2. Optionally scale equality Jacobians and residuals.
3. `ns_factorization(...)` on every node.
4. Backward `riccati_recursion(...)`.
5. `compute_primal_sensitivity(...)`.
6. Forward `fwd_linear_rollout(...)`.
7. If inequalities exist, start predictor mode.
8. Finalize primal step and compute line-search bounds.
9. If inequalities exist, run corrector step and resolve.
10. Optionally run iterative refinement.
11. Finalize dual Newton step.
12. Unscale dual step and restore Jacobians/residuals to original units.
13. Backup primal and dual states for line search.
14. Run line-search trial loop:
    restore backed-up state
    apply affine step to primal and soft-constraint states
    evaluate values
    evaluate KKT residuals
    accept or backtrack
15. On acceptance, update derivatives if needed and store `kkt_current = kkt_trial`.

## Nullspace / Riccati Solve Data Flow

### 1. Projected dynamics update

`ns_factorization(...)` starts by calling:

- `update_projected_dynamics()`
- `activate_lag_jac_corr()`

This makes sure the stage QP sees all pending gradient corrections from IPM, PMM, restoration, or refinement.

### 2. Copy base blocks

`Q_ux`, `Q_yx`, `Q_xx`, `Q_yy` and their modification blocks are copied into working matrices like:

- `u_0_p_K`
- `y_0_p_K`
- `V_xx`
- `V_yy`

### 3. Build equality Jacobian stacks

For hard equalities:

- `s_c_stacked = [s_y * F_u ; c_u]`
- `s_c_stacked_0_K = [s_x + s_y * F_x ; c_x]`

LU factorization of `s_c_stacked` gives rank information and equality elimination data.

### 4. Constrainedness branch

Cases:

- no equality constraints: unconstrained setup
- rank 0: unconstrained setup
- rank = `nu`: fully constrained
- otherwise: constrained with nontrivial nullspace `Z_u`

In the constrained case:

- `Z_u = kernel(s_c_stacked)`
- `Z_y = F_u * Z_u`
- `Q_zz = Z_u^T * (Q_uu + Q_uu_mod) * Z_u`
- `u_y_K = solve(s_c_stacked_0_K)`
- `y_y_K = F_x + F_u * u_y_K`

### 5. Residual correction setup

`ns_factorization_correction(...)` builds:

- `s_c_stacked_0_k`
- `u_y_k`
- `y_y_k`
- `z_0_k`

These capture the feedforward correction induced by equality residuals.

### 6. Backward Riccati recursion

`riccati_recursion(...)`:

- symmetrizes `V_yy`
- forms `y_0_p_k`, `y_0_p_K`
- augments `Q_zz` with future-state terms via `V_yy`
- solves the reduced LLT system in `Q_zz`
- updates `Q_x` and `V_xx`
- propagates first-order and second-order value terms into the previous node’s `Q_y` and `V_yy`

Cross-stage propagation uses the permutation from one stage’s `__y` layout to the next stage’s `__x` layout.

### 7. Forward rollout

Forward rollout reconstructs primal steps:

- `trial_prim_step[__u] = d_u.k + d_u.K * trial_prim_step[__x]`
- `trial_prim_step[__y] = d_y.k + d_y.K * trial_prim_step[__x]`
- next stage `__x` tangent is populated from current stage `__y`

### 8. Dual step recovery

`finalize_dual_newton_step(...)` computes:

- `d_lbd_f` from `Q_y`, `V_yy`, and current primal step
- equality duals from the LU solve or GN reconstruction
- `trial_dual_step[__dyn]` by applying inverse-transpose `f_y^{-T}`

In normal constrained mode:

- solve `lu_eq_.transpose().solve(...)` for hard-equality multipliers

In restoration GN mode:

- recover `dlam = (h + J_u du + J_x dx) / rho_eq`

## KKT Information And Residual Accounting

[`compute_kkt_info(...)`](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp) aggregates across all nodes:

- `cost`
- `objective = cost - mu * log_slack_sum`
- `obj_fullstep_dec`
- `inf_prim_res`
- `prim_res_l1`
- `inf_dual_res`
- `avg_dual_res`
- `inf_comp_res`
- primal/dual step norms
- max dual norms
- IPM diagnostics like `max_diag_scaling`

Important dual-residual detail:

- `inf_dual_res` is not a raw stationarity norm
- it is IPOPT-style scaled by
  `s_d = max(s_max, ||lambda||_1 / n_constr) / s_max`

Cross-stage dual residual for state/costate consistency is formed using:

- current stage `lag_jac_[__y]`
- next stage `lag_jac_[__x]`
- `permutation_from_y_to_x(...)`

## Line Search Flow

`moto` supports:

- IPOPT-style filter line search
- simple merit-function backtracking

### Filter line search

Core logic is in [`src/solver/sqp_impl/line_search.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/line_search.cpp).

Trial acceptance uses:

- filter dominance against stored points
- IPOPT switching condition
- Armijo condition in switching mode
- otherwise, sufficient progress against the current iterate
- optional flat-objective acceptance

Key details:

- stored filter points contain primal residual, dual residual, and barrier objective
- barrier objective is recomputed with current `mu`
- `fullstep_dec = obj_fullstep_dec - mu * barrier_dir_deriv`
- `fullstep_dec < 0` must be checked before `pow(...)` to avoid NaNs
- backtracking can be linear or geometric
- failure fallback is either minimum step or best trial

The SOC scaffolding exists but is intentionally not implemented:

- `second_order_correction()` is empty
- dispatch paths still exist and should not be removed casually

### Merit backtracking

Alternative line search uses:

- `merit = prim_res_l1^2 + sigma * avg_dual_res^2`

with an Armijo condition based on a finite-difference directional derivative estimate from the full step.

## Scaling Flow

[`src/solver/sqp_impl/scaling.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/scaling.cpp) applies cached in-place row scaling.

Currently scaled:

- hard equalities excluding dynamics
- specifically `__eq_x` and `__eq_xu`

Intentionally not scaled:

- `__dyn`, because `jac_[__y]` aliases `f_y` and would corrupt projected-dynamics LU usage
- IPM inequalities, because their Jacobians and duals are managed inside the IPM model
- cost gradients, because `Q_y` is propagated across stages during the backward recursion, so in-place scaling would contaminate cross-stage first-order accumulation

Recompute policy:

- scales are recomputed on first use
- scales are recomputed when `inf_prim_step >= 1 / update_ratio_threshold`
- otherwise cached scales are reused

Application:

- residual rows are multiplied by row scales
- Jacobian rows are multiplied by row scales

Unscaling after the QP solve:

- residuals are divided back
- Jacobian rows are divided back
- dual steps are multiplied by the same row scales
- accumulated dual variables are not rescaled

## Inequality / Soft-Constraint Flow

Shared dispatch lives in:

- [`include/moto/solver/ineq_soft.hpp`](/home/harper/Documents/moto/include/moto/solver/ineq_soft.hpp)
- [`src/solver/ineq_soft_impl.cpp`](/home/harper/Documents/moto/src/solver/ineq_soft_impl.cpp)

This layer:

- iterates all soft and inequality constraints in a node
- binds Newton-step views
- calls type-specific hooks for initialization, predictor/corrector, line search, and state backup

### IPM inequalities

`ipm_constr` stores:

- raw constraint value `g_`
- residual `r_s_`
- slack
- multiplier
- NT diagonal scaling
- scaled residuals
- predictor/corrector terms

IPM derivative flow:

- `value_impl`: set `g_`, form `v_ = g + slack`, update complementarity-related vectors
- `jacobian_impl`: build NT scaling and scaled residuals
- `propagate_jacobian`: add barrier-induced gradient terms into `lag_jac_corr_`
- `propagate_hessian`: add `J^T D J` into Hessian blocks

IPM Newton-step flow:

- `finalize_newton_step`: compute `d_slack` and `d_multiplier`
- `update_ls_bounds`: clip primal and dual alpha so slack and multipliers stay positive
- `finalize_predictor_step`: collect Mehrotra affine-step stats
- `apply_corrector_step`: switch scaled residuals to the corrected barrier target
- `apply_affine_step`: update slack and multiplier during line search

### PMM soft equalities

`pmm_constr` implements a Schur-complement PMM model:

- `g_` stores raw residual `h = C(x)`
- Jacobian propagation adds `(1/rho) J^T h`
- Hessian propagation adds `(1/rho) J^T J`
- dual Newton step is `dlam = (J du + h) / rho`

PMM line-search state:

- only multiplier backup/restore is needed
- no slack variables exist

## Predictor-Corrector And Iterative Refinement

### IPM predictor-corrector

If inequalities are present:

- first solve produces an affine predictor step
- worker-local stats are merged
- adaptive `mu` update may happen
- the solver reruns a correction solve with updated barrier data
- line-search bounds are recomputed afterward

### Iterative refinement

[`src/solver/sqp_impl/iterative_refinement.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/iterative_refinement.cpp) does:

1. finalize dual step
2. compute KKT stationarity residuals
3. aggregate residual norms
4. if needed, inject `kkt_stat_err_` into `lag_jac_corr_` via `first_order_correction_start(...)`
5. rerun factorization/backward/forward correction passes
6. add corrections into `trial_prim_step`
7. restore original Jacobian state
8. recompute line-search bounds

This is a true correction solve on the linearized KKT system, not a full relinearization of the nonlinear problem.

## Restoration Mode

[`src/solver/sqp_impl/restoration.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/restoration.cpp) implements a Gauss-Newton-style feasibility-restoration mode.

Current status:

- restoration logic exists
- the automatic trigger block in `update()` is commented out
- it can still be called from code

Restoration setup:

- snapshot `u_ref` and `y_ref`
- compute per-component relative scaling `sigma = 1 / max(|ref|, 1)`
- allocate `restoration_aux_data` with `rho_eq`
- inject proximal gradient terms into `lag_jac_corr_`
- inject diagonal proximal Hessian terms into `ns_sqp::data::primal_prox_hess_diagonal_`

Restoration factorization behavior:

- dynamics stays hard
- `__eq_x` and `__eq_xu` are treated like PMM soft constraints in Gauss-Newton mode
- original cost terms remain intact
- equality residuals add PMM-like gradient and Hessian contributions
- Riccati solve proceeds in unconstrained mode while LU data is preserved for dual recovery

Exit conditions:

- accept if primal infeasibility improves sufficiently
- mark `infeasible_stationary` if dual residual falls below tolerance without primal feasibility
- otherwise mark `restoration_failed`

## Diagnostics

Useful built-in diagnostics:

- `print_stats(...)`
- `print_scaling_info()`
- `print_dual_res_breakdown()`
- `print_licq_info()`

`print_licq_info()` performs a global LICQ check using forward nullspace propagation and approximately active inequalities.

Verbose output rules:

- logging should remain gated behind `settings.verbose`
- avoid unconditional printing in hot solver paths

## Python Bindings

Bindings live in `bindings/` and use nanobind.

Important files:

- [`bindings/setup_bindings.cpp`](/home/harper/Documents/moto/bindings/setup_bindings.cpp)
- [`bindings/definition/ns_sqp.cpp`](/home/harper/Documents/moto/bindings/definition/ns_sqp.cpp)

When adding or changing settings, enum values, or public solver surface area:

- update the header
- update the implementation
- update bindings and generated stubs if needed

## Practical Editing Rules

- follow nearby style instead of introducing a new one
- prefer understanding aliasing before editing matrices or vectors in place
- be careful with any change touching `Q_y`, `f_y`, or stage permutations
- remember many sparse/dense objects are views into shared storage, not owned copies
- do not copy `settings_t`
- do not assume soft constraints are only inequalities; PMM soft equalities use the same dispatch layer
- do not remove commented or dormant solver hooks like restoration or SOC infrastructure without checking intended roadmap
- when changing C++ code that affects Python examples, always wait for the full build to finish before running Python tests
- do not trust a Python test run started while `moto` / `moto_pywrap` is still linking; stale modules can easily give misleading results

## Common Pitfalls

- confusing `cost_jac_` with `lag_jac_`
- forgetting that `update_approximation()` snapshots `cost_jac_` before adding `J^T lambda`
- modifying `__dyn` scaling and breaking projected-dynamics solves
- assuming `__eq_x` / `__ineq_x` differentiate against `__x`
- touching line-search or IPM state without handling backup/restore
- changing a settings struct without updating bindings
- treating [`example/quadruped/run.py`](/home/harper/Documents/moto/example/quadruped/run.py) as canonical; it is often used for experiments
- seeing `warning: substitution in generic_constr ... go2_q_nxt` in quadruped logs means the example is still using the legacy `ocp.create()` path, not the newer `node_ocp / edge_ocp` modeling path

## Suggested Reading Order For Solver Work

If you are new to the repo and need to debug solver behavior, read in this order:

1. [`include/moto/solver/ns_sqp.hpp`](/home/harper/Documents/moto/include/moto/solver/ns_sqp.hpp)
2. [`src/solver/sqp_impl/ns_sqp_impl.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp)
3. [`src/ocp/node_data.cpp`](/home/harper/Documents/moto/src/ocp/node_data.cpp)
4. [`src/solver/nsp_impl/presolve.cpp`](/home/harper/Documents/moto/src/solver/nsp_impl/presolve.cpp)
5. [`src/solver/nsp_impl/backward.cpp`](/home/harper/Documents/moto/src/solver/nsp_impl/backward.cpp)
6. [`src/solver/nsp_impl/rollout.cpp`](/home/harper/Documents/moto/src/solver/nsp_impl/rollout.cpp)
7. [`src/solver/sqp_impl/line_search.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/line_search.cpp)
8. [`src/solver/ipm_impl/ipm_constr.cpp`](/home/harper/Documents/moto/src/solver/ipm_impl/ipm_constr.cpp)
9. [`src/solver/soft_impl/pmm_constr.cpp`](/home/harper/Documents/moto/src/solver/soft_impl/pmm_constr.cpp)

That path covers most bugs involving assembly, factorization, rollout, globalization, and inequality handling.

## Current Refactor Status

As of the current working tree, the OCP layer has been partially refactored to introduce:

- `ocp_base` as the shared storage / activation / flattening container
- `ocp` as the generic legacy-compatible problem type
- `node_ocp` as a thin node-local wrapper
- `edge_ocp` as a thin transition-local wrapper that can bind start/end node problems

What is already true:

- [`include/moto/ocp/problem.hpp`](/home/harper/Documents/moto/include/moto/ocp/problem.hpp) now documents the intended role split between these types
- clone logic was deduplicated into `ocp_base::refresh_after_clone(...)`
- quadruped still matches the historical baseline after the refactor:
  - build with `cmake -E env CCACHE_DISABLE=1 cmake --build build -j4`
  - run with `python example/quadruped/run.py`
  - expected result is convergence in `46` iterations with objective about `3.060e+02`

What is not true yet:

- the quadruped example is not yet using the new modeling layer as its primary construction path
- the presence of `x -> y` substitution warnings in quadruped output confirms that it still goes through the legacy stage formulation path
- `node_ocp / edge_ocp` are still intentionally thin wrappers; they do not yet replace the legacy lowering path end-to-end

Recommended next step from here:

1. Validate the new node/edge modeling path on a small example first
2. Confirm that the small example does not rely on legacy pure-`x -> y` substitution
3. Only then migrate [`example/quadruped/run.py`](/home/harper/Documents/moto/example/quadruped/run.py) to the new modeling path

This order is important because quadruped is too large to use as the first proving ground for new node/edge semantics.
