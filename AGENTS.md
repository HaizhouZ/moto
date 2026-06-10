# moto AGENTS Guide

## Purpose

This file is a maintainer-oriented map of the `moto` codebase. It is meant to help an agent or human contributor answer:

- where core data lives
- how an OCP stage is represented
- how raw function evaluations become a QP
- how the SQP / Riccati / line-search pipeline moves data each iteration
- which conventions are easy to violate when editing solver code

This guide was derived from the code itself, especially:

- `CLAUDE.md`
- `include/moto/ocp/graph_model.hpp`
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

- [`include/moto/core/fields.hpp`](/home/harper/Documents/moto/include/moto/core/fields.hpp): field taxonomy like `__x`, `__u`, `__y`, `__s`, `__dyn`, `__eq_x`, `__ineq_xu`
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
- [`src/solver/sqp_impl/iterative_refinement.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/iterative_refinement.cpp): residual correction
- [`include/moto/solver/ipm/ipm_constr.hpp`](/home/harper/Documents/moto/include/moto/solver/ipm/ipm_constr.hpp): IPM inequality implementation
- [`include/moto/solver/soft_constr/pmm_constr.hpp`](/home/harper/Documents/moto/include/moto/solver/soft_constr/pmm_constr.hpp): PMM soft equality implementation
- [`docs/restoration.md`](/home/harper/Documents/moto/docs/restoration.md): restoration design note; useful for the elastic KKT / condensation math, but not a complete description of the current overlay-based implementation
- [`bindings/`](/home/harper/Documents/moto/bindings): Python bindings
- [`example/`](/home/harper/Documents/moto/example): manual examples
- [`unittests/`](/home/harper/Documents/moto/unittests): Catch2 tests
- [`include/moto/ocp/graph_model.hpp`](/home/harper/Documents/moto/include/moto/ocp/graph_model.hpp): graph-first modeling layer
- [`include/moto/solver/linear_runtime_graph.hpp`](/home/harper/Documents/moto/include/moto/solver/linear_runtime_graph.hpp): internal linear solver traversal storage

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
- current CMake test registration includes `sym_test` and `graph_model_compose_test`

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
KMP_AFFINITY='noverbose,granularity=fine,scatter' OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MOTO_PROFILE_SQP=1 MOTO_SQP_BENCH_RUNS=1 MOTO_SQP_MAX_ITER=50 python example/quadruped/run.py
```

Current practical note from local quadruped profiling:

- when benchmarking `example/quadruped/run.py` with `OMP_NUM_THREADS=10`, keeping `Eigen::setNbThreads(1)` in `ns_sqp` construction was measurably better than leaving Eigen's internal thread count unconstrained
- the most comparable local check was:
  - `KMP_AFFINITY='noverbose,granularity=fine,scatter' OMP_NUM_THREADS=10 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MOTO_PROFILE_SQP=1 MOTO_SQP_BENCH_RUNS=1 MOTO_SQP_MAX_ITER=50 python example/quadruped/run.py`
- on that workload, the version with `Eigen::setNbThreads(1)` reduced total profiled update time from roughly `474 ms` to roughly `415 ms` in one run, and from `526 ms` to `374 ms` after additional iterative-refinement experiments; exact values do vary run-to-run, but the direction was consistently favorable enough to keep the setting in `ns_sqp`
- a more recent warm run of the current quadruped setup converged in `35` iterations with about `112 ms` total profiled update time and about `3.2 ms / iter`
- in that run there was no backtracking:
  - `trial_evals = 35`
  - every iteration had exactly one accepted full-step trial
- the dominant globalization cost in that profile was not filter bookkeeping itself:
  - `run_globalization` was about `0.55 ms / iter`
  - `update_approx_accepted` was about `0.50 ms / iter`
  - `evaluate_trial_point` was only about `0.05 ms / iter`
- practical implication:
  - for current filter runs, the expensive part is often relinearizing the accepted point for the next SQP iteration, not the scalar acceptance logic in [`src/solver/sqp_impl/line_search.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/line_search.cpp)

Useful benchmarking / profiling environment variables in `example/quadruped/run.py`:

- `MOTO_SQP_MAX_ITER`: iterations requested per `sqp.update(...)`
- `MOTO_SQP_BENCH_RUNS`: number of timed hot-start runs
- `MOTO_PROFILE_SQP`: print the last run's SQP profile summary
- `MOTO_SQP_BENCH_VERBOSE`: if set, print verbose logs for every timed run; otherwise only the last timed run is verbose
- `MOTO_PARALLEL_BLOCK_ORDER={forward,reverse}`: override chunk assignment order in `parallel_for`
- `MOTO_DEBUG_SOLVER_PROBS`: print head/tail solver problems for the realized graph
- `MOTO_DEBUG_GRAPH_LAYOUT`: print flattened solver graph dimensions and dynamics count

Current benchmark-script caveats:

- [`example/quadruped/run.py`](/home/harper/Documents/moto/example/quadruped/run.py) no longer has a dedicated warm-up loop
- benchmark timing now measures only the configured `MOTO_SQP_BENCH_RUNS`
- `get_profile_report()` still reports the most recent `update(...)` call only
- the script still defaults to `display = True`, so headless or solver-only profiling often uses a temporary no-visualization copy rather than editing the file in place

Current default parallel behavior:

- chunk order defaults to `forward`
- solver callbacks receive the logical chunk id used by `parallel_for`
- chunking follows the order of the provided view; backward passes use the
  reversed traversal view from `solver::backward_edges(...)`

## Field System

Fields encode the semantic role of symbols and functions.

Primary primal fields:

- `__x`: current state
- `__u`: control
- `__y`: next state

Solver-managed non-primal storage:

- `__s`: internal shared slack storage reserved for solver-owned IPM state and other solver-private storage
- `__p`: non-decision parameters

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
- terminal pure-`x` terms are added through the final stage endpoint view:
  `stages[-1].ed.add(term)`
- `stage_ocp` is the public modeling container
- `stage_ocp.st` and `stage_ocp.ed` are lightweight endpoint views for pure
  `x` terms
- `__dyn`, `u` terms, and mixed interval terms belong on `stage_ocp.add(...)`
- any required lowering from node semantics to solver storage should happen during graph-aware compose, not during generic expression finalization

Why this is easy to trip over:

- a user may write a stage-local constraint `g(x_k)` and expect it to remain attached to the current `x`
- any hidden remap changes formulation, so it must be explicit in logs and scoped by graph topology
- the intended semantics are stage-centric:
  - interval terms stay on the solver interval unchanged
  - graph-start terms are authored explicitly through `sqp.start_node` and
    materialize on the first solver `x`
  - stage start endpoint terms lower onto the incoming interval's solver `y`
    whenever the stage has a predecessor
  - end endpoint terms lower by cached `x -> y` remap and materialize on solver `y`
- this decision cannot be made correctly by a local `finalize()` on a standalone expression or a standalone problem

Useful field groups used throughout the solver:

- `primal_fields = {__x, __u, __y}`
- `__s` is intentionally not part of `primal_fields`
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

There are two related problem containers that matter in practice:

- `ocp`
  - generic container used by C++ solver internals
  - not a Python modeling entry point
- `stage_ocp`
  - public modeling container
  - accepts `__dyn`, `u` terms, mixed interval terms, and pure `x` terms
  - pure `x` terms added with `stage.add(...)` are interval-local and
    materialize on solver `x`
  - pure `y` terms are rejected; users should write the expression on `x` and
    add it with `stage.ed.add(...)`

The key invariant is:

- users model interval semantics on `stage_ocp`
- users model endpoint-only state terms through `stage.st` or `stage.ed`
- graph compose produces internal `ocp` interval problems
- the SQP solver ultimately consumes composed interval problems

## Graph Modeling

The current recommended API is graph-first.

Core type:

- [`graph_model`](/home/harper/Documents/moto/include/moto/ocp/graph_model.hpp)

Intended semantics:

- `stage_ocp.add(...)` holds interval terms such as dynamics, control costs, and
  mixed constraints
- `stage_ocp.st.add(...)` holds pure start-state terms
- `stage_ocp.ed.add(...)` holds pure end-state terms
- when a new phase is appended, pure `x` terms on the new phase's `stage.st`
  are part of the same graph node as the previous phase's `stage.ed`; both
  endpoint sources lower onto the previous interval's solver `y`, and the
  `stage.st` terms are not duplicated on the new phase's outgoing `x`
- end-node pure-`x` terms lower onto the current interval's solver `y`
- terminal terms are ordinary endpoint terms on the last graph-owned stage:
  `stages[-1].ed.add(term)`

Important compose rules that are now covered by unit tests:

- inactive `u` symbols prune the composed interval's primal `u`
- `stage.add(x_only)` materializes on solver `x`
- `sqp.start_node.add(x_only)` materializes on the first solver `x`
- `stage.st.add(x_only)` lowers onto the previous interval's solver `y` when
  the stage is connected after a predecessor, including appended phase
  boundaries
- `stage.ed.add(x_only)` materializes on solver `y`
- endpoint pure-`x` costs and constraints lower onto the relevant interval `y`
- invalid endpoint terms involving `u`, `y`, or `__dyn` are rejected clearly
- returned graph-owned stage handles are mutable and invalidate solver caches
- edits to a prototype after `add_stage(...)` do not affect graph-owned clones
- codegen finalization of lowered/materialized clones must be serialized or uniquely named to avoid `.so` races
- current implementation chooses serialization plus reuse:
  - finalized clones intentionally share stable generated symbol names
  - same-name codegen is serialized with a per-function mutex in [`src/utils/codegen.cpp`](/home/harper/Documents/moto/src/utils/codegen.cpp)
  - compiled `.so` and `.json` outputs are written through `.tmp` files and renamed atomically
- graph realization now lives with `graph_model` itself:
  - `graph_model::compose_stage(...)` handles interval composition and endpoint materialization
  - `ns_sqp::realize_runtime(...)` consumes composed intervals to build internal solver storage

## SQP Graph Ownership

`ns_sqp` owns a default `graph_model`, but the public modeling API is exposed
directly on `sqp` through `start_node()`, `add_stage(...)`, `add_stages(...)`,
and `flatten_nodes()`. Solver storage is realized lazily from that model.

Current restoration detail:

- `ns_sqp` caches separate restoration runtime storage built from overlay problems
- the cached restoration runtime is rebuilt only when the modeled path or restoration settings change
- this avoids rebuilding the restoration overlay on every restoration entry

Current recommended Python flow:

```python
sqp = moto.sqp(n_job=10)

stage = moto.stage_ocp.create()
stage.add(model.dyn)
stage.add(path_cost_or_constraint)

stages = sqp.add_stage(stage, N)
stages[-1].ed.add(model.terminal_cost)

flat_nodes = sqp.flatten_nodes()
```

Important current path semantics:

- `sqp.add_stage(stage, N)` appends exactly `N` graph-owned stage clones
- the first append starts from `sqp.start_node`
- repeated `add_stage(...)` calls append from the current graph tail
- `sqp.add_stages(start_node, stage, N)` appends from an explicit node view
- there is no hidden terminal tail edge anymore
- returned stages are graph-owned mutable clones
- endpoint edits on graph-owned stages invalidate solver/runtime caches

Current division of responsibility:

- user edits the `sqp` graph through `add_stage(...)`, `add_stages(...)`, and
  returned graph-owned stage handles
- `graph_model` owns path composition and topology realization policy
- `sqp.flatten_nodes()` returns realized solver stages for initialization and debug inspection
- Python top-level exports only the modeling/solver surface; low-level runtime/data bindings stay behind `moto._moto_pywrap` for debugging

Useful modeling entry points:

- `sqp.start_node`
- `sqp.add_stage(...)`
- `sqp.add_stages(...)`
- `sqp.flatten_nodes()`

The design direction is:

- keep linear runtime storage internal
- keep the Python modeling surface on `sqp`
- avoid re-exposing low-level runtime graph machinery

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
- if an expression is terminal, the user writes `stages[-1].ed.add(term)`
- if an expression is a path-state equality that the solver wants on predecessor storage, that should be decided during graph compose, not by hidden mutation of the authored expression

That makes stage-local modeling harder than it needs to be.

Recommended usage:

- create a stage prototype with `moto.stage_ocp.create()`
- add dynamics and interval terms with `stage.add(...)`
- add start-state-only terms with `stage.st.add(...)`
- add end-state or terminal state-only terms with `stage.ed.add(...)`
- build solver paths through `sqp.add_stage(stage, N)`
- inspect realized stages through `sqp.flatten_nodes()` when needed

### Best Internal Mental Model

If the solver keeps the triplet, the cleanest interpretation is:

- `x`: state owned by the node
- `u`: action owned by the outgoing edge
- `y`: predicted outgoing state copy used only by the solver

That is better than presenting all three as peer modeling variables.

## Model Graph And Runtime Storage

Recent refactor work exposed an important design constraint:

- `graph_model` is the modeling-side linear path graph
- `ns_sqp` owns one default `graph_model`
- linear runtime storage is internal solver traversal machinery, not a public modeling object
- Python graph-building APIs live on `sqp`
- graph realization policy should live on `graph_model`
- `ns_sqp` should consume a realized graph model rather than mirror its topology or lowering API

This matters for lowering:

- lowering is not fundamentally "move a term from one node to another node"
- it is "assign an endpoint-authored term to the correct solver interval storage"
- in the current strict interpretation:
  - interval terms stay on the current composed interval
  - graph-start pure-`x` terms are authored explicitly through
    `sqp.start_node` and materialize on the first solver `x`
  - stage start endpoint pure-`x` terms lower to the incoming interval's solver
    `y` once the stage is connected after a predecessor
  - end endpoint pure-`x` terms lower to solver `y`
  - at a connected boundary, `prev.ed` and `next.st` are two authoring views of
    the same graph node and their terms are composed onto the predecessor `y`
  - there is no generic predecessor-edge remap beyond endpoint `x -> y` lowering

Because of that, graph-aware compose should be centered on stages:

- `stage_ocp` remains the authoring surface for interval costs and constraints
- endpoint views provide side-specific state-only placement
- ownership of `u` should stay with the outgoing edge, not with an incoming-edge compatibility convention

Practical implication for future work:

- do not keep adding semantic policy inside `ns_sqp`
- do not reintroduce finalize-time silent substitution
- prefer a graph-level lowering/composition pass that:
  - sees the whole modeled graph
  - keeps interval role terms unchanged
  - materializes graph-start endpoint terms on `x`
  - lowers endpoint pure-`x` terms onto `y` when that endpoint is an interval
    end or an appended phase boundary
  - then materializes solver problems in a form compatible with internal linear solver storage

Current status of the refactor:

- Python uses `stage_ocp` directly as the modeling prototype
- Python graph construction goes through `sqp.add_stage(...)` or `sqp.add_stages(...)`
- `node_ocp`, `edge_ocp`, and `add_terminal(...)` are no longer public APIs
- quadruped runs through the modeled composition path and still converges under `MOTO_SQP_MAX_ITER=50`

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
- `ipm_constr`: slack, NT scaling, barrier residuals, IPM predictor/corrector logic; no longer `final`
- `pmm_constr`: soft-equality PMM Schur-complement terms

Restoration note:

- restoration is active through the overlay-based path in [`src/solver/sqp_impl/restoration.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/restoration.cpp)
- restoration overlay problems are assembled in [`src/solver/restoration/resto_overlay.cpp`](/home/harper/Documents/moto/src/solver/restoration/resto_overlay.cpp)
- [`docs/restoration.md`](/home/harper/Documents/moto/docs/restoration.md) remains useful for the local elastic KKT derivation, but current entry/exit, logging, and cleanup behavior live in code

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
- `lag_hess_`: views into the stage’s dense Hessian blocks

This is the bridge from function-local derivatives to the global per-stage QP data structures.

Important current convention:

- callbacks should write function-local first-order data through `jac_`
- `func_approx_data` no longer carries a direct alias into assembled `lag_jac_`
- assembled stage gradients remain owned by [`lag_data`](/home/harper/Documents/moto/include/moto/ocp/impl/lag_data.hpp) and are formed during stage assembly

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
- `lag_jac_corr_` is solver-owned scratch for pending corrections from IPM, PMM, or refinement
- line search uses `cost_jac_` for `augmented_objective_fullstep_dec`
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

Important mode distinction:

- `eval_val` updates values and residual summaries only
- `eval_derivatives` updates Jacobians and Hessians but does not refresh primal-residual summaries by itself
- in current filter line search, trial points are often checked with `eval_val` only, and the accepted point is then relinearized with `eval_derivatives` so the next SQP iteration sees a fresh QP model

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
  - these are now `virtual` on `data_base` and overridden by `ns_sqp::data`
  - `ns_sqp::data` override first applies the base snapshots, then dispatches `ineq_soft::backup_trial_state` / `restore_trial_state`
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

- `aux_` can still hold solver-private mode-specific state when needed
- the active restoration runtime is no longer carried through this hook; overlay restoration is cached on the active graph state instead

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

- active runtime storage is realized from `model_graph_` and accessed internally through `ns_sqp::active_data()`
- `mem_`: node-data memory pool
- `riccati_solver_`: `generic_solver`
- `settings`
- `kkt_last`
- `iter_last`

Public update return value:

- `ns_sqp::result_type`
  - inherits the latest `kkt_info` summary
  - carries iteration metadata separately in `iter`

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
2. refresh `settings.has_ineq_soft` and `settings.has_ipm_ineq` from the active graph
3. for every node:
   call `setup_workspace_data(...)` on each constraint
   bind soft-constraint runtime views with `solver::ineq_soft::bind_runtime(...)`
   evaluate values and derivatives with `update_approximation(eval_all)`
5. compute initial `kkt_info`
6. reset scaling caches
7. print stats header and iteration-0 stats if verbose

Current soft-constraint lifecycle:

- `solver::ineq_soft::bind_runtime(node_data*)` binds runtime views such as `prim_step_` and `d_multiplier_`
- individual soft constraints lazily initialize themselves from `value_impl()` via `solver::ineq_soft::ensure_initialized(...)`
- when restoration / equality-init rebuilds require a fresh soft state, callers explicitly use `solver::ineq_soft::bind_and_invalidate(...)` inside the existing per-node parallel loop

## SQP Iteration Flow

The core iteration routine is `ns_sqp::sqp_iter(...)`.

Its runtime sequence is:

1. Reset line-search worker state.
2. Optionally scale equality Jacobians and residuals.
3. `ns_factorization(...)` on every node.
4. Backward `riccati_recursion(...)`.
5. `compute_primal_sensitivity(...)`.
6. Forward `fwd_linear_rollout(...)`.
7. If true IPM inequalities exist, start predictor mode.
8. Finalize primal step and compute line-search bounds.
9. If true IPM inequalities exist, run corrector step and resolve.
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

Accepted-trial detail that matters for performance:

- in filter mode, the accepted point commonly reaches `accept_trial_point(...)` with only value information
- the solver then performs a full `eval_derivatives` pass on the accepted point before the next SQP iteration
- in merit-backtracking mode, trial derivatives may already be available, so acceptance can skip that extra derivative refresh

## SQP Profiling

`ns_sqp` now keeps a wall-clock profile report for the most recent `update(...)` call.

Relevant surface area:

- [`include/moto/solver/ns_sqp.hpp`](/home/harper/Documents/moto/include/moto/solver/ns_sqp.hpp): profile phase enums and report structs
- [`src/solver/sqp_impl/ns_sqp_impl.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp): top-level profiling scopes
- [`src/solver/sqp_impl/iterative_refinement.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/iterative_refinement.cpp): refinement-specific profiling scopes
- [`bindings/definition/ns_sqp.cpp`](/home/harper/Documents/moto/bindings/definition/ns_sqp.cpp): Python exposure via `get_profile_report()`

What the report tracks:

- total update time
- initialization time
- per-phase total / average milliseconds and call counts
- per-iteration totals
- trial evaluation counts

The most useful currently exposed phases for bottleneck work are:

- `solve_direction`
- `riccati_recursion`
- `run_globalization`
- `evaluate_trial_point`
- `apply_affine_step`
- `update_res_stat`
- `accept_trial_point`
- `update_approx_accepted`
- `iterative_refinement`
- `iterative_refinement_step`
- `correction_post_factorization`
- `correction_riccati_recursion`
- `correction_fwd_rollout`

Current profiling takeaway for quadruped:

- if `trial_evaluations` is close to the number of SQP iterations and `ls_steps` stay at zero, line search is not spending time on repeated backtracking trials
- in that regime, a large `run_globalization` cost usually means accepted-point bookkeeping and relinearization, not the filter predicates themselves

## Nullspace / Riccati Solve Data Flow

### 1. Projected dynamics update

`ns_factorization(...)` starts by calling:

- `update_projected_dynamics()`
- `activate_lag_jac_corr()`

This makes sure the stage QP sees all pending gradient corrections from IPM, PMM, or refinement.

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

## KKT Information And Residual Accounting

The old monolithic `compute_kkt_info(...)` path has been split. The current main entry points are:

- `update_primal_info(...)`
- `update_step_info(...)`
- `update_stat_info(...)`

Current division of responsibility:

- `update_primal_info(...)`
  - `cost`
  - `augmented_objective`
  - `barrier_value`
  - `ls_objective`
  - `inf_res`
  - `res_l1`
  - `inf_comp`
- `update_step_info(...)`
  - line-search directional data such as:
  - `augmented_objective_fullstep_dec`
  - `ls_objective_fullstep_dec`
  - primal/dual step norms
- `update_stat_info(...)`
  - `dual.inf_res`
  - `dual.lambda_l1`
  - `dual.n_constr`
  - max dual norms
  - restoration-local dual/complementarity residual aggregation when restoration is active

Important dual-residual detail:

- `dual.inf_res` is not a raw stationarity norm
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

- stored filter points contain primal residual, dual residual, and line-search objective
- barrier value is recomputed with current `mu`
- `ls_objective_fullstep_dec = augmented_objective_fullstep_dec - search_barrier_dir_deriv`
- `fullstep_dec < 0` must be checked before `pow(...)` to avoid NaNs
- backtracking can be linear or geometric
- failure fallback is either minimum step or best trial

Current practical interpretation:

- the scalar filter / Armijo logic in [`src/solver/sqp_impl/line_search.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/line_search.cpp) is usually not the hot part
- when globalization looks expensive in current quadruped profiles, the cost typically comes from:
  - restoring trial state
  - applying the affine step
  - re-evaluating the accepted point's derivatives for the next SQP iteration

The old empty SOC scaffold has been removed. Do not add solver settings,
actions, or dispatch branches unless they execute real algorithmic work.

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

### Internal slack storage

Current solver direction:

- `__s` exists as solver-managed shared slack storage
- it is not exposed as a user modeling field and does not participate in code generation like `__x/__u/__y`
- it is not part of the global Riccati primal state

## Predictor-Corrector And Iterative Refinement

### IPM predictor-corrector

If true IPM inequalities are present:

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

## Restoration Note

- the legacy restoration implementation was removed, but restoration is still active through the overlay-based path in [`src/solver/sqp_impl/restoration.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/restoration.cpp)
- restoration overlay problems are built by [`src/solver/restoration/resto_overlay.cpp`](/home/harper/Documents/moto/src/solver/restoration/resto_overlay.cpp)
- the active graph state caches the realized restoration runtime and invalidates it when the modeled graph or restoration settings change
- [`docs/restoration.md`](/home/harper/Documents/moto/docs/restoration.md) is still useful as historical design context, but it is not a complete description of the current overlay implementation

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

Current profiling-related Python entry points on `moto.sqp`:

- `reset_profile()`
- `get_profile_report()`

## Practical Editing Rules

- follow nearby style instead of introducing a new one
- prefer understanding aliasing before editing matrices or vectors in place
- be careful with any change touching `Q_y`, `f_y`, or stage permutations
- remember many sparse/dense objects are views into shared storage, not owned copies
- do not copy `settings_t`
- do not assume soft constraints are only inequalities; PMM soft equalities use the same dispatch layer
- do not keep dormant solver hooks as public API; either implement the algorithmic work or remove the misleading surface
- when changing C++ code that affects Python examples, always wait for the full build to finish before running Python tests
- do not trust a Python test run started while `moto` / `moto_pywrap` is still linking; stale modules can easily give misleading results
- do not compare first-run wall time across commits without separating solver time from codegen / shared-library compilation time
- remember that same-name codegen is intentionally serialized today; apparent loss of "multithreaded codegen" may be reuse and race avoidance rather than a regression in the solver itself

## Common Pitfalls

- confusing `cost_jac_` with `lag_jac_`
- forgetting that `update_approximation()` snapshots `cost_jac_` before adding `J^T lambda`
- modifying `__dyn` scaling and breaking projected-dynamics solves
- assuming `__eq_x` / `__ineq_x` differentiate against `__x`
- touching line-search or IPM state without handling backup/restore
- assuming `eval_val` and `eval_derivatives` have the same bookkeeping side effects
- blaming [`src/solver/sqp_impl/line_search.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/line_search.cpp) predicates for globalization cost before checking `update_approx_accepted`
- changing a settings struct without updating bindings
- treating [`example/quadruped/run.py`](/home/harper/Documents/moto/example/quadruped/run.py) as canonical; it is often used for experiments
- seeing `warning: substitution in generic_constr ... go2_q_nxt` means some path has reintroduced generic expression substitution instead of graph-level lowering

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

As of the current working tree, the OCP graph modeling API has been refactored to:

- `ocp_base` as the shared storage / activation / flattening container
- `ocp` as the generic internal problem type
- `stage_ocp` as the public modeling container
- `node_view` as the lightweight endpoint handle exposed through `stage.st`,
  `stage.ed`, and `sqp.start_node`

What is already true:

- Python exposes `moto.stage_ocp.create()`, `stage.add(...)`, `stage.st.add(...)`,
  `stage.ed.add(...)`, `sqp.start_node`, `sqp.add_stage(...)`,
  `sqp.add_stages(...)`, and `sqp.flatten_nodes()`
- Python no longer exposes `moto.node_ocp`, `moto.edge_ocp`, `sqp.graph`,
  `sqp.graph.add_path(...)`, `sqp.graph.flatten_nodes(...)`, or
  `add_terminal(...)`
- clone logic is shared through `ocp_base::refresh_after_clone(...)`
- graph-owned stages are cloned on append, mutable, and invalidate solver caches
- endpoint terms lower through cached `x -> y` remaps, so repeated stages reuse
  function entities instead of generating per-stage copies
- current regression checks:
  - `cmake --build build -j8`
  - `ctest --test-dir build --output-on-failure`
  - clean `gen/`, `MOTO_DISPLAY=0 MOTO_SQP_MAX_ITER=50 python example/quadruped/run.py`
    converges at iteration `22` with `28` generated shared libraries
  - clean `gen/`, `python example/arm/run.py --max-iter 100 --n-job 4`
    returns the current restoration result at iteration `87` with `14`
    generated shared libraries

Recommended next step from here:

1. Keep adding focused compose tests only when new topology behavior is added;
   the current linear append API is already covered by regression tests.
2. Add debug tooling or golden tests for node/stage initialization so future path-authoring changes can be validated quickly
3. If incoming-edge authoring is ever needed again, expose it as an explicit modeling helper rather than relying on implicit compose behavior

This order helps keep future graph-semantics changes explicit and testable instead of rediscovering them through quadruped regressions.

## Todo

- add rank-based backward degeneration test and propagation
- add automatic inertia correction after ill-conditioning is detected
