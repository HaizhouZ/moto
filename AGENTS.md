# moto Agent Guide

## Scope

This is the maintainer map for the current `moto` tree. Keep it focused on
stable architecture and invariants. Do not add one-off benchmark numbers,
temporary refactor plans, or descriptions of deleted APIs; those belong in
issues, commits, or profiling notes.

`moto` is a C++20 trajectory optimizer with:

- graph-first symbolic OCP modeling
- CasADi derivative generation and sparsity detection
- precompiled Eigen-based sparse panel kernels
- a nonsmooth SQP outer solver
- a nullspace/Riccati stagewise QP solve
- IPM inequalities, PMM soft equalities, and restoration overlays
- nanobind Python bindings

## Non-Negotiable Invariants

- The public model is stage-centric: users author `x_k`, `u_k`, and terminal
  `x_N`; internal solver storage may use `x/u/y`.
- Endpoint lowering is graph policy. Never silently substitute `x -> y` in a
  standalone expression or standalone problem finalizer.
- `expr_handle` copies share identity. Copying a handle is not cloning.
- Symbol cloning and function remapping are different operations.
- Finalization must be idempotent and must leave every expression ready before
  runtime storage consumes it.
- Generated function names are stable artifact identities. Do not create a new
  code-generated function per repeated stage or per remap.
- Dynamics-local `P F` construction and OCP-wide linear fusion are separate
  layers.
- `dense_dynamics` is the general dense fallback; it must not enter the sparse
  Euler projection-generation path.
- Solver matrices are often views into shared storage. Confirm ownership and
  aliasing before writing in place.
- Keep Eigen internal threading at one inside SQP parallel regions.
- Use at most six build jobs in this workspace.

## Repository Map

Modeling and expressions:

- [`include/moto/core/fields.hpp`](/home/harper/Documents/moto/include/moto/core/fields.hpp)
- [`include/moto/core/expr.hpp`](/home/harper/Documents/moto/include/moto/core/expr.hpp)
- [`include/moto/ocp/sym.hpp`](/home/harper/Documents/moto/include/moto/ocp/sym.hpp)
- [`include/moto/ocp/problem.hpp`](/home/harper/Documents/moto/include/moto/ocp/problem.hpp)
- [`include/moto/ocp/graph_model.hpp`](/home/harper/Documents/moto/include/moto/ocp/graph_model.hpp)
- [`include/moto/ocp/graph_composer.hpp`](/home/harper/Documents/moto/include/moto/ocp/graph_composer.hpp)
- [`include/moto/ocp/impl/func.hpp`](/home/harper/Documents/moto/include/moto/ocp/impl/func.hpp)
- [`include/moto/ocp/impl/func_data.hpp`](/home/harper/Documents/moto/include/moto/ocp/impl/func_data.hpp)

Dynamics and linear backend:

- [`include/moto/ocp/dynamics.hpp`](/home/harper/Documents/moto/include/moto/ocp/dynamics.hpp)
- [`include/moto/ocp/lifted.hpp`](/home/harper/Documents/moto/include/moto/ocp/lifted.hpp)
- [`src/ocp/lifted.cpp`](/home/harper/Documents/moto/src/ocp/lifted.cpp)
- [`include/moto/ocp/dynamics/semi_implicit_euler.hpp`](/home/harper/Documents/moto/include/moto/ocp/dynamics/semi_implicit_euler.hpp)
- [`include/moto/ocp/dynamics/dense_dynamics.hpp`](/home/harper/Documents/moto/include/moto/ocp/dynamics/dense_dynamics.hpp)
- [`include/moto/core/sparse_matrix.hpp`](/home/harper/Documents/moto/include/moto/core/sparse_matrix.hpp)
- [`include/moto/core/linear_backend.hpp`](/home/harper/Documents/moto/include/moto/core/linear_backend.hpp)
- [`src/core/linear_backend.cpp`](/home/harper/Documents/moto/src/core/linear_backend.cpp)

Runtime approximation storage:

- [`include/moto/ocp/impl/node_data.hpp`](/home/harper/Documents/moto/include/moto/ocp/impl/node_data.hpp)
- [`include/moto/ocp/impl/lag_data.hpp`](/home/harper/Documents/moto/include/moto/ocp/impl/lag_data.hpp)
- [`src/ocp/node_data.cpp`](/home/harper/Documents/moto/src/ocp/node_data.cpp)

Solver:

- [`include/moto/solver/ns_sqp.hpp`](/home/harper/Documents/moto/include/moto/solver/ns_sqp.hpp)
- [`include/moto/solver/data_base.hpp`](/home/harper/Documents/moto/include/moto/solver/data_base.hpp)
- [`include/moto/solver/linear_runtime_graph.hpp`](/home/harper/Documents/moto/include/moto/solver/linear_runtime_graph.hpp)
- [`include/moto/solver/ns_riccati/ns_riccati_data.hpp`](/home/harper/Documents/moto/include/moto/solver/ns_riccati/ns_riccati_data.hpp)
- [`src/solver/sqp_impl/`](/home/harper/Documents/moto/src/solver/sqp_impl)
- [`src/solver/nsp_impl/`](/home/harper/Documents/moto/src/solver/nsp_impl)
- [`include/moto/solver/ipm/`](/home/harper/Documents/moto/include/moto/solver/ipm)
- [`include/moto/solver/soft_constr/`](/home/harper/Documents/moto/include/moto/solver/soft_constr)
- [`src/solver/restoration/`](/home/harper/Documents/moto/src/solver/restoration)

Bindings, examples, and tests:

- [`bindings/`](/home/harper/Documents/moto/bindings)
- [`example/helpers.py`](/home/harper/Documents/moto/example/helpers.py)
- [`example/toy/`](/home/harper/Documents/moto/example/toy)
- [`example/arm/`](/home/harper/Documents/moto/example/arm)
- [`example/quadruped/`](/home/harper/Documents/moto/example/quadruped)
- [`unittests/`](/home/harper/Documents/moto/unittests)

## Build And Validation

Use a Release/native build for performance work and no more than six jobs:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DWITH_NATIVE_OPT=ON
cmake --build build -j6
ctest --test-dir build --output-on-failure -j6
```

Useful focused checks:

```bash
MOTO_SYNC_CODEGEN=1 ./build/unittests/semi_implicit_euler_test
./build/unittests/linear_backend_test
./build/unittests/graph_model_compose_test
python example/toy/initial_state_optimization.py
python example/toy/restoration.py
python example/quadruped/quaternion_test.py
python example/quadruped/run.py --no-display --acceleration-control \
  --horizon 4 --steps 1 --nodes-per-step 2 --max-iter 1
```

Quadruped smoke test with controlled threading:

```bash
KMP_AFFINITY='noverbose,granularity=fine,scatter' \
OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python example/quadruped/run.py --no-display --horizon 4 --steps 1 \
  --nodes-per-step 2 --max-iter 1 --configuration-velocity next
```

Build and Python validation rules:

- Wait for `moto_pywrap` to finish linking before running Python.
- A first run may include CasADi and linear-backend compilation. Do not compare
  it directly with hot solver time.
- Use `MOTO_PROFILE_SQP=1` and `get_profile_report()` for solver timing.
- Do not increase build parallelism to compensate for slow code generation.
- Do not delete unrelated contents of `gen/`; generated artifacts may belong
  to another active test or user run.

## Field System

Primary primal fields:

- `__x`: current state
- `__u`: interval input
- `__y`: predicted next-state copy used by solver algebra
- `__l`: explicit user-authored lifted primal variables

Other symbol storage:

- `__p`: parameters, not decision variables
- `__s`: solver-managed private slack/storage
- `__usr_var`: user-defined nonstandard storage

Main function fields:

- `__dyn`: dynamics residual
- `__lift`: grouped lifted-variable equality residual
- `__cost`: costs
- `__eq_x`, `__eq_xu`: hard equalities
- `__ineq_x`, `__ineq_xu`: inequalities
- `__eq_x_soft`, `__eq_xu_soft`: soft equalities

Do not treat `__s` as a public primal block. `x/u` are unlifted, while `y/l`
are eliminated relative to them in the local QP. All four remain explicit
nonlinear primal fields.

## Public Modeling Surface

Python modeling starts from `moto.sqp` and `moto.stage()`:

```python
sqp = moto.sqp(n_job=6)
stage = moto.stage()
stage.add(dynamics)
stage.add(interval_cost)
stages = sqp.add_stage(stage, horizon)
stages[-1].ed.add(terminal_cost)
nodes = sqp.nodes
```

Placement rules:

- `stage.add(...)`: interval dynamics, input terms, mixed terms, and ordinary
  path-state terms evaluated on the interval's current `x`
- `stage.st.add(...)`: state-only term on the phase start boundary
- `stage.ed.add(...)`: state-only term on the phase end boundary
- `sqp.start_node.add(...)`: state-only term on the graph's initial state

Endpoint views reject terms involving `u`, authored `y`, or dynamics. Users
write endpoint expressions on `x`; graph composition performs the necessary
solver-storage lowering.

`sqp.add_stage(stage, N)` appends `N` graph-owned stage copies from the current
tail. `sqp.add_stages(node, stage, N)` appends from an explicit graph boundary.
Returned stages are mutable graph-owned copies; editing them invalidates cached
realization. Editing the original prototype later does not mutate those copies.

`sqp.nodes` realizes and returns the ordered solver-stage list for initialization
and debugging. It is not the modeling API and should not absorb graph semantic
policy.

## Graph Composition

`graph_model` owns topology and mutation revision tracking.
`graph_composer` turns a topology snapshot into internal `ocp` intervals.
`ns_sqp` consumes the composed result and reconciles runtime storage.

Lowering rules:

- interval terms remain on their authored interval
- graph-start terms materialize on the first solver `x`
- a connected stage-start boundary lowers onto the predecessor interval's `y`
- a stage-end boundary lowers onto that interval's `y`
- `prev.ed` and `next.st` describe the same connected graph boundary
- inactive symbols are pruned when composed stages are copied

Endpoint lowering uses cached function remaps. It must not mutate the authored
function and must not generate a unique compiled artifact for every stage.
Enable `MOTO_TRACE_COMPOSE=1` only when debugging lowering/remap decisions.

## Expression Identity, Clone, Remap, And Reuse

Keep these semantics distinct:

- handle/share: `expr::handle()` and copied `expr_handle` values refer to the
  exact same UID and object
- symbol clone: `sym.clone(name)` creates a new logical symbol with a fresh UID;
  cloning an `x` state also creates its paired `y`
- stage copy: `stage.copy(...)` creates an independent container while sharing
  immutable expression handles until graph lowering needs a remapped function
- function remap: changes the symbols used to address an already finalized
  function implementation; it is not symbolic re-derivation
- remap reuse: `reuse_remap(...)` caches by normalized source/target UID pairs
  and returns the same remapped handle for an equivalent mapping

Function cloning is an implementation detail. Do not expose a generic public
`func.clone()` API. If a user needs a distinct symbolic expression, build it
from cloned symbols; if only argument identity changes, remap the function.

## Symbols And Automatic Argument Inference

Every symbol created through `moto.sym` is registered by UID in
`global_registry`. CasADi-backed functions infer arguments by querying the
symbolic primitives in the output and resolving them through this registry.

Consequences:

- `constr.create(name, expression)`, `cost.from_scalar(...)`,
  `cost.from_vector(...)`, `ineq.bounds(...)`, and dynamics constructors do not
  require a duplicated argument list for ordinary use
- explicitly supplied symbols are registered before inference
- inferred arguments are UID-based, not name-based
- unused arguments are removed during function finalization unless explicitly
  exempted
- constants must remain constants; do not turn every numeric weight/reference
  into a symbol

Parameters used in values, weights, references, lower bounds, or upper bounds
may be numeric scalars/vectors or explicitly supplied symbols. Partial bounds
are represented by the bounded expression/symbol passed to `ineq.bounds`, not
by forcing a full-state bound vector.

## Finalization And Code Generation

Expression finalization establishes dimensions, dependencies, derivative
sparsity, and generated callbacks. Problem finalization then:

1. maintains dynamics-compatible primal ordering when enabled
2. rebuilds flattened field layouts
3. marks the problem finalized
4. builds its `ocp_linear_profile` from finalized function sparsity

`wait_until_ready()` waits for every expression's generated implementation and
then finalizes the problem layout. Runtime objects must not be created from a
partially ready problem.

Generated names are stable function names. Same-name codegen is serialized in
[`src/utils/codegen.cpp`](/home/harper/Documents/moto/src/utils/codegen.cpp),
and compiled outputs are published atomically. Repeated stages and cached
remaps intentionally reuse those artifacts.

Do not put graph topology decisions into `generic_func::finalize_impl()` or
`ocp_base::finalize()`; neither has enough context to place endpoint terms.

## Lifted Groups, Structured Euler, And Dense Dynamics

`generic_dynamics` is the common dynamics/lifting group. Its active `__y`
arguments and any explicitly marked `__l` arguments remain authored nonlinear
primals, while their local QP directions are eliminated relative to `x/u`.
Remap/substitution must preserve that lifted identity.

A stage with explicit `__l` owns grouped `__lift` subconstraints and must
provide one MX elimination graph for the coupled rows `[__dyn; __lift]` and
columns `[__y; __l]`. The graph supplies projected `x/u/residual` responses and
a forward linear action; the transpose action is derived from the same graph.
CasADi supplies symbolic DAG and sparsity metadata only. The linear backend
owns panel storage, factor reuse, lowering, and runtime execution. Do not add a
second dense assembled-pivot fallback.

The nullspace solver consumes those projected responses, contracts all
`x/u/y/l` gradient and Hessian contributions, and recovers explicit `l` steps
and both multiplier blocks after rollout. Lifting never performs symbolic
nonlinear substitution or removes the authored primal/dual variables.

All dynamics implement the `generic_dynamics` projection interface consumed by
the solver:

- `F_x = F_y^{-1} f_x`
- `F_u = F_y^{-1} f_u`
- `F_0 = F_y^{-1} f`
- application of `F_y^{-T}` to multiplier vectors

`semi_implicit_euler` is the only structured Euler class:

- `state_t::pos_vel` / Python `state.pos_vel` is the default complete
  position-plus-velocity dynamics
- `state_t::pos` / Python `state.pos` is the complete position-only dynamics
  used for kinematic optimization
- the position block uses the supplied configuration structure, including the
  small orientation block
- the position-velocity mode uses the semi-implicit block-triangular inverse

During dynamics finalization, CasADi forms tangent Jacobians, the symbolic
inverse, and the final projected expressions. Their output sparsity is split
into dense, diagonal, and identity panels. Runtime data writes generated
projected Jacobians directly into sparse `proj_f_x_` and `proj_f_u_` storage;
residual and inverse-transpose products use precompiled linear-backend kernels.

Do not hand-code SpGEMM for this path. CasADi owns symbolic product formation
and sparsity discovery; the backend owns runtime panel execution.

`dense_dynamics` is independent. It gathers dense `F_y`, factors it with the
dense BLASFEO LU path, and solves projections at runtime. It must not generate
or cache symbolic `P F` products.

The helper [`example/helpers.py`](/home/harper/Documents/moto/example/helpers.py)
provides Pinocchio-aware state creation and the standard `pos_vel` residual so
examples do not repeat direct manifold `integrate` / `difference` calls.

## OCP Linear Profiles And Backend

After all functions are finalized, `ocp_base::build_linear_profile()` collects
active Jacobian and Hessian panels into stage-global offsets. `node_data`
allocates sparse matrices from that profile and lazily compiles OCP-level
kernels.

Current backend responsibilities include:

- sparse/dense products and transpose products
- sparse/sparse product profiles
- dense writes
- weighted Gram products
- rowwise scaling and norms
- batched constraint-gradient assembly
- batched soft-constraint Jacobian-step products
- batched first- and second-order soft-constraint condensation

The backend emits straight-line C++ specialized to runtime dimensions and
panel layouts, compiles it once, caches it, and invokes it through compact
pointer arrays. Generated kernels use Eigen maps and preserve aligned maps for
panel data that has the aligned-storage guarantee.

Fusion is stage/OCP-level where pointer lifetimes and output order allow it.
Do not fall back to per-constraint handwritten matrix traversal when the same
operation belongs in a batch plan. Do not fuse unrelated dense and diagonal
storage merely to claim a larger kernel; fusion must preserve the detected
sparsity and avoid copies/permutations whose cost exceeds the saved dispatch.

## Runtime Approximation Data

`node_data` owns one runtime stage:

- `prob_`: finalized static `ocp`
- `sym_`: serialized symbol values
- `dense_`: assembled cost, constraints, Jacobians, Hessians, and projections
- `shared_`: UID-keyed custom/precompute state
- `sparse_`: one function-local approximation map per active function
- `linear_plan_`: lazily compiled stage-level batch kernels

`func_arg_map` maps function arguments into serialized symbol storage.
`func_approx_data` maps function-local value/Jacobian/Hessian outputs into the
stage storage planned from `ocp_linear_profile`.

`lag_data` contains, among other fields:

- constraint residuals and sparse Jacobian blocks
- constraint duals and complementarity residuals
- `cost_` and `lag_`
- `cost_jac_`: cost gradient only
- `lag_jac_`: active base Lagrangian gradient
- `lag_jac_corr_`: pending solver correction
- upper-triangular Lagrangian Hessian blocks
- Hessian-modification blocks
- `proj_f_x_`, `proj_f_u_`, and `proj_f_res_`

`node_data::update_approximation()` currently:

1. resets requested value/derivative outputs
2. executes precompute callbacks
3. evaluates active functions
4. condenses soft constraints through the stage batch plan
5. executes postcompute callbacks
6. establishes the base cost/Lagrangian gradient
7. adds constraint value and `J^T lambda` contributions
8. updates primal and complementarity residual summaries when values are active

Value-only and derivative-only updates do not have identical side effects.
Line-search trial evaluation and accepted-point relinearization must use the
appropriate mode.

## Solver Flow

The main loop is in
[`src/solver/sqp_impl/ns_sqp_impl.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp).
At a high level:

1. realize/reconcile runtime stages from the graph model
2. initialize primal/dual/IPM state when required
3. update nonlinear approximations
4. apply scaling
5. form projected dynamics and equality/nullspace data
6. run backward Riccati recursion
7. run forward rollout and recover dual steps
8. perform IPM correction and iterative refinement when enabled
9. globalize with filter or merit backtracking
10. accept/reject, restore state as needed, and relinearize the accepted point

`data_base` aliases solver names such as `Q_x`, `Q_u`, `Q_y`, `Q_xx`, and
`Q_uu` onto `lag_data`; these are not independent matrices. Trial-state backup
and restore are virtual because inequality/soft runtimes must checkpoint their
own state together with the base primal and dual values.

The nullspace/Riccati layer consumes projected dynamics and equality blocks.
Be especially careful with `F_x`, `F_u`, `Q_y`, `Q_zz`, nullspace bases, and
forward/backward traversal order.

## Initial State, Inequalities, And Restoration

`settings.initial_state` has two modes:

- `fixed`: the initial state is not optimized
- `optimized`: the initial state participates in the SQP step

The executable demo is
[`example/toy/initial_state_optimization.py`](/home/harper/Documents/moto/example/toy/initial_state_optimization.py).

Inequalities are converted through the registered IPM implementation. Soft
equalities use the PMM layer and share the soft-constraint dispatch and
condensation machinery. Do not assume every `ineq_soft` object is an inequality.

Restoration is an overlay runtime, not mutation of the authored graph. The
solver caches restoration overlay storage and invalidates it when model or
restoration settings change. Current implementation files are:

- [`src/solver/sqp_impl/restoration.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/restoration.cpp)
- [`src/solver/restoration/resto_overlay.cpp`](/home/harper/Documents/moto/src/solver/restoration/resto_overlay.cpp)
- [`example/toy/restoration.py`](/home/harper/Documents/moto/example/toy/restoration.py)

[`docs/restoration.md`](/home/harper/Documents/moto/docs/restoration.md) explains
the elastic KKT derivation, but code is authoritative for entry, exit, caching,
and cleanup behavior.

## Runtime Traversal And Threading

`linear_runtime_graph` is internal contiguous solver storage. Its `nodes()`
accessor returns the ordered list directly; there is no flattening operation. It provides
forward, backward, adjacent, zipped, sequential, and parallel views. It is not
a public modeling graph.

Parallel callbacks receive a logical worker/chunk ID. Backward passes should
use the reversed view supplied by `solver::backward_edges(...)`; do not reverse
indices again inside callbacks. Keep Eigen, OpenBLAS, and MKL internal thread
counts at one to avoid nested parallelism.

## Python Bindings

Bindings use nanobind. The public package surface is defined by:

- [`bindings/definition/public_api.py`](/home/harper/Documents/moto/bindings/definition/public_api.py)
- [`bindings/package_init.py`](/home/harper/Documents/moto/bindings/package_init.py)
- [`bindings/definition/sqp.py`](/home/harper/Documents/moto/bindings/definition/sqp.py)
- [`bindings/definition/var.py`](/home/harper/Documents/moto/bindings/definition/var.py)

Low-level `node_data`, `lag_data`, and runtime-graph machinery are debugging
details and should not be promoted to top-level modeling APIs.

When changing a public enum, constructor, setting, or return type:

1. update C++ declaration and implementation
2. update the nanobind definition
3. rebuild the extension and generated `.pyi`
4. import through the installed `moto` package, not only the raw extension
5. run at least one example using the changed surface

For external libraries that do not recognize the derived `moto.var`, pass
`var.sx`. Prefer helpers for Pinocchio manifold operations.

Robot example visualization uses `viser` and `viser.extras.ViserUrdf` through
the shared `ViserRobot` helper in `example/helpers.py`. Do not reintroduce
MeshCat or initialize an `example_robot_data` viewer during model construction.

## Editing Checklist

Before editing:

- locate the authoritative owner of the behavior
- inspect aliases and sparse-panel ownership
- check whether the change is modeling-time, finalize-time, dynamics-local, or
  OCP-runtime work
- inspect the worktree and preserve unrelated user changes

Before committing:

- run `git diff --check`
- build Release/native with `-j6`
- run focused tests for the changed subsystem
- run `ctest --output-on-failure -j6` for cross-cutting changes
- run a Python smoke test after the binding has finished linking
- check `git status --short`

Common mistakes:

- moving endpoint terms during standalone function finalization
- confusing handle sharing with cloning
- generating a new artifact for an equivalent function remap
- treating `y` as a user-owned peer state
- using dense dynamics to infer structured Euler projections
- rebuilding `P F` in the OCP-wide backend
- reintroducing per-constraint handwritten sparse traversal
- losing aligned panel guarantees in generated Eigen maps
- copying `settings_t`; it contains reference members
- confusing `cost_jac_`, `lag_jac_`, and `lag_jac_corr_`
- scaling `__dyn` as if it were an ordinary hard equality
- forgetting inequality/soft state during line-search backup and restore
- running Python against a stale or partially linked extension
- reporting cold codegen time as hot solver regression

## Reading Order

For graph/modeling work:

1. `include/moto/core/expr.hpp`
2. `include/moto/ocp/sym.hpp`
3. `include/moto/ocp/impl/func.hpp`
4. `include/moto/ocp/problem.hpp`
5. `include/moto/ocp/graph_model.hpp`
6. `src/ocp/graph_composer.cpp`

For approximation/backend work:

1. `include/moto/core/sparse_matrix.hpp`
2. `include/moto/core/linear_backend.hpp`
3. `src/core/linear_backend.cpp`
4. `include/moto/ocp/impl/func_data.hpp`
5. `src/ocp/node_data.cpp`

For solver work:

1. `include/moto/solver/ns_sqp.hpp`
2. `src/solver/sqp_impl/ns_sqp_impl.cpp`
3. `src/solver/nsp_impl/presolve.cpp`
4. `src/solver/nsp_impl/backward.cpp`
5. `src/solver/nsp_impl/rollout.cpp`
6. `src/solver/sqp_impl/line_search.cpp`
7. IPM/soft/restoration implementation for the relevant mode
