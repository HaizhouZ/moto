# Build and solve an OCP

Moto models an OCP as an ordered vector of interval stages. Each stage authors
the current state `x`, interval input `u`, and predicted next state `xn`;
`sqp.ed` places state-only terms on the final $x_N$. Graph composition performs
the internal endpoint lowering.

## Complete example

This example drives a bounded double integrator toward the origin over 12
intervals:

```python
import casadi as cs
import moto
import numpy as np

x0 = np.array([1.0, 0.0])
x, xn = moto.sym.states("x", 2, default_val=x0)
u = moto.sym.inputs("u", 1, default_val=0.0)

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.0], [0.1]])

dyn = moto.dense_dynamics.create("dyn", xn.sx - A @ x.sx - B @ u.sx)
running = moto.cost.from_vector(
    "running", cs.vertcat(x.sx, u.sx), weight=[1.0, 1.0, 0.1]
)
terminal = moto.cost.from_vector("terminal", x, weight=10.0)
u_box = moto.ineq.bounds("u_box", u, -0.5, 0.5)

stage = moto.stage()
stage.add([dyn, running, u_box])

sqp = moto.sqp(n_job=1)
sqp.stages.extend([stage.copy() for _ in range(12)])
sqp.ed.add(terminal)

nodes = sqp.nodes

sqp.settings.prim_tol = 1e-8
sqp.settings.dual_tol = 1e-8
sqp.settings.comp_tol = 1e-8

result = sqp.update(50, verbose=True)
assert result.solved, result.result

x_trajectory = [np.asarray(node.value[x]).reshape(-1) for node in nodes]
x_trajectory.append(np.asarray(nodes[-1].value[xn]).reshape(-1))
u_trajectory = [np.asarray(node.value[u]).reshape(-1) for node in nodes]
print("iterations:", result.num_iter)
print("final state:", x_trajectory[-1])
```

Accessing `sqp.nodes` realizes the authored graph. Finish ordinary model edits
before that point. Later structural edits are supported, but the next
`sqp.nodes` access must reconcile runtime storage.

## Defaults, initial guesses, and parameters

`states`, `inputs`, and `params` accept `default_val`. Moto copies that value
into every node where the symbol is active when runtime storage is first
created:

```python
x, xn = moto.sym.states("x", 2, default_val=np.array([1.0, 0.0]))
u = moto.sym.inputs("u", 1, default_val=0.0)
target = moto.sym.params("target", 2, default_val=np.zeros(2))
```

A scalar default is broadcast to the symbol dimension. An array default must
match the symbol's storage dimension. The default can also be changed before
the graph is realized:

```python
x.default_value = np.array([0.5, 0.0])
u.default_value = 0.0
nodes = sqp.nodes
```

Changing `default_value` after `sqp.nodes` has been accessed does not rewrite
existing node storage. Modify realized values through `node.value` instead:

```python
nodes[0].value[x] = np.array([1.2, -0.1])

for node in nodes:
    node.value[target] = np.array([0.0, 0.0])
```

Runtime assignments must match the symbol dimension; scalar assignment is only
valid for a one-dimensional symbol. `node.value[...]` returns writable NumPy
storage, so individual entries can also be changed:

```python
nodes[0].value[x][1] = 0.25
```

The state pair returned by `moto.sym.states` represents the current state `x`
and the interval terminal state `xn`. Their shared default is convenient for a
constant initial guess. For a nonconstant trajectory guess, initialize both
and keep adjacent intervals consistent:

```python
guess = np.linspace([1.0, 0.0], [0.0, 0.0], len(nodes) + 1)
for k, node in enumerate(nodes):
    node.value[x] = guess[k]
    node.value[xn] = guess[k + 1]
    node.value[u] = np.zeros(1)
```

With the default `fixed` initial-state mode, `nodes[0].value[x]` is the fixed
$x_0$. In `optimized` mode it is an initial guess for an optimization
variable.

## Change cost weights and references

`cost.from_vector` builds

$$
\frac{1}{2}(v-r)^T\operatorname{diag}(w)(v-r),
$$

and automatically uses the weighted Gauss--Newton Hessian of that residual.
`cost.from_scalar` instead forms the scalar quadratic tracking expression and
uses ordinary exact second-order differentiation. For a nonlinear scalar
residual, that exact Hessian can be indefinite; use `from_vector` when a
Gauss--Newton residual model is intended.

Numeric `weight` and `reference` arguments are defaults, not baked-in
constants: every cost exposes the resulting parameter symbols as `cost.weight`
and `cost.reference`.

The constructor arguments are the simplest way to set those defaults. They can
also be changed through the exposed symbols before `sqp.nodes` is first
accessed:

```python
running.weight.default_value = np.array([2.0, 0.5, 0.05])
running.reference.default_value = np.zeros(3)
```

For example, the costs from the complete example can be retuned after the
nodes are realized:

```python
for node in nodes:
    node.value[running.weight] = np.array([2.0, 0.5, 0.05])
    node.value[running.reference] = np.array([0.0, 0.0, 0.0])

# sqp.ed is stored on the last interval's terminal boundary.
nodes[-1].value[terminal.weight] = np.array([50.0, 50.0])
nodes[-1].value[terminal.reference] = np.array([0.0, 0.0])
```

This changes only node-local numeric data. It does not rebuild the symbolic
graph or regenerate derivatives.

Use explicit parameter symbols when several expressions should intentionally
share the same setting:

```python
q = moto.sym.params("q", 2, default_val=np.array([1.0, 0.2]))
x_ref = moto.sym.params("x_ref", 2, default_val=np.zeros(2))
r = moto.sym.params("r", 1, default_val=0.05)

state_cost = moto.cost.from_vector(
    "state_cost", x.sx, weight=q, reference=x_ref
)
control_cost = moto.cost.from_scalar(
    "control_cost", u.sx, weight=r, reference=0.0
)
```

Every function using `q` or `x_ref` in one node reads the same node-local
value. That is useful for shared tuning, but it is also why a running target
and a different terminal target should use distinct symbols or the separate
`running.reference` and `terminal.reference` handles.

Weights have the residual's tangent dimension, while references have the
tracked value's storage dimension. These dimensions are usually identical for
Euclidean variables but can differ for manifold-valued states. Use
`cost.from_scalar` for a one-element tracked value and `cost.from_vector` for
two or more elements.

## Use per-node schedules and update between solves

Parameters are node-local even when all stages share the same symbolic
expression. This supports time-varying references and weights without making
copies of the cost:

```python
references = np.linspace([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], len(nodes))

for k, node in enumerate(nodes):
    node.value[running.reference] = references[k]
    node.value[running.weight] = np.array([1.0 + k, 1.0, 0.1])

result = sqp.update(50, verbose=False)
```

For repeated solves, change parameters and call `update` again. The current
primal trajectory remains the next initial guess:

```python
for node in nodes:
    node.value[running.reference] = np.array([0.25, 0.0, 0.0])

result = sqp.update(20, verbose=False)
```

`sqp.settings.ipm.warm_start` controls reuse of the inequality solver's barrier
state. It is separate from retaining the current primal values:

```python
sqp.settings.ipm.warm_start = True
```

Set it only when consecutive problems are close enough for their inequality
multipliers and slacks to be useful.

The following operations are numeric updates and do not trigger code
generation:

- changing `node.value[symbol]`
- changing `cost.weight` or `cost.reference` through `node.value`
- changing an initial trajectory guess
- changing solver tolerances or iteration limits

Changing expression formulas, dimensions, stage membership, or active
expressions is a structural edit. The next graph realization may need to
finalize new expressions and generate new kernels.

## Place terms on stages and boundaries

| API | Meaning |
| --- | --- |
| `stage.add(term)` | Dynamics, input terms, mixed terms, and ordinary path-state terms evaluated on every occurrence of that interval. |
| `sqp.st.add(term)` | A state-only term on the graph's initial $x_0$. |
| `stage.st.add(term)` | A state-only term on that stage occurrence's connected incoming boundary. It is not a substitute for `sqp.st`. |
| `stage.ed.add(term)` | A state-only term on that stage occurrence's outgoing boundary. |
| `sqp.ed.add(term)` | A state-only term on the stable terminal $x_N$, including after horizon edits. |

Endpoint terms must be state-only. Write them on `x`; graph composition maps
them to the appropriate solver storage. A running and terminal cost with the
same formula must be created as two separately named expressions.

## Phase variants and graph editing

Phase variants reuse one prototype while changing active expressions:

```python
bounded = stage.copy()
unbounded = stage.copy(disable=[u_box])

phase_sqp = moto.sqp(n_job=1)
phase_sqp.stages.extend([bounded.copy() for _ in range(10)])
phase_sqp.stages.extend([unbounded.copy() for _ in range(5)])
phase_sqp.ed.add(terminal)

phase_sqp.stages[3].disable(u_box)
phase_sqp.stages[3].enable(u_box)
```

`sqp.stages` is the mutable ordered stage container. Replacement and
MPC-style shifts use ordinary container operations, while every inserted
occurrence must be an independent stage object:

```python
phase_sqp.stages[4] = unbounded.copy()
del phase_sqp.stages[0]
phase_sqp.stages.append(bounded.copy())
nodes = phase_sqp.nodes
```

The default initial-state mode is `fixed`. To optimize $x_0$, select
`moto.sqp.initial_state_mode.optimized` and place its state-only objective or
constraint on `sqp.st`; see the
[initial-state example](https://github.com/HaizhouZ/moto/blob/main/example/toy/initial_state_optimization.py).

Costs support scalar and vector construction through `cost.from_scalar(...)`
and `cost.from_vector(...)`. Inequality boxes use `ineq.bounds(...)`, including
bounds on a selected symbol or subvector.

## Hard-equality projection backend

Eigen `FullPivLU` remains the default. To use the panel-major complete-pivot LU
backend for hard-equality projection:

```python
sqp.settings.equality_projection = moto.sqp.equality_projection_backend.panel_lu
# Restore the default:
sqp.settings.equality_projection = moto.sqp.equality_projection_backend.eigen
```

The selected backend factors each stage's hard-equality geometry once and reuses
it for particular solutions, the input nullspace and multiplier recovery,
including correction solves. Changes apply at the next factorization; no model
rebuild or derivative generation is needed. This option does not change the
dynamics solver, Cholesky, globalization or regularization. The panel backend
uses BLASFEO's configured routines, not a forced hardware target. It is rank
revealing but does not promise an orthonormal nullspace, minimum-norm solutions
or bitwise equality with Eigen near numerical rank boundaries. Benchmark your
own stage dimensions; backend speed is not an end-to-end SQP speedup guarantee.
Small matrices can still factor more slowly than Eigen even when the complete
projection workflow is faster because of the reused triangular solves.
Conda BLASFEO is sufficient: pivot search uses a no-copy Eigen reduction and
does not depend on a newer BLASFEO vector norm. Matrix updates and triangular
solves still use the public BLASFEO interfaces. When several installations
exist, point CMake's `blasfeo_DIR` at the intended exported package.

## Worker limits

Set both process and solver limits before constructing the solver:

```bash
OMP_NUM_THREADS=6 \
OMP_DYNAMIC=FALSE \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python example/arm/run.py --n-job 6
```

The effective count is `min(n_job, OMP_NUM_THREADS, work_items)`. Check the
normalized solver limit with `sqp.n_job`. Eigen is forced to one internal
thread, and the BLAS variables avoid nested threading. Build concurrency and
runtime worker counts are independent; for short horizons, benchmark against
`n_job=1`.
