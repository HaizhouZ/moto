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

x, xn = moto.sym.states("x", 2)
u = moto.sym.inputs("u", 1)

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.0], [0.1]])
x0 = np.array([1.0, 0.0])

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
for node in nodes:
    node.value[x] = x0
    node.value[xn] = x0
    node.value[u] = 0.0

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
