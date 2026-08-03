# Moto

Moto is a C++20/Python trajectory optimizer. It combines a graph-first
multiple-shooting model, sparse generated derivatives, a nonsmooth SQP method,
and a stagewise nullspace/Riccati QP solver. Dense factorizations use BLASFEO;
static sparse linear operations use precompiled Eigen-based kernels.

# Requirements

1. Eigen 3.4+
2. CasADi 3.7+
3. BLASFEO
4. OpenMP
5. libfmt
6. magic_enum
7. A C++20 compiler

## Compilation Notes

```bash
conda create -n moto python=3.11 casadi eigen magic_enum fmt re2 nanobind \
  nlohmann_json pinocchio example-robot-data \
  example-robot-data-loaders mujoco libblasfeo -c conda-forge
conda activate moto
python -m pip install "viser[urdf]"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib"
export BLASFEO_LIB_DIR=/absolute/path/to/blasfeo/lib

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DWITH_NATIVE_OPT=ON
cmake --build build -j6
cmake --install build
ctest --test-dir build --output-on-failure -j6
```

For representative performance, use a Release build with
`WITH_NATIVE_OPT=ON`. Compiler architecture flags should be consistent across
Moto, CasADi, Pinocchio, and BLASFEO. GCC 13.2+ is recommended for Zen 4
AVX-512.

## SQP Tutorial

The public model is stage-centric: users author `x_k`, `u_k`, and terminal
`x_N`; the solver lowers this graph to its internal `x/u/y` representation.
This complete double-integrator example builds a 12-stage OCP, initializes it,
solves it, and reads the result:

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
stage.add(dyn)
stage.add(running)
stage.add(u_box)

sqp = moto.sqp(n_job=4)
stages = sqp.add_stage(stage, 12)
stages[-1].ed.add(terminal)

# Accessing nodes realizes the graph. Finish editing stages before this point.
nodes = sqp.nodes
for node in nodes:
    node.value[x] = x0
    if node.prob.dim(moto.field.field___y):
        node.value[xn] = x0

sqp.settings.prim_tol = 1e-8
sqp.settings.dual_tol = 1e-8
sqp.settings.comp_tol = 1e-8

result = sqp.update(50, verbose=True)
assert result.solved, result.result

x_trajectory = [np.asarray(node.value[x]).reshape(-1) for node in nodes]
x_trajectory.append(np.asarray(nodes[-1].value[xn]).reshape(-1))
u_trajectory = [np.asarray(node.value[u]).reshape(-1) for node in nodes]
print(result.num_iter, result.inf_prim_res, result.inf_dual_res)
```

Use `sqp.start_node.add(...)` for initial-state terms, `stage.st.add(...)` for
phase-start state terms, and `stage.ed.add(...)` for phase-end or terminal
state terms. Interval dynamics, controls, and mixed terms belong on
`stage.add(...)`.

Costs support explicit scalar- and vector-valued construction through
`cost.from_scalar(...)` and `cost.from_vector(...)`. Inequality boxes use
`ineq.bounds(...)`, including bounds on only a selected symbol or subvector.

### SQP workers: which setting wins?

Pass the desired solver worker count to the constructor. It is fixed for that
solver instance:

```python
sqp = moto.sqp(n_job=6)
print(sqp.n_job)  # effective constructor-time cap, after OpenMP normalization
```

There are three limits. For a parallel loop with `work_items` stages, Moto
requests:

```text
worker_count = min(n_job, OpenMP maximum threads, work_items)
```

In other words, neither setting wins alone:

- `n_job` is the per-SQP worker ceiling and is the setting application code
  should use.
- `OMP_NUM_THREADS` controls the OpenMP process ceiling observed when the SQP
  is constructed. If it is lower than `n_job`, Moto clamps `sqp.n_job` to it.
- A loop cannot use more workers than it has stages/items. The OpenMP runtime
  can reduce the team further when dynamic teams or `OMP_THREAD_LIMIT` apply.

Set the environment before starting Python, and disable dynamic team sizing
when a reproducible worker count matters:

```bash
OMP_NUM_THREADS=6 \
OMP_DYNAMIC=FALSE \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python example/arm/run.py --n-job 6
```

Examples of the resulting SQP cap:

| `OMP_NUM_THREADS` | `moto.sqp(n_job=...)` | `sqp.n_job` |
|---:|---:|---:|
| 8 | 6 | 6 |
| 4 | 6 | 4 |
| 8 | 1 | 1 |

`ns_sqp` explicitly sets Eigen's internal thread count to one, so Eigen does
not create a second thread team inside Moto's stage-parallel regions.
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` likewise prevent nested threading
in those libraries when they are present. These variables do not choose the
SQP worker count.

Two similarly named controls are independent of solver execution:

- `cmake --build build -j6` selects build-system compilation concurrency.
- generated-function compilation currently uses the OpenMP maximum, not the
  particular SQP object's `n_job`; `n_job` controls runtime OCP traversal.

For small horizons, `n_job=1` is often fastest because worker dispatch costs
more than the available stage work. For arm and quadruped problems, benchmark
`1`, the number of physical cores, and a few intermediate values using the
same Release build and warm-start state. Do not compare first-run codegen or
Viser URDF loading with hot `sqp.update(...)` time.

## Euler Dynamics

`semi_implicit_euler` is the single structured Euler dynamics class. Its state
mode controls the supplied inverse structure:

```python
# Complete position + velocity semi-implicit dynamics (default).
dyn = moto.semi_implicit_euler.create("robot_dyn", pos_vel_residual)

# Complete position-only Euler dynamics for kinematic optimization.
kin_dyn = moto.semi_implicit_euler.create(
    "kinematic_dyn",
    position_residual,
    state=moto.semi_implicit_euler.state.pos,
)
```

The two modes are `state.pos` and `state.pos_vel`; the latter is the default.
Both use CasADi-generated projected Jacobians and the same sparse panel backend.
Use `dense_dynamics` only as the general fallback when no structured sparse
inverse is supplied.

## Run

Run the maintained examples:

```bash
python example/toy/run.py
python example/toy/initial_state_optimization.py
python example/toy/restoration.py
python example/arm/run.py
python example/quadruped/run.py
python example/quadruped/run.py --acceleration-control  # acceleration-only gait
python example/quadruped/mpc.py
```

`--display` starts a Viser server and replays the optimized URDF trajectory in
the browser. MeshCat is not used.

## Pinocchio Interop

Some external Python bindings accept `casadi.SX` but do not recognize Moto's
derived `var` type. Pass the `.sx` view when calling those APIs directly:

```python
q = moto.sym.params("q")
func(..., q.sx, ...)
```

For the standard Pinocchio workflow, prefer the state and dynamics helpers in
`example/helpers.py`; they provide manifold-aware state creation, velocity
generation, integration, and difference without exposing those calls at each
modeling site.

## Citation

If you use Moto, please cite the Hippo paper published in IEEE Robotics and
Automation Letters ([IEEE DOI: 10.1109/LRA.2026.3682524](https://doi.org/10.1109/LRA.2026.3682524)):

```bibtex
@article{zhao2026hippo,
  author  = {Haizhou Zhao and Ludovic Righetti and Majid Khadiv},
  title   = {Hippo: High-Performance Interior-Point and Projection-Based Solver
             for Generic Constrained Trajectory Optimization},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  volume  = {11},
  number  = {6},
  pages   = {6752--6759},
  doi     = {10.1109/LRA.2026.3682524}
}
```
