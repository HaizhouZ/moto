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

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DWITH_NATIVE_OPT=ON
cmake --build build -j6
cmake --install build
ctest --test-dir build --output-on-failure -j6
```

### BLASFEO discovery

The conda setup above installs BLASFEO with:

```bash
conda install -c conda-forge libblasfeo
```

No BLASFEO path variable is needed for an activated conda environment. Moto
first consumes BLASFEO's own CMake config target and otherwise searches for
`blasfeo.h` and `libblasfeo` under `$CONDA_PREFIX/include` and
`$CONDA_PREFIX/lib`. A successful configure prints either:

```text
-- Found BLASFEO via config target: blasfeo
```

or the fallback library path:

```text
-- Found BLASFEO: .../lib/libblasfeo.so
```

For a BLASFEO installation outside conda and the standard system prefixes,
pass its installation root—not the library file or library directory:

```bash
cmake -S . -B build \
  -DBLASFEO_ROOT=/opt/blasfeo \
  -DCMAKE_BUILD_TYPE=Release
```

That root must contain `include/blasfeo.h` and either `lib/libblasfeo.so` or
`lib/libblasfeo.a`. The old `BLASFEO_LIB_DIR` environment variable is removed.
Installed Moto packages also ship the same finder, so downstream CMake users
can use `find_package(moto REQUIRED)` without recreating BLASFEO lookup logic.

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
sqp.stages.extend([stage.copy() for _ in range(12)])
sqp.ed.add(terminal)

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

Set both limits before constructing the solver:

```bash
OMP_NUM_THREADS=6 \
OMP_DYNAMIC=FALSE \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python example/arm/run.py --n-job 6
```

The effective count is
`min(n_job, OMP_NUM_THREADS, work_items)`: `n_job` is the per-solver limit and
`OMP_NUM_THREADS` is the process limit. Check the normalized solver limit with
`sqp.n_job`. Eigen is forced to one internal thread; the BLAS variables above
avoid nested threading.

Build concurrency (`cmake --build ... -j6`) and codegen compilation are
separate from SQP workers. For short horizons, benchmark against `n_job=1`.

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
