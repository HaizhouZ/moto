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
  nlohmann_json pinocchio meshcat-python meshcat-shapes example-robot-data \
  example-robot-data-loaders mujoco libblasfeo -c conda-forge
conda activate moto

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

## Modeling

The public model is stage-centric: users author `x_k`, `u_k`, and terminal
`x_N`; the solver lowers this graph to its internal `x/u/y` representation.

```python
import casadi as cs
import moto

x, xn = moto.sym.states("x", 2)
u = moto.sym.inputs("u", 1)

dyn = moto.dense_dynamics.create("dyn", xn.sx - x.sx - cs.vertcat(u.sx, 0))
running = moto.cost.from_vector("running", cs.vertcat(x.sx, u.sx))
terminal = moto.cost.from_vector("terminal", x, weight=10.0)
u_box = moto.ineq.bounds("u_box", u, -1.0, 1.0)

stage = moto.stage()
stage.add(dyn)
stage.add(running)
stage.add(u_box)

sqp = moto.sqp(n_job=1)
stages = sqp.add_stage(stage, 20)
stages[-1].ed.add(terminal)
nodes = sqp.flatten_nodes()
```

Use `sqp.start_node.add(...)` for initial-state terms, `stage.st.add(...)` for
phase-start state terms, and `stage.ed.add(...)` for phase-end or terminal
state terms. Interval dynamics, controls, and mixed terms belong on
`stage.add(...)`.

Costs support explicit scalar- and vector-valued construction through
`cost.from_scalar(...)` and `cost.from_vector(...)`. Inequality boxes use
`ineq.bounds(...)`, including bounds on only a selected symbol or subvector.

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

Set the OpenMP worker count explicitly; keep Eigen/BLAS internal threading at
one when profiling solver parallelism:

```bash
export KMP_AFFINITY='noverbose,granularity=fine,scatter'
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

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
