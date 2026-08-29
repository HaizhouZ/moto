# Manifold states and Euler dynamics

## Define a manifold state

Use `casadi_manifold` when a state's stored coordinates and tangent directions
are different, or when addition and subtraction do not define valid state
updates. Supply CasADi SX expressions for

$$
\operatorname{integrate}(q,dq)=q\oplus dq,
\qquad
\operatorname{difference}(q_0,q_1)=q_1\ominus q_0.
$$

This unit-circle example stores two coordinates but has one tangent direction:

```python
import casadi as cs
import moto
import numpy as np

q_base = cs.SX.sym("circle_base", 2)
dq = cs.SX.sym("circle_step")
q_other = cs.SX.sym("circle_other", 2)

integrated = cs.vertcat(
    cs.cos(dq) * q_base[0] - cs.sin(dq) * q_base[1],
    cs.sin(dq) * q_base[0] + cs.cos(dq) * q_base[1],
)
difference = cs.atan2(
    q_base[0] * q_other[1] - q_base[1] * q_other[0],
    q_base[0] * q_other[0] + q_base[1] * q_other[1],
)

q, qn = moto.casadi_manifold.create(
    "circle_q",
    q_base,
    dq,
    integrated,
    q_other,
    difference,
    np.array([1.0, 0.0]),
)
omega = moto.sym.inputs("circle_omega")
q_predicted = q.symbolic_integrate(q.sx, 0.01 * omega.sx)
position_residual = q.symbolic_difference(qn.sx, q_predicted)
dynamics = moto.semi_implicit_euler.create(
    "circle_dynamics",
    position_residual,
    state=moto.semi_implicit_euler.state.pos,
)

manifold_stage = moto.stage()
manifold_stage.add(dynamics)
manifold_stage.wait_until_ready()
assert q.dim == 2 and q.tdim == 1
```

The constructor arguments after `name` are the base coordinate symbol, tangent
step symbol, integrated coordinate expression, second coordinate symbol,
difference expression, and optional default value. Their dimensions are
`dim`, `tdim`, `dim`, `dim`, and `tdim`. The returned `q` and `qn` are paired
current/next state symbols and otherwise use the ordinary `moto.sym` API.

## Structured Euler dynamics

`semi_implicit_euler` is the structured Euler dynamics class. Its state mode
selects the supplied inverse structure:

```python
# Complete position + velocity semi-implicit dynamics (default).
dyn = moto.semi_implicit_euler.create("robot_dyn", pos_vel_residual)

# Complete position-only dynamics for kinematic optimization.
kin_dyn = moto.semi_implicit_euler.create(
    "kinematic_dyn",
    position_residual,
    state=moto.semi_implicit_euler.state.pos,
)
```

The modes are `state.pos` and `state.pos_vel`; `pos_vel` is the default. Both
use CasADi-generated projected Jacobians and the sparse panel backend. Use
`dense_dynamics` as the general fallback when no structured sparse inverse is
available.

## Pinocchio interop

External Python bindings may accept `casadi.SX` but not recognize Moto's
derived `var`. Pass `.sx` when calling those APIs directly:

```python
q = moto.sym.params("q")
func(..., q.sx, ...)
```

For Pinocchio models, prefer `pinocchio_states(model, name, default)` and
`semi_implicit_dynamics(...)` from
[`example/helpers.py`](https://github.com/HaizhouZ/moto/blob/main/example/helpers.py).
They pass Pinocchio's complete configuration-space `integrate` and
`difference` expressions to `casadi_manifold`, including floating-base
quaternion joints.
