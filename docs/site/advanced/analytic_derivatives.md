# Analytic derivatives

Moto normally differentiates CasADi SX expressions automatically. When an
external library already provides a cheaper derivative, such as Pinocchio's
ABA or RNEA derivatives, attach that symbolic block before finalizing the
stage:

```python
import casadi as cs
import moto
import numpy as np

x_d, xn_d = moto.sym.states("analytic_x", 2)
u_d = moto.sym.inputs("analytic_u", 1)

g_expr = cs.vertcat(
    x_d.sx[0] ** 2 + u_d.sx[0],
    cs.sin(x_d.sx[1]),
)
g = moto.constr.create("analytic_g", g_expr)

J_x = cs.vertcat(
    cs.horzcat(2 * x_d.sx[0], 0),
    cs.horzcat(0, cs.cos(x_d.sx[1])),
)
J_u = cs.SX([1, 0])
g.set_analytic_jacobian(x_d, J_x)
g.set_analytic_jacobian(u_d, J_u)

dx = cs.SX.sym("dx", x_d.tdim)
g_perturbed = cs.substitute(
    g_expr,
    x_d.sx,
    x_d.symbolic_integrate(x_d.sx, dx),
)
J_x_ad = cs.substitute(
    cs.jacobian(g_perturbed, dx),
    dx,
    cs.SX.zeros(x_d.tdim, 1),
)
jacobian_error = cs.Function(
    "analytic_g_error", [x_d.sx, u_d.sx], [J_x - J_x_ad]
)
np.testing.assert_allclose(
    np.asarray(jacobian_error([0.4, -0.2], [0.3])),
    0.0,
    atol=1e-12,
)

h_expr = x_d.sx[0] * u_d.sx[0]
h = moto.constr.create(
    "analytic_h",
    h_expr,
    order=moto.approx_order.approx_order_second,
)
h.set_analytic_jacobian(x_d, cs.horzcat(u_d.sx[0], cs.SX(0)))
h.set_analytic_jacobian(u_d, cs.reshape(x_d.sx[0], 1, 1))
h.set_analytic_hessian(x_d, u_d, cs.SX([1, 0]))

derivative_stage = moto.stage()
derivative_stage.add([g, h])
derivative_stage.wait_until_ready()
```

For output dimension `m`, an analytic Jacobian for argument `a` has shape
`(m, a.tdim)`. Its columns are tangent directions, so a manifold state uses the
derivative of $f(a\oplus da)$ at $da=0$, not necessarily the ambient
derivative with `a.dim` columns.

For a scalar-output function,
`set_analytic_hessian(a, b, H_ab)` receives a block with shape
`(a.tdim, b.tdim)` representing $d/ db\,(df/da)$. Moto derives the transposed
reverse orientation. Every supplied block must be a `casadi.SX` expression;
unspecified blocks still use automatic differentiation.

For dynamics, provide derivatives of the authored residual with its original
signs. Projection through $F_y$ or a lifted elimination graph occurs later.
