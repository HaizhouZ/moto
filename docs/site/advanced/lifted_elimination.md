# Custom lifted elimination graphs

A lifted group keeps `y`, explicitly lifted variables `l`, and their
multipliers in the nonlinear model while eliminating their local QP directions
relative to `x/u`. For grouped residuals $h=[dyn; lift]$, the builder supplies
an inverse action with rows and columns ordered as `[y; l]`.

The following group has $y-x-l=0$ and $l-u=0$. Its lifted Jacobian is block
triangular, so the inverse action can be expressed as two substitutions:

```python
import casadi as cs
import moto

x_l, y_l = moto.sym.states("elimination_x", 1)
u_l = moto.sym.inputs("elimination_u", 1)
l = moto.sym.lifted("elimination_l", 1)

dynamics = moto.semi_implicit_euler.create(
    "elimination_dynamics",
    y_l.sx - x_l.sx - l.sx,
    moto.semi_implicit_euler.state.pos,
)
coupling = moto.lifted.create(
    "elimination_coupling",
    l.sx - u_l.sx,
    [l],
)

def elimination(system):
    dyn_y = system.jac(dynamics, y_l).mx
    dyn_l = system.jac(dynamics, l).mx
    lift_y = system.jac(coupling, y_l).mx
    lift_l = system.jac(coupling, l).mx
    assert lift_y.nnz() == 0

    def solve(rhs):
        rhs_y = rhs[:1, :]
        rhs_l = rhs[1:, :]
        dl = rhs_l / cs.repmat(lift_l, 1, rhs.size2())
        dy = (rhs_y - dyn_l @ dl) / cs.repmat(
            dyn_y, 1, rhs.size2()
        )
        return cs.vertcat(dy, dl)

    return system.eliminate(solve)

dynamics = dynamics.with_elimination_graph(elimination, [coupling])

lifted_stage = moto.stage()
lifted_stage.add(dynamics)
lifted_sqp = moto.sqp(n_job=1)
lifted_sqp.stages.append(lifted_stage.copy())
_ = lifted_sqp.nodes
```

Select blocks by equation and symbol handles with
`system.jac(equation, variable)`. Structurally zero blocks retain their full
shape. `system.residual(equation)` provides an equation residual when a
structured solve needs it. `system.eliminate(solve)` applies the same solve to
the packed `[h_x, h_u, h]` columns and forward-action right-hand side, then
derives the transpose action from that graph.

The solve returns the unsigned response $h_l^{-1}rhs$. Do not insert the minus
sign from

$$
dl=-h_l^{-1}(h_xdx+h_udu+h),
$$

because the nullspace solver owns that sign. Preserve RHS row and column
dimensions.

## Regularization parameters

Regularize an otherwise absent square block inside the builder:

```python
block = system.jac(equation, lifted_variable)
regularization = block.param(1e-6, name="contact_regularization")
regularized_block = block.add_diag(regularization)
```

The dynamics group exposes these symbols through
`dynamics.elimination_parameters`. The builder is pure modeling code and may
be invoked more than once during finalization; it must not perform runtime work
or depend on invocation count.

CasADi supplies MX dependency and sparsity metadata. Moto lowers the finished
graph to persistent linear-backend storage and kernels, so Python is not called
in the SQP hot path and users do not provide sparsity patterns. See the
[larger sparse example](https://github.com/HaizhouZ/moto/blob/main/example/toy/lifted_sparse_elimination.py).
