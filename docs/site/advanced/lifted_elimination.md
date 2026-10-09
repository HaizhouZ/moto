# Custom lifted elimination graphs

Use a lifted elimination graph when a dynamics equation is coupled to auxiliary
interval variables and equality constraints whose QP directions can be removed
by a structured local solve. Declare those variables and constraints normally:

```python
acceleration = moto.sym.inputs("acceleration", nv)
force_balance = moto.constr.create("force_balance", residual)
```

There is no separate lifted-variable or lifted-constraint constructor. Passing
the ordinary handles to `with_elimination_graph(...)` marks their role in that
group:

```python
dynamics = dynamics.with_elimination_graph(
    elimination,
    variables=[acceleration, contact_wrench],
    constraints=[force_balance, contact_kinematics],
)
```

Do this before adding any selected handle to a stage. The same handles remain
valid for costs, bounds, warm starts, and stage activation;
Moto does not create replacement symbols or constraints. Internally, the
selected directions are ordered as `l` and the selected residuals as `lift`.
Together with the predicted state `y` and dynamics residual `dyn`, the local
system is

$$
h(x,u,y,l) = [dyn; lift].
$$

The nonlinear variables and multipliers remain explicit. Only their local QP
directions are eliminated relative to the remaining `x/u` directions. The
builder supplies an inverse action whose rows and columns are ordered as
`[y; l]`.

The following group has $y-x-a=0$ and $a-u=0$. The ordinary input `a` becomes
the group's lifted auxiliary variable when the group is created. Its grouped
Jacobian is block triangular, so the inverse action can be expressed as two
substitutions:

```python
import casadi as cs
import moto

x_l, y_l = moto.sym.states("elimination_x", 1)
u_l = moto.sym.inputs("elimination_u", 1)
a = moto.sym.inputs("elimination_a", 1)

dynamics = moto.semi_implicit_euler.create(
    "elimination_dynamics",
    y_l.sx - x_l.sx - a.sx,
    moto.semi_implicit_euler.state.pos,
)
coupling = moto.constr.create(
    "elimination_coupling",
    a.sx - u_l.sx,
)

def elimination(system):
    dyn_y = system.jac(dynamics, y_l).mx
    dyn_a = system.jac(dynamics, a).mx
    lift_y = system.jac(coupling, y_l).mx
    lift_a = system.jac(coupling, a).mx
    assert lift_y.nnz() == 0

    def solve(rhs):
        rhs_y = rhs[:1, :]
        rhs_l = rhs[1:, :]
        da = rhs_l / cs.repmat(lift_a, 1, rhs.size2())
        dy = (rhs_y - dyn_a @ da) / cs.repmat(
            dyn_y, 1, rhs.size2()
        )
        return cs.vertcat(dy, da)

    return system.eliminate(solve)

dynamics = dynamics.with_elimination_graph(
    elimination, variables=[a], constraints=[coupling]
)

lifted_stage = moto.stage()
lifted_stage.add(dynamics)
lifted_sqp = moto.sqp(n_job=1)
lifted_sqp.stages.append(lifted_stage.copy())
_ = lifted_sqp.nodes
```

`with_elimination_graph(...)` returns the configured dynamics group, so keep
the returned value as shown above. The source dynamics object is unchanged.
Each selected variable must be an ordinary interval input, and each selected
constraint must be an ordinary hard equality. A handle cannot be assigned to a
lifted group after it has already entered a problem.

Select blocks by equation and symbol handles with
`system.jac(equation, variable)`. Structurally zero blocks retain their full
shape. `system.residual(equation)` provides an equation residual when a
structured solve needs it. `system.eliminate(solve)` applies the same solve to
the packed `[h_x, h_u, h]` columns and forward-action right-hand side, then
derives the transpose action from that graph. The builder is pure symbolic
modeling code: it may be called more than once during finalization, must not
mutate external state, and is never called in the SQP hot path.

The solve returns the unsigned response $h_l^{-1}rhs$. Do not insert the minus
sign from

$$
dl=-h_l^{-1}(h_xdx+h_udu+h),
$$

because the nullspace solver owns that sign. Preserve RHS row and column
dimensions.

## Reusable matrix solves

Use `system.solve(matrix)` to create a reusable factor, then apply it to each
right-hand side:

```python
factor = system.solve(matrix)
response = factor.solve(rhs)
```

For a matrix known to be symmetric positive definite, use
`system.solve(matrix, spd=True)`. This is a mathematical declaration, not a
request for automatic regularization; a failed Cholesky factorization raises
an error.

Equivalent matrix solves share their factorization within one graph
linearization. Larger blocks (dimension greater than three) use BLASFEO
row-pivot LU or Cholesky; scalar and fixed-size 2x2/3x3 helpers retain their
existing paths. Factors and RHS workspace belong to the graph instance and
reuse allocated capacity across iterations. No additional SQP setting is
needed; `settings.equality_projection` controls ordinary hard-equality
projection separately, not these elimination-graph solves.

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
