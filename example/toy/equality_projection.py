"""Compare selectable hard-equality projection on a redundant constrained OCP."""

import moto
import casadi as cs
import numpy as np

x, xn = moto.sym.states("projection_demo_x", 2, default_val=0.0)
u = moto.sym.inputs("projection_demo_u", 2, default_val=0.0)
dyn = moto.dense_dynamics.create("projection_demo_dyn", xn.sx - x.sx - u.sx)
residual = u.sx[0] + u.sx[1] - 1.0
equality = moto.constr.create(
    "projection_demo_eq", cs.vertcat(residual, 2.0 * residual, 0.0)
)
cost = moto.cost.from_vector("projection_demo_cost", u, weight=[1.0, 2.0])
bound = moto.ineq.bounds("projection_demo_bound", u, -2.0, 2.0)
stage = moto.stage()
stage.add([dyn, equality, cost, bound])

for backend in (moto.sqp.equality_projection_backend.eigen,
                moto.sqp.equality_projection_backend.panel_lu):
    sqp = moto.sqp(n_job=2)
    sqp.stages.extend([stage.copy() for _ in range(3)])
    sqp.settings.equality_projection = backend
    sqp.settings.regularization.validate_direction = True
    sqp.settings.prim_tol = sqp.settings.dual_tol = sqp.settings.comp_tol = 1e-8
    result = sqp.update(30, verbose=False)
    print(f"{backend}: solved={result.solved}, iterations={result.num_iter}, "
          f"primal={result.inf_prim_res:.3e}, dual={result.inf_dual_res:.3e}")
    assert result.solved, result.result
    for node in sqp.nodes:
        np.testing.assert_allclose(node.value[u], [2.0/3.0, 1.0/3.0], atol=1e-7)
