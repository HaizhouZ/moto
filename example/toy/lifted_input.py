#!/usr/bin/env python3

"""Small analytic check for stage-local lifted-variable elimination."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import moto
import casadi as cs
import numpy as np


TARGET = 3.0


def main():
    x, xn = moto.sym.states("lifted_demo_x", 1)
    u = moto.sym.inputs("lifted_demo_u", 1)
    lifted = moto.sym.lifted("lifted_demo_l", 1)

    # y - x - l = 0 and l - u = 0.  The QP keeps l and both equality
    # multipliers in its authored model, but eliminates l together with y in
    # the stage lifting operator.
    dynamics = moto.semi_implicit_euler.create(
        "lifted_demo_semi_dynamics",
        xn.sx - x.sx - lifted.sx,
        moto.semi_implicit_euler.state.pos,
    )
    lifting = moto.lifted.create(
        "lifted_demo_constraint", lifted.sx - u.sx, [lifted]
    )
    dynamics.add_subconstraint(lifting)

    def elimination(system):
        def solve(rhs):
            return cs.vertcat(rhs[:1, :] + rhs[1:, :], rhs[1:, :])

        return moto.lifted.elimination(
            solve(system.h_x()),
            solve(system.h_u()),
            solve(system.h()),
            [],
            solve(system.action_rhs),
        )

    dynamics = dynamics.set_elimination_graph(elimination)

    stage = moto.stage()
    stage.add(dynamics)
    stage.add(moto.cost.from_scalar("lifted_demo_u_cost", u))
    stage.add(moto.cost.from_scalar("lifted_demo_l_cost", lifted))

    sqp = moto.sqp(n_job=1)
    stages = sqp.add_stage(stage, 1)
    stages[-1].ed.add(
        moto.cost.from_scalar(
            "lifted_demo_target_cost", x, reference=TARGET
        )
    )
    sqp.settings.restoration.enabled = False
    sqp.settings.prim_tol = 1e-10
    sqp.settings.dual_tol = 1e-10
    sqp.settings.comp_tol = 1e-10

    node = sqp.nodes[0]
    node.value[x] = np.zeros(1)
    node.value[xn] = np.zeros(1)
    node.value[u] = np.zeros(1)
    node.value[lifted] = np.zeros(1)

    result = sqp.update(10, verbose=False)
    expected = TARGET / 3.0
    values = {
        "u": float(node.value[u][0]),
        "l": float(node.value[lifted][0]),
        "y": float(node.value[xn][0]),
    }

    print(f"result   : {result.result}")
    print(f"iterations: {result.num_iter}")
    print(f"expected : {expected:.12f}")
    print(f"values   : {values}")

    assert result.solved, f"lifted-input solve failed: {result.result}"
    np.testing.assert_allclose(list(values.values()), expected, atol=1e-8)


if __name__ == "__main__":
    main()
