#!/usr/bin/env python3

"""Sparse lifted Schur graph with block-local regularization."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import casadi as cs
import moto


DIM = 8


def main():
    x, y = moto.sym.states("sparse_schur_x", DIM)
    u = moto.sym.inputs("sparse_schur_u", DIM)
    lifted = moto.sym.lifted("sparse_schur_l", DIM)

    dynamics_residual = y.sx - x.sx - u.sx
    dynamics = moto.semi_implicit_euler.create(
        "sparse_schur_dynamics",
        dynamics_residual,
        moto.semi_implicit_euler.state.pos,
    )
    constraint_residual = lifted.sx - y.sx
    constraint = moto.lifted.create(
        "sparse_schur_constraint", constraint_residual, [lifted]
    )

    def elimination(system):
        missing = system.jac(dynamics, lifted)
        assert missing.mx.shape == (DIM, DIM)
        assert missing.mx.nnz() == 0

        regularization = missing.param(1e-6)
        regularized = missing.add_diag(regularization)
        lift_l = system.jac(constraint, lifted).mx
        lift_y = system.jac(constraint, y).mx
        schur_diagonal = cs.diag(lift_l - lift_y @ regularized)

        def solve(rhs):
            y_base = rhs[:DIM, :]
            reduced = rhs[DIM:, :] - lift_y @ y_base
            lifted_step = reduced / cs.repmat(
                schur_diagonal, 1, rhs.size2()
            )
            return cs.vertcat(
                y_base - regularized @ lifted_step, lifted_step
            )

        return system.eliminate(solve)

    dynamics = dynamics.with_elimination_graph(elimination, [constraint])

    stage = moto.stage()
    stage.add(dynamics)
    sqp = moto.sqp(n_job=1)
    sqp.stages.extend([stage.copy() for _ in range(2)])
    _ = sqp.nodes

    parameters = dynamics.elimination_parameters
    assert len(parameters) == 1
    assert parameters[0].default_value[0] == 1e-6


if __name__ == "__main__":
    main()
