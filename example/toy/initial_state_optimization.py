#!/usr/bin/env python3

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import casadi as cs
import moto
import numpy as np

from example.helpers import visit_nodes

N = 3
TARGET = 2.0

x, xn = moto.sym.states("initial_state_demo_x", 1)
u = moto.sym.inputs("initial_state_demo_u", 1)

dynamics = moto.dense_dynamics.create(
    "initial_state_demo_dynamics",
    xn.sx - x.sx - u.sx,
)
control_cost = moto.cost.from_scalar(
    "initial_state_demo_control_cost", u, weight=2.0
)
initial_cost = moto.cost.from_scalar(
    "initial_state_demo_target_cost", x, weight=2.0, reference=TARGET
)


def solve(mode, *, optimize):
    sqp = moto.sqp(n_job=1)
    stage = moto.stage()
    stage.add(dynamics)
    stage.add(control_cost)
    sqp.add_stage(stage, N)
    sqp.start_node.add(initial_cost)

    sqp.settings.initial_state = mode
    sqp.settings.restoration.enabled = False
    sqp.settings.prim_tol = 1e-9
    sqp.settings.dual_tol = 1e-9
    sqp.settings.comp_tol = 1e-9

    nodes = sqp.nodes

    def initialize(node, _):
        node.value[x] = np.zeros(1)
        node.value[xn] = np.zeros(1)
        node.value[u] = np.zeros(1)

    visit_nodes(nodes, initialize)
    result = sqp.update(10, verbose=False) if optimize else None
    states = np.array([node.value[x][0] for node in nodes] + [nodes[-1].value[xn][0]])
    controls = np.array([node.value[u][0] for node in nodes])
    return result, states, controls


def main():
    _, fixed_states, _ = solve(
        moto.sqp.initial_state_mode.fixed,
        optimize=False,
    )
    optimized, optimized_states, optimized_controls = solve(
        moto.sqp.initial_state_mode.optimized,
        optimize=True,
    )

    print(f"fixed x[0]     : {fixed_states[0]:.9f}")
    print(f"optimized x[0] : {optimized_states[0]:.9f}")
    print(f"target          : {TARGET:.9f}")
    print(f"optimized states: {optimized_states}")
    print(f"optimized inputs: {optimized_controls}")

    assert optimized.solved, f"optimized initial-state solve failed: {optimized.result}"
    np.testing.assert_allclose(fixed_states[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(optimized_states, TARGET, atol=1e-7)
    np.testing.assert_allclose(optimized_controls, 0.0, atol=1e-7)


if __name__ == "__main__":
    main()
