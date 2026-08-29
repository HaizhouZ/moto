#!/usr/bin/env python3

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import moto
import numpy as np

def main():
    x, xn = moto.sym.states("restoration_demo_x", 1)
    u = moto.sym.inputs("restoration_demo_u", 1)

    dynamics = moto.dense_dynamics.create(
        "restoration_demo_dynamics", xn.sx - x.sx - u.sx
    )
    target = moto.constr.create("restoration_demo_target", x.sx - 1.0)
    tracking = moto.cost.from_scalar(
        "restoration_demo_tracking", u, weight=50.0, reference=0.5
    )

    sqp = moto.sqp(n_job=1)
    stage = moto.stage()
    stage.add(dynamics)
    stage.add(tracking)
    sqp.stages.extend([stage.copy() for _ in range(2)])
    sqp.ed.add(target)

    nodes = sqp.nodes
    for index, node in enumerate(nodes):
        node.value[x] = np.full(1, 0.0 if index == 0 else 2.0)
        node.value[xn] = np.full(1, 2.0)
        node.value[u] = np.zeros(1)

    # Deliberately make the normal-phase Armijo test too strict. Its Newton
    # step is rejected down to alpha_min, which exercises restoration. Inside
    # restoration the filter accepts feasibility progress and returns to the
    # original problem.
    sqp.settings.ls.constr_vio_min_frac = 10.0
    sqp.settings.ls.s_phi = 1.0
    sqp.settings.ls.s_theta = 1.0
    sqp.settings.ls.armijo_dec_frac = 3.0
    sqp.settings.ls.max_steps = 5
    sqp.settings.ls.enable_flat_obj_accept = False
    sqp.settings.restoration.enabled = True
    sqp.settings.restoration.max_iter = 20
    sqp.settings.restoration.alpha_min_factor = 0.2
    sqp.settings.restoration.restoration_improvement_frac = 0.9
    sqp.settings.restoration.rho_eq = 0.1
    sqp.settings.prim_tol = 1e-8
    sqp.settings.dual_tol = 1e-8
    sqp.settings.comp_tol = 1e-8

    recovery = sqp.update(2, verbose=True)
    assert recovery.solved, f"restoration did not recover: {recovery.result}"
    assert recovery.inf_prim_res < 2.0

    # Continue from the recovered point with ordinary globalization settings.
    sqp.settings.ls.armijo_dec_frac = 1e-4
    sqp.settings.ls.enable_flat_obj_accept = True
    sqp.settings.restoration.alpha_min_factor = 5e-2
    result = sqp.update(20, verbose=False)
    terminal_state = float(nodes[-1].value[xn][0])
    states = np.array([node.value[x][0] for node in nodes])
    next_states = np.array([node.value[xn][0] for node in nodes])
    controls = np.array([node.value[u][0] for node in nodes])

    print("\nrestoration demo summary")
    print(f"  recovery   : {recovery.result}")
    print(f"  recovered r: {recovery.inf_prim_res:.3e}")
    print(f"  result     : {result.result}")
    print(f"  iterations : {result.num_iter}")
    print(f"  primal res : {result.inf_prim_res:.3e}")
    print(f"  dual res   : {result.inf_dual_res:.3e}")
    print(f"  terminal x : {terminal_state:.9f}")
    print(f"  states     : {states}")
    print(f"  next states: {next_states}")
    print(f"  controls   : {controls}")

    assert result.solved, f"restoration demo failed: {result.result}"
    assert recovery.num_iter == 2, "the configured normal step did not enter restoration"
    np.testing.assert_allclose(terminal_state, 1.0, atol=1e-7)
    np.testing.assert_allclose(controls, 0.5, atol=1e-7)


if __name__ == "__main__":
    main()
