#!/usr/bin/env python3

import casadi as cs
import moto
import numpy as np


def build_ill_conditioned_sqp(
    *,
    control_scale: float = 1e-2,
    enable_restoration: bool = True,
):
    nx, nu = 2, 1
    horizon = 12

    # Weak control authority plus all-stage x/x_next initialization gives a
    # feasible but numerically awkward rollout.
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    B = np.array([[0.0], [control_scale]])
    x_init = np.array([10.0, 0.0])

    x, xn = moto.sym.states("x", nx)
    u = moto.sym.inputs("u", nu)

    dyn = moto.dense_dynamics.create("toy_ill_dyn", [x, xn, u], xn.sx - A @ x.sx - B @ u.sx)
    running_cost = moto.cost.create(
        "toy_ill_running_cost",
        [x, u],
        0.5 * cs.sumsqr(x.sx) + 1e-4 * cs.sumsqr(u.sx),
    ).set_diag_hess()
    terminal_cost = moto.cost.create(
        "toy_ill_terminal_cost",
        [x],
        1e-2 * cs.sumsqr(x.sx),
    ).set_diag_hess()
    u_box = moto.constr.create(
        "toy_ill_u_box",
        [u],
        cs.vertcat(u.sx - 0.2, -u.sx - 0.2),
    ).cast_ineq()

    sqp = moto.sqp(n_job=1)
    modeled = sqp.create_graph()

    stage_prob = moto.node_ocp.create()
    stage_prob.add(running_cost)
    stage_prob.add(u_box)

    terminal_prob = stage_prob.clone()
    terminal_prob.add_terminal(terminal_cost)

    start_node = modeled.create_node(stage_prob)
    terminal_node = modeled.create_node(terminal_prob)
    for edge in modeled.add_path(start_node, terminal_node, horizon):
        edge.add(dyn)

    def init(node: moto.sqp.data_type):
        node.value[x] = x_init.copy()
        if node.prob.dim(moto.field___y) > 0:
            node.value[xn] = x_init.copy()

    sqp.apply_forward(init)

    s = sqp.settings
    s.prim_tol = 1e-8
    s.dual_tol = 1e-8
    s.comp_tol = 1e-8
    s.ls.method = moto.ns_sqp.search_method_filter
    s.ls.max_steps = 2
    s.restoration.enabled = enable_restoration
    s.restoration.max_iter = 8

    return sqp


def run_case(label: str, **kwargs):
    sqp = build_ill_conditioned_sqp(**kwargs)
    kkt = sqp.update(40, verbose=True)
    print(f"\n=== {label} ===")
    print(f"result     : {kkt.result}")
    print(f"num_iter   : {kkt.num_iter}")
    print(f"prim_res   : {kkt.inf_prim_res:.3e}")
    print(f"dual_res   : {kkt.inf_dual_res:.3e}")
    print(f"comp_res   : {kkt.inf_comp_res:.3e}")
    print(f"solved     : {kkt.solved}")
    return kkt


def main():
    relaxed = run_case(
        "Feasible Ill-Conditioned Baseline",
        control_scale=1e-2,
        enable_restoration=True,
    )

    print("\nSummary")
    print("  baseline is feasible and uses the normal filter settings.")
    print("  this script is meant to be a numerically awkward but still well-posed baseline.")

    if not relaxed.solved:
        raise AssertionError(f"expected relaxed ill-conditioned case to solve, got {relaxed.result}")


if __name__ == "__main__":
    main()
