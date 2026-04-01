#!/usr/bin/env python3

import casadi as cs
import moto
import numpy as np
import os


def build_infeasible_sqp():
    x, xn = moto.sym.states("x", 1)
    u = moto.sym.inputs("u", 1)

    init_eq = moto.constr.create("toy_infeas_x0_eq", [x], x.sx)
    u_zero = moto.constr.create("toy_infeas_u_eq", [u], u.sx)
    term_eq = moto.constr.create("toy_infeas_xN_eq", [x], x.sx - 1.0)
    dyn = moto.dense_dynamics.create("toy_infeas_integrator_dyn", [x, xn, u], xn.sx - x.sx - u.sx)
    running_cost = moto.cost.create(
        "toy_infeas_running_cost",
        [x, u],
        1e-3 * cs.sumsqr(x.sx) + 1e-3 * cs.sumsqr(u.sx),
    ).set_diag_hess()
    terminal_cost = moto.cost.create(
        "toy_infeas_terminal_cost",
        [x],
        1e-3 * cs.sumsqr(x.sx),
    ).set_diag_hess()

    sqp = moto.sqp(n_job=1)
    modeled = sqp.create_graph()

    start_prob = moto.node_ocp.create()
    start_prob.add(init_eq)
    start_prob.add(u_zero)
    start_prob.add(running_cost)

    terminal_prob = moto.node_ocp.create()
    terminal_prob.add(u_zero)
    terminal_prob.add(running_cost)
    terminal_prob.add_terminal(terminal_cost)
    terminal_prob.add_terminal(term_eq)

    start_node = modeled.create_node(start_prob)
    terminal_node = modeled.create_node(terminal_prob)
    for edge in modeled.add_path(start_node, terminal_node, 1):
        edge.add(dyn)

    def init(node: moto.sqp.data_type):
        node.value[x] = np.array([0.0])
        node.value[u] = np.array([0.0])
        if node.prob.dim(moto.field___y) > 0:
            node.value[xn] = np.array([0.0])

    sqp.apply_forward(init)
    sqp.settings.prim_tol = 1e-9
    sqp.settings.dual_tol = 1e-9
    sqp.settings.comp_tol = 1e-9
    sqp.settings.ls.method = moto.ns_sqp.search_method_filter
    sqp.settings.restoration.enabled = True
    sqp.settings.restoration.max_iter = 8

    return sqp


def main():
    sqp = build_infeasible_sqp()
    verbose = os.getenv("MOTO_INFEAS_VERBOSE", "0") == "1"
    kkt = sqp.update(20, verbose=verbose)

    print("\n=== Infeasible Restoration Probe ===")
    print(f"result     : {kkt.result}")
    print(f"num_iter   : {kkt.num_iter}")
    print(f"prim_res   : {kkt.inf_prim_res:.3e}")
    print(f"dual_res   : {kkt.inf_dual_res:.3e}")
    print(f"comp_res   : {kkt.inf_comp_res:.3e}")
    print(f"solved     : {kkt.solved}")

    detected = kkt.result in (
        moto.ns_sqp.iter_result_infeasible_stationary,
        moto.ns_sqp.iter_result_restoration_failed,
    )
    print(f"detected infeasibility-like exit: {detected}")

    if not detected:
        print("note: this case no longer exits through the restoration-based infeasibility path.")


if __name__ == "__main__":
    main()
