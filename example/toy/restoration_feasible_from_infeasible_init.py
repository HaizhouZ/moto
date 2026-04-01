#!/usr/bin/env python3

import casadi as cs
import moto
import numpy as np
import os


def build_feasible_sqp():
    x, xn = moto.sym.states("x", 1)
    u = moto.sym.inputs("u", 1)

    term_eq = moto.constr.create("toy_feas_init_xN_eq", [x], x.sx - 1.0)
    dyn = moto.dense_dynamics.create("toy_feas_init_integrator_dyn", [x, xn, u], xn.sx - x.sx - u.sx)

    running_cost = moto.cost.create(
        "toy_feas_init_running_cost",
        [x, u],
        1e-2 * cs.sumsqr(x.sx) + 1e-2 * cs.sumsqr(u.sx),
    ).set_diag_hess()
    terminal_cost = moto.cost.create(
        "toy_feas_init_terminal_cost",
        [x],
        1e-2 * cs.sumsqr(x.sx - 1.0),
    ).set_diag_hess()

    sqp = moto.sqp(n_job=1)
    modeled = sqp.create_graph()

    stage_prob = moto.node_ocp.create()
    stage_prob.add(running_cost)

    terminal_prob = moto.node_ocp.create()
    terminal_prob.add(running_cost)
    terminal_prob.add_terminal(terminal_cost)
    terminal_prob.add_terminal(term_eq)

    start_node = modeled.create_node(stage_prob)
    terminal_node = modeled.create_node(terminal_prob)
    for edge in modeled.add_path(start_node, terminal_node, 4):
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

    return sqp, x, u


def main():
    sqp, x, u = build_feasible_sqp()
    verbose = os.getenv("MOTO_FEAS_VERBOSE", "0") == "1"
    kkt = sqp.update(20, verbose=verbose)

    values = {"x": [], "u": []}

    def grab(node: moto.sqp.data_type):
        values["x"].append(np.asarray(node.value[x], dtype=float).reshape(-1))
        values["u"].append(np.asarray(node.value[u], dtype=float).reshape(-1))

    sqp.apply_forward(grab)

    print("\n=== Feasible From Infeasible Init Probe ===")
    print(f"result     : {kkt.result}")
    print(f"num_iter   : {kkt.num_iter}")
    print(f"prim_res   : {kkt.inf_prim_res:.3e}")
    print(f"dual_res   : {kkt.inf_dual_res:.3e}")
    print(f"comp_res   : {kkt.inf_comp_res:.3e}")
    print(f"solved     : {kkt.solved}")
    print(f"x[0]       : {values['x'][0]}")
    print(f"u[0]       : {values['u'][0]}")
    print(f"x[N]       : {values['x'][-1]}")

    if not kkt.solved:
        print("note: if this fails, restoration/normal globalization is still too aggressive on a feasible-but-infeasible-init case.")


if __name__ == "__main__":
    main()
