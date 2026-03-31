#!/usr/bin/env python3

import casadi as cs
import moto
import numpy as np

np.set_printoptions(precision=4, suppress=True)

nx, nu = 2, 1
N = 12

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.0], [0.1]])
x0 = np.array([1.0, 0.0])

x, xn = moto.sym.states("x", nx)
u = moto.sym.inputs("u", nu)

dyn = moto.dense_dynamics.create(
    "double_integrator_dyn",
    [x, xn, u],
    xn.sx - A @ x.sx - B @ u.sx,
)

running_cost = (
    moto.cost.create(
        "running_cost",
        [x, u],
        0.5 * cs.sumsqr(x.sx) + 0.05 * cs.sumsqr(u.sx),
    )
    .set_diag_hess()
)

terminal_cost = (
    moto.cost.create(
        "terminal_cost",
        [x],
        5.0 * cs.sumsqr(x.sx),
    )
    .set_diag_hess()
)

u_limit = 0.5
u_box = moto.constr.create(
    "u_box",
    [u],
    cs.vertcat(u.sx - u_limit, -u.sx - u_limit),
).cast_ineq()


def build_sqp():
    sqp = moto.sqp(n_job=1)
    modeled = sqp.create_graph()

    stage_node_prob = moto.node_ocp.create()
    stage_node_prob.add(running_cost)
    stage_node_prob.add(u_box)

    terminal_node_prob = stage_node_prob.clone()
    terminal_node_prob.add_terminal(terminal_cost)

    stage_node = modeled.create_node(stage_node_prob)
    terminal_node = modeled.create_node(terminal_node_prob)

    for edge in modeled.add_path(stage_node, terminal_node, N):
        edge.add(dyn)

    flat_nodes = modeled.flatten_nodes()
    print("Stage problem")
    flat_nodes[0].prob.print_summary()
    print("Terminal problem")
    flat_nodes[-1].prob.print_summary()

    def init(node: moto.sqp.data_type):
        node.value[x] = x0.copy()
        if node.prob.dim(moto.field___y) > 0:
            node.value[xn] = x0.copy()

    sqp.apply_forward(init)
    sqp.settings.prim_tol = 1e-8
    sqp.settings.dual_tol = 1e-8
    sqp.settings.comp_tol = 1e-8
    return sqp


def main():
    sqp = build_sqp()
    kkt = sqp.update(50, verbose=True)

    values = {"x": [], "u": []}

    def grab(node: moto.sqp.data_type):
        values["x"].append(np.asarray(node.value[x], dtype=float).reshape(-1))
        values["u"].append(np.asarray(node.value[u], dtype=float).reshape(-1))

    sqp.apply_forward(grab)

    print(f"result   : {kkt.result}")
    print(f"num_iter : {kkt.num_iter}")
    print(f"prim_res : {kkt.inf_prim_res:.2e}")
    print(f"dual_res : {kkt.inf_dual_res:.2e}")
    print(f"x[0]     : {values['x'][0]}")
    print(f"u[0]     : {values['u'][0]}")

    assert kkt.solved, f"toy modeled OCP failed: {kkt.result}"


if __name__ == "__main__":
    main()
