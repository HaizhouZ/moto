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


def build_modeled_ocp():
    model = moto.graph_model()
    stage_node = model.add_node()
    stage_node.add(running_cost)
    stage_node.add(u_box)

    terminal_node = model.add_node(stage_node.prob.clone())
    stage_edge = model.connect(stage_node, terminal_node)
    stage_edge.add(dyn)

    stage_prob = stage_edge.compose()
    terminal_prob = stage_prob.clone()
    terminal_prob.add_terminal(terminal_cost)
    return stage_prob, terminal_prob


def build_sqp():
    stage_prob, terminal_prob = build_modeled_ocp()

    print("Stage problem")
    stage_prob.print_summary()
    print("Terminal problem")
    terminal_prob.print_summary()

    sqp = moto.sqp(n_job=1)
    graph = sqp.graph
    head = graph.set_head(graph.add(sqp.create_node(stage_prob)))
    tail = graph.set_tail(graph.add(sqp.create_node(terminal_prob)))
    graph.add_edge(head, tail, N)

    def init(node: moto.sqp.data_type):
        node.value[x] = x0.copy()
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
