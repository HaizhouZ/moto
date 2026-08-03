#!/usr/bin/env python3

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import casadi as cs
import moto
import numpy as np

from example.helpers import add_terms, collect_node_values, visit_nodes

np.set_printoptions(precision=4, suppress=True)

nx, nu = 2, 1
N = 12

A = np.array([[1.0, 0.1], [0.0, 1.0]])
B = np.array([[0.0], [0.1]])
x0 = np.array([1.0, 0.0])

x, xn = moto.sym.states("x", nx)
u = moto.sym.inputs("u", nu)

dyn = moto.dense_dynamics.create(
    "toy_base_double_integrator_dyn",
    xn.sx - A @ x.sx - B @ u.sx,
)

running_cost = moto.cost.from_vector(
    "toy_base_running_cost",
    cs.vertcat(x.sx, u.sx),
    weight=np.array([1.0, 1.0, 0.1]),
)

terminal_cost = moto.cost.from_vector(
    "toy_base_terminal_cost",
    x,
    weight=10.0,
)

u_limit = 0.5
u_box = moto.ineq.bounds("toy_base_u_box", u, -u_limit, u_limit)


def build_sqp():
    sqp = moto.sqp(n_job=1)

    stage_prob = moto.stage()
    add_terms(stage_prob, dyn, running_cost, u_box)

    stages = sqp.add_stage(stage_prob, N)
    stages[-1].ed.add(terminal_cost)

    nodes = sqp.nodes
    print("Stage problem")
    nodes[0].prob.print_summary()
    print("Terminal problem")
    nodes[-1].prob.print_summary()

    def init(node: moto.sqp.data_type, _):
        node.value[x] = x0.copy()
        if node.prob.dim(moto.field.field___y) > 0:
            node.value[xn] = x0.copy()

    visit_nodes(nodes, init)
    sqp.settings.prim_tol = 1e-8
    sqp.settings.dual_tol = 1e-8
    sqp.settings.comp_tol = 1e-8
    return sqp, nodes


def main():
    sqp, nodes = build_sqp()
    sys.stdout.flush()
    kkt = sqp.update(50, verbose=True)
    sys.stdout.flush()

    x_values, u_values = collect_node_values(nodes, x, u)
    x_values = [value.reshape(-1) for value in x_values]
    u_values = [value.reshape(-1) for value in u_values]

    print(f"result   : {kkt.result}")
    print(f"num_iter : {kkt.num_iter}")
    print(f"prim_res : {kkt.inf_prim_res:.2e}")
    print(f"dual_res : {kkt.inf_dual_res:.2e}")
    print(f"x[0]     : {x_values[0]}")
    print(f"u[0]     : {u_values[0]}")

    assert kkt.solved, f"toy modeled OCP failed: {kkt.result}"


if __name__ == "__main__":
    main()
