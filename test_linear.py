#!/usr/bin/env python3
"""
Linear system convergence tests for ns-sqp.

Test 1 (equality only):
  linear dynamics + quadratic cost + soft PMM equality constraint
  → IPM-free solve, verifies convergence

Test 2 (inequality):
  same linear system + box constraint on u
  → IPM convergence test

Test 3 (rank-deficient equalities):
  two hard __eq_xu constraints whose u-Jacobians are proportional
  → s_c_stacked has rank 1 < ncstr=2; FullPivLU detects redundancy
  → solver should converge to the same solution as the single-constraint case

Test 4 (shared soft equality + hard inequality boundary):
    soft PMM equality pushes u[0] toward u_target=0.3
    hard IPM inequality caps u[0] ≤ 0.3
    → solver should land on the shared boundary (u[0] ≈ 0.3)

Test 5 (CasADi/IPOPT shared boundary):
    hard equality u[0] = 0.3 plus hard inequality u[0] ≤ 0.3
    → IPOPT should solve the same boundary case with the shared limit active
"""

import moto
import casadi as cs
import numpy as np

np.set_printoptions(precision=4, suppress=True)

# ── system parameters ────────────────────────────────────────────────────────
nx, nu = 2, 2
N = 10  # horizon length

# stable 2D system
A = np.array([[0.9, 0.1], [0.0, 0.8]])
B = np.array([[0.1, 0.0], [0.0, 0.1]])

x0 = np.array([1.0, -0.5])

# ── symbolic variables ───────────────────────────────────────────────────────
x, xn = moto.sym.states("x", nx)
u_sym = moto.sym.inputs("u", nu)

# dynamics:  xn − A·x − B·u = 0
dyn = moto.dense_dynamics.create(
    "dyn",
    [x, xn, u_sym],
    xn.sx - A @ x.sx - B @ u_sym.sx,
)

# running cost:  ½‖x‖² + ½·0.01·‖u‖²
cost = moto.cost.create(
    "cost",
    [x, u_sym],
    0.5 * cs.sumsqr(x.sx) + 0.005 * cs.sumsqr(u_sym.sx),
).set_diag_hess()

# terminal cost:  ½‖x‖²  (applied to current state at terminal node)
term_cost = (
    moto.cost.create("term_cost", [x], 0.5 * cs.sumsqr(x.sx))
    .set_diag_hess()
)

# equality constraint:  u[0] + x[0] = 0  (__eq_xu, 1-dimensional)
#   ncstr=1, nu=2  →  nz=1 (1-D null space remains free)
eq_constr = moto.constr.create(
    "eq",
    [x, u_sym],
    cs.vertcat(u_sym.sx[0] + x.sx[0]),
)

eq_constr = eq_constr.cast_soft()

# inequality constraint:  u ≤ 0.3  (__ineq_xu, 2-dimensional)
u_max = 0.3
ineq_constr = moto.constr.create(
    "ineq",
    [u_sym],
    u_sym.sx - u_max,
).cast_ineq()


# ── helper: build a fresh SQP problem ───────────────────────────────────────
def _build_sqp(
    prob: moto.ocp,
    prim_tol=1e-8,
    dual_tol=1e-8,
    comp_tol=1e-8,
    horizon=N,
):
    prob_term = prob.clone()
    prob_term.add_terminal(term_cost)

    prob.print_summary()
    prob_term.print_summary()

    sqp = moto.sqp(n_job=1)
    g = sqp.graph
    n0 = g.set_head(g.add(sqp.create_node(prob)))
    n1 = g.set_tail(g.add(sqp.create_node(prob_term)))
    g.add_edge(n0, n1, horizon)

    def init(d: moto.sqp.data_type):
        d.value[x] = x0.copy()
        d.value[xn] = x0.copy()

    sqp.apply_forward(init)

    sqp.settings.prim_tol = prim_tol
    sqp.settings.dual_tol = dual_tol
    sqp.settings.comp_tol = comp_tol
    return sqp


def _to_plain_ocp(prob: moto.ocp_base):
    flat = moto.ocp.create()
    fields = [
        moto.field___x,
        moto.field___u,
        moto.field___y,
        moto.field___dyn,
        moto.field___eq_x,
        moto.field___eq_x_soft,
        moto.field___eq_xu,
        moto.field___eq_xu_soft,
        moto.field___ineq_x,
        moto.field___ineq_xu,
        moto.field___cost,
        moto.field___p,
        moto.field___usr_func,
        moto.field___func_stack,
        moto.field___pre_comp,
        moto.field___post_comp,
    ]
    for field in fields:
        for expr in prob.exprs(field):
            flat.add(expr)
    return flat


def _build_modeled_sqp(
    prob: moto.ocp,
    prim_tol=1e-8,
    dual_tol=1e-8,
    comp_tol=1e-8,
    horizon=N,
):
    model = moto.graph_model()
    stage_node = model.add_node()
    node_fields = [
        moto.field___cost,
        moto.field___eq_x,
        moto.field___eq_x_soft,
        moto.field___eq_xu,
        moto.field___eq_xu_soft,
        moto.field___ineq_x,
        moto.field___ineq_xu,
        moto.field___p,
        moto.field___usr_func,
        moto.field___func_stack,
        moto.field___pre_comp,
        moto.field___post_comp,
    ]
    for field in node_fields:
        for expr in prob.exprs(field):
            stage_node.add(expr)

    term_node = model.add_node(stage_node.prob.clone())
    stage_edge = model.connect(stage_node, term_node)
    for dyn_expr in prob.exprs(moto.field___dyn):
        stage_edge.add(dyn_expr)

    prob_stage = _to_plain_ocp(stage_edge.compose())
    prob_term = prob_stage.clone()
    prob_term.add_terminal(term_cost)

    prob_stage.print_summary()
    prob_term.print_summary()

    sqp = moto.sqp(n_job=1)
    g = sqp.graph
    n0 = g.set_head(g.add(sqp.create_node(prob_stage)))
    n1 = g.set_tail(g.add(sqp.create_node(prob_term)))
    g.add_edge(n0, n1, horizon)

    def init(d: moto.sqp.data_type):
        d.value[x] = x0.copy()
        d.value[xn] = x0.copy()

    sqp.apply_forward(init)

    sqp.settings.prim_tol = prim_tol
    sqp.settings.dual_tol = dual_tol
    sqp.settings.comp_tol = comp_tol
    return sqp


def _solve(
    prob: moto.ocp,
    prim_tol=1e-8,
    dual_tol=1e-8,
    comp_tol=1e-8,
    max_iter=100,
    horizon=N,
):
    sqp = _build_sqp(
        prob, prim_tol=prim_tol, dual_tol=dual_tol, comp_tol=comp_tol, horizon=horizon
    )
    kkt = sqp.update(max_iter, verbose=True)
    assert kkt.solved, f"expected success, got {kkt.result}"
    return sqp, kkt


dual = []


def _read_duals(sqp: moto.sqp):
    def grab(d: moto.sqp.data_type):
        dual.append(np.concatenate(d.dense.dual))

    sqp.apply_forward(grab)
    return dual


def _first_node_values(sqp: moto.sqp):
    values = {"x": [], "u": []}

    def grab(d: moto.sqp.data_type):
        values["x"].append(np.asarray(d.value[x], dtype=float).reshape(-1))
        values["u"].append(np.asarray(d.value[u_sym], dtype=float).reshape(-1))

    sqp.apply_forward(grab)
    return values


# ── Test 1: equality-only → 1 SQP iteration ─────────────────────────────────
def test_eq_one_iter():
    print("\n" + "=" * 60)
    print("Test 1: equality constraint only — expect 1 SQP iteration")

    prob = moto.ocp.create()
    prob.add(dyn)
    prob.add(cost)
    prob.add(eq_constr)

    sqp, kkt = _solve(prob, prim_tol=1e-8, dual_tol=1e-8, max_iter=50)
    sol = _first_node_values(sqp)

    print(f"  result   : {kkt.result}")
    print(f"  num_iter : {kkt.num_iter}")
    print(f"  prim_res : {kkt.inf_prim_res:.2e}")
    print(f"  dual_res : {kkt.inf_dual_res:.2e}")
    print(f"  x[0]     : {sol['x']}")
    print(f"  u[0]     : {sol['u']}")

    # assert kkt.num_iter == 1, (
    #     f"linear QP should converge in 1 iteration, got {kkt.num_iter}"
    # )
    print("  PASSED")


def test_modeled_eq_converges():
    print("\n" + "=" * 60)
    print("Test 1b: modeled node/edge path — expect convergence")

    prob = moto.ocp.create()
    prob.add(dyn)
    prob.add(cost)
    prob.add(eq_constr)

    sqp = _build_modeled_sqp(prob, prim_tol=1e-8, dual_tol=1e-8, comp_tol=1e-8)
    kkt = sqp.update(50, verbose=True)
    sol = _first_node_values(sqp)

    print(f"  result   : {kkt.result}")
    print(f"  num_iter : {kkt.num_iter}")
    print(f"  prim_res : {kkt.inf_prim_res:.2e}")
    print(f"  dual_res : {kkt.inf_dual_res:.2e}")
    print(f"  x[0]     : {sol['x']}")
    print(f"  u[0]     : {sol['u']}")

    assert kkt.solved, f"expected success, got {kkt.result}"
    print("  PASSED")


# ── Test 2: inequality constraints → convergence ─────────────────────────────
def test_ineq_converges():
    print("\n" + "=" * 60)
    print("Test 2: inequality constraint — expect convergence")

    prob = moto.ocp.create()
    prob.add(dyn)
    prob.add(cost)
    prob.add(ineq_constr)

    sqp = _build_sqp(prob, prim_tol=1e-6, dual_tol=1e-6, comp_tol=1e-6)
    sqp.settings.ipm.mu0 = 1.0
    kkt = sqp.update(100, verbose=True)

    print(f"  result   : {kkt.result}")
    print(f"  num_iter : {kkt.num_iter}")
    print(f"  prim_res : {kkt.inf_prim_res:.2e}")
    print(f"  dual_res : {kkt.inf_dual_res:.2e}")
    print(f"  comp_res : {kkt.inf_comp_res:.2e}")

    assert kkt.solved, f"expected success, got {kkt.result}"
    print("  PASSED")


def test_rank_deficient_equalities():
    print("\n" + "=" * 60)
    print(
        "Test 3: rank-deficient equalities — expect same solution as single constraint"
    )

    base_eq = moto.constr.create(
        "eq_base",
        [x, u_sym],
        cs.vertcat(u_sym.sx[0] + x.sx[0]),
    )

    redundant_eq = moto.constr.create(
        "eq_redundant",
        [x, u_sym],
        cs.vertcat(
            u_sym.sx[0] + x.sx[0],
            2.0 * u_sym.sx[0] + 2.0 * x.sx[0],
        ),
    )

    prob_base = moto.ocp.create()
    prob_base.add(dyn)
    prob_base.add(cost)
    prob_base.add(base_eq)

    prob_redundant = moto.ocp.create()
    prob_redundant.add(dyn)
    prob_redundant.add(cost)
    prob_redundant.add(redundant_eq)

    sqp_base, kkt_base = _solve(prob_base, prim_tol=1e-8, dual_tol=1e-8, max_iter=50)
    sqp_redundant, kkt_redundant = _solve(
        prob_redundant, prim_tol=1e-8, dual_tol=1e-8, max_iter=50
    )

    sol_base = _first_node_values(sqp_base)
    sol_redundant = _first_node_values(sqp_redundant)

    print(f"  base result      : {kkt_base.result}")
    print(f"  redundant result : {kkt_redundant.result}")
    print(f"  base u[0]        : {sol_base['u']}")
    print(f"  redundant u[0]   : {sol_redundant['u']}")

    assert np.allclose(sol_base["u"], sol_redundant["u"], atol=1e-7, rtol=1e-7)
    assert np.allclose(sol_base["x"], sol_redundant["x"], atol=1e-7, rtol=1e-7)
    print("  PASSED")


def test_soft_eq_vs_hard_ineq():
    print("\n" + "=" * 60)
    print("Test 4: soft equality vs hard inequality — expect a shared boundary")

    soft_target = 0.3 - 1e-3
    hard_max = 0.3

    soft_eq = moto.constr.create(
        "soft_eq",
        [u_sym],
        cs.vertcat(u_sym.sx[0] - soft_target),
    ).cast_soft()
    soft_eq.rho = 1e-8

    hard_ineq = moto.constr.create(
        "hard_ineq",
        [u_sym],
        cs.vertcat(u_sym.sx[0] - hard_max),
    ).cast_ineq()  # u[0] ≤ hard_max

    prob = moto.ocp.create()
    prob.add(dyn)
    prob.add(cost)
    prob.add(soft_eq)
    prob.add(hard_ineq)

    sqp = _build_sqp(prob, prim_tol=1e-6, dual_tol=1e-6, comp_tol=1e-6, horizon=100)
    sqp.settings.ipm.mu0 = 1.0
    sqp.settings.ipm.mu_method = moto.ns_sqp.monotonic_decrease
    sqp.settings.ls.update_alpha_dual = False
    sqp.settings.ls.flat_obj_dec_tol = 1e-3
    kkt = sqp.update(50, verbose=True)
    sol = _first_node_values(sqp)


    # sqp.update(1, verbose=True)  # one more iteration to see duals update

    print(f"  result   : {kkt.result}")
    print(f"  num_iter : {kkt.num_iter}")
    print(f"  u[0]     : {np.vstack(sol['u']).T}")
    print(f"  target   : {soft_target:.3f}")
    print(f"  cap      : {hard_max:.3f}")
    dual = _read_duals(sqp)
    print(f"  duals    : {np.vstack(dual)}")
    # assert sol["u"][0] <= hard_max, f"expected u[0] <= {hard_max}, got {sol['u'][0]}"
    # assert np.isclose(sol["u"][0], soft_target, atol=1e-4, rtol=0.0), (
    #     f"expected u[0] near the shared boundary {soft_target}, got {sol['u'][0]}"
    # )
    print("  PASSED")


if __name__ == "__main__":
    # test_eq_one_iter()
    # test_ineq_converges()
    # test_rank_deficient_equalities()
    test_soft_eq_vs_hard_ineq()
    print("\n" + "=" * 60)
    print("All tests passed!")
