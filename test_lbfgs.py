#!/usr/bin/env python3
"""
Structured L-BFGS Hessian correction convergence test.

Compares two solvers on the same nonlinear OCP:

  Baseline : second-order state cost + first-order control cost.
             Q_uu = 0 from the control cost; Q_xx ≠ 0 seeds the Riccati
             backward pass.  The missing Q_uu = 0.2·I degrades convergence.

  L-BFGS  : same cost formulation, with structured L-BFGS enabled.
             Pairs (s, y#) capture the per-stage control curvature 0.2·I
             that is invisible to the Riccati recursion, injecting it
             additively into Q_zz and accelerating convergence.

Why split state / control cost?
  With nu = ny = 2 and F_u = 0.2·I, the Schur complement
    M = V_yy − V_yy·F_u·(F_u^T·V_yy·F_u)^{-1}·F_u^T·V_yy = 0
  exactly.  A pure terminal-seeded backward pass gives V_xx = 0 at every
  intermediate stage, making Q_zz = 0 and crashing the LLT solve.
  Adding a second-order state cost ensures Q_xx ≥ 2·I at every stage, so
  V_xx = Q_xx + 0 ≥ 2·I and the backward pass stays well-conditioned.
  L-BFGS then learns only the missing Q_uu = 0.2·I part.

System: nonlinear 2-D, no inequality constraints (clean curvature signal).

  state  x = [x1, x2],   input  u = [u1, u2]

Dynamics:
  x1_next = 0.9·x1 + 0.2·u1 + 0.1·sin(x1)
  x2_next = 0.9·x2 + 0.2·u2 + 0.05·x1·x2

State cost — second-order (seeds Q_xx, stabilises backward Riccati):
  (1 − cos(x1))^2 + x2^2

Control cost — first-order (Q_uu = 0; L-BFGS learns 0.2·I):
  0.1·(u1^2 + u2^2)

Terminal cost — second-order:
  5·((1 − cos(x1))^2 + x2^2)

Initial x0 = [1.5, 0.5], warm-started from open-loop simulation with u=0.
"""

import time

import casadi as cs
import moto
import numpy as np

np.set_printoptions(precision=4, suppress=True, linewidth=120)

# ── parameters ────────────────────────────────────────────────────────────────
a, b = 0.9, 0.2
N = 30  # horizon long enough that per-stage Hessian matters
w_u = 0.5    # large enough that missing Q_uu=w_u*I is significant vs propagated ~0.04·I
# Large initial deviation creates a genuinely nonlinear OCP that requires
# many SQP iterations so L-BFGS has time to inject the missing Q_uu.
x0 = np.array([0.6, 0.4])

# ── symbolic variables ────────────────────────────────────────────────────────
x, xn = moto.sym.states("x", 2)
u_sym = moto.sym.inputs("u", 2)

x1, x2 = x.sx[0], x.sx[1]
x1n, x2n = xn.sx[0], xn.sx[1]
u1, u2 = u_sym.sx[0], u_sym.sx[1]

# ── dynamics ──────────────────────────────────────────────────────────────────
dyn = moto.dense_dynamics.create(
    "nldyn",
    [x, xn, u_sym],
    cs.vertcat(
        x1n - (a * x1 + b * u1 + 0.1 * cs.sin(x1)),
        x2n - (a * x2 + b * u2 + 0.05 * x1 * x2),
    ),
)

# ── state cost — SECOND-ORDER, globally convex quadratic ──────────────────────
# (1-cos x1)^2 has negative Hessian for |x1|>π/2, crashing the LLT for large x0.
# Use a simple quadratic x1^2+x2^2 so Q_xx = 2·I everywhere.  The dynamics are
# still nonlinear (sin, x1·x2 coupling), giving genuine SQP iteration complexity.
state_cost = moto.cost.create(
    "state_cost",
    [x],
    x1**2 + x2**2,
    moto.approx_order.approx_order_second,  # ← provides Q_xx=2·I, always PD
)

# ── control cost — FIRST-ORDER only (Q_uu = 0; L-BFGS must learn 0.2·I) ───────
# approx_order_first: gradient only.  The missing curvature 0.2·I is exactly what
# the structured L-BFGS recovers from the (s, y#) secant pairs.
ctrl_cost = moto.cost.create(
    "ctrl_cost",
    [u_sym],
    w_u * (u1 - 0.1)**2 + w_u * (u2 - 0.2)**2,
    moto.approx_order.approx_order_first,  # ← GN / gradient-only for u
)

ctrl_cost_reg = moto.cost.create(
    "ctrl_cost_reg",
    [u_sym],
    0.01 * (u1**2 + u2**2),
    moto.approx_order.approx_order_second,  # ← small regulariser to keep Q_uu PD
)

# ── terminal cost — SECOND-ORDER quadratic (extra weight on final state) ───────
term_cost = moto.cost.create(
    "term",
    [x],
    5.0 * (x1**2 + x2**2),
    moto.approx_order.approx_order_second,
).as_terminal()

# ── OCP definitions ───────────────────────────────────────────────────────────
prob = moto.ocp.create()
prob.add(dyn)
prob.add(state_cost)
prob.add(ctrl_cost)
prob.add(ctrl_cost_reg)  # small regulariser to keep Q_uu PD

prob_term = prob.clone()
prob_term.add(term_cost)


# ── warm-start trajectory (open-loop u=0 simulation) ─────────────────────────
def _simulate(x0: np.ndarray, N: int) -> list:
    xs = [x0.copy()]
    for _ in range(N):
        xi = xs[-1]
        xs.append(
            np.array(
                [
                    a * xi[0] + 0.1 * np.sin(xi[0]),
                    a * xi[1] + 0.05 * xi[0] * xi[1],
                ]
            )
        )
    return xs


xs_init = _simulate(x0, N)


# ── solver factory ────────────────────────────────────────────────────────────
def _build(enable_lbfgs: bool, max_pairs: int = 10) -> moto.sqp:
    sqp = moto.sqp(n_job=1)
    g = sqp.graph
    n0 = g.set_head(g.add(sqp.create_node(prob)))
    n1 = g.set_tail(g.add(sqp.create_node(prob_term)))
    g.add_edge(n0, n1, N)

    idx = [0]

    def init(d: moto.sqp.data_type):
        i = idx[0]
        d.value[x] = xs_init[i].copy()
        d.value[xn] = xs_init[min(i + 1, len(xs_init) - 1)].copy()
        d.value[u_sym] = np.zeros(2)
        idx[0] += 1

    sqp.apply_forward(init)

    sqp.settings.prim_tol = 1e-6
    sqp.settings.dual_tol = 1e-6
    sqp.settings.comp_tol = 1e-6

    # CRITICAL: preserve μ and IPM slacks across update(1) calls.
    # Without warm_start=True, initialize() resets μ=μ0 on every call,
    # so the IPM never converges when calling update(1) in a loop.
    sqp.settings.ipm.warm_start = False
    sqp.settings.ls.enabled = False
    sqp.settings.ls.max_steps = 5
    sqp.settings.no_except = True

    sqp.settings.lbfgs.enabled = enable_lbfgs
    sqp.settings.lbfgs.max_pairs = max_pairs

    return sqp


# ── solve-and-record helper ───────────────────────────────────────────────────
def _solve_record(sqp: moto.sqp, max_iter: int = 200):
    """Run up to max_iter SQP steps, recording one KKT snapshot per step."""
    kkt = sqp.update(max_iter, verbose=True)
    return kkt


def _extract_traj(sqp: moto.sqp):
    xs, us = [], []

    def grab(d: moto.sqp.data_type):
        xs.append(np.asarray(d.value[x], dtype=float).ravel())
        us.append(np.asarray(d.value[u_sym], dtype=float).ravel())

    sqp.apply_forward(grab)
    return np.array(xs), np.array(us)


# ── run ───────────────────────────────────────────────────────────────────────
MAX_ITER = 200

print("\n" + "=" * 70)
print("L-BFGS convergence test  (split state/ctrl cost, structured L-BFGS)")
print(f"  N={N}, nx=2, nu=2, no inequality constraints")
print(f"  Dynamics: nonlinear  (sin(x1), x1·x2 coupling)")
print("  State cost:   (1-cos x1)^2+x2^2  [2nd-order, Q_xx seeds Riccati]")
print("  Control cost: 0.1*u^2            [1st-order, Q_uu=0, L-BFGS learns 0.2*I]")
print("  Terminal: 5*((1-cos x1)^2+x2^2) [2nd-order]")
print(f"  x0 = {x0}  (F_x[1,1]={a+0.05*x0[0]:.3f} < 1 → stable backward pass)")
print(f"  warm_start=False (μ not preserved across update(1) calls)")
print("=" * 70)

print("RUNNING BASELINE"); sqp_base = _build(enable_lbfgs=False)
t0 = time.perf_counter()
kkt_base = _solve_record(sqp_base, MAX_ITER)
t_base = time.perf_counter() - t0
xs_base, us_base = _extract_traj(sqp_base)

print("RUNNING LBFGS"); sqp_lbfgs = _build(enable_lbfgs=True, max_pairs=10)
t0 = time.perf_counter()
kkt_lbfgs = _solve_record(sqp_lbfgs, MAX_ITER)
t_lbfgs = time.perf_counter() - t0
xs_lbfgs, us_lbfgs = _extract_traj(sqp_lbfgs)