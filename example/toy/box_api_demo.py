#!/usr/bin/env python3

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import casadi as cs
import moto
import numpy as np


def main():
    x = moto.sym.states("x", 4)[0]
    dt = moto.sym.params("dt", 1)
    p_lb = moto.sym.params("p_lb", 4)
    p_ub = moto.sym.params("p_ub", 4)

    boxes = [
        moto.ineq.create(
            "x_box_numeric",
            [x],
            x.sx,
            np.array([-1.0, 0.0, -2.0, 1.0]),
            np.array([2.0, 3.0, 4.0, 5.0]),
        ),
        moto.ineq.create(
            "x_box_mixed",
            [x],
            x.sx,
            np.array([-np.inf, 0.5, -1.0, -np.inf]),
            np.array([2.0, np.inf, 3.0, 4.0]),
        ),
    ]

    g = cs.vertcat(x.sx[0] + x.sx[1], cs.sin(x.sx[2]))
    boxes.append(moto.ineq.create("g_box", [x], g, np.array([-1.0, -0.2]), np.array([1.0, 0.8])))
    boxes.append(moto.ineq.create("x_box_symbolic", [x, p_lb, p_ub], x.sx, p_lb.sx, p_ub.sx))
    boxes.append(moto.ineq.create("dt_box_scalar", [dt], dt.sx, 1e-3, 0.1))

    sel = cs.vertcat(x.sx[0], x.sx[3])
    boxes.append(moto.ineq.create("x_box_slice", [x], sel, np.array([-1.0, 2.0]), np.array([4.0, 5.0])))

    print("native box inequality examples")
    for box in boxes:
        print(f"  {box.name:<16} dim={box.dim}")

    try:
        moto.ineq.create("bad_primal_bound", [x], x.sx, x.sx - 1.0, x.sx + 1.0)
    except RuntimeError as exc:
        print("  bad_primal_bound rejected:", str(exc).splitlines()[0])
    else:
        raise AssertionError("box bounds that depend on primal variables should be rejected")


if __name__ == "__main__":
    main()
