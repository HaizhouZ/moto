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
    p_lb = moto.sym.params("p_lb", 4)
    p_ub = moto.sym.params("p_ub", 4)

    box_numeric = moto.ineq.create(
        "x_box_numeric",
        [x],
        x.sx,
        np.array([-1.0, 0.0, -2.0, 1.0]),
        np.array([2.0, 3.0, 4.0, 5.0]),
    )

    box_mixed = moto.ineq.create(
        "x_box_mixed",
        [x],
        x.sx,
        np.array([-np.inf, 0.5, -1.0, -np.inf]),
        np.array([2.0, np.inf, 3.0, 4.0]),
    )

    g = cs.vertcat(x.sx[0] + x.sx[1], cs.sin(x.sx[2]))
    box_nonlinear = moto.ineq.create(
        "g_box",
        [x],
        g,
        np.array([-1.0, -0.2]),
        np.array([1.0, 0.8]),
    )

    box_symbolic = moto.ineq.create(
        "x_box_symbolic",
        [x, p_lb, p_ub],
        x.sx,
        p_lb.sx,
        p_ub.sx,
    )

    sel = cs.vertcat(x.sx[0], x.sx[3])
    box_linear_slice = moto.ineq.create(
        "x_box_linear_slice",
        [x],
        sel,
        np.array([-1.0, 2.0]),
        np.array([4.0, 5.0]),
    )

    print("box_numeric.dim      =", box_numeric.dim)
    print("box_mixed.dim        =", box_mixed.dim)
    print("box_nonlinear.dim    =", box_nonlinear.dim)
    print("box_symbolic.dim     =", box_symbolic.dim)
    print("box_linear_slice.dim =", box_linear_slice.dim)


if __name__ == "__main__":
    main()
