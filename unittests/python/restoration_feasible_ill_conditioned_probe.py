#!/usr/bin/env python3

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from example.toy.restoration_ill_conditioned import build_ill_conditioned_sqp


def main():
    sqp = build_ill_conditioned_sqp(
        control_scale=1e-2,
        enable_restoration=True,
    )
    kkt = sqp.update(40, verbose=False)

    assert kkt.solved, f"expected feasible ill-conditioned probe to solve, got {kkt.result}"
    assert kkt.inf_prim_res < 1e-8, f"expected small primal residual, got {kkt.inf_prim_res}"
    assert kkt.inf_dual_res < 1e-7, f"expected small dual residual, got {kkt.inf_dual_res}"
    assert kkt.num_iter >= 5, f"expected a nontrivial solve, got only {kkt.num_iter} iterations"


if __name__ == "__main__":
    main()
