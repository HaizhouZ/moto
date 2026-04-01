#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "example" / "toy"))
from run import build_sqp


def main():
    sqp = build_sqp()
    kkt = sqp.update(50, verbose=False)

    assert kkt.result == type(kkt.result).iter_result_success or kkt.solved, (
        f"expected success on feasible bad-init toy problem, got {kkt.result}"
    )
    assert kkt.inf_prim_res < 1e-8, f"expected small primal residual, got {kkt.inf_prim_res}"
    assert kkt.inf_dual_res < 1e-6, f"expected small dual residual, got {kkt.inf_dual_res}"


if __name__ == "__main__":
    main()
