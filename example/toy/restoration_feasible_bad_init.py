#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import build_sqp


def main():
    sqp = build_sqp()
    verbose = os.getenv("MOTO_FEAS_BAD_INIT_VERBOSE", "0") == "1"
    kkt = sqp.update(50, verbose=verbose)

    print("\n=== Feasible But Infeasible Init Probe ===")
    print(f"result     : {kkt.result}")
    print(f"num_iter   : {kkt.num_iter}")
    print(f"prim_res   : {kkt.inf_prim_res:.3e}")
    print(f"dual_res   : {kkt.inf_dual_res:.3e}")
    print(f"comp_res   : {kkt.inf_comp_res:.3e}")
    print(f"solved     : {kkt.solved}")


if __name__ == "__main__":
    main()
