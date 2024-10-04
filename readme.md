<p align="center">
  <img src="asset/logo.svg" alt="logo"/>
</p>

**Atri** is a high-performance multi-threaded trajectory optimizer. It exploits the temporal and spatial sparisty of implicit multiple-shooting formulation and the highly efficient `BLASFEO`.

# Requirements

1. Eigen 3 (> 3.4)
2. CasADi (optional)
3. BLASFEO
4. OpenMP
5. libfmt
6. magic_enum
7. CXX compiler that supports `>C++17`

## Compilation Notes

1. To achieve the optimal performance, please use `-march=native` and `-O3`. 
2. For AMD Ryzen ZEN4 AVX512, please use `GCC >= 13.2`.

# List of scheduled features
Core
1. Modularized modelling of generalized expressions (symbolics/functions)
2. Directed graph representation of any causal process without loops, including list, tree and network.
3. Exploiting block sparsity in derivatives
4. Hard equality constraints (implicit dynamics/contacts) and soft equality constraints (kinematics tasks), balance between speed and feasibility, also different semantics
5. CasADi supports

Below are advanced

6. Hessian scaling, useful for better numerical robustness
7. Enough robustness for single-precision, extreme speed
8. precompute and better CasADi supports (automatic sparsity)
9.  Second-order cone inequality constraints (friction cone)
10. Homotopy complementarity constraints
11. Equality constraint infeasibility detection and handling