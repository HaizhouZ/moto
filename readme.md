**Atri** is a high-performance multi-threaded trajectory optimizer. It exploits the temporal and spatial sparisty of implicit multiple-shooting formulation and the highly efficient `BLASFEO`.

# Requirements

1. Eigen 3 (> 3.4)
2. CasADi (>3.7 optional)
3. BLASFEO
4. OpenMP
5. libfmt
6. magic_enum
7. CXX compiler that supports `>= C++20`

## Compilation Notes

1. To achieve the optimal performance, please use `-march=native` and `-O3`. 
2. For AMD Ryzen ZEN4 AVX512, please use `GCC >= 13.2`.
3. use `export KMP_AFFINITY=noverbose,granularity=fine,"scatter" OMP_NUM_THREADS={n_threads}`

# Todo list

1. automatic handling of expression derivative
2. arm test
3. per node class data
4. state input respectively node and edge (remodeling)
5. automatic handling of constraints and symbol field (especially for y)
6. vjp AD
7. precompute interface
8. improve speed with constr
9. semi-implicit euler helper (cpp)
10. more model helpers
11. KKT residual
12. line search