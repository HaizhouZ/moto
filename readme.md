**Atri** is a high-performance multi-threaded trajectory optimizer. It exploits the temporal and spatial sparisty of implicit multiple-shooting formulation and the highly efficient `BLASFEO`.

# Requirements

1. Eigen 3 (> 3.4)
2. CasADi (optional)
3. BLASFEO
4. OpenMP
5. libfmt
6. magic_enum
7. CXX compiler that supports `>= C++20`

## Compilation Notes

1. To achieve the optimal performance, please use `-march=native` and `-O3`. 
2. For AMD Ryzen ZEN4 AVX512, please use `GCC >= 13.2`.
3. use `export KMP_AFFINITY='scatter'` and set the right number of `OMP_NUM_THREADS`

# Todo list

1. refactor solver
2. double integrator test
3. CasADi supports
4. precompute interface
5. quadruped test