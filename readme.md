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

# Todo list

1. multipler jacobian product
2. terminal cost handling (nodes of different problem!)

3. CasADi supports