**Moto** is a high-performance multi-threaded trajectory optimizer. It exploits the temporal and spatial sparisty of implicit multiple-shooting formulation and the highly efficient `BLASFEO`.

# Requirements

1. Eigen 3 (> 3.4)
2. CasADi (>3.7 optional)
3. BLASFEO
4. OpenMP
5. libfmt
6. magic_enum
7. CXX compiler that supports `>= C++20`

## Compilation Notes

```bash
 conda create -n moto python=3.11 casadi eigen magic_enum fmt re2 nanobind nlohmann_json pinocchio meshcat-python meshcat-shapes example-robot-data example-robot-data-loaders mujoco libblasfeo -c conda-forge
 conda activate moto
 # install blasfeo following official instructions!
 export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
 export LIBRARY_PATH=$CONDA_PREFIX/lib
 export BLASFEO_LIB_DIR=<absolute full path to your blasfeo lib>
 mkdir build
 cd build
 cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
 # it is recommended to use cmake-gui to check the options: cmake-gui .
 make install -j 12
```

1. To achieve the optimal performance, please use `WITH_NATIVE_OPT=ON` and `-O3` (**must be consistent with casadi/pinocchio/blasfeo**) and `CMAKE_BUILD_TYPE=Release`
2. For AMD Ryzen ZEN4 AVX512, please use `GCC >= 13.2`.

## Run
To set the number of threads of OpenMP:
```bash
export KMP_AFFINITY=noverbose,granularity=fine,"scatter" OMP_NUM_THREADS=<number of threads>
```
To run the example:
```bash
python example/arm/run.py
python example/quadruped/run.py
python example/quadruped/mpc.py
```

## Notes:
Due to the limitation of nanobind, the `var` class is not well implemented in python binding. For some libraries such as `Pinocchio`, the `casadi.SX` variables should be passed to their apis via
```python
q = moto.state.param('q')
func(..., q.sx, ...) # for example, the boost binding of pinocchio does not recognize the moto.var (Derived from casadi.SX)
```