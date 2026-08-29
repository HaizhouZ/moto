# Installation

## Requirements

Moto requires:

1. Eigen 3.4+
2. CasADi 3.7+
3. BLASFEO
4. OpenMP
5. libfmt
6. magic_enum
7. A C++20 compiler

Create a conda environment containing the C++ and Python dependencies:

```bash
conda create -n moto python=3.11 casadi eigen magic_enum fmt re2 nanobind \
  nlohmann_json pinocchio example-robot-data \
  example-robot-data-loaders mujoco libblasfeo -c conda-forge
conda activate moto
python -m pip install "viser[urdf]"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

## Build and install

Use a Release/native build for representative solver performance:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DWITH_NATIVE_OPT=ON
cmake --build build -j4
cmake --install build
ctest --test-dir build --output-on-failure -j4
```

Wait for `moto_pywrap` to finish linking before importing `moto` from Python.
A first example run may also compile CasADi and linear-backend artifacts, so it
should not be used as a hot solver benchmark.

## BLASFEO discovery

The conda setup installs BLASFEO with:

```bash
conda install -c conda-forge libblasfeo
```

No BLASFEO path variable is needed in an activated conda environment. Moto
first consumes BLASFEO's CMake config target and otherwise searches for
`blasfeo.h` and `libblasfeo` below `$CONDA_PREFIX/include` and
`$CONDA_PREFIX/lib`. A successful configure prints either:

```text
-- Found BLASFEO via config target: blasfeo
```

or the fallback library path:

```text
-- Found BLASFEO: .../lib/libblasfeo.so
```

For an installation outside conda and standard system prefixes, pass its root:

```bash
cmake -S . -B build \
  -DBLASFEO_ROOT=/opt/blasfeo \
  -DCMAKE_BUILD_TYPE=Release
```

The root must contain `include/blasfeo.h` and either
`lib/libblasfeo.so` or `lib/libblasfeo.a`. Installed Moto packages ship the
same finder, so downstream CMake projects can use `find_package(moto REQUIRED)`.

Compiler architecture flags should be consistent across Moto, CasADi,
Pinocchio, and BLASFEO. GCC 13.2+ is recommended for Zen 4 AVX-512.
