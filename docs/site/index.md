# Moto

Moto is a C++20/Python trajectory optimizer. It combines graph-first
multiple-shooting modeling, sparse generated derivatives, a nonsmooth SQP
method, and a stagewise nullspace/Riccati QP solver.

The documentation is organized by task:

- [Install Moto](installation.md) and verify a Release/native build.
- [Build and solve an OCP](tutorials/first_problem.md) with the stage-centric
  Python API.
- Model [manifold states and structured Euler dynamics](modeling/manifolds.md).
- Supply [analytic derivatives](advanced/analytic_derivatives.md) or a
  [custom lifted elimination graph](advanced/lifted_elimination.md).
- Browse the generated [Python API](reference/python.md) and
  [C++ API](reference/cpp.md).

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

installation
tutorials/first_problem
examples
```

```{toctree}
:maxdepth: 2
:caption: Modeling
:hidden:

modeling/manifolds
advanced/analytic_derivatives
advanced/lifted_elimination
```

```{toctree}
:maxdepth: 2
:caption: API reference
:hidden:

reference/python
reference/cpp
```

## Citation

If you use Moto, please cite the Hippo paper published in IEEE Robotics and
Automation Letters ([DOI 10.1109/LRA.2026.3682524](https://doi.org/10.1109/LRA.2026.3682524)):

```bibtex
@article{zhao2026hippo,
  author  = {Haizhou Zhao and Ludovic Righetti and Majid Khadiv},
  title   = {Hippo: High-Performance Interior-Point and Projection-Based Solver
             for Generic Constrained Trajectory Optimization},
  journal = {IEEE Robotics and Automation Letters},
  year    = {2026},
  volume  = {11},
  number  = {6},
  pages   = {6752--6759},
  doi     = {10.1109/LRA.2026.3682524}
}
```
