# Examples

Run the maintained examples from the repository root:

```bash
python example/toy/run.py
python example/toy/initial_state_optimization.py
python example/toy/lifted_input.py
python example/toy/lifted_sparse_elimination.py
python example/toy/restoration.py
python example/arm/run.py
python example/quadruped/run.py
python example/quadruped/run.py --acceleration-control
python example/quadruped/mpc.py
```

`--display` starts a Viser server and replays the optimized URDF trajectory in
the browser. MeshCat is not used.

The source is organized into
[`toy`](https://github.com/HaizhouZ/moto/tree/main/example/toy),
[`arm`](https://github.com/HaizhouZ/moto/tree/main/example/arm), and
[`quadruped`](https://github.com/HaizhouZ/moto/tree/main/example/quadruped)
examples.
