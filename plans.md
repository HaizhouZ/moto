# Architecture Cleanup Notes

## Current Direction

- Keep modeling graph policy in `graph_model`.
- Keep runtime traversal/storage in `linear_runtime_graph`.
- Keep solver internals out of the Python top-level API.
- Prefer stage-owned data flow over solver-local helper layers.
- Treat line-search fallback as explicit policy, not hidden recovery.

## Completed In This Cleanup

- Removed the old public graph-building layer from Python.
- Removed public `graph_model` / `model_node` / `model_edge` exports.
- Replaced user flow with `sqp.graph.add_path(start_node, end_node, edge, n_edges)`.
- Removed stale graph callback/building APIs from bindings.
- Removed commented-out binding and example debug residue.
- Removed the old `jac_sparsity_ops` helper layer.
- Routed remaining panel Jacobian utilities through `func_approx_data`.
- Kept `node_data` as the central stage approximation path.

## Remaining Nontrivial Work

- Replace `settings.no_except` exception suppression with explicit error-code based handling.
- Decide whether the remaining `func_approx_data` panel operations should move lower into `sparse_mat` / `spmm`.
- Continue shrinking direct solver access to low-level constraint state when a clean constraint-owned interface exists.

## Validation Baseline

- Build: `cmake --build build -j8`
- Tests: `ctest --test-dir build --output-on-failure`
- Arm regression: `python example/arm/run.py --max-iter 50 --n-job 10`
- Quadruped regression: `MOTO_DISPLAY=0 MOTO_SQP_MAX_ITER=50 MOTO_SQP_BENCH_RUNS=1 python example/quadruped/run.py`
