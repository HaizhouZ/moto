# Stage Term Placement Equivalence

Opened 2026-08-29.

## Requested outcome

- [x] Treat the observed `stage.add(x)` versus connected `stage.st(x)`
  convergence difference as an implementation incompleteness, not as an
  example-formulation choice.
- [x] Locate the first non-equivalent value, gradient, Hessian, projection, or
  Riccati contribution at a connected stage boundary.
- [x] Make equivalent state-only terms produce equivalent SQP/QP algebra after
  graph lowering.
- [x] Add focused regression coverage for value, first-order, second-order,
  and solver-direction equivalence.
- [x] Verify the 100-stage lifted-acceleration Go2 formulation through both
  authoring placements and require the same convergence behavior.
- [x] Do not accept the one-step `q_nom` shift as a sufficient explanation for
  the observed 19-to-100 iteration divergence; compare a fully matched Go2
  problem and locate the first numerical divergence.

## Workflow

1. Inspect the stage-graph contract, endpoint-lowering contract, graph
   composer, approximation storage aliases, and NSP traversal.
2. Build a minimal two-stage oracle with the same state-only expression placed
   on interval `x` and on the connected start boundary.
3. Compare composed fields and runtime value/Jacobian/Hessian storage before
   comparing the SQP direction.
4. Fix the authoritative lowering or solver ownership layer; do not retain an
   example-only workaround as the resolution.
5. Run focused C++ tests, complete CTest, and both Go2 authoring variants.

## References

- `docs/contracts/stage_graph_modeling.md`
- `docs/contracts/vocabulary.md`
- `include/moto/ocp/graph_model.hpp`
- `src/ocp/graph_composer.cpp`
- `include/moto/ocp/impl/lag_data.hpp`
- `src/solver/nsp_impl/presolve.cpp`
- `src/solver/nsp_impl/backward.cpp`

## Findings

| id | severity | finding | evidence | suggested action | done-status | user-action |
| --- | --- | --- | --- | --- | --- | --- |
| STP-1 | high/correctness | The integrated lifted presolve graph silently lost the accumulated `Q_xx/Q_xx_mod` contribution to `V_xx`; placing the same term on the connected boundary stored it in `Q_yy` and bypassed the defect. | In the fully matched Go2 A/B, moving only the cost made the first refinement `y` stationarity error grow from `1.16e-11` to `2.80e2`. `MOTO_VERIFY_NSP_GRAPH=1` reported `Qzz=0`, `z0K=1.39e-17`, and `Vxx=200`, isolating the mismatch to the state Hessian. The sparse matrix can contain additive overlapping panels, while one MX input has only one value per structural coordinate. | Keep additive `Q_xx/Q_xx_mod` accumulation on the sparse-panel dense-write path and let the integrated graph produce the remaining Schur terms. Retain a lifted-presolve regression with overlapping state Hessian panels. | done | y |

## Evidence

- Initial scenario isolation changed only the placement of the state cost and
  joint limits; elimination graph code and dispatcher binaries were held
  fixed.
- The first scenario isolation was insufficient: placement also determines
  which physical knot consumes each node-local parameter value. The dedicated
  same-knot oracle removed that confounder and found no value/derivative/NSP
  discrepancy.
- A full 100-stage Go2 A/B subsequently removed the same confounder by matching
  each physical knot's reference and support, including an independent terminal
  reference. The 19-versus-100 divergence remained, proving that the reference
  shift is not a sufficient explanation and that the richer solver path still
  contains a placement-equivalence defect.
- Splitting cost and joint-limit placement isolated the dominant failure to the
  state-cost Hessian path. Before the fix, the first Newton step changed from
  `23.73` to `380.60`; moving only joint limits did not reproduce that failure
  once the first-stage support was matched.
- The integrated presolve graph now computes the Schur terms and the existing
  sparse-panel dense-write kernels accumulate `Q_xx` and `Q_xx_mod` into
  `V_xx`. The graph artifact version was advanced so stale generated plans are
  not reused.
- After the fix, all four placement splits had identical initial KKT data,
  Newton step, refinement residual, and first line-search result. The complete
  100-stage boundary and direct formulations both converged in 19 iterations,
  with matching final objective and residuals.
- `graph_model_compose_test` now contains a two-stage quadratic same-knot
  regression. All 219 assertions in its 25 test cases passed after the
  addition.
