# Lifted Weighted-Gram Experiment

Date opened: 2026-08-29

## Direct lifted-direction recovery

- [x] Remove `d_l.K` materialization while retaining `d_y.K`.
- [x] Recover Newton `dl` directly from the complete `dx` and `du`.
- [x] Recover correction `dl` directly without the residual term.
- [x] Verify focused C++ tests and both ordinary/acceleration Python paths.
- [x] Measure the hot Go2 effect under controlled threading.

## Remaining acceleration-gap localization

- [x] Measure ordinary lifted and lifted-acceleration with identical hot-run
  conditions.
- [x] Split `ns_factorization` into its existing algorithmic subphases without
  changing their order or algebra.
- [x] Attribute the absolute per-iteration delta to those subphases.
- [x] Remove temporary diagnostics after recording evidence.

## Requested outcome

- [x] Run a strict same-layout benchmark of the current lifted Hessian
  contraction and a direct weighted-Gram implementation.
- [x] Keep the production solver unchanged during the experiment.
- [x] Verify numerical equivalence.
- [x] Report whether measured benefit justifies production integration.

## Workflow

1. Fix Go2 lifted-acceleration dimensions and diagonal panel boundaries.
2. Generate identical aligned stage data for both candidates.
3. Verify block outputs against each other.
4. Warm both candidates.
5. Measure repeated 100-stage traversals with one Eigen thread.
6. Record raw medians and speedup.

For remaining-gap localization:

1. Add timing-only scopes around existing NSP phase boundaries.
2. Rebuild with four jobs and wait for `moto_pywrap` to link.
3. Warm each formulation in-process and compare the final profile report.
4. Repeat independent processes to reject scheduling noise.
5. Remove timing-only code and rebuild before reporting.

## References

- `docs/contracts/experimental_lifted_weighted_gram.md`
- `docs/contracts/lifted_direction_recovery.md`
- `docs/math/lifted_weighted_gram.md`
- `docs/math/lifted_direction_recovery.md`
- `docs/contracts/vocabulary.md`

## Evidence

Temporary benchmark source: `/tmp/moto_lifted_gram_bench.cpp`.

Build:

```text
g++ -std=c++20 -O3 -DNDEBUG -march=native -ffp-contract=fast \
  -fopenmp-simd -DEIGEN_DONT_PARALLELIZE -I/usr/include/eigen3 \
  /tmp/moto_lifted_gram_bench.cpp -Lbuild \
  -Wl,-rpath,/home/harper/Documents/moto/build -lmoto \
  -o /tmp/moto_lifted_gram_bench
```

Runtime used one Eigen/OpenMP/BLAS thread. Each reported sample is the median
of nine hot measurements, each traversing 100 stage buffers for 300 rounds.
Three independent process results were:

| run | generated schedule (us/stage) | full Gram (us/stage) | speedup |
| --- | ---: | ---: | ---: |
| 1 | 3.750133 | 2.994282 | 1.252432 |
| 2 | 3.748787 | 2.992656 | 1.252662 |
| 3 | 3.992300 | 3.182625 | 1.254405 |

The generated-schedule candidate calls the same exported panel helpers as the
whole-graph JIT: ten diagonal-panel scaling calls followed by the two
input-row products and one state-state product. The full-Gram candidate packs
the same response values, performs an upper-triangular symmetric rank update,
mirrors the result, and extracts the same full output blocks.

Maximum absolute difference printed as `0.000000` and was below the benchmark
exit threshold of `1e-11` in every run.

The direct Gram is consistently about 25% faster for this isolated
contraction, saving approximately `0.76 us/stage`, or `0.076 ms` over 100
stages. This is a real but small fraction of the observed lifted-acceleration
gap. The experiment therefore does not justify claiming that weighted Gram
will remove that gap. Production integration remains unapproved and was not
performed.

## Remaining-gap evidence

The controlled comparison used the default 100-stage problem, six solver
workers, one Eigen/BLAS thread, max-iter=1, and two updates in one process.
The first update warmed lazy kernels. Five independent process pairs produced
these median phase times for the second update:

| phase | ordinary (ms) | acceleration (ms) | delta (ms) |
| --- | ---: | ---: | ---: |
| sqp_iter_total | 3.130 | 3.259 | 0.129 |
| solve_direction | 2.437 | 2.557 | 0.120 |
| ns_factorization | 0.955 | 1.026 | 0.071 |
| riccati_recursion | 1.188 | 1.193 | 0.005 |
| post_solve | 0.078 | 0.080 | 0.002 |
| fwd_linear_rollout | 0.084 | 0.087 | 0.003 |
| finalize_primal_step | 0.123 | 0.156 | 0.033 |

Thus ns_factorization and finalize_primal_step account for 0.104 ms,
about 81% of the median iteration delta. The latter executes the same
recovery formula in both formulations, but the acceleration formulation
recovers 30 lifted rows rather than 12.

Temporary per-operation TSC instrumentation was then enabled in the generated
whole-graph kernel and run sequentially over the same 100 stages. Across the
three active-layout kernel variants, average cycles per stage were:

| integrated graph region | ordinary cycles | acceleration cycles | delta cycles |
| --- | ---: | ---: | ---: |
| elimination and solves | 51,598 | 52,146 | 548 |
| lifted Hessian contraction | 21,116 | 28,760 | 7,644 |
| total | 72,714 | 80,906 | 8,192 |

At the measured approximately 2.495 GHz TSC rate, elimination adds only about
0.22 us/stage. Hessian contraction adds about 3.06 us/stage and accounts
for approximately 93% of the integrated-graph delta.

The acceleration contraction contains:

- 30x30 * 30x48 for the lifted Hessian response;
- 12x30 * 30x48 for input and input-state output blocks;
- 36x30 * 30x36 for the full state-state output.

The ordinary formulation has corresponding inner dimension 12. In one
operation-level repetition, the state-state product alone increased from
approximately 1.33 us/stage to 2.22 us/stage.

All temporary NSP and generated-kernel timers were removed. The clean rebuild
used four jobs. Both four-stage Python formulation smoke tests completed
without runtime errors, linear_backend_test passed all 141 assertions, and
git diff --check passed.

## Direct lifted-direction recovery evidence

The NSP sensitivity storage now contains only `d_u` and `d_y`. Newton lifted
directions are recovered as `-l_0 - l_x dx - l_u du`; correction directions
use the same two sparse panel actions without `l_0`. No `d_l.k` or `d_l.K`
read or write remains in the NSP implementation.

The build completed with four jobs. Focused tests passed 141 linear-backend
assertions and 78 semi-implicit-Euler assertions. After rebuilding every test
executable affected by the changed data layout, CTest passed 17/17. The first
CTest attempt used a stale `restoration_test` object with the old
`ns_riccati_data` layout and segfaulted; rebuilding that target resolved the
ABI mismatch.

The 100-stage ordinary lifted Go2 converged in 26 iterations. The 100-stage
lifted-acceleration Go2 converged in 19 iterations. Both used six solver
workers, one Eigen/BLAS thread, configuration velocity `next`, and a maximum
of 100 iterations. Lifted KKT residuals after iterative refinement remained
at approximately `1e-9` or below near convergence.

Five independent hot process pairs used two one-iteration updates per process.
The median `finalize_primal_step` times were:

| formulation | before direct recovery (ms) | after direct recovery (ms) |
| --- | ---: | ---: |
| ordinary lifted | 0.123 | 0.124665 |
| lifted acceleration | 0.156 | 0.136538 |
| acceleration minus ordinary | 0.033 | 0.011873 |

The individual post-change samples were `0.124665, 0.132976, 0.127634,
0.121104, 0.112797` ms for ordinary lifted and `0.150192, 0.136538,
0.137726, 0.125858, 0.130607` ms for lifted acceleration. The acceleration
recovery improved by about `0.019 ms`; ordinary recovery remained within
measurement variation. Temporary profile-output changes were removed.

## Findings

| id | severity | finding | evidence | suggested action | done-status | user-action |
| --- | --- | --- | --- | --- | --- | --- |
| LWG-1 | performance | The remaining integrated-graph delta is not caused by acceleration elimination; it is dominated by the larger lifted Hessian contraction. | Elimination delta 548 cycles/stage versus contraction delta 7,644 cycles/stage. | If authorized, prototype symmetric/block-output contraction lowering; expected gain is bounded by the earlier 0.076 ms OCP microbenchmark. | open | |
| LWG-2 | performance | Final lifted-direction recovery is the second measurable acceleration cost. | Direct recovery reduced the five-pair median acceleration-minus-ordinary recovery delta from 0.033 ms to 0.011873 ms; both full Go2 formulations retained convergence. | Test delayed/direct l step recovery that avoids materializing d_l.K while preserving the existing rollout semantics. | done | y |
