# Lifting Sparse-Path Review

Date opened: 2026-08-29

## Requested outcome

- [x] Independently review the integrated lifting implementation again.
- [x] Remove code that is incorrect, stale, redundant, or unreachable.
- [x] Trace every sparse-matrix use path from symbolic sparsity to NSP runtime.
- [x] Report potential performance problems with expected impact and
  implementation difficulty.
- [x] Rebuild and validate all affected C++ and Python paths with at most four
  build jobs.

## Compact compiler refactor follow-up

Requested 2026-08-29:

- [x] Refactor the retained MX translator and linear e-graph implementation
  into a sufficiently compact production path.
- [x] Preserve lifted elimination algebra, sparse-panel ownership, factor
  reuse, generated-kernel behavior, and the existing public modeling surface.
- [x] Remove implementation duplication made unnecessary by the compact path.
- [x] Validate correctness and hot Go2 performance with at most four build
  jobs.

The authorized change is structural only. It does not authorize changes to
the lifted direction equations, NSP traversal, convergence thresholds, or
solver globalization.

## Workflow

1. Re-read the lifting contracts, mathematical identities, vocabulary, and
   previous integration findings.
2. Trace symbolic Jacobian/Hessian sparsity into OCP profiles, physical panel
   allocation, runtime binding, translated graph execution, and NSP consumers.
3. Audit ownership, aliasing, zero/identity/diagonal handling, graph caching,
   materialization, and dense fallbacks.
4. Record each finding before changing production code.
5. Remove or repair only findings whose required change is already authorized
   by this cleanup request and does not alter the documented mathematics.
6. Run focused, full, and scenario validation; leave unresolved performance
   proposals open with implementation difficulty.

## References

- `docs/contracts/vocabulary.md`
- `docs/contracts/lifted_direction_recovery.md`
- `docs/contracts/linear_graph_lowering.md`
- `docs/contracts/experimental_lifted_weighted_gram.md`
- `docs/math/lifted_direction_recovery.md`
- `docs/math/lifted_weighted_gram.md`
- `docs/records/lifting_integration_review.md`

## Findings

| id | severity | finding | evidence | suggested action | done-status | user-action |
| --- | --- | --- | --- | --- | --- | --- |
| LSP-1 | correctness | `sparse_matrix::resize()` rejects a panel whose end is exactly equal to the resized matrix boundary in assertion-enabled builds. | The insertion contract accepts `row + rows <= rows_` and every full-field panel ends exactly at the matrix dimension, but all three resize checks use strict `<`. | Change the checks to `<=` and add a boundary regression test. | done | y |
| LSP-2 | correctness/API | The unconstrained template assignment for `sparse_matrix` silently writes only selected dense panels, ignores row offsets and all diagonal/identity panels, and leaves unmatched panels unchanged. | `operator=(const rhs_type&)` uses only `rhs.middleCols(...)` under `panel.rows_ == rhs.rows()`; no production call site exists. | Remove this misleading unused interface; callers must use explicit backend writes or bound views. | done | y |
| LSP-3 | correctness | A generated lifted projection cache is invalidated by value evaluation but not by Jacobian-only evaluation. A derivative-only refresh can therefore reuse projections and factors built from the previous linearization or parameter value. | `value_impl()` clears `projection_ready`; `jacobian_impl()` is not overridden, while accepted-point and equality-initialization paths call `update_approximation(eval_derivatives)`. | Invalidate after generated Jacobian evaluation and regression-test a parameter change followed by derivative-only refresh. | done | y |
| LSP-4 | cleanup | The lifted path contains a no-op `update_lifting_residual()` API/call and stale commented constructor/debug fragments. | An empty method is called from correction presolve, while the unified dynamics residual update already refreshes both `y/l`; old allocation and factorizer comments no longer describe live code. | Remove them without changing the documented algebra or execution order. | done | y |
| LSP-5 | performance | Integrated presolve is sparse only while the user elimination graph and Hessian operands remain panelized; dense factor/RHS materialization is mandatory at every generic `solve` boundary. | Translator keeps panel views through products, then `materialize_solve_inputs()` creates contiguous dense factor storage and may pack RHS columns before Eigen LLT/LU. | Keep user graphs factored into physically meaningful smaller solves. Automatic structured factor discovery would be high difficulty and is not cleanup scope. | open | |
| LSP-6 | performance | The whole NSP graph is available only for unconstrained stages. Adding ordinary hard constraints falls back to dense nullspace bases plus many separately dispatched sparse/dense products. | `graph_presolve` requires `rank_status::unconstrained && !ncstr`; the fallback in `presolve.cpp` materializes `Z_l`, `l_y_K`, and invokes each Hessian block separately. | Treat constrained NSP graph capture as a separate high-difficulty project because rank-revealing LU/kernel dimensions are runtime-dependent. | open | |
| LSP-7 | performance | Static layout planning is greedy and recognizes only same-row adjacent dense panels, adjacent structured diagonals, and exact additive overlaps. Partial diagonal overlaps and alternative dense tilings remain separate logical panels. | `make_sparse_layout_plan()` merges the first eligible pair and restarts; backend exact-overlap fusion cannot combine arbitrary partial overlaps into one call. | Add diagnostics first; cost-aware retiling is medium-to-high difficulty because fewer calls can increase copied area, lose alignment, or add zero FLOPs. | open | |
| LSP-8 | performance | Generic lifted forward/transpose actions dispatch once per RHS column. | Both generated and integrated `solve_stage_lifted_system` loops copy one column into a fixed one-column graph entry. Current dual recovery uses one column, so hot impact is low today. | Batch action entries only when a real multi-column solver consumer appears; medium implementation difficulty. | open | |
| LSP-9 | performance | The linear e-graph optimizes multiplication association and transpose/identity equivalence only; it does not globally optimize additions, solve placement, panel layout, cache traffic, or parallel scheduling. | Its private saturation pass has associativity, transpose, and identity rules; extraction uses lexicographic scalar-product/panel/temp counts. | Do not present it as a whole-NSP optimizer. Extending it safely is high research/implementation difficulty. | open | |
| LSP-10 | correctness | The implicit `sparse_matrix` copy constructor shares a JIT cache whose bound pointers still address the source matrix; assignment into an empty target also omits the static binding plan. | `jit_cache_` is a shared pointer to cached pointer arrays, while only copy assignment is customized and its empty-target branch does not copy `planned_`. | Define explicit copy/move semantics: copy layout/value/binding state but never the cache; move may transfer the cache with its storage. Add regressions for cached-copy independence and planned binding copies. | done | y |
| LSP-11 | performance/cleanup | NSP initialization precompiles several kernels that have no matching runtime call, and prepares sparse×sparse hard-constraint products under the wrong operand/cache identity. | `prepare_weighted_gram(F_u)` has no runtime `weighted_gram`; `F_x`/`Q_yx` sparse products have no consumer; hard geometry calls `right_multiply(J, P)` but preparation used dense-other requests or `prepare_sparse_product(J, P, times)`. | Remove dead preparations and prepare the exact `right_times` sparse×sparse calls used by `build_lifted_hard_geometry()`. Low implementation difficulty; reduces cold compilation without changing hot algebra. | done | y |
| LSP-12 | performance | Every stage prepares the generic NSP fallback kernels before the OCP-level integrated graph is attached, even when an unconstrained lifted stage will execute the integrated graph instead. | `ns_sqp::initialize()` calls `prepare_linear_backend()` in parallel, then `prepare_ocp_linear_graph()`; integrated `ns_factorization()` skips most prepared fallback operations. Disk/source caches limit duplication, but cold initialization still compiles unused variants. | Attach the OCP graph first and make fallback preparation path-aware. Medium difficulty because correction, KKT, constrained, verification, restoration, and overlay kernels must remain prepared. | open | |
| LSP-13 | cleanup/architecture | The MX translator retains both a generated whole-kernel runtime and a virtual interpreted runtime with independent factor and solve buffers, although every supported test and Go2 graph lowers to the whole kernel. | `casadi_mx_graph_instance::run()` branches on `plan->whole_kernel`; tracing all linear-backend tests and lifted Go2 produced no whole-JIT rejection. | Make successful whole-kernel lowering part of the component contract and remove the duplicate algebra evaluator while retaining panel binding and persistent workspace support. | done | y |
| LSP-14 | cleanup/architecture | The small linear e-graph is exposed as a public core class and split across a public header, implementation, and MX adapter even though only the adapter is a production consumer. | Repository references outside its own implementation are two white-box tests; production reaches it only through `optimize_casadi_mx_graph()`. | Preserve saturation/extraction behavior but make the implementation private to the MX optimizer and replace white-box tests with graph-level behavior tests. | done | y |
| LSP-15 | cleanup/diagnostic | The optional runtime CasADi reference checker cannot model cross-entry persistence: an action deliberately uses the factor and shared branches from entry zero, while the checker reevaluates the complete graph from all current inputs and reports a false mismatch. | `MOTO_CHECK_SPARSE_GRAPH=1 linear_backend_test` failed only in the test that changes factor/branch inputs between presolve and action; ordinary mathematical oracle tests pass. | Remove the invalid duplicate runtime checker; keep graph-level dense-reference tests as the correctness oracle. | done | y |

## Sparse-matrix path

| phase | representation and owner | runtime behavior | sparsity retained? |
| --- | --- | --- | --- |
| Function finalization | CasADi SX derivative generation records dense, diagonal, identity, and panel Hessian/Jacobian blocks in each `generic_func`. Generated callbacks emit only those outputs. | No runtime matrix assembly is involved yet. | Yes, at function-local block granularity. |
| OCP profile | `ocp_base::build_linear_profile()` shifts local blocks into stage-global field coordinates and calls `make_sparse_layout_plan()`. Jacobians use distinct bindings; Hessians use additive bindings so exact overlaps share storage. | One immutable profile is reused by every runtime stage with that OCP. | Yes; partial overlaps remain separate panels. |
| Physical storage | `lag_data` resizes and plans every constraint Jacobian, Hessian/modification block, and projected dynamics matrix. `func_approx_data` and dynamics-specific data bind function outputs directly to planned dense/diag/eye panel storage. | Function callbacks write through `matrix_ref`; there is no per-iteration gather into a full sparse matrix. Packed diagonal segments keep each logical segment aligned. | Yes. Identity panels are storage-free unless made dynamic by scaling. |
| Lifted symbolic graph | `build_linear_profile()` exposes real Jacobian panels as MX leaves, keeps structural zeros as shaped zero blocks, adds parameter leaves, derives `response_x/u/residual/action`, and infers projection/intermediate layouts from MX output sparsity. | CasADi provides DAG and sparsity metadata only. | Yes until a user-authored generic solve or an explicitly dense output. |
| Graph lowering | The MX translator preserves panel aliases, CSE branches, lazy products, sparse RHS columns, and factor reuse. It materializes persistent sparse intermediates in `sparse_matrix` workspace and lowers products to aligned/unaligned panel helpers selected statically from offsets. | The whole graph is one callable entry when all operations lower; factor state persists across presolve/action entries. | Yes for products/views; factor matrices and some RHS values become contiguous dense buffers at solve boundaries. |
| Integrated NSP presolve | One OCP/layout-signature graph combines lifted projection with `Q_zz`, `z_0_K`, and `V_xx` contraction. Stage instances bind persistent raw panels, projection panels, Hessian panels, workspace, and dense Riccati outputs through pointer arrays. | Entry 0 refreshes factors and all presolve outputs; entries 1/2 reuse factors for forward/transpose actions. | Lifted `l_x/l_u` remain sparse. `Z_y`, `y_y_K`, `Q_zz`, `z_0_K`, and `V_xx` are intentionally dense consumers. `Z_l/l_y_K` are not materialized on this path. |
| Generic/constrained NSP | `linear_backend` separately JITs sparse×dense, sparse×sparse, transpose, dense-write, and rowwise kernels cached on matrix/layout identity. | Used for hard-constraint geometry, constrained nullspaces, corrections, rollout, KKT residuals, and recovery. | Sparse operands are retained, but nullspace bases and Riccati work matrices are dense. |

## Performance assessment

| priority | issue | expected effect | implementation difficulty |
| --- | --- | --- | --- |
| 1 | Lifted Hessian contraction grows with all lifted rows and dominates the acceleration-formulation delta. | Hot-path, measurable. The existing isolated weighted-Gram experiment saved about `0.076 ms` per 100-stage traversal, so it is useful but cannot remove the whole gap. | Medium-high. A production version must preserve symmetry, additive Hessian panels, output aliases, and correction semantics; current contract does not authorize adoption. |
| 2 | Integrated graph is disabled as soon as ordinary hard constraints create a constrained input nullspace. | Potentially high for models with such constraints: dense `Z_u/Z_l/l_y` materialization and many dispatches return. No effect on the current unconstrained lifted Go2 presolve. | High. Rank and nullity are runtime results of rank-revealing factorization, so graph shape and storage cannot be fixed from symbolic sparsity alone. |
| 3 | Generic `solve` is a dense boundary even when surrounding operands are sparse. | High if the user gives one monolithic `h_l` solve; low for the current Go2 graph, which manually exposes smaller structured factors. | Low for a user-authored factor graph; high for automatic block-factor discovery and reordering. |
| 4 | Greedy panel planning cannot optimize partial overlaps or trade kernel count against enlarged dense tiles/cache traffic. | Layout-dependent and currently unquantified. It can create extra panel pairs, especially in additive Hessians. | Medium-high. A safe optimizer needs measured/backend-aware costs and alignment constraints, not unconditional merging. |
| 5 | Generic fallback preparation runs before the integrated OCP graph is known. | Cold initialization only. Fresh-process Go2 initialization remains about `0.95 s` even with cached artifacts; only part of that is linear-kernel planning/compilation. | Medium. Reorder graph attachment and classify exactly which correction/KKT kernels still need eager preparation. |
| 6 | Forward/transpose lifted actions are one-column graph entries. | Low now because dual and inverse-transpose consumers are vectors; potentially relevant for future batched refinement/recovery. | Medium. Requires variable/fixed RHS-width graph variants and pointer layouts. |
| 7 | The e-graph searches only multiplication association and transpose/identity forms with a simple static cost tuple. | Small/local improvements only; it cannot choose global solve placement, panel layout, or parallel schedule. | High/research-level to broaden while keeping compile time bounded. |

## Evidence

### Validation

- Release/native build completed with `cmake --build build -j4`; the Python
  extension and stub finished linking.
- `ctest --test-dir build --output-on-failure -j4`: 17/17 passed.
- `linear_backend_test`: 44 test cases and 144 assertions passed.
- `semi_implicit_euler_test`: 7 test cases and 84 assertions passed,
  including derivative-only lifted projection invalidation.
- Four-stage Python smoke tests passed for ordinary lifted contact and lifted
  acceleration formulations.
- Full 100-stage lifted contact Go2 converged in 26 iterations. Excluding the
  first iteration, median profiled iteration time was `2.296 ms`.
- Full 100-stage lifted-acceleration Go2 converged in 19 iterations. Excluding
  the first iteration, median profiled iteration time was `2.668 ms`.
- The existing Go2-sized elimination microbenchmark measured the translated
  graph at `23.958 us` versus `23.344 us` for the hand-written fused reference
  in this run (`1.026x`), indicating that elimination dispatch itself is not
  the main current hot-path gap.

### Cleanup result

- LSP-1, LSP-2, LSP-3, LSP-4, LSP-10, and LSP-11 are resolved.
- LSP-13 is resolved by making whole-kernel generation mandatory and removing
  the interpreted factor/solve/product runtime and its duplicate precompiles.
- LSP-14 is resolved by keeping the same e-graph saturation and extraction
  inside the MX optimizer and deleting its public header and separate PIMPL
  implementation.
- LSP-15 is resolved by deleting the invalid cross-entry runtime reference
  evaluator; dense mathematical oracles remain in the backend tests.
- LSP-5 through LSP-9 and LSP-12 remain design/performance findings; no
  unapproved mathematical or solver-flow change was made.

### Compact compiler evidence

- Retained compiler implementation decreased from 3,696 to 3,220 physical
  lines across the MX translator and e-graph files, a reduction of 476 lines
  (`12.9%`). Two implementation files/public surfaces were deleted.
- The generated runtime is now the only algebra execution path. Runtime
  binding, persistent sparse workspace, direct outputs, and generated factor
  state remain.
- The matrix-chain test still reports `1600 -> 160` scalar products. The
  final Go2-sized elimination microbenchmark measured `23.499 us` for the
  graph and `22.835 us` for the hand-written fused reference (`1.029x`).
- Release/native build completed with `-j4`; CTest passed 17/17.
- Full ordinary lifted Go2 converged in 26 iterations; with six solver workers,
  the final median iteration excluding the first was `2.348 ms`.
- Full lifted-acceleration Go2 converged in 19 iterations; the median iteration
  excluding the first was `2.565 ms`.
