# Linear Backend Compaction

Opened 2026-08-29.

## Requested outcome

- [x] Analyze graph-construction and linear-backend ownership and execution
  paths.
- [x] Remove obsolete, duplicate, and unnecessarily layered logic.
- [ ] Reduce the scoped implementation footprint to approximately one half
  without moving the same logic to uncounted files or compressing formatting.
- [x] Preserve panel sparsity, alignment, factor reuse, artifact identity,
  generated operation order, numerical results, and hot execution speed.
- [x] Validate C++ tests, Python demos, complete lifted Go2 convergence, cold
  preparation time, and hot NSP timing.

## Scope And Baseline

The measured backend/compiler scope at `c54ffd8` was:

- `include/moto/core/linear_backend.hpp`: 559 lines
- `src/core/linear_backend.cpp`: 3462 lines
- `src/core/casadi_mx_graph_translator.hpp`: 55 lines
- `src/core/casadi_mx_graph_translator.cpp`: 3017 lines
- `src/core/casadi_mx_egraph.hpp`: 17 lines
- `src/core/casadi_mx_egraph.cpp`: 516 lines
- `src/core/spmm_analysis.cpp`: 390 lines

The seven-file surface totaled 8016 physical lines. The stage-topology and
composition implementation was audited separately: its two headers and two
implementation files total 338 lines and already have one production path.
Moving implementation into a new file, generated include, macro wall, or
denser formatting does not count as reduction.

After cleanup the backend/compiler scope is 6063 lines: 321 in the public
header, 2661 in the general backend implementation, 55 in the private
translator header, and 3026 in the translator. This is a 1953-line (24.4%)
reduction. The literal half-size target was not reached: the remaining code is
the live static-kernel implementation, not a second compiler path. Reaching
4008 lines requires reducing supported graph semantics, replacing static
generated kernels with a runtime interpreter, or a separate architectural
rewrite; those are not behavior-preserving cleanup.

## Workflow

1. Inventory every public entry point, production caller, test-only caller,
   generated helper, cache, and ownership boundary.
2. Identify the single required graph-lowering path and the non-graph backend
   operations still required by OCP approximation and structured dynamics.
3. Update the lowering contract before changing ownership or removing a
   production path.
4. Delete dead APIs and combine equivalent analysis, source emission, kernel,
   cache, and invocation abstractions.
5. Re-measure source footprint after each coherent deletion; do not trade code
   size against hot execution or cold preparation regressions.
6. Run contract, mathematical, and scenario validation and record the final
   footprint and timing.

## References

- `docs/contracts/vocabulary.md`
- `docs/contracts/linear_graph_lowering.md`
- `docs/contracts/lifted_elimination_graph.md`
- `docs/records/elimination_graph_interface_review.md`
- `include/moto/core/linear_backend.hpp`
- `src/core/linear_backend.cpp`
- `src/core/casadi_mx_graph_translator.hpp`
- `src/core/casadi_mx_graph_translator.cpp`

## Findings

| id | severity | finding | evidence | suggested action | done-status | user-action |
| --- | --- | --- | --- | --- | --- | --- |
| LBC-1 | high/complexity | Graph lowering and the general linear backend had separate graph/general panel-product helper ABIs and interleaved obsolete compiler layers. | Baseline `wc -l` above; the translator and general backend emitted equivalent Eigen panel products through different exported helpers. | Reuse the alignment-specialized general pair helpers from graph-generated kernels and retain one runtime product ABI. | done | y |
| LBC-2 | high/cleanup | Several public compilation APIs had no production caller and existed only for unit tests or not at all. | Caller scan found old SpGEMM, sparse-output product, weighted-Gram, indexed-product, and panel-program APIs outside the production path. | Delete those APIs and test production entry points instead. | done | y |
| LBC-3 | high/architecture | The graph translator and backend both modeled panel programs, exposing translator-private types publicly. | `panel_program_*` and indexed sparse-product types occupied the public backend header. | Keep graph panel programs private to the translator; expose only layouts and executable kernels. | done | y |
| LBC-4 | medium/performance | Replacing static panel calls with an interpreted descriptor would reduce source at the cost of hot execution. | Previous review measured roughly 258 loops; O0 execution regressed from about 23 us to 56 us and static descriptor execution also regressed. | Retain static generated calls. This is the main reason cleanup cannot safely halve the remaining implementation. | accepted | y |
| LBC-5 | medium/architecture | Stage graph topology/composition is already compact and has no duplicate realization path. | `graph_model` and `graph_composer` total 338 lines; mutation, identity tracking, endpoint lowering, and composition caching are each owned once. | Leave this path unchanged; folding it into `problem.cpp` would hide ownership rather than remove logic. | done | y |

## Verification

- Baseline commit: `c54ffd8`.
- Baseline validation: 134 linear-backend assertions, 18/18 CTests, and the
  lifted acceleration/contact Go2 case converging in 19 iterations.
- Baseline final cached Go2 profile: `initialize_total=56.9 ms`,
  `sqp_iter_total=2.892 ms/call`, `ns_factorization=0.814 ms/call`.
- Final Release/native build completed with `-j4`.
- Final C++/Python suite: 18/18 CTests passed; focused backend coverage is
  97 assertions in 35 cases, structured Euler is 83/8, and graph composition
  is 219/25.
- Direct Python runs passed for `initial_state_optimization.py` and
  `restoration.py`.
- Lifted acceleration/RNEA/contact Go2 converged in 19 iterations on every
  final run. Three cached profiles measured `sqp_iter_total` at 3.056, 3.071,
  and 3.117 ms/call (median 3.071 ms), with `ns_factorization` at 0.862,
  0.870, and 0.897 ms/call. Initialization was 57.4-59.1 ms.
- A same-machine, same-example detached build of baseline `c54ffd8` measured
  `sqp_iter_total` at 3.142, 3.088, and 3.078 ms/call (median 3.088 ms) and
  `ns_factorization` at 0.895, 0.888, and 0.884 ms/call (median 0.888 ms).
  The cleaned implementation is therefore 0.6% faster overall and 2.0%
  faster in factorization in this paired run; there is no measured hot-path
  regression.
