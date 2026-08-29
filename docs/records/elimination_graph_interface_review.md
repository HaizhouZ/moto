# Elimination Graph Interface Review

Opened 2026-08-29.

## Requested outcome

- [x] Review the user-facing friendliness of the elimination graph interface.
- [x] Trace authoring, validation, sparsity, parameter, factorization, and runtime-value workflows.
- [x] Record concrete usability findings without changing the production interface.
- [x] Measure current JIT execution speed and identify interface choices that force duplicate work or dispatch.
- [x] Replace the primary interface with handle-indexed blocks and one canonical solve.
- [x] Configure subconstraints and elimination as one value operation.
- [x] Remove generated runtime flag dispatch and validate the integrated hot path.
- [x] Reduce first-process initialization and uncached dispatcher compilation.
- [ ] Separate runtime-graph preparation, equality initialization, and first execution in profiling.
- [x] Persist and reload the final lowered panel plan rather than only the optimized MX DAG.
- [x] Attach the OCP integrated graph to the equality-init runtime before its local SQP solve.
- [ ] Remove irrelevant dispatcher vectorization passes without changing hot execution speed.
- [ ] Remove generated elementwise loops for contiguous panel copies and zero fills.
- [ ] Coarsen panel programs before emission so the generated dispatcher is
  call glue while preserving the Go2-sized fused-reference hot speed.
- [x] Re-evaluate `-O2` plus explicit SIMD on the current coarsened dispatcher:
  require `<0.5 s` compilation per Go2 artifact and at most `2%` hot
  `ns_factorization` regression before production adoption.
- [x] Validate the adopted dispatcher policy on the 100-stage lifted-
  acceleration Go2 formulation: cold/cached initialization, hot
  `ns_factorization`, and convergence within 100 iterations.
- [x] Compact the pending elimination-graph commit without changing its public
  interface, solver algebra, generated operation order, or verified Go2 result.
- [x] Optimize linear-backend plan construction and JIT preparation time while
  preserving generated operations, hot execution speed, and numerical results.

## Workflow

1. Read the public C++/Python surface and existing lifted contracts and math.
2. Compare toy and Go2 authoring paths with the declared elimination model.
3. Exercise discoverability, error behavior, parameter access, and graph validation.
4. Benchmark generated elimination against its fused reference and inspect integrated stage scheduling.
5. Record findings and suggested actions.
6. Implement the redesign authorized on 2026-08-29 and close findings only after verification.
7. Measure plan lookup, MX optimization/lowering, source preparation,
   compilation/loading, and graph instantiation separately; optimize the
   dominant cold and cached paths; re-run hot and scenario checks.

## References

- `docs/contracts/vocabulary.md`
- `docs/contracts/linear_graph_lowering.md`
- `docs/contracts/lifted_direction_recovery.md`
- `docs/contracts/equality_multiplier_initialization.md`
- `docs/math/lifted_direction_recovery.md`
- `include/moto/ocp/lifted.hpp`
- `bindings/definition/functional.cpp`
- `example/toy/lifted_sparse_elimination.py`
- `example/helpers.py`

## Findings

| id | severity | finding | evidence | suggested action | done-status | user-action |
| --- | --- | --- | --- | --- | --- | --- |
| EGI-1 | high | The result requires four independently authored inverse responses but validates only their shapes, so projection and action algebra can silently disagree. | `lifted.elimination` takes `response_x/u/residual/action` separately; `src/ocp/problem.cpp:747-764` performs shape checks only. A Python graph with zero projections and identity `response_action` finalized successfully (`incoherent_graph_accepted`). | Make one user-provided linear solve/action the canonical result and derive all responses and the transpose action from it; optionally retain the raw response constructor as an expert API. | done | y |
| EGI-2 | high | Python cannot use the existing named block API and must retain exact raw SX residual expressions for `system.jac`, which is fragile and ambiguous for repeated equal rows. | C++ declares both `block(field, field)` and `block(name, name)` in `include/moto/ocp/lifted.hpp:64-67`, but `bindings/definition/functional.cpp:261-286` binds neither. All examples use `system.jac(raw_sx, variable)`. A valid repeated-row residual produced `ValueError: lifted Jacobian equation is ambiguous`. | Bind identity-based `system.jac(equation_handle, variable_handle)` and `system.residual(equation_handle)` as the normal path; keep raw-SX lookup only as an expert sub-row tool. | done | y |
| EGI-3 | medium | The builder is deferred, multi-shot, and undocumented as pure: parameters do not exist until realization and one three-stage setup invoked the callback six times. | Python probe observed `elimination_parameters` and callback count change from `0/0` before `sqp.nodes` to `1/6` afterward. `derive_elimination_graph()` invokes the stored callback during OCP profile construction (`src/ocp/problem.cpp:688`). | Define the builder as a pure multi-shot callback and canonicalize derived parameters/profile data by layout identity. | open | y |
| EGI-4 | medium | Configuration mixes mutation and value semantics: `add_subconstraint()` mutates while `set_elimination_graph()` returns a new object and leaves the source unchanged. Dropping that return fails only during realization. | `include/moto/ocp/lifted.hpp:149-165` documents different ownership behavior; the examples must reassign. The probe that discarded the return later failed with `dynamics ... requires an explicit lifted elimination graph`. | Add one immutable `with_elimination_graph(builder, subconstraints)` operation and migrate public examples. | done | y |
| EGI-5 | medium | The public surface does not document row/column order, signs, response semantics, callback lifetime, factor reuse, or parameter access; generated stubs provide signatures only. | Repository search finds no public contract/tutorial for `set_elimination_graph`; the authoritative ordering is only inferable from implementation and examples. `build/bindings/moto_pywrap.pyi:342-462` has no semantic docstrings. | Add a user-facing elimination-graph contract and one minimal structured example defining `[dyn; lift]`, `[y; l]`, `h_l^{-1}[h_x,h_u,h]`, action semantics, and parameter lifecycle. | done | y |
| EGI-6 | low | Regularization creation is locally convenient, but `add_diag(number)` hides the created parameter handle and the only discovery API is a post-finalization list. | `block.add_diag` accepts either a symbol or value, while `elimination_parameters` is list-only and initially empty; the sparse toy forces `sqp.nodes` before inspecting it. | Preserve `block.param(...); block.add_diag(param)` as the documented controllable path and expose stable parameter handles. | open | y |
| EGI-7 | low | `intermediates` is mandatory constructor ceremony even when empty, while its storage purpose and user retrieval model are undocumented; the generated `subconstraints` stub also leaks a C++ template type. | Every example passes `[]`; `build/bindings/moto_pywrap.pyi:415-456` requires the positional list and types `subconstraints` as `list["moto::utils::shared<...>"]`. | Make intermediates optional, document persistence, and correct the generated annotation. | done | y |
| EGI-8 | high/performance | The current lifted JIT consumes too much of the iteration budget even though its generated elimination algebra is close to the hand-written equivalent. | The Go2-sized isolated graph measured `24.938 us` versus `24.303 us` fused (`1.026x`). In hot 100-stage/6-worker runs, lifted `ns_factorization` was about `0.90-1.06 ms` per call versus about `0.50-0.52 ms` for explicit RNEA/contact. The hand-written elimination lower bound alone predicts about `0.405 ms` per parallel traversal. | Treat execution cost as an interface invariant: the handle-based/canonical solve API must lower to the same or smaller DAG, batch compatible RHS automatically, reuse factors across projection/action entries, and add no runtime Python or virtual dispatch. Further gains require reducing the structured elimination workload, not merely changing syntax. | done | y |
| EGI-11 | high/correctness | The unified quadruped example changed the standard Go2 state-term support without changing its next-knot reference schedule, and the SQP result state could retain a successful-restoration marker after later exhausting the iteration budget. | With state terms on `stage.add`, `x_i` consumed the node's `q_nom(i+1)`; with the standard `stage.st` placement, the lowered term evaluates `y_i=x_{i+1}` against that reference. The changed formulation did not converge within 100 iterations, while restoring its original support produced a real `Converged!` at iteration 19. A separate same-knot quadratic oracle proved the `x/y` NSP representations equivalent. An unconverged 20-iteration run also exposed the stale restoration-success marker. | Preserve the standard Go2 knot/reference pairing, guard same-knot placement equivalence, and reset the transient restoration result before resuming normal SQP iterations. | done | y |
| EGI-9 | medium/performance | Generated graph calls retain runtime flag dispatch for transpose/alignment even though those flags are compile-time constants, and the integrated graph remains highly fragmented. | The current 132,802-byte integrated source contains 72 dense/dense, 14 sparse/dense, 15 dense/sparse, and one sparse/sparse pair helper call. Each external helper enters `dispatch_graph_flags4()` (`src/core/linear_backend.cpp:503-528`), preventing cross-DSO constant propagation. | Route generated calls to precompiled flag-specialized helper symbols, while keeping matrix dimensions/layout metadata static where available. Benchmark first: isolated evidence suggests hot gain will be modest, but this should reduce branch/dispatch cost without increasing JIT source size. | done | y |
| EGI-10 | medium/performance | Cold graph realization is unsuitable for online compilation and the current profile cannot separate graph compilation from approximation setup. | A cold 100-stage lifted Go2 run spent `4.993 s` in initialization (`4.403 s` in `initialize_setup_eval`) and emitted a 132,802-byte integrated source; the hot repeated update initialized in about `4.7 ms`. The only hot timing exposed is the enclosing `ns_factorization`. | Keep artifact reuse mandatory, shrink generated source through precompiled helpers, and expose compile/load time plus entry-0/action timings separately from solver phases. | open | y |

## Verification

- Read all applicable lifted contracts, math, prior reviews, C++ API, Python bindings/stubs, toy examples, and the Go2 structured graph.
- Existing positive behavior: structural-zero blocks retain shape; sparsity is inferred; local regularization parameters are supported; repeated factor solves share factorization; `spd=True` selects the LLT path; transpose action is derived automatically; output shape errors are explicit.
- Both `python example/toy/lifted_sparse_elimination.py` and `python example/toy/lifted_input.py` passed as baseline scenario checks.
- Initial performance commands used one Eigen/BLAS thread and six SQP workers.
  The explicit/lifted comparison is for `ns_factorization`, not whole-iteration
  convergence because their line-search paths differed. The final dispatcher
  policy is recorded below and supersedes the initial `-O3` baseline.
- The redesigned Go2 graph emitted about 101 KB and 80 specialized pair calls, down from about 133 KB and 102 pair calls. Generated calls contain no runtime pair flags.
- In matched hot 100-stage/6-worker profiles, lifted `ns_factorization` averaged about `0.855 ms` and explicit RNEA/contact about `0.510 ms`, leaving about `0.345 ms` lifted overhead.
- A full 100-stage run and an A/B run using the pre-redesign builder both reached the same existing `infeasible_stationary` result after 50 iterations. The redesign did not introduce that convergence behavior; it remains outside this interface/performance change.
- A cached-artifact fresh process still spent `717.7 ms` in initialization versus `875.6 ms` with the old builder. Direct compilation of the specialized 101 KB dispatcher took `2.05 s` at `-O3`, `0.84 s` at `-O2`, `0.40 s` at `-O1`, and `0.17 s` at `-O0`; dispatcher runtime equivalence remains to be measured before changing production flags.
- Reusing the immutable source OCP when it has no hard `__eq_x`/`__eq_xu`
  removed the redundant equality-overlay finalization. Together with a
  persistent optimized-MX cache, cached-artifact fresh Go2 processes fell from
  `717.7 ms` to `202-212 ms` initialization. A repeated measured split was
  `158.0 ms` OCP graph lowering and `47.4 ms` equality initialization.
- Lowering dispatcher optimization was rejected by execution measurements:
  the Go2-sized graph/fused ratio regressed from about `1.02x` at `-O3` to `1.294x`
  at `-O2` and `1.428x` at `-O1`. Production dispatchers therefore retain
  `-O3`; the optimized MX DAG is cached instead.
- Equality initialization now attaches instances of the same immutable
  OCP-level integrated plan to its independent runtime stages. On the
  100-stage Go2 case, equality initialization fell from about `47 ms` to
  `26 ms`; its actual local solve fell from about `36 ms` to `5.2-5.6 ms`.
- The final lowered panel plan is persisted as versioned CBOR metadata. It
  retains only runtime binding operations and constant CSC addresses; mutable
  workspace and factor state remain stage-local. A cached-artifact fresh Go2
  process initialized in `53.2 ms`, with `20.4 ms` OCP-plan loading and
  `26.0 ms` equality initialization. The Go2-sized hot graph remained
  `1.013x` the fused Eigen reference in the verification run.
- Direct compiler profiling attributes the dispatcher delay to optimization,
  not linking: GCC `-O3` compilation took `2.03 s` and linking `0.01 s` for a
  101 KB Go2 dispatcher; Clang `-O3` took `3.68 s`. GCC `-O3` with loop and
  SLP vectorization disabled took `0.83 s`, while disabling IPA constant
  propagation did not help (`1.94 s`). The dispatcher delegates arithmetic to
  precompiled helpers, so vectorization passes are candidates for removal only
  if the hot graph benchmark remains unchanged.
- Disabling vectorization was rejected: the hot graph/fused ratio regressed to
  `1.248x`. Replacing contiguous copy/fill loops with compiler memory builtins
  was also rejected: the generated source still contained hundreds of
  arithmetic loops, GCC compilation increased to `2.34 s`, and the hot ratio
  moved to `1.053x`. Neither experiment remains in production.
- Compiling the dispatcher globally at `-O2` while forcing SIMD on arithmetic
  loops was also rejected. Cold test wall time fell to `1.53 s`, but the
  Go2-sized graph/fused ratio regressed to `1.127x`; production remains `-O3`.
- Replacing generated loops with one precompiled helper call per loop reduced
  cold wall time but regressed the hot graph/fused ratio to `1.137x` because it
  emitted roughly 864 cross-DSO calls. A static-descriptor executor reduced the
  generated file to data plus coarse calls, but lost constant propagation and
  regressed further to `1.310x`. Both experiments were removed. The remaining
  viable route is to coarsen the lowered panel programs before code emission,
  so the dispatcher becomes glue without turning static operands into runtime
  metadata.
- Explicit SIMD on every generated panel loop recovered the `-O2` hot ratio
  (`1.003x`), but cold wall time remained `2.01 s`, so it did not improve the
  actual problem. At `-O1`, the same source regressed to `1.475x`. Production
  therefore keeps the original `-O3` straight-line kernel until lowering emits
  substantially fewer panel operations.
- Canonicalizing read-only operands through panel-alias chains and recognizing
  periodic instruction groups reduced the isolated Go2-sized dispatcher from
  `87.8 KB` to `11.6 KB`. Its direct GCC `-O3` compilation fell from `5.25 s`
  to `0.26 s`, while five hot graph/fused ratios remained between `1.027x` and
  `1.037x` (about `24.1 us` for the graph).
- The full Go2 integrated graph exposed a second source-expansion problem: one
  sparse solve emitted hundreds of scalar RHS pack/unpack statements. Lowering
  those copies to compressed run metadata plus precompiled backend pack/unpack
  calls reduced each of the two stage-layout dispatchers from about `87 KB` to
  `42 KB` and direct GCC `-O3` compilation from `1.26 s` to `0.90 s`. A fully
  cold small-Go2 `initialize_ocp_linear` fell from `2.54 s` to `1.85 s`.
- Five cached small-Go2 runs measured `ns_factorization` at `0.426-0.445 ms`
  per call after compressed solve packing, within the prior `0.429-0.472 ms`
  range. The generated integrated files still contain about 290 static panel
  loops and 92 precompiled product calls, so dispatcher coarsening remains an
  open item rather than a completed call-glue conversion.
- Compiling distinct stage-layout artifacts concurrently was rejected. The two
  GCC processes competed for the same CPU resources and did not reduce cold
  wall time; the experiment was removed, and artifact preparation remains
  serial and deterministic.
- The post-coarsening compiler A/B used the acceptance conditions now recorded
  in `docs/contracts/linear_graph_lowering.md`; measurements on the former
  101 KB source did not decide the new policy.
- On the current source, the two integrated Go2 dispatchers compiled in
  `0.89-0.91 s` with production `-O3` and `0.30 s` with `-O2` plus explicit
  SIMD. The isolated elimination dispatcher compiled in `0.12 s` with the
  candidate policy. All measurements were sequential on the same host.
- Seven-process 100-stage A/B runs measured a candidate median
  `ns_factorization` time of `2.802 ms` per call. The immediately restored O3
  batch measured `2.805 ms` median (with one `4.557 ms` scheduling outlier),
  so the candidate did not regress the hot NSP path.
- The isolated graph median improved from `24.081 us` with O3 to `23.794 us`
  with O2 plus explicit SIMD. After production adoption, five graph/fused runs
  measured a `23.849 us` graph median and a `1.010x` median ratio.
- Production applies explicit SIMD only to generated fixed-stride panel loops.
  The intermediate O2 dispatcher policy was superseded by the measured O1
  policy below. All other linear-backend JIT artifacts remain O3, and
  nondefault compiler options participate in the artifact key. The lowered-plan
  cache remains version 12.
- A cold 100-stage lifted-contact run measured `735.4 ms` in
  `initialize_ocp_linear`; a cached fresh process measured `15.5 ms`. The two
  integrated production sources compiled directly in `0.30 s` and `0.31 s`.
- Production verification passed 134 linear-backend assertions, all 18 CTests,
  both lifted Python demos, and the 100-stage Go2 scenario. The latter retained
  the currently recorded pre-existing `infeasible_stationary` result after 50
  iterations; this compiler-policy change did not alter solver formulation or
  operation order.
- The acceleration convergence regression was independent of dispatcher
  compilation: both O2 plus explicit SIMD and O3 followed the same failed
  trajectory after the unified example moved state terms to `stage.add` but
  retained a next-knot `q_nom` schedule. Restoring the standard Go2 knot and
  reference pairing through `stage.st` produced a real `Converged!` at
  iteration 19. A separate same-knot oracle later confirmed that this was a
  formulation-support mismatch, not an `x/y` NSP-equivalence defect.
- Seven cached two-factorization acceleration samples measured median
  `ns_factorization` times of `3.092 ms` with O2 plus explicit SIMD and
  `3.108 ms` with O3. A post-fix full converged run averaged `1.209 ms` over 20
  calls; the differing full-run averages are trajectory/scheduling dependent
  and are not evidence of a SIMD defect.
- After a successful restoration, the transient success marker is now cleared
  when normal SQP iterations continue. A deliberately regressed 20-iteration
  formulation therefore returned `exceed_max_iter` with `solved=False`, while
  the existing restoration-only demo retained its successful-recovery result.
- Commit compaction removed the obsolete generic pair dispatcher, unified
  solve copy-run coalescing, and removed temporary subphase profiling. The
  tracked diff fell by 198 net lines. All 18 CTests passed, and the full lifted
  acceleration/contact Go2 case still converged in 19 iterations.
- Named graph keys now use the finalized artifact identity, dimensions, entry
  partition, SPD-factor count, and panel layouts directly. They no longer
  serialize an SPD CasADi function merely to locate an already lowered plan.
  Version-2 named caches are loaded once through their legacy key and copied to
  the stable key, preserving existing artifacts without retaining the
  serialization cost in later processes.
- Current 38 KB Go2 dispatchers compile in about `0.20 s` with
  `-O1 -ftree-loop-vectorize -fvect-cost-model=unlimited -fstrict-aliasing`,
  versus about `0.29 s` with O2. Five isolated hot samples measured the graph
  at `22.25-23.72 us` and the fused Eigen reference at `21.72-23.28 us`; the
  four non-first samples had a `1.018-1.023x` ratio.
- O0 dispatcher compilation reduced the isolated compile interval from about
  `0.09 s` to `0.05 s`, but regressed the hot graph from about `23 us` to
  `56 us` (`2.4x` the fused reference), so it was rejected. A conservative
  generated-loop fusion experiment reduced only 7 of 258 loops and did not
  change integrated compilation time; it was also removed.
- With all non-linear generated artifacts copied and `gen/linear_backend`
  empty, the small Go2 initialization measured `1234.7 ms`; the comparable
  pre-change run measured `1841.2 ms`. Its next fresh process initialized in
  `13.4 ms`. The full 100-stage lifted acceleration/contact Go2 case converged
  in 19 iterations; its final cached fresh process initialized in `56.9 ms`
  and averaged `2.892 ms` over the profiled SQP iteration calls.
- Final verification passed 134 linear-backend assertions, 86 structured-Euler
  assertions, all 18 CTests, and the full 100-stage Go2 convergence scenario.
