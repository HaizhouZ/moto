# Linear Graph Lowering Contract

## Objective

Lower a finalized CasADi MX linear graph into one reusable linear-backend
program without retaining a second runtime evaluator.

## Workflow

1. `compile_graph` batches independent solve and product consumers.
2. The internal linear-algebra optimizer may replace multiplication and
   transpose subgraphs only with algebraically equivalent expressions selected
   from the same exact sparsity and panel-layout analysis.
3. The translator imports the optimized MX graph into its internal operation
   representation, establishes aliases, materializations, persistent factor
   slots, entry schedules, and workspace ownership.
4. Before source emission, alias-compatible views remain views and contiguous
   panel assignments are coalesced across adjacent lowering operations. A
   materialization is retained only when a factorization, a reused branch, or
   the measured product schedule requires owned contiguous storage.
5. Every non-binding operation in every entry must lower to the generated
   whole-kernel program. Failure to lower is a compilation error.
6. A graph instance binds caller panels and persistent workspace once, then
   executes only the generated whole-kernel entry. Factor state persists in
   the generated program across entries and is refreshed by entry zero.

## Artifact Preparation

1. One finalized artifact identity produces one immutable lowered plan and one
   generated dispatcher artifact. A supplied identity covers the complete MX
   graph, including factorization metadata, so repeated cache lookup does not
   reserialize that graph. A legacy named cache may serialize once to recover
   its old key; successful recovery publishes the same plan under the stable
   key used by later processes.
2. Equivalent stage instances share that plan and dispatcher. They allocate
   only their mutable workspace and factor state.
3. Generated dispatchers call precompiled numerical helpers. Their compilation
   must use the least optimization work that preserves measured runtime
   performance; numerical helper optimization is owned by the main build. The
   level is a measured property, not an assumption based on source size.
   Fixed-stride panel loops carry an explicit SIMD contract. MX dispatchers use
   `-O1` plus loop vectorization, an unlimited vector cost model, and strict
   aliasing; other linear-backend JIT artifacts retain their own independently
   measured compiler policy.
4. Cache layers are explicit: an optimized-DAG hit does not repeat MX
   optimization, while a fully lowered-plan hit does not repeat graph
   translation or source compilation. Neither operation is repeated per stage
   instance.
## Ownership

- CasADi owns symbolic DAG identity and symbolic sparsity metadata.
- The linear backend owns panel layout analysis, algebraic extraction,
  generated source, factor state, and runtime execution.
- The caller owns input and output panel storage.
- A graph instance owns or borrows its persistent sparse workspace.

## Invariants

- The optimizer and translator are internal implementation details, not public
  modeling APIs.
- Inputs already backed by sparse panels are bound directly and are not
  regenerated or repacked unless a solve requires dense factor/RHS storage.
- Common factors and graph branches are represented once and reused.
- Runtime execution has one production path; there is no interpreted algebra
  fallback with different storage or scheduling semantics.
- Generated graph identity remains stable for equivalent finalized artifacts.
- Compiler options that affect generated binary behavior participate in cache
  identity; changing them cannot silently reuse an incompatible artifact.
