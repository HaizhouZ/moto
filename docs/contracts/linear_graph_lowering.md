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
4. Every non-binding operation in every entry must lower to the generated
   whole-kernel program. Failure to lower is a compilation error.
5. A graph instance binds caller panels and persistent workspace once, then
   executes only the generated whole-kernel entry. Factor state persists in
   the generated program across entries and is refreshed by entry zero.

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
