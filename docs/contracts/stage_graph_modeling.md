# Stage Graph Modeling Contract

## Objective

Provide one stage-centric modeling surface from authored stage prototypes to
the graph-owned stages consumed by composition.

## Workflow

1. A user authors an interval prototype with `stage.add(...)`.
2. The user inserts independent stage copies directly into `sqp.stages`, the
   graph-owned `std::vector` exposed in Python by nanobind's standard vector
   binding.
3. `sqp.st`, `sqp.ed`, and `sqp.stages` expose the graph boundaries and the
   ordinary ordered stage container independently. There is no second name for
   either graph boundary.
4. The graph derives each interval's incoming and outgoing boundaries directly
   from neighboring entries in `sqp.stages`; it does not maintain a second
   interval-topology representation.
5. The graph composer lowers boundary terms and realizes solver runtime nodes.

## Term Placement

- Ordinary path dynamics, input terms, mixed terms, and path-state terms belong
  to the interval through `stage.add(...)` and are evaluated on its current
  state.
- Terms that exist only at the initial graph boundary belong to `sqp.st`.
- Terms that exist only at a stage or phase start boundary belong to
  `stage.st`.
- Terms that exist only at a stage or phase end boundary belong to `stage.ed`.
- A terminal term belongs to the stable graph boundary `sqp.ed`; it remains at
  the graph tail when the interval vector is shifted or replaced.

Boundary placement must not be used as a substitute for ordinary path
placement.

For one fixed connected knot, the solver representation is invariant: a
state-only term on the next interval's current `x` and the same term lowered
from that interval's `stage.st` onto the predecessor's `y` must produce the
same value, first- and second-order contribution, and SQP direction after the
`y -> x` transfer. This equivalence assumes the same numerical values for all
non-primal arguments. It does not make the two authoring forms select the same
set of knots across a horizon: interval placement includes every authored
interval occurrence, while start-boundary placement applies only where that
boundary is connected by composition.

Non-primal arguments of a lowered boundary term are read from the runtime
interval that owns the lowered `y`. A time-varying parameter schedule must
therefore be indexed by the physical knot represented by that runtime
interval, not merely by the source stage object's vector index.

An expression handle has one placement identity within a stage. A
mathematically identical running and terminal term must therefore be authored
as two expressions with stable, distinct names; copying a handle does not
create a terminal expression.

## Ownership And Access

- The graph owns every stage pointer inserted into `sqp.stages`.
- `sqp.stages` is the graph's ordinary mutable `std::vector` of stage pointers,
  exposed with nanobind's standard vector binding. Its stage objects remain
  mutable modeling objects.
- Index assignment, deletion, insertion, and append edit the linear stage order
  directly. Each occurrence has a distinct stage pointer; duplicate pointers
  are rejected because they cannot preserve occurrence identity across a
  homogeneous horizon shift.
- Mutating a graph-owned stage invalidates cached composition and runtime
  realization.
- Unmodified copies of one stage share a composition identity. The composer may
  therefore share one immutable composed OCP for equal interior placements;
  editing one copy first gives that copy a new identity.
- `sqp.nodes` exposes composed runtime data and is not a modeling-stage
  accessor.
- `moto.stage()` is the public stage factory. The implementation types backing
  stages and endpoints are not top-level package APIs, and endpoint handles do
  not expose their retained owner.
- `stage.copy(disable=..., enable=...)` creates a phase variant in one step.
  `stage.disable(...)` and `stage.enable(...)` update an existing authored or
  graph-owned stage. The internal active-status configuration object is not a
  user API.

## Composition, Lowering, And Active Status

Each interval is composed in this order:

1. current-stage interval terms
2. graph-start terms on the first interval
3. current stage-end terms lowered from `x` to `y`
4. next stage-start terms, or graph-end terms, lowered from `x` to `y`
5. one combined active-layout resolution and finalization

Enable/disable predicates are resolved in the authored stage; only active terms
are placed, and later OCP copies do not re-evaluate those predicates in another
stage's symbol context.

The composed primal layout is the union of placed terms' computational
arguments. Direct placement preserves identity; lowering maps source `x` status
to paired `y`; active wins when placements share a target symbol.

Composition applies this resolved status in one pass without running generic
dependency pruning. Later explicit status changes may prune a function with no
active computational primal argument, but do not re-run its authored predicate.

Lowered functions retain lineage to the user-defined function and parameter
handles. Runtime function-data lookup accepts either function handle, and its
argument lookup maps source `x` to lowered `y`. This is contextual:
`node.value[x]` always remains the interval's current state.

## Sequential Construction

- A requested horizon is partitioned exactly; zero-length phases are omitted.
- A sequential caller appends independent `stage.copy()` objects directly to
  `sqp.stages`.
- `sqp.stages` contains exactly the requested number of intervals.
- There is no separate stage/phase insertion API or public start-node argument.

## Invariants

- Endpoint lowering is performed only by graph composition.
- Authored endpoint function handles remain valid for composed runtime data
  lookup after lowering.
- Runtime realization never replaces or exposes a different object as the
  authored graph-owned stage.
- Structural editing directly uses native container operations on
  `sqp.stages`. Shift therefore retains the pointer identity of the surviving
  stages and only introduces the explicitly copied tail stages.
- Modeling mutation and solver realization are not concurrent operations.
  Concurrent readers may share an already realized graph.
- Runtime reconciliation preserves one runtime node for each surviving stage
  occurrence even when equivalent occurrences share one immutable composed OCP.
- Building or shifting 500-occurrence topology targets 50 microseconds,
  including graph/composition allocations but excluding stage authoring,
  codegen, runtime-node allocation, and solver work.
