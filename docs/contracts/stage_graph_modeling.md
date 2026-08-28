# Stage Graph Modeling Contract

## Objective

Provide one stage-centric modeling surface from authored stage prototypes to
the graph-owned stages consumed by composition.

## Workflow

1. A user authors an interval prototype with `stage.add(...)`.
2. The user copies prototypes directly into `sqp.stages`, the graph-owned
   `std::vector` exposed in Python by nanobind's standard vector binding.
3. `sqp.st`, `sqp.ed`, and `sqp.stages` expose the graph boundaries and the
   ordinary ordered stage container independently.
4. The graph composer lowers boundary terms and realizes solver runtime nodes.

## Term Placement

- Ordinary path dynamics, input terms, mixed terms, and path-state terms belong
  to the interval through `stage.add(...)` and are evaluated on its current
  state.
- Terms that exist only at the initial graph boundary belong to
  `sqp.start_node`.
- Terms that exist only at a stage or phase start boundary belong to
  `stage.st`.
- Terms that exist only at a stage or phase end boundary belong to `stage.ed`.
- A terminal term belongs to the stable graph boundary `sqp.ed`; it remains at
  the graph tail when the interval vector is shifted or replaced.

Boundary placement must not be used as a substitute for ordinary path
placement.

An expression handle has one placement identity within a stage. A
mathematically identical running and terminal term must therefore be authored
as two expressions with stable, distinct names; copying a handle does not
create a terminal expression.

## Ownership And Access

- The authored prototype remains independent from every graph-owned copy.
- The graph owns every stage pointer inserted into `sqp.stages`.
- `sqp.stages` is the graph's ordinary mutable `std::vector` of stage pointers,
  exposed with nanobind's standard vector binding. Its stage objects remain
  mutable modeling objects.
- Index assignment, deletion, insertion, and append edit the linear stage order
  directly. Existing graph-owned pointers retain identity; newly supplied
  stages must be independent copies, and duplicate pointers are rejected when
  the graph is next realized.
- Mutating a graph-owned stage invalidates cached composition and runtime
  realization.
- `sqp.nodes` exposes composed runtime data and is not a modeling-stage
  accessor.

## Sequential Construction

- A requested horizon is partitioned exactly; zero-length phases are omitted.
- A sequential caller appends independent `stage.copy()` objects directly to
  `sqp.stages`.
- `sqp.stages` contains exactly the requested number of intervals.
- There is no separate stage/phase insertion API or public start-node argument.

## Invariants

- Endpoint lowering is performed only by graph composition.
- Copying a prototype into the graph does not mutate the prototype.
- Runtime realization never replaces or exposes a different object as the
  authored graph-owned stage.
- Structural editing directly uses native container operations on
  `sqp.stages`. Shift therefore retains the pointer identity of the surviving
  stages and only introduces the explicitly copied tail stages.
