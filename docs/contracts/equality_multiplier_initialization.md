# Equality Multiplier Initialization Contract

## Objective

Initialize equality multipliers by solving one local SQP direction on an
overlay runtime while preserving the authored primal point and inequality
state.

## Overlay Construction

1. Dynamics, lifted equalities, existing soft equalities, inequalities, costs,
   and their immutable finalized function data are shared with the source
   stage formulation.
2. Hard `__eq_x` and `__eq_xu` constraints are replaced by equality-init soft
   constraints in a distinct overlay formulation.
3. If a source stage has no hard `__eq_x` or `__eq_xu` constraints, its overlay
   formulation is structurally identical and reuses the same immutable
   finalized OCP object. The overlay runtime still owns independent primal,
   dual, approximation, workspace, and factor state.
4. Equivalent source stages reuse finalized functions and generated artifacts;
   overlay construction must not regenerate unchanged dynamics or lifting
   graphs.

## Initialization Workflow

1. Synchronize source primal values and equality/inequality multiplier state
   into the overlay runtime.
2. Prepare overlay-local workspace and evaluate the overlay approximation.
   When the overlay formulation shares the source OCP linear profile, attach
   an instance of the same OCP-level integrated graph before the local SQP
   direction is solved; do not enter the per-stage projection fallback.
3. Solve one equality-initialization SQP direction without globalization or
   restoration.
4. Commit equality and soft-equality multipliers to the source runtime.
5. Refresh source derivatives when requested. Do not commit overlay primal or
   inequality state.

## Invariants

- Reusing a source OCP without hard state/input equalities is an immutable
  formulation reuse only; mutable runtime storage is never shared.
- Equality initialization does not alter the nonlinear model or convergence
  thresholds.
- Overlay caching is keyed by model revision, initial-state mode, and equality
  initialization settings.
- Sharing an immutable integrated graph plan does not share its mutable
  workspace, pointer bindings, factor state, or outputs.
