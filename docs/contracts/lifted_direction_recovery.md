# Lifted Direction Recovery Contract

## Objective

Recover the explicit lifted primal direction without materializing a lifted
state-feedback matrix.

## Workflow

1. The parallel postsolve materializes the state feedback `d_y.K` because the
   sequential state rollout consumes it and the correction rollout reuses it.
2. The sequential rollout recovers `dy` and propagates it to the next stage's
   `dx`.
3. After state rollout, input recovery computes the complete `du` from its
   input policy.
4. Lifted direction recovery applies the stored sparse lifted responses
   directly to the complete `dx` and `du`.
5. The Newton direction includes the projected lifted residual. The correction
   direction does not.

## Storage And Ownership

- `d_y.K` is a persistent state feedback matrix owned by NSP postsolve.
- The NSP must not allocate or materialize `d_l.K`.
- `l_x`, `l_u`, and `l_0` remain owned by the generic dynamics/lifting group.
- The linear backend applies `l_x` and `l_u` using their existing sparse panel
  layouts.

## Invariants

- Lifted direction recovery occurs only after the matching complete input
  direction is available.
- Newton and correction recovery use the same linear action and differ only by
  the residual term.
- The authored lifted primal remains explicit; this scheduling change performs
  no nonlinear substitution or condensation.
