# Lifting Integration Review

Date opened: 2026-08-29

## Requested outcome

- [x] Identify the first local commit that introduced lifting.
- [x] Review the complete lifting change from `origin/dev` through the current
  worktree.
- [x] Resolve blocking correctness, ownership, stale-code, and validation
  findings authorized by the integration request.
- [x] Fold later local lifting work into the first lifting commit.
- [x] Verify that the resulting branch contains one local lifting commit and a
  clean, validated implementation state.

## Workflow

1. Preserve the current worktree and identify the history boundary.
2. Review modeling, elimination graph, linear backend, NSP projection,
   rollout, correction, refinement, and examples as one integrated change.
3. Record findings before changing history.
4. Resolve integration blockers and rerun focused and full validation.
5. Rewrite only the two local commits above `origin/dev`, combining them with
   the reviewed worktree into the first lifting commit.
6. Inspect the final commit and worktree after the rewrite.

## References

- `docs/contracts/lifted_direction_recovery.md`
- `docs/math/lifted_direction_recovery.md`
- `docs/contracts/vocabulary.md`

## Findings

| id | severity | finding | evidence | suggested action | done-status | user-action |
| --- | --- | --- | --- | --- | --- | --- |
| LIR-1 | high | The integrated NSP graph cache keyed lifted stages only by dimensions and panel layouts. Two user elimination graphs with identical shapes and sparsity could therefore reuse the first graph's algebra. | `nsp_layout_signature` omitted graph identity and the named `compile_graph` path intentionally does not serialize graph expressions. | Derive a content identity from the finalized lifted MX graph, include it in the NSP signature, and use it in both generated projection and integrated-presolve artifact identities. | done | |
| LIR-2 | medium | Repeated fine-grained `system.jac(equation, variable)` calls created duplicate logical MX inputs and duplicate runtime panel bindings. | The Jacobian factory had no block cache; a regression test observed two bindings for one identical equation slice and variable. | Cache blocks by function UID, equation slice, and variable UID, and require one binding in the lifted graph test. | done | |
| LIR-3 | medium | An inequality depending on both predicted state `y` and explicit lifted primal `l` was inferred as `__ineq_x`, even though `l` is interval-local. | The field inference selected `__ineq_xu` only for `has_l && !has_y`. | Classify every ordinary inequality with an `__l` dependency as `__ineq_xu` and add a field-inference test. | done | |
| LIR-4 | test gap | Analytic mixed Hessians had no orientation regression test, despite normalization by symbol UID and storage by primal-field order. | Existing coverage validated analytic Jacobians only. | Compare a rectangular analytic `x,u` Hessian block with symbolic AD and verify the stored `Q_ux` orientation. | done | |

## Evidence

The first local lifting commit is `565626f Support analytic derivatives and
lifted elimination`; `da414e7 Add graph-ready SpMM analysis` is the only later
committed change. They and the reviewed worktree were folded into one commit
above `c7ed4be`.

Release/native validation used four build jobs. Focused linear-backend,
semi-implicit-Euler, and Jacobian-sparsity tests passed; CTest passed 17/17;
the toy lifted elimination demo and quadruped smoke passed; and the default
full lifted Go2 problem converged in 26 iterations with a limit of 100.
