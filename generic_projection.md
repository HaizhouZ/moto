# Generic Projection Plan

## Goal

Design a user-pluggable hard-constraint projection system for the SQP / Riccati solver that:

- supports dense, sparse, and sequential special-purpose backends
- supports both nullspace projection and pseudo-inverse solves
- integrates into the current recursive stagewise solver
- does not force every backend to materialize the same internal representation

The main mathematical requirement is to handle hard equalities of the form

A_k \, \delta u_k + B_k \, \delta x_k + a_k = 0

at each stage `k`, where `A_k` is the stacked hard-constraint Jacobian with respect to the control-like decision block, `B_k` is the stacked Jacobian with respect to the incoming state block, and `a_k` is the stacked residual.

The current implementation realizes this through a dense rank-revealing factorization plus explicit nullspace basis construction. The new design should preserve the same mathematics while allowing other realizations.

## Revision Against Current `nsp_impl`

After reviewing the current implementation in:

- [presolve.cpp](src/solver/nsp_impl/presolve.cpp)
- [backward.cpp](src/solver/nsp_impl/backward.cpp)
- [postsolve.cpp](src/solver/nsp_impl/postsolve.cpp)
- [rollout.cpp](src/solver/nsp_impl/rollout.cpp)
- [ns_riccati_data.cpp](src/solver/nsp_impl/ns_riccati_data.cpp)
- [projection_default_stage.cpp](src/solver/nsp_impl/projection_default_stage.cpp)
- [projection_dense_reference.cpp](src/solver/nsp_impl/projection_dense_reference.cpp)

the most important correction is:

the current solver does **not** project over a generic stage constraint system written directly as

A_k \delta u_k + B_k \delta x_k + a_k = 0

on the raw stage Jacobians.

Instead, it first projects dynamics through the stage-local `y`-Jacobian inverse and then builds the hard-equality elimination system in those projected coordinates.

Concretely, the current stacked equality system is

- Abar = [ s_{y,k} F_{u,k}
           c_{u,k}        ]

- Bbar = [ s_{x,k} + s_{y,k} F_{x,k}
           c_{x,k}                 ]

- abar = [ s_k + s_{y,k} F_{0,k}
           c_k                 ]

where:

- `F_x, F_u, F_0` are the already projected dynamics quantities produced by `update_projected_dynamics()` and `update_projected_dynamics_residual()`
- `s_y, s_x` come from `__eq_x`
- `c_u, c_x` come from `__eq_xu`

This is exactly the current `s_c_stacked`, `s_c_stacked_0_K`, and `s_c_stacked_0_k` construction, now assembled by the helpers in [projection_default_stage.cpp](src/solver/nsp_impl/projection_default_stage.cpp) and consumed from [presolve.cpp](src/solver/nsp_impl/presolve.cpp).

So the projection backend must be defined around this **projected hard-equality operator**, not around raw constraint Jacobians in isolation.

## Feasibility Assessment

The design is feasible, but only under a scoped interpretation of "unify".

What is feasible without sacrificing the current hot path is:

- unify the **elimination-plan abstraction**
- keep the existing dynamics projection as a specialized first stage
- compose one or more hard-equality elimination blocks after that stage
- keep each elimination block numerically specialized

What is not advisable is:

- flattening dynamics projection and all hard-constraint handling into one generic matrix backend
- forcing every backend to materialize one common dense operator or basis

The current solver already demonstrates the right composition pattern:

1. project dynamics through the `__dyn` hooks
2. build the projected hard-equality operator
3. eliminate that operator with the current dense LU/nullspace path
4. run the reduced Riccati recursion

So the right generalization is to treat the node solve as an ordered elimination plan.

The first feasible generalized plan is:

1. `DynamicsProjection`
2. `CustomEqualityElimination(projected subset)`
3. `DefaultEqualityElimination(remaining hard equalities)`

This preserves the current specialized dynamics code path while allowing mixed custom-projection and default-LU behavior.

Important scope limit:

the current code already supports multiple `__dyn` expressions aggregated into one node-level projected dynamics object, but it does **not** yet expose multiple independently scheduled dynamics-elimination blocks. Supporting those later is still feasible, but it requires a second refactor of projected-dynamics storage and `x/y` mapping utilities rather than only a new backend.

## Current Branch Status

This worktree now implements the first usable projector-driven layout path.

What is implemented:

- projector specs live on `ocp_base` and are built through `problem.projector()` plus explicit group handles
- composed interval problems enable layout compilation on the solver-owned clone before `finalize()`
  - `graph_model::compose_interval(...)`
  - direct `edge_ocp::compose(...)`
- finalize-time layout compilation:
  - reorders `__u`
  - reorders hard constraints in `__eq_x` and `__eq_xu`
  - checks incompatible projector ownership
  - emits compiled hard-constraint block metadata
- the existing dense projected hard-equality stage consumes that compiled block metadata automatically
- dense dynamics now tolerate reordered control layout by packing exclusive `u` blocks locally and scattering back to the finalized global order
- Python bindings expose:
  - `problem.projector()`
  - `projector.group()`
  - `group.require_primal(...)`
  - `group.require_constraint(...)`
  - `group.require_before(...)`
  - `problem.compiled_hard_constraint_blocks`

Regression coverage now includes:

- merged node+edge projector requirements during interval compose
- incompatible projector ownership rejection
- propagation of compiled hard-constraint block order into `ns_riccati_data`
- explicit `__eq_x(x,y)` projected-row coverage
- a Python smoke test for the projector API on the composed interval problem

What is still not implemented:

- projector-specific numeric backends
- sparse or sequential projection backends
- recursive staged elimination of constraint subsets
- backend preferences on the public projector API

The older row-block assembly helper is still present internally because the dense equality stage uses it to consume the compiled hard-constraint order, but it is not the public feature anymore. The public feature is now pre-finalize projector-driven layout compilation on the solver-owned composed problem.

## Projector-Driven Layout

The clean target is not "manipulate the stacked Jacobian after assembly".

The clean target is:

1. a projector declares layout requirements
2. the solver OCP creator checks whether all requested projectors are compatible
3. the solver OCP creator applies the compiled primal/constraint reordering on the solver-owned composed clone
4. that clone is finalized once
5. the solver uses the ordinary projected assembly on the already-ordered problem

This shifts ordering into the cheapest and safest layer:

- before flattened indices are frozen
- before dense/sparse blocks are assembled
- without adding per-iteration permutation overhead to the hot path

### Projector Requirements

In the implemented path, a projector declares two kinds of requirements:

- primal ordering requirements
- constraint ordering requirements

Typical primal requirements are:

- keep this `u` subset contiguous
- place group `A` before group `B`
- preserve authored order inside a group

Typical constraint requirements are:

- these hard constraints belong to the projector
- keep these constraint rows contiguous
- place group `contact` before group `closure`
- preserve authored order inside each group

In the current implementation, all requirements are hard requirements:

- incompatible ownership or missing references fail layout compilation
- there is no soft-priority fallback layer yet

### Compatibility Check

The solver OCP creator should gather all active projectors for a realized stage and compile one layout plan.

It should reject incompatible sets, for example when:

- two projectors demand contradictory relative order for the same `u` groups
- two projectors require incompatible hard-contiguous placements for the same constraint rows
- a projector refers to a constraint subset that is not representable after compose
- a projector depends on field semantics that do not survive realization

The output of this step is currently:

- one compiled primal order for `__u`
- one compiled hard-constraint layout for `__eq_x` and `__eq_xu`
- explicit diagnostics when the projector set is incompatible

### Where Reordering Belongs

This reordering now happens:

- on the solver-owned composed interval problem
- after graph compose / realization
- before finalize computes flattened indices

More precisely:

- `__u` is reordered directly on the composed clone before index flattening
- `__eq_x` and `__eq_xu` are each reordered within their own field lists
- cross-field hard-constraint order is carried by compiled block metadata and then consumed by the dense projected equality stage

It should **not** happen:

- on the user-authored problem object
- by rewriting the stacked Jacobian every SQP iteration
- by exposing raw permutations as a public API

### Scope For The First Version

The current supported projector-layout feature targets:

- `__u` primal ordering
- hard-constraint ordering metadata for `__eq_x` / `__eq_xu`

It should not try to reorder:

- `x`
- `y`
- dynamics blocks

until there is a concrete need and a safe graph-level design for those cases.

## Layout Order Versus Recursive Elimination

Because the terminal here does not reliably render LaTeX, the key distinction is written below in plain text.

### 1. Current dense-reference math

After dynamics projection, the current solver builds one stacked projected hard-equality system:

- Abar = [ s_y * F_u
           c_u       ]

- Bbar = [ s_x + s_y * F_x
           c_x             ]

- abar = [ s + s_y * F_0
           c             ]

Then it solves the full stacked system globally:

- Abar * U = Bbar
- Abar * u = abar
- Abar * Z = 0

and uses:

- du = Z * z - U * dx - u

with the induced state step:

- Z_y   = F_u * Z
- Y_p   = F_x - F_u * U
- y_p   = F_0 - F_u * u
- dy    = Z_y * z - Y_p * dx - y_p

This is the current `nsp_impl` structure. In the dense-reference path, the solver still uses this full stacked solve.

### 2. What row/block order means today

Suppose a row permutation `Pi` is applied to the assembled projected system:

- Aperm = Pi * Abar
- Bperm = Pi * Bbar
- aperm = Pi * abar

If the solver still solves the full permuted system globally, then the mathematical problem is unchanged:

- Aperm * U = Bperm
- Aperm * u = aperm
- Null(Aperm) = Null(Abar)

Only the factorization layout changes.

So the currently implemented "ordered blocks" mean:

- the assembled projected-equality rows can be grouped and ordered explicitly
- the dense LU sees that order
- pivoting / cache behavior / sparse heuristics can depend on that order
- but the actual particular solve and nullspace solve are still global on the whole stacked system

This is a layout / factorization-order feature, not yet a recursive elimination algorithm.

### 3. What true staged elimination would mean

Now split the full projected system into two blocks:

- Abar = [ A1
           A2 ]

- Bbar = [ B1
           B2 ]

- abar = [ a1
           a2 ]

Assume block 1 is to be eliminated first. A true staged elimination must compute an admissible parameterization satisfying block 1:

- du = T1 * w - U1 * dx - u1

with:

- A1 * T1 = 0
- A1 * U1 = B1
- A1 * u1 = a1

Then block 2 is not solved in `du` anymore. It is solved in the reduced coordinates `w`:

- (A2 * T1) * w = (B2 - A2 * U1) * dx + (a2 - A2 * u1)

If block 2 is also eliminated, write:

- w = T2 * z - U2 * dx - u2

with:

- (A2 * T1) * T2 = 0
- (A2 * T1) * U2 = B2 - A2 * U1
- (A2 * T1) * u2 = a2 - A2 * u1

and compose:

- du = T1 * T2 * z - (U1 + T1 * U2) * dx - (u1 + T1 * u2)

This is a genuine recursive elimination law.

### 4. Why arbitrary block order is not enough

Arbitrary row reordering does not automatically define a valid staged elimination.

The failure mode is simple:

- if A1 already has full column rank in `u`, then T1 = 0
- block 1 picks one particular affine law:
  - du = -U1 * dx - u1
- then block 2 has no freedom left

For block 2 to remain satisfied, it must be true that:

- A2 * U1 = B2
- A2 * u1 = a2

for the same `U1` and `u1` chosen from block 1.

That only holds when block 2 is already implied by block 1 after projection, or when the staged elimination policy is designed to preserve that compatibility. It is not true for an arbitrary row partition.

So:

- "choose a row order" is easy and safe
- "solve blocks recursively in that order" is stronger and only valid for carefully chosen elimination stages

### 5. Practical conclusion for the current branch

The current implementation in this worktree therefore does the following:

- compile explicit equality-row groups and order
- assemble the projected hard-equality system in that order
- factor and solve the full assembled system globally in the dense-reference backend

It does not yet claim that any user-supplied row grouping is a valid recursive projector.

A future recursive backend should only be allowed to stage-eliminate subsets that come with stronger semantics, for example:

- the existing dynamics projection stage
- a structured constraint family with a known admissible elimination law
- a backend-specific sparse or sequential block policy that can prove compatibility of later blocks

## Staged Elimination Model

The clean mathematical model for a true recursive backend is still an ordered sequence of elimination stages inside one node.

Before stage `i`, suppose the active control-like variable is parameterized as

`dq = T_i * w_i - U_i * dx - u_i`

where:

- `dx` is the incoming state perturbation
- `w_i` is the current reduced coordinate
- `T_i` is the admissible map from reduced coordinates to the current primal space
- `U_i`, `u_i` encode the current particular affine part

Then stage `i + 1` eliminates some new constraint block and produces

`w_i = S_i * w_{i+1} - V_i * dx - v_i`

which composes to

`dq = T_i * S_i * w_{i+1} - (U_i + T_i * V_i) * dx - (u_i + T_i * v_i)`

This is already what the current dynamics/constraints split is doing conceptually:

- dynamics projection builds the first admissible parameterization
- projected hard-equality elimination reduces it further
- Riccati recursion runs on the final reduced coordinate

For future mixed "project some constraints, default-LU the rest" support, this staged model remains the correct composition mechanism, but only when each stage is a semantically valid elimination stage rather than just an arbitrary row block.

## Core Principle

The correct abstraction is not "give me a nullspace basis".

The correct abstraction is:

- solve a particular constrained step
- project arbitrary vectors into the admissible subspace
- solve the reduced Newton system on that admissible subspace
- recover duals consistently

An explicit basis `Z_k` is only one possible implementation of that abstraction.

Two caveats matter:

1. A notation like `A_k^+` should be read as a backend-defined generalized inverse action, not automatically as the Moore-Penrose pseudo-inverse.
2. A notation like `P_k = I - A_k^+ A_k` is only one possible realization of a nullspace projector. It depends on the chosen generalized inverse and is orthogonal only in special cases such as Moore-Penrose or orthonormal-basis constructions.

## Stage Equality Model

For a fixed stage, after dynamics projection, write the hard constraints in condensed form:

A_k \, \delta u_k = - B_k \, \delta x_k - a_k

where:

- `A_k \in \mathbb{R}^{m_k \times n_{u,k}}`
- `B_k \in \mathbb{R}^{m_k \times n_{x,k}}`
- `a_k \in \mathbb{R}^{m_k}`

The admissible control step is any `\delta u_k` satisfying the affine constraint above.

If `A_k` has rank `r_k`, then:

- `r_k = 0`: unconstrained stage
- `0 < r_k < n_{u,k}`: partially constrained stage
- `r_k = n_{u,k}`: fully constrained control block

The generic projection system must classify these cases, because the recursion changes structurally across them.

It must also distinguish:

- which constraints belong to the projected hard-constraint set
- how those constraints are internally ordered for factorization

Those are separate decisions.

One important codebase-specific note:

- if a hard equality is authored as `generic_constr(x, y, ...)` with field left `__undefined`, current finalization classifies it as `__dyn`
- the projected-state-equality path discussed in this note is exercised when the constraint is explicitly kept in `__eq_x`
- the current tests therefore use an explicit `__eq_x` field for the mixed `x,y` projected-state regression

## Particular Plus Nullspace Decomposition

The standard decomposition is:

\delta u_k = \delta u_k^{\mathrm{p}} + \delta u_k^{\mathrm{n}}

with:

A_k \, \delta u_k^{\mathrm{p}} = - B_k \, \delta x_k - a_k

and

A_k \, \delta u_k^{\mathrm{n}} = 0

When an explicit nullspace basis is available, one may write

\delta u_k^{\mathrm{n}} = Z_k z_k, \qquad A_k Z_k = 0

and

\delta u_k^{\mathrm{p}} = - A_k^{+}(B_k \delta x_k + a_k)

so that

\delta u_k = Z_k z_k - A_k^{+}(B_k \delta x_k + a_k)

But the important point is that the solver only needs the *action* of these maps. It does not fundamentally need `Z_k` itself.

## Projection-Operator View

If a backend exposes a generalized right-inverse action `A_k^{+}`, one may define a candidate nullspace projector

P_k := I - A_k^{+} A_k

Then any admissible step can be represented as

\delta u_k = -A_k^{+}(B_k \delta x_k + a_k) + P_k w_k

for an arbitrary free vector `w_k`.

This view is more general than the basis form:

- dense backend: `P_k = Z_k Z_k^\top` if an orthonormal basis is used
- sparse backend: `P_k` may be applied through operator solves without ever forming `Z_k`
- sequential backend: `P_k` may be realized by recursive eliminations or specialized transforms

This is an important representation, but it should not itself be the full mathematical contract of the pluggable system, because:

- for a non-Moore-Penrose generalized inverse, `P_k` depends on the chosen complement
- many efficient backends never materialize `P_k`
- many efficient backends prefer an admissible coordinate map `T_k` such that

\delta u_k^{\mathrm{n}} = T_k s_k, \qquad A_k T_k = 0

So the abstract object should be an admissible-step representation, not necessarily an explicit projector.

## Reduced Newton System

Let the local quadratic model in `(\delta x_k, \delta u_k, \delta y_k)` be represented in the current solver notation. After projected dynamics elimination, the part relevant to the control block is a quadratic model in `\delta u_k`, together with coupling terms to `\delta x_k` and `\delta y_k`.

The projection system must produce the reduced Newton equation on the admissible subspace:

H_k^{\mathrm{red}} s_k = - g_k^{\mathrm{red}}

where:

- `s_k` is the free projected step variable
- `H_k^{\mathrm{red}}` is the Hessian restricted to the admissible subspace
- `g_k^{\mathrm{red}}` is the projected gradient including particular-solution corrections

In the explicit-basis realization:

H_k^{\mathrm{red}} = Z_k^\top H_k Z_k

g_k^{\mathrm{red}} = Z_k^\top \left(g_k - H_k A_k^{+}(B_k \delta x_k + a_k)\right)

In a more general admissible-coordinate form, the same reduced system is represented by a backend-defined map `T_k`:

H_k^{\mathrm{red}} = T_k^\top H_k T_k

If a backend prefers to work with a projector `P_k`, then one may think of the reduced operator as the restriction of

P_k^\top H_k P_k

to the admissible subspace, but that full-space operator is singular whenever constraints are active. So the solver still needs either:

- an explicit coordinate chart `T_k`, or
- a direct constrained reduced solve

rather than only a projector action.

The solver should depend only on the existence of:

- reduced Hessian action
- reduced solve
- reconstruction of the primal step

not on whether the backend used `Z_k`, a sparse QR, or a sequential elimination.

## Recursive Projection Plan

The recursive solver needs more than a one-shot projection. It needs the affine dependence of the admissible step on the incoming state perturbation.

In the current implementation, the reduced stage solve is built after projected dynamics elimination, so the hard constraints imply an affine admissible control law in the projected coordinates of the form

\delta u_k = K_k \, \delta x_k + k_k

for an affine map consisting of:

- a feedback-like gain from `\delta x_k`
- an offset induced by the current residual
- possibly an added free nullspace correction

The current solver expresses this in the form analogous to:

\delta u_k = Z_k z_k - U_{y,k} \, \delta x_k - u_{y,k}

\delta y_k = Z^y_k z_k - Y_k \, \delta x_k - y_k

More precisely, the current dense implementation computes particular affine maps by solving

\bar A_k U_k^{\mathrm p} = \bar B_k,
\qquad
\bar A_k u_k^{\mathrm p} = \bar a_k,

which correspond to the current `u_y_K` and `u_y_k`, and then propagates them through projected dynamics to obtain the current `y_y_K` and `y_y_k`.

The exact sign convention should be taken directly from the solver reconstruction formulas in [postsolve.cpp](src/solver/nsp_impl/postsolve.cpp), because the implementation stores the propagated particular part and the final affine law separately and combines them as

\delta u_k = T_k s_k - u_k^{\mathrm p}(\delta x_k),
\qquad
\delta y_k = T_k^y s_k - y_k^{\mathrm p}(\delta x_k),

rather than storing one already-combined feedback matrix.

U_k^{\mathrm{p}} := A_k^{+} B_k, \qquad u_k^{\mathrm{p}} := A_k^{+} a_k

and then optimizes only over admissible corrections.

The generic recursive projection plan is:

1. Stage constraint solve

A_k U^{\mathrm{p}}_k = B_k, \qquad A_k u^{\mathrm{p}}_k = a_k

so that

\delta u_k^{\mathrm{p}} = - U^{\mathrm{p}}_k \delta x_k - u^{\mathrm{p}}_k

2. Admissible free correction

Represent the remaining freedom by an internal free variable `s_k`, via either:

- basis form: `\delta u_k^{\mathrm{n}} = Z_k s_k`
- projector form: `\delta u_k^{\mathrm{n}} = P_k w_k`
- sequential form: `\delta u_k^{\mathrm{n}} = T_k s_k` for some backend-defined admissible map `T_k`

For the solver, the sequential form is the most general one. Basis and projector forms are special cases of this admissible-coordinate view.

3. Dynamics propagation

If the local projected dynamics map is

\delta y_k = F^x_k \delta x_k + F^u_k \delta u_k + f_k

then substitution gives

\delta y_k =
\left(F^x_k - F^u_k U^{\mathrm{p}}_k\right)\delta x_k
- F^u_k u^{\mathrm{p}}_k
+ F^u_k \delta u_k^{\mathrm{n}}
+ f_k

This is the recursive object the backward pass needs.

4. Backward projected solve

The reduced cost-to-go is formed on the admissible free variable, regardless of representation. This yields:

- a feedback map for the free variable
- a feedforward offset
- an update to the local value function

5. Forward reconstruction

Given `\delta x_k`, reconstruct:

- the particular step
- the free correction
- `\delta u_k`
- `\delta y_k`

This is the correct notion of "recursive projection". The recursion should be expressed in terms of admissible affine maps, not tied to one nullspace matrix implementation.

## Constraint Set Selection

The user problem is not "project everything". It is "project this selected hard-constraint family".

Write that selected subset as

A_k^{\mathrm{proj}} \delta u_k + B_k^{\mathrm{proj}} \delta x_k + a_k^{\mathrm{proj}} = 0

and only this subset is consumed by the projection backend. Other constraints may remain in:

- the ordinary equality multiplier system
- soft-constraint handling
- IPM inequality handling

This is a structural modeling decision, not just a numerical one.

The user-facing API should therefore let a user declare:

- which constraints belong to the projected set
- optional grouping labels within that set
- optional ordering hints for specialized backends

The API should not require the user to permute dense matrices manually.

## Variable And Constraint Order

Order matters for efficiency, but it should matter in a controlled way.

There are three different notions of order:

1. Authored order

The order in which users define variables and constraints in the modeling layer.

2. Solver layout order

The order used by the solver-owned composed OCP clone after layout compilation and before finalize. This is the order in which flattened primal indices and hard-constraint rows are created.

In the current implementation, the projected hard-constraint Jacobian is still assembled in the canonical block form

A_k = \begin{bmatrix} s_u \\ c_u \end{bmatrix}, \qquad
B_k = \begin{bmatrix} s_x + s_y F_x \\ c_x \end{bmatrix}

but the intended long-term design is that the stage's primal and hard-constraint ordering should already have been compiled into the solver-owned OCP before this assembly occurs.

3. Factorization order

The row/column order actually used by the backend for pivoting, fill reduction, elimination, or structured recursion.

The design should preserve a strict separation:

- users control membership in the projected set
- projectors declare ordering requirements
- the solver OCP creator compiles one solver layout order and checks compatibility
- the backend may reorder internally and must own the associated permutation metadata

This matters because:

- dense rank-revealing factorizations may prefer numerical pivoting
- sparse factorizations may prefer fill-reducing permutations
- sequential specialized backends may require semantic block order

So the API should accept projector requirements and ordering hints, but the backend must still remain free to apply an additional internal permutation when stability or sparsity requires it.

## Efficiency Implications

A mathematically generic design can still be computationally poor if it forces the wrong intermediate objects.

The system should avoid:

- forcing explicit dense `Z_k` on sparse problems
- materializing projector matrices `P_k`
- rebuilding symbolic factorizations every SQP iteration when sparsity is unchanged
- coupling authored order directly to factorization order
- projecting a large mixed constraint set when only a structured subset benefits from elimination

The preferred efficiency rules are:

1. Separate symbolic and numeric phases

For sparse and sequential backends, symbolic analysis and ordering decisions should be cached across SQP iterations whenever the stage sparsity pattern is unchanged.

2. Distinguish assembly order from factorization order

Backends should consume canonical assembled blocks and perform their own row/column permutations internally.

3. Keep reduced coordinates backend-local

If a backend has a good admissible coordinate map `T_k`, the solver should not force conversion to an explicit basis `Z_k`.

4. Allow partial projection

Users should be able to project only the hard constraints that structurally benefit from elimination.

5. Prefer affine-map outputs

The solver really needs the affine maps analogous to the current `u_{y,k}`, `u_{y,K}`, `y_{y,k}`, `y_{y,K}`, plus reduced-step solves. Those should be the optimized outputs. Everything else is backend-internal.

## Pseudo-Inverse Requirements

The pseudo-inverse solve must be defined by properties, not by algorithm:

\delta u_k^{\mathrm{p}} = -A_k^{+} r_k, \qquad r_k := B_k \delta x_k + a_k

Required properties:

- primal feasibility:

A_k \delta u_k^{\mathrm{p}} = -r_k

- consistency with rank deficiency:
  - if `r_k \in \operatorname{range}(A_k)`, solve exactly
  - otherwise detect and report loss of consistency

- backend policy:
  - dense backend may use rank-revealing LU/QR/SVD semantics
  - sparse backend may use sparse QR or sparse least-norm solves
  - sequential backend may exploit structure and avoid generic pseudo-inverse formation

The API should therefore expose "solve particular constrained step" rather than "return pseudo-inverse matrix".

For efficiency, it should also support batched affine solves:

A_k U_k^{\mathrm{p}} = B_k, \qquad A_k u_k^{\mathrm{p}} = a_k

because solving these together can be substantially cheaper than solving one right-hand side at a time.

## Nullspace Requirements

Similarly, the nullspace component should be specified by properties:

A_k \delta u_k^{\mathrm{n}} = 0

\delta u_k^{\mathrm{n}} \in \mathcal{N}(A_k)

If a backend exposes a basis `Z_k`, then:

\mathcal{N}(A_k) = \operatorname{range}(Z_k)

If not, then it must at least support:

- reconstructing an admissible correction from backend-local reduced coordinates
- applying the reduced Hessian on the admissible subspace
- solving the reduced system there

That is enough for the solver.

## Dual Recovery

After the primal step is determined, the hard-constraint multipliers satisfy a transpose solve of the form

A_k^\top \lambda_k = \rho_k

where `\rho_k` is assembled from stationarity residual terms after primal reconstruction.

This is mathematically distinct from the primal particular solve, but it should live in the same backend because:

- it reuses the same factorization
- rank handling must stay consistent
- sparse/sequential backends may have a specialized transpose solve path

The recursive solver should ask only for:

- transpose constrained solve for multiplier recovery

and not know how the backend realized it.

## Recommended Backend Contract

Each equality-elimination backend should be built around the following mathematical operations.

This contract applies **after** the current dynamics-projection stage has produced the projected operator seen by the equality stage. In other words:

- dynamics projection remains a specialized elimination stage
- the backend contract below applies to equality-elimination blocks composed after that stage

Within that scope, each backend should support:

1. Analyze and factor hard constraints

Input:

- `A_k`
- optional structural metadata
- optional grouping / ordering hints

Output:

- rank classification
- reusable factorization state
- internal permutation metadata when applicable

2. Solve particular affine constraints

Given `r_k`, return `\delta u_k^{\mathrm{p}}` such that

A_k \delta u_k^{\mathrm{p}} = -r_k

3. Build admissible-step representation

Provide one of:

- explicit basis `Z_k`
- admissible coordinate map `T_k`
- projector action `P_k v`
- direct constrained reduced-solve machinery that makes separate projection unnecessary

4. Build or apply reduced Hessian

Given control Hessian action `H_k`, provide:

- reduced Hessian action
- optionally an explicit reduced matrix

5. Solve reduced projected Newton system

Given reduced rhs, solve for the free projected step

6. Recover duals

Solve the transpose system needed for multiplier updates

This contract is general enough for all three backend families.

## Backend Families

### 1. Dense reference backend

Representation:

- rank-revealing dense factorization
- explicit nullspace basis `Z_k`
- explicit reduced Hessian `Z_k^\top H_k Z_k`

Purpose:

- preserve current behavior
- serve as regression oracle
- simplest path for initial integration

### 2. Sparse operator backend

Representation:

- sparse rank-revealing factorization
- operator-form projector and constrained solves
- optional reduced basis only when economical

Purpose:

- avoid densifying large sparse hard-constraint systems
- exploit sparse structure in robotics / contact / graph problems

### 3. Sequential specialized backend

Representation:

- recursive elimination or structure-aware transforms
- may never construct a global `A_k` explicitly in dense form

Purpose:

- exploit known problem structure
- support high-performance special cases

This backend should still satisfy the same mathematical contract.

## Integration Plan For The Solver

The projector-driven layout path described above is already landed in this branch.

The phased plan below refers to the next solver/backend refactor: making the
numeric equality elimination layer explicit and backend-polymorphic while
preserving the current dense-reference behavior.

### Phase 1: Introduce An Elimination Plan Without Changing Kernels

Refactor the node solve so that it is represented as a fixed elimination plan:

- specialized dynamics projection stage
- one dense reference equality-elimination stage

with no algorithmic change inside either stage.

The equality stage should ask for:

- rank status
- particular maps with respect to residual and state perturbation
- admissible free-step solve
- dual transpose solve

but the dense backend internally reproduces the existing `Z_u`, `Z_y`, `Q_{zz}`, and pseudo-inverse logic.

No change in algorithm yet.

### Phase 2: Split Equality Handling Into Ordered Blocks

Allow more than one equality-elimination block after the fixed dynamics stage.

The first new supported composition should be:

- custom projected subset
- default dense LU block for the remaining hard equalities

Composition should occur by staged affine elimination, not by trying to solve the two subsets independently.

### Phase 3: Replace Basis-Dependent Interfaces

Reduce direct dependence of the Riccati layer on explicit `Z_u` storage.

The backward pass should work with:

- reduced Hessian solve
- admissible-step reconstruction
- affine propagation maps

This is the necessary step before adding a sparse/sequential backend.

### Phase 4: Add Sparse Operator Backend

Implement the same contract without requiring explicit basis construction unless needed.

### Phase 5: Add Specialized Recursive Equality Backends

Allow a backend whose admissible map is already recursive or block-structured, as long as it provides the same stage-local mathematical outputs.

### Deferred: Multiple Independently Scheduled Dynamics Blocks

This should not be part of the next solver/backend refactor.

Current feasibility:

- multiple `__dyn` terms are already aggregated through the existing `generic_dynamics` hooks
- projected dynamics storage is still node-global
- `x/y` mapping utilities currently assume aligned dynamics lists between neighboring stages

So a future "multiple dynamics blocks in the elimination plan" design is feasible, but it requires a second refactor of projected-dynamics storage and traversal utilities rather than only a new equality backend.

## Validation Plan

The projection subsystem should be validated by mathematical identities first.

The current branch already has regression coverage for projector-driven layout
compilation itself. The items below are mainly the validation target for the
next solver/backend refactor.

For each backend and each stage rank case:

1. Feasibility of particular solve

\|A_k \delta u_k^{\mathrm{p}} + r_k\|

should be near machine precision when consistent.

2. Nullspace feasibility

\|A_k \delta u_k^{\mathrm{n}}\|

should be near machine precision.

3. Projector consistency

P_k^2 \approx P_k, \qquad A_k P_k \approx 0

where meaningful.

4. Reduced solve consistency

The reconstructed primal step should satisfy the same KKT equations as the dense reference backend.

5. Ordering robustness

Changing authored constraint order should not change the mathematical solution, except for expected floating-point differences. Backends may still choose different internal permutations for efficiency.

6. Partial projection consistency

If only a subset of hard constraints is projected, the combined projected-plus-unprojected solve should match the reference KKT system for the same modeling choice.

7. Recursive equivalence

The full backward/forward sweep should match the current dense implementation on existing test problems up to numerical tolerance.

8. Dual consistency

Recovered projected hard multipliers should satisfy the transpose stationarity equations used for dual recovery.

9. Explicit `__eq_x(x,y)` projected-row coverage

At least one regression should explicitly keep a mixed `x,y` hard equality in `__eq_x` and verify that the assembled projected row contains:

- the direct `s_x` contribution
- the projected `s_y * F_x` contribution
- the projected `s_y * F_u` contribution
- the projected residual contribution `s + s_y * F_0`

This is distinct from the auto-detected `generic_constr(x, y)` path, which currently finalizes as `__dyn`.

## User-Facing API Plan

The user-facing API should let users attach projectors declaratively.

The important user action is:

- declare a projector and its layout requirements

The important non-goal is:

- making users manage matrix permutations or factorization details directly

A reasonable user-level model is:

- each projector names the hard constraints it owns
- each projector declares primal and constraint ordering requirements
- the solver OCP creator compiles the compatible layout before finalize

The authored order should be preserved for diagnostics and reproducibility, but it should not be treated as the final solver or backend factorization order.

## Implemented Public Interface

The public interface should be smaller than the internal machinery. Users should not have to think in terms of:

- assembled operators
- nullspace bases
- symbolic factorization caches
- backend state lifetimes

The public surface should expose only three concepts:

1. projector membership
2. projector ordering requirements
3. compatibility-checked composition during solver problem creation

Everything else should be compiled from those declarations.

### Implemented C++ Surface

The current C++ builder is:

```cpp
auto state = node_prob->projector().group();
state.require_constraint(expr_list{eq_state});

auto proj = edge_prob->projector();
auto drive = proj.group();
drive.require_primal(var_list{u1});
drive.require_constraint(expr_list{eq_u1});

auto balance = proj.group();
balance.require_primal(var_list{u0});
balance.require_constraint(expr_list{eq_u0});

drive.require_before(state);
state.require_before(balance);
```

The same pattern can be used on `node_ocp` and `edge_ocp`. During interval composition, projector specs from participating problems are merged onto the solver-owned composed interval problem. That composed clone then compiles:

- the finalized `__u` order
- the finalized hard-constraint order for `__eq_x` and `__eq_xu`
- the hard-constraint block metadata consumed by the dense equality stage

### Implemented Python Surface

The Python binding matches the C++ builder closely:

```python
state = node_prob.projector().group()
state.require_constraint([eq_state])

proj = edge_prob.projector()
drive = proj.group()
drive.require_primal([u1])
drive.require_constraint([eq_u1])

balance = proj.group()
balance.require_primal([u0])
balance.require_constraint([eq_u0])

drive.require_before(state)
state.require_before(balance)
```

After compose, the finalized layout can be inspected through:

```python
blocks = composed.compiled_hard_constraint_blocks
```

Each block currently exposes:

- `field`
- `source_begin`
- `source_count`
- `group`

### Current Compatibility Model

The current implementation enforces:

- unique ownership of each reordered control across groups
- unique ownership of each hard constraint across groups
- explicit group ordering through `group.require_before(...)`
- preservation of authored order inside each group
- rejection of references that do not survive graph compose into the realized interval problem

At the moment, compatibility is a hard check. There is no soft-priority resolution layer yet.

### Practical Example

The regression in `unittests/graph_model_compose.cpp` demonstrates the intended use:

- a node-level projector contributes a projected-state hard constraint group
- an edge-level projector contributes reordered controls and `__eq_xu` groups
- interval composition merges both sources
- the composed problem is finalized in the requested compatible order
- `ns_riccati_data` sees the same group order through its internal default elimination stage

This is the currently supported user model. Backend selection and recursive projector execution are still future work.

## Canonical Assembly Plan

Before any equality backend is called, the solver should assemble a canonical stage-local projected hard-equality problem.

For each projection group `g` at stage `k`, assemble:

A_{k,g} \, \delta u_k + B_{k,g} \, \delta x_k + a_{k,g} = 0

plus any metadata needed for structure-aware backends:

- row provenance: which original constraints produced each assembled row
- block provenance: whether rows came from `__eq_x` or `__eq_xu`
- column provenance: which control variables / control blocks each column corresponds to
- optional semantic tags such as contact, closure, holonomic, or user-defined labels

This canonical assembly layer should be solver-owned and backend-independent.

In the projector-driven design, this assembly should consume the already reordered solver-owned OCP clone. In today's code, the natural starting point is the existing projected stacked form assembled in [projection_default_stage.cpp](src/solver/nsp_impl/projection_default_stage.cpp) and used from [presolve.cpp](src/solver/nsp_impl/presolve.cpp):

\bar A_k = \begin{bmatrix} s_y F_u \\ c_u \end{bmatrix}, \qquad
\bar B_k = \begin{bmatrix} s_x + s_y F_x \\ c_x \end{bmatrix}, \qquad
\bar a_k = \begin{bmatrix} s + s_y F_0 \\ c \end{bmatrix}

The new design should keep this as the canonical reference assembly for the first backend.

The important architectural distinction is:

- dynamics projection is not part of this assembly step; it is the specialized elimination stage that runs before it
- equality backends consume the already projected operator produced by that stage

## Backend Interface Plan

The backend interface should expose stage-local projected-equality elimination algebra, not global solver policy.

Conceptually, one backend instance is responsible for one assembled projected-equality set `(A_{k,g}, B_{k,g}, a_{k,g})` produced after the specialized dynamics stage.

### Analyze Phase

Input:

- canonical assembled operator `A_{k,g}`
- structural metadata
- optional ordering hints

Output:

- rank status
- numeric rank
- internal row/column permutations
- reusable factorization state
- backend-local symbolic cache

This phase should be separable into:

- symbolic analyze
- numeric factorize

whenever the backend benefits from that distinction.

### Solve Phase

The solver should then ask for four families of outputs.

1. Particular affine solves

Return objects equivalent to:

U_{k,g}^{\mathrm{p}} \approx A_{k,g}^{+} B_{k,g}, \qquad
u_{k,g}^{\mathrm{p}} \approx A_{k,g}^{+} a_{k,g}

without requiring the backend to form `A_{k,g}^{+}` explicitly.

2. Admissible reduced-coordinate machinery

Return one of:

- explicit basis `Z_{k,g}`
- admissible map `T_{k,g}`
- direct reduced-solve closure

3. Reconstruction operators

Given reduced coordinates and `\delta x_k`, reconstruct:

- `\delta u_k`
- any propagated `\delta y_k` contribution needed by the solver

4. Dual recovery solves

Given the post-primal stationarity rhs, solve the transpose system needed to recover projected hard multipliers.

### What The Solver Should Not Ask For

The solver should not require:

- an explicit pseudo-inverse matrix
- an explicit projector matrix
- a dense nullspace basis in every backend

Those are implementation details.

## Recommended Internal Data Shapes

The current `ns_riccati_data::nullspace_data` mixes:

- solver-consumed affine maps
- backend-specific factorization artifacts
- one specific basis-based realization

That should be split.

### Solver-Visible Projection Outputs

For each stage, the solver should keep only the quantities it actually consumes:

- rank classification
- particular affine maps with respect to residual and state perturbation
- reduced-step feedforward and feedback results
- primal reconstruction outputs
- dual reconstruction outputs

These are conceptually the generalized versions of the current:

- `u_y_k`, `u_y_K`
- `y_y_k`, `y_y_K`
- `z_k`, `z_K`
- `d_lbd_s_c`

### Backend-Owned Auxiliary State

Everything else should live behind a backend-owned state object:

- LU / QR / LDLT / SVD factors
- sparse symbolic analysis
- permutations
- explicit basis matrices if present
- structured elimination trees

The existing `aux_` hook in [ns_riccati_data.hpp](include/moto/solver/ns_riccati/ns_riccati_data.hpp) can host this during an initial refactor, but a dedicated `projection_state` object would be cleaner.

## Integration Into The Current Solver

The current integration seam is [generic_solver.hpp](include/moto/solver/ns_riccati/generic_solver.hpp), especially:

- `ns_factorization`
- `ns_factorization_correction`
- `compute_primal_sensitivity`
- `finalize_dual_newton_step`

The integration plan should be incremental.

In the scoped-feasible design, this integration still assumes:

- one specialized dynamics-projection stage feeding the node-level projected operator
- one or more equality-elimination blocks after that stage

### Step 1: Encapsulate The Current Dense Path

Keep the current algebra in [projection_default_stage.cpp](src/solver/nsp_impl/projection_default_stage.cpp) and [presolve.cpp](src/solver/nsp_impl/presolve.cpp), but move the following backend-specific actions behind a dense reference backend:

- stacked equality factorization
- rank detection
- `kernel()` construction
- batched solves for `u_y_k` and `u_y_K`
- reduced Hessian assembly `Q_zz`

No behavioral change yet.

### Step 2: Replace Direct Basis Dependence At The Solver Boundary

Refactor the backward and forward passes so they consume:

- reduced-step solve results
- affine admissible-step maps
- reconstruction closures

instead of reaching directly into a basis-specific storage layout.

The dense backend may still internally use:

- `Z_u`
- `Z_y`
- `Q_zz`

but those should stop being the only representable form at the boundary.

### Step 3: Support Multiple Projection Groups

Once the single-group case is stable, allow multiple projection groups per stage.

There are two possible policies:

1. monolithic assembly

All projected rows are stacked into one operator and one backend instance handles them together.

2. grouped assembly

Each group is analyzed separately and their admissible maps are composed.

The next backend-polymorphic implementation should use monolithic assembly for simplicity. Grouped composition is mathematically attractive for structure, but it is harder to make robust.

### Step 4: Correction And Refinement

The correction path must reuse the same factorization whenever the projected constraint Jacobian has not structurally changed.

In particular, the correction path in [presolve.cpp](src/solver/nsp_impl/presolve.cpp) currently reuses:

- the equality factorization
- the nullspace structure
- the reduced Hessian solve path

Any generic projection backend must preserve that reuse pattern, or correction steps will become unnecessarily expensive.

## Efficiency Plan By Backend Type

### Dense Backend

Use the current code as the reference backend first.

Good properties:

- simple
- behavior already known
- good for small/medium dense stage systems

Likely improvements later:

- better rank-revealing choice than `FullPivLU` if needed
- less temporary allocation
- tighter separation between factorization and reconstruction

### Sparse Backend

The sparse backend should optimize for repeated factorization with fixed sparsity pattern.

Core plan:

- canonical sparse assembly
- symbolic analysis once per structural pattern
- numeric factorization every SQP iteration
- backend-local permutations for fill reduction
- operator-form reduced solves unless an explicit basis is clearly cheap

This backend should avoid converting sparse assembled constraints into dense `matrix` objects except in fallback mode.

### Sequential Specialized Backend

This backend is appropriate when the projected constraint set already has a meaningful internal order, for example:

- contact stacks
- kinematic closure chains
- articulated-body style recursions
- block lower-triangular elimination patterns

In that case, the ordering hints supplied by the user should be treated as semantic group hints, not as a promise of final factorization order.

The backend may then build a specialized admissible map `T_k` directly from that structure.

## Practical Design Recommendation

The first user-visible version should deliberately be conservative:

- support projector-driven layout only for `__u` and hard-equality order
- compile that layout in the solver OCP creation layer before finalize
- keep dynamics projection as one specialized fixed stage
- keep the dense solver algebra unchanged after finalize
- preserve existing solver numerics

Only after that should you expose:

- multiple compatible projectors per stage
- sparse operator backends
- structure-specialized sequential backends
- multiple independently scheduled dynamics blocks

That ordering keeps the abstraction honest. If the next backend-polymorphic implementation already tries to expose all projection policies at once, it will almost certainly hard-code the wrong boundary.

## Concrete C++ Interface Sketch

This section proposes a concrete integration shape for the current codebase. The goal is to make the next solver/backend refactor small enough to land while still leaving room for sparse and sequential backends.

The design should introduce a dedicated projection namespace, for example:

```cpp
namespace moto::solver::projection {}
```

with four layers:

- elimination plan
- canonical assembled stage problem
- backend interface
- solver-visible projection outputs

### 0. Elimination Plan

The node solve should be driven by a small compiled elimination plan rather than one monolithic backend choice.

```cpp
namespace moto::solver::projection {

enum class stage_kind : uint8_t {
    dynamics_projection,
    equality_elimination,
};

struct elimination_stage {
    stage_kind kind = stage_kind::equality_elimination;
    std::string group;
    std::string backend = "auto";
    int priority = 0;
};

struct elimination_plan {
    std::vector<elimination_stage> stages;
};

} // namespace moto::solver::projection
```

For the next backend-polymorphic step, the compiled plan is effectively:

1. `dynamics_projection`
2. `equality_elimination` for all hard equalities

The first generalization should allow:

1. `dynamics_projection`
2. `equality_elimination` for a custom projected subset
3. `equality_elimination` for the remaining default-LU subset

### 1. Canonical Stage Problem

The backend should consume one assembled stage-local projection problem. A reasonable initial shape is:

```cpp
namespace moto::solver::projection {

struct assembled_problem {
    size_t nx = 0;
    size_t nu = 0;
    size_t ny = 0;
    size_t nrow = 0;

    // Canonical assembled hard-constraint blocks:
    // A du + B dx + a = 0
    matrix A_dense;
    matrix B_dense;
    vector a_dense;

    // Optional sparse mirrors for sparse backends.
    sparse_mat A_sparse;
    sparse_mat B_sparse;

    // Provenance / semantics for debugging and specialized backends.
    std::vector<uint32_t> row_source_ids;
    std::vector<uint32_t> col_source_ids;
    std::vector<std::string> row_tags;
};

} // namespace moto::solver::projection
```

For the dense-first version, only `A_dense`, `B_dense`, and `a_dense` need to be populated. The sparse members can remain empty.

This object should be built by a solver-owned assembly function, not by a backend.

### 2. User-Level Projection Spec

The backend should not directly inspect modeling objects to decide membership. That should be resolved before assembly into a small solver-owned spec.

```cpp
namespace moto::solver::projection {

struct order_hint {
    enum class mode : uint8_t {
        none,
        preserve_authored,
        grouped,
        backend_default,
    };
    mode value = mode::backend_default;
};

struct group_spec {
    std::string name;
    std::string backend = "auto";
    order_hint ordering;
};

struct stage_spec {
    // The next backend-polymorphic step may allow at most one group.
    std::vector<group_spec> groups;
};

} // namespace moto::solver::projection
```

The next backend-polymorphic step can ignore most of this and just support one implicit group, but the type should leave room for extension.

### 3. Rank Classification

The backend needs to report more than a bare integer rank.

```cpp
namespace moto::solver::projection {

enum class rank_case : uint8_t {
    unconstrained,
    constrained,
    fully_constrained,
    inconsistent,
};

struct analyze_info {
    rank_case rank_status = rank_case::unconstrained;
    size_t numeric_rank = 0;
    size_t nullity = 0;
    bool structurally_reusable = false;
};

} // namespace moto::solver::projection
```

`inconsistent` is useful even if the next dense-backend refactor never expects it in normal operation, because sparse/sequential backends may detect structural or numerical inconsistency explicitly.

### 4. Solver-Visible Projection Outputs

The solver should store only the outputs it actually consumes in the Riccati recursion and rollout.

```cpp
namespace moto::solver::projection {

struct stage_outputs {
    analyze_info info;

    // Particular affine maps:
    // du_p = -U_p * dx - u_p
    // dy_p = -Y_p * dx - y_p   (after propagation through F_u)
    matrix U_p;
    vector u_p;
    matrix Y_p;
    vector y_p;

    // Reduced-step solve result in backend-local coordinates.
    matrix K_red;
    vector k_red;

    // Optional explicit reconstruction maps for dense backends.
    matrix T_u;
    matrix T_y;

    // Dual recovery output.
    vector dlbd_proj;
};

} // namespace moto::solver::projection
```

This is intentionally close to the current meanings of:

- `u_y_K`, `u_y_k`
- `y_y_K`, `y_y_k`
- `z_K`, `z_k`
- `d_lbd_s_c`

but without assuming that the reduced coordinates come from an explicit nullspace basis.

For the next backend refactor, `K_red` and `k_red` may still be the dense nullspace coordinates currently stored in `z_K` and `z_k`.

### 5. Backend-Owned State

The backend state should be opaque to the solver.

```cpp
namespace moto::solver::projection {

struct backend_state {
    virtual ~backend_state() = default;
};

} // namespace moto::solver::projection
```

Dense backend examples:

- `Eigen::FullPivLU<matrix>`
- explicit `Z_u`, `Z_y`
- explicit reduced Hessian storage

Sparse backend examples:

- symbolic factorization
- numeric factorization handles
- permutations

Sequential backend examples:

- elimination tree
- local transforms
- block metadata

### 6. Backend Interface

The backend should expose lifecycle, factorization, reduced solve, reconstruction, and dual recovery.

```cpp
namespace moto::solver::projection {

struct backend {
    virtual ~backend() = default;

    virtual std::unique_ptr<backend_state> create_state() const = 0;

    virtual void analyze_pattern(backend_state& state,
                                 const assembled_problem& prob,
                                 const stage_spec& spec) = 0;

    virtual analyze_info factorize(backend_state& state,
                                   const assembled_problem& prob,
                                   const stage_spec& spec) = 0;

    virtual void build_particular_maps(backend_state& state,
                                       const assembled_problem& prob,
                                       stage_outputs& out) = 0;

    virtual void build_reduced_system(backend_state& state,
                                      const assembled_problem& prob,
                                      const matrix& H_uu,
                                      const matrix& H_yterm,
                                      stage_outputs& out) = 0;

    virtual void solve_reduced_step(backend_state& state,
                                    const assembled_problem& prob,
                                    const vector& rhs_k,
                                    const matrix& rhs_K,
                                    stage_outputs& out) = 0;

    virtual void reconstruct_primal(backend_state& state,
                                    const assembled_problem& prob,
                                    const matrix_ref dx_K,
                                    const vector_ref dx_k,
                                    stage_outputs& out,
                                    matrix_ref du_K,
                                    vector_ref du_k,
                                    matrix_ref dy_K,
                                    vector_ref dy_k) = 0;

    virtual void recover_duals(backend_state& state,
                               const assembled_problem& prob,
                               const vector& stationarity_rhs,
                               stage_outputs& out) = 0;
};

} // namespace moto::solver::projection
```

This looks large, but in practice the first dense backend can implement most of it by wrapping the current code paths.

Two implementation notes:

1. `analyze_pattern(...)` may be a no-op for the dense backend.
2. `build_reduced_system(...)` should accept exactly the Hessian pieces the current solver already forms, rather than asking the backend to reconstruct solver algebra from scratch.

### 7. Dense Reference Backend Mapping

The current dense path already computes nearly everything required by the proposed interface.

Initial mapping:

- `assembled_problem.A_dense` maps to current `s_c_stacked`
- `assembled_problem.B_dense` maps to current `s_c_stacked_0_K`
- `assembled_problem.a_dense` maps to current `s_c_stacked_0_k`
- `backend_state` owns current `lu_eq_`, `Z_u`, `Z_y`, `Q_zz`
- `stage_outputs.U_p` maps to current `u_y_K`
- `stage_outputs.u_p` maps to current `u_y_k`
- `stage_outputs.Y_p` maps to current `y_y_K`
- `stage_outputs.y_p` maps to current `y_y_k`
- reduced-step outputs map to current `z_K`, `z_k`
- dual recovery maps to current transpose solve using `lu_eq_.transpose().solve(...)`

That means the next backend refactor can be almost entirely a packaging change.

## Proposed Changes To `ns_riccati_data`

The current `nullspace_data` struct is too specific to the dense-basis realization. A staged migration is safer than deleting it immediately.

### Transitional Layout

Keep the current members during the next backend refactor, but add a new projection slot:

```cpp
namespace moto::solver::projection {
struct stage_outputs;
struct backend_state;
struct assembled_problem;
}

struct ns_riccati_data : public data_base {
    // existing members ...

    projection::assembled_problem proj_prob_;
    projection::stage_outputs proj_out_;
    std::unique_ptr<projection::backend_state> proj_state_;

    // legacy dense-path cache, kept temporarily during migration
    nullspace_data nsp_;
};
```

This lets the solver migrate incrementally:

- first populate both `nsp_` and `proj_out_`
- then switch downstream code to read `proj_out_`
- then delete or shrink `nsp_`

### Final Layout

Once migration is complete, the basis-specific storage should move out of `ns_riccati_data` entirely:

```cpp
struct ns_riccati_data : public data_base {
    // dimensions and solver aliases ...

    projection::assembled_problem proj_prob_;
    projection::stage_outputs proj_out_;
    std::unique_ptr<projection::backend_state> proj_state_;
};
```

Any dense-only objects such as `Z_u` or `lu_eq_` would then live in the concrete dense backend state.

## Proposed Changes To `generic_solver`

The current `generic_solver` can remain the orchestration layer. It should gain a projection backend handle.

```cpp
namespace moto::solver::ns_riccati {

struct generic_solver {
    std::shared_ptr<projection::backend> projection_backend_;

    virtual ns_riccati_data create_data(node_data* full_data);
    virtual void ns_factorization(ns_riccati_data* cur, bool gauss_newton = false);
    virtual void ns_factorization_correction(ns_riccati_data* cur);
    virtual void riccati_recursion(ns_riccati_data* cur, ns_riccati_data* prev);
    virtual void riccati_recursion_correction(ns_riccati_data* cur, ns_riccati_data* prev);
    virtual void compute_primal_sensitivity(ns_riccati_data* cur);
    virtual void compute_primal_sensitivity_correction(ns_riccati_data* cur);
    virtual void fwd_linear_rollout(ns_riccati_data* cur, ns_riccati_data* next);
    virtual void fwd_linear_rollout_correction(ns_riccati_data* cur, ns_riccati_data* next);
    virtual void finalize_primal_step(ns_riccati_data* cur);
    virtual void finalize_dual_newton_step(ns_riccati_data* cur);
    virtual void finalize_primal_step_correction(ns_riccati_data* cur);
    virtual void apply_affine_step(ns_riccati_data* cur, workspace_data* cfg);
};

} // namespace moto::solver::ns_riccati
```

The solver should own backend selection policy, but the backend should own projection algebra.

### `ns_factorization(...)`

The new flow should be:

1. canonical stage assembly
2. backend analyze/factorize
3. build particular affine maps
4. build reduced system
5. activate gradient corrections
6. correction-path residual solve

Conceptually:

```cpp
void generic_solver::ns_factorization(ns_riccati_data* cur, bool gauss_newton) {
    auto& d = *cur;

    cur->update_projected_dynamics();
    assemble_projection_problem(cur->proj_prob_, d);

    auto& be = *projection_backend_;
    if (!cur->proj_state_) {
        cur->proj_state_ = be.create_state();
        be.analyze_pattern(*cur->proj_state_, cur->proj_prob_, projection_spec_for(d));
    }

    cur->proj_out_.info = be.factorize(*cur->proj_state_, cur->proj_prob_, projection_spec_for(d));
    be.build_particular_maps(*cur->proj_state_, cur->proj_prob_, cur->proj_out_);
    be.build_reduced_system(*cur->proj_state_, cur->proj_prob_, /* Hessian terms */, /* Y terms */, cur->proj_out_);

    cur->activate_lag_jac_corr();
    ns_factorization_correction(cur);
}
```

The dense backend implementation can internally continue to produce the same intermediate values as today.

### `ns_factorization_correction(...)`

This method should stop reconstructing backend-specific factorization state. It should only:

- refresh residual-dependent rhs terms
- ask the backend for correction-stage particular solves
- update the solver-visible affine maps

That preserves the current reuse pattern.

### `compute_primal_sensitivity(...)`

This method should depend only on:

- reduced-step solve result
- primal reconstruction

For the dense backend, it may still internally compute:

\delta u_k = Z_k z_k - u_k^{\mathrm{p}}, \qquad
\delta y_k = Z^y_k z_k - y_k^{\mathrm{p}}

but the generic solver should consume only the reconstructed outputs.

### `finalize_dual_newton_step(...)`

This method should ask the backend to solve the transpose system for the projected hard multipliers, using the same factorization state already built during `ns_factorization(...)`.

## Assembly Helper Functions

To keep the backend interface clean, canonical assembly should live in solver-owned helper functions.

The first useful helpers are:

```cpp
namespace moto::solver::projection {

void assemble_problem(assembled_problem& out, ns_riccati_data& d);

void assemble_dense_A(matrix& out, ns_riccati_data& d);
void assemble_dense_B(matrix& out, ns_riccati_data& d);
void assemble_dense_a(vector& out, ns_riccati_data& d);

stage_spec projection_spec_for(const ns_riccati_data& d);

} // namespace moto::solver::projection
```

This preserves one clear responsibility split:

- solver assembles canonical stage algebra
- backend chooses factorization strategy
- recursion consumes only solver-visible outputs

## First Refactor Sequence

The safest code-change order is:

1. Add new projection namespace and types with no behavior change.
2. Add canonical assembly helpers that reproduce the current stacked dense matrices.
3. Add dense reference backend that wraps current `FullPivLU` / `kernel()` logic.
4. Add `proj_prob_`, `proj_out_`, and `proj_state_` to `ns_riccati_data`.
5. Modify `ns_factorization(...)` to populate the new structures while still filling `nsp_`.
6. Switch `compute_primal_sensitivity(...)` and dual recovery to consume `proj_out_`.
7. Remove basis-specific dependencies from the solver boundary.
8. Only then consider sparse or sequential backends.

This order keeps regression risk low and gives a clean place to compare the new interface against the current dense implementation at every step.

## Practical Recommendation For The Next Coding Pass

The next implementation pass should not try to solve sparse/sequential backends yet.

It should do exactly this:

- define a minimal projector layout spec
- let the solver OCP creator gather projector requirements
- check projector compatibility there
- reorder the solver-owned composed OCP clone before finalize
- keep the dense projected assembly and solve path otherwise unchanged

If that lands cleanly, the boundary is probably right. If it still needs runtime stacked-Jacobian manipulation to express ordering, then the layout decision is happening too late.
