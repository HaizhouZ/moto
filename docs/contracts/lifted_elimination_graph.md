# Lifted Elimination Graph Contract

## Purpose

A lifted elimination graph describes how a grouped dynamics constraint eliminates
the local QP directions associated with `y` and explicitly lifted variables `l`.
It is a symbolic description of one linear inverse action, not a second dynamics
model and not a runtime callback mechanism.

For the grouped residual `h(x, u, y, l) = [dyn; lift]`, directions are ordered as

```text
dlifted = [dy; dl],   dunlifted = [dx; du].
```

The graph represents applications of `h_l^-1`, where `h_l` is the Jacobian of
`h` with respect to `[y; l]`. It supplies projected responses to the nullspace
solver without removing `y`, `l`, or their multipliers from the nonlinear model.

## Authoring Model

The normal user interface is identity based:

- an equation block is selected by its dynamics or constraint handle;
- a variable block is selected by its symbol handle;
- `system.jac(equation, variable)` returns that whole Jacobian block;
- `system.residual(equation)` returns that whole residual block.

Names and strings are diagnostics only. Equal symbolic expressions in different
equation rows remain distinguishable because their owning function handles are
different. Raw symbolic-expression lookup may exist as an expert sub-row tool,
but it is not the primary interface and may reject ambiguous matches.

The user provides one pure symbolic solve operation:

```python
def elimination(system):
    def solve(rhs):
        return structured_inverse_action(rhs)

    return system.eliminate(solve)
```

`system.eliminate(solve)` applies the same operation to the packed projection
right-hand side `[h_x, h_u, h]` and to the forward-action right-hand side. It
splits the packed response into state, input, and residual responses. The
transpose action is derived from this same graph. A normal user therefore cannot
supply inconsistent projection and action implementations.

Explicit intermediates are optional. They declare graph values that must remain
available to later generated entries; they are not declarations of temporary
matrix storage and are not required for ordinary factor reuse.

## Group Configuration

Dynamics, lifted subconstraints, and the elimination builder are configured as
one value operation:

```python
group = dynamics.with_elimination_graph(elimination, [contact_constraint])
```

The returned group owns the configuration. The source dynamics and constraints
remain unchanged. Compatibility mutation or raw-response APIs may remain, but
examples and new code use the coherent value operation.

The builder is a pure modeling callback. Finalization may invoke it more than
once while deriving compatible profiles. It must not depend on invocation count,
mutate external model state, or perform runtime numerical work. Parameters made
through graph blocks are canonicalized by identity across equivalent invocations.

## Algebra And Signs

The grouped linearization is

```text
h_l dlifted + h_x dx + h_u du = -h.
```

The elimination graph computes unsigned inverse responses

```text
R_x = h_l^-1 h_x,
R_u = h_l^-1 h_u,
R_0 = h_l^-1 h.
```

The solver owns the minus signs in

```text
dlifted = -R_x dx - R_u du - R_0.
```

All blocks retain their declared dimensions even when structurally zero.
Sparsity belongs to the resulting expressions and is inferred during
finalization; users do not provide sparsity patterns.

## Factorization And Regularization

The graph may expose structured factorizations and may add user-controlled
regularization to a structurally absent diagonal block. A regularizer is an
ordinary model parameter created from the selected block and used directly in
the graph.

Equivalent solves of the same matrix share one factorization. Compatible
right-hand sides may be packed into one solve. Factorization results and branch
values with multiple consumers are persistent generated-graph data, while
single-use expressions may remain lazy.

The user declares mathematical structure when it is not derivable from symbolic
identity, such as an SPD solve. Numeric positive-definiteness is not inferred at
model-construction time.

## Lowering And Runtime

CasADi MX supplies the symbolic DAG, dependency identity, shape, and sparsity.
It does not execute elimination at solver runtime. Finalization lowers the graph
into the existing linear backend:

- sparse and dense panels bind directly to OCP-planned storage;
- factorization and solve entries reuse generated temporary storage;
- compile-time transpose, alignment, and fixed-size properties do not become
  runtime branch decisions;
- Python callbacks, string lookup, CasADi evaluation, and per-entry virtual
  dispatch are forbidden in the hot path.

The canonical solve interface must lower to no more work than the equivalent
expert response graph. Validation covers isolated generated entries and the
integrated nullspace traversal; cold artifact generation is reported separately.

## Validation Requirements

Finalization rejects:

- an equation or variable handle outside the grouped system;
- solve outputs with incompatible row or column dimensions;
- a graph missing its forward action;
- incompatible repeated declarations of the same regularization parameter.

Tests cover structural zeros, repeated equal residual expressions, packed
right-hand sides, factor reuse, transpose consistency, source immutability,
Python callback absence at runtime, and numerical agreement with a dense solve.
