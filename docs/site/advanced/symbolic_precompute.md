# Reuse symbolic work with precompute

Use `moto.precompute` when several costs or constraints need the same expensive
CasADi SX subgraph. A precompute evaluates that subgraph once per solver node
and approximation update, stores its value and total first derivatives in
node-local caches, and lets independent consumers retain their own residual
rows, multipliers, and activation rules.

Precompute is runtime sharing, not a new optimization variable and not a
request to combine consumers into one generated function.

## Create and consume cached values

Create the shared SX outputs, then build ordinary costs and constraints from
the returned cache variables:

```python
import casadi as cs
import moto

x, xn = moto.sym.states("precompute_x", 2)
u = moto.sym.inputs("precompute_u", 1)

shared = moto.precompute.create(
    "shared_features",
    [cs.vertcat(x.sx[0] * x.sx[1], cs.sin(x.sx[0]))],
)
features = shared.outputs[0]

dynamics = moto.dense_dynamics.create(
    "precompute_dynamics", xn.sx - x.sx - cs.vertcat(u.sx[0], 0)
)
tracking = moto.cost.from_vector(
    "cached_tracking", features.sx, weight=[2.0, 0.5]
)
coupling = moto.constr.create(
    "cached_coupling", features.sx[0] + u.sx[0]
)

stage = moto.stage()
stage.add([dynamics, tracking, coupling])
```

Do not call `stage.add(shared)`. Adding a consumer automatically adds the
producer and its complete upstream dependency chain. Moto executes producers
before their consumers regardless of the order in which the consumers were
added.

`shared.outputs` contains one public cache variable per SX output passed to
`create`. Moto privately creates the derivative caches. Do not create
`field___usr_var` symbols, set cache values through `node.value`, or manually
remap output caches.

## Chain precomputes

A downstream precompute may consume an upstream cache:

```python
raw = moto.precompute.create(
    "raw_features", [cs.vertcat(x.sx[0] ** 2, x.sx[1] ** 2)]
)
summary = moto.precompute.create(
    "feature_summary",
    [
        cs.vertcat(
            cs.sum1(raw.outputs[0].sx),
            raw.outputs[0].sx[0] - raw.outputs[0].sx[1],
        )
    ],
)
summary_cost = moto.cost.from_vector(
    "summary_cost", summary.outputs[0].sx, weight=3.0
)

stage = moto.stage()
stage.add(summary_cost)
```

Adding `summary_cost` inserts both `raw` and `summary` in topological order.
Each remains a separate generated artifact, so a long dependency chain does
not become one large translation unit. Cycles are rejected during problem
finalization.

## Reuse one implementation on new symbols

If code retains the original producer, instantiate it positionally on another
set of public input symbols:

```python
x_other, _ = moto.sym.states("precompute_x_other", 2)
shared_other = shared.instantiate([x_other])
features_other = shared_other.outputs[0]
```

The input sequence follows the original producer's inferred input order. An
input-only mapping is also accepted:

```python
shared_other = shared.instantiate([(x, x_other)])
```

Equivalent input mappings return the same instance and cache symbols. Prefer
`instantiate` over manually calling `reuse_remap` on a precompute: instantiation
allocates and registers all public and private caches for you.

## Reuse across scenes with a canonical source

Application code that constructs structurally equivalent expressions for
different scenes should use the process-wide canonical helper:

```python
def cached_robot_features(q, model_signature):
    return moto.precompute.canonical(
        "robot_features",
        identity=("robot_features_v1", model_signature, q.dim, q.tdim),
        inputs=(q,),
        output_factory=lambda: (
            cs.vertcat(q.sx[0] ** 2, cs.sin(q.sx[1])),
        ),
    )[0]

q_a, _ = moto.sym.states("robot_a_q", 2)
q_b, _ = moto.sym.states("robot_b_q", 2)

features_a = cached_robot_features(q_a, ("two_dof",))
features_b = cached_robot_features(q_b, ("two_dof",))
```

For the first `(directory, prefix, identity)` key, `output_factory` authors the
canonical source. Later calls with the same key do not evaluate the factory;
Moto instantiates the source on the current inputs. Compilation occurs on
finalization, including when an implementation must be ready for remapping.

Both `moto.precompute.canonical(...)` and `moto.func.canonical(...)` accept a
keyword-only `codegen` context. Pass the same context used by the consuming
stage and solver. Each directory has independent canonical sources; omitting
the context selects the current working directory's `gen/`. The generated
function names and structural identity rules are unchanged.

The identity must be hashable and must describe every structural property that
changes the symbolic graph: dimensions, model or frame layout, selected
operations, and algorithm version. Do not use a scene name as the identity
when several scenes share the same structure, and do not reuse one identity
for structurally different graphs.

## Placement and derivatives

A precompute inherits placement from its consumer. For example, a state-only
consumer added through `stage.ed.add(...)` carries its producer to the same
endpoint, including the graph's internal `x`-to-`y` lowering. Do not add a
second producer explicitly at another placement.

Moto automatically propagates total tangent Jacobians through one or more
precompute levels. Supported consumers are:

- first-order constraints and dynamics;
- Gauss-Newton vector costs;
- functions that use cached values together with direct primal inputs.

Exact-second-order consumers through a precompute cache are rejected because
second-derivative caches are not currently generated. Use a Gauss-Newton cost,
keep the exact-second-order expression direct, or provide a formulation that
does not cross the cache boundary.

## When precompute helps

Precompute is useful when a substantial operation—robot kinematics, repeated
geometry, feature extraction, or another shared SX graph—feeds multiple
independent consumers at the same node. It avoids repeated runtime evaluation
and repeated differentiation while preserving consumer sparsity and identity.

For a small expression used by only one consumer, direct SX is usually simpler.
A precompute still executes once per node and approximation update; it is not a
global constant cache. Numeric constants, weights, and references should remain
ordinary constants or parameters.
