# Graph User API Cleanup

Opened 2026-08-29.

## Requested outcome

- [x] Audit the user-visible graph construction API.
- [x] Keep one stage-centric construction and editing path.
- [x] Remove duplicate aliases and implementation-facing graph types.
- [x] Preserve stage mutation, replacement, deletion, shift, endpoint lowering,
  and active-expression selection.
- [x] Rebuild bindings and validate C++, Python, and quadruped scenarios.

## Workflow

1. Inspect the documented stage graph and every Python binding/export.
2. Define the retained public path in the stage-graph contract.
3. Remove aliases and internal owner/configuration objects from the top-level
   package.
4. Migrate examples and benchmarks to the retained API.
5. Verify topology editing, lowering, binding stubs, and complete Go2 behavior.

## References

- `docs/contracts/stage_graph_modeling.md`
- `docs/contracts/vocabulary.md`
- `include/moto/ocp/problem.hpp`
- `include/moto/ocp/graph_model.hpp`
- `bindings/definition/node_data.cpp`
- `bindings/definition/ns_sqp.cpp`
- `bindings/definition/public_api.py`

## Findings

| id | severity | finding | evidence | suggested action | done-status | user-action |
| --- | --- | --- | --- | --- | --- | --- |
| GUA-1 | medium/API | `sqp.start_node` and `sqp.st` expose the same graph boundary. | Both bindings forward to the same `graph_model` start-stage endpoint; graph tests compare their owners for equality. | Retain `sqp.st` and remove `start_node`. | done | y |
| GUA-2 | high/API | Phase variants require the implementation-facing `active_status_config` plus the redundant `with_status` method. | The quadruped model calls `stage.with_status(moto.active_status_config(deactivate_list=...))`; the configuration merely forwards to `stage_ocp::copy`. | Make `stage.copy(disable=..., enable=...)` the variant constructor and provide `stage.disable/enable` for an existing stage. | done | y |
| GUA-3 | medium/API | Top-level `stage_ocp`, `endpoint`, and `active_status_config` expose implementation type names beside the intended `moto.stage()` factory. | `PUBLIC_BINDINGS` exports all three even though ordinary examples only need `moto.stage`, returned endpoint handles, and stage methods. | Stop exporting implementation types at package top level. | done | y |
| GUA-4 | medium/ownership | `endpoint.stage` exposes the shared owner used to keep the endpoint alive. | No production caller uses it; only identity tests and the graph-editing demo inspect it. | Keep owner retention private and test stable terminal behavior through composed results. | done | y |

## Verification

- Release/native build completed with four jobs.
- 19/19 CTests pass, including the new Python public-API/editing regression.
- `graph_model_compose_test`: 25 cases and 214 assertions pass.
- `initial_state_optimization.py` and `stage_graph_editing.py` pass through the
  installed package.
- The 100-stage lifted acceleration/RNEA/contact Go2 case still converges in
  19 iterations.
