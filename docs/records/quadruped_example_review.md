# Quadruped Example Review

Date opened: 2026-08-29

## Requested outcome

- [x] Review why the quadruped examples have become unnecessarily large.
- [x] Check their stage-construction workflow against the public graph/stage
  modeling surface.
- [x] Identify the smallest refactor that removes duplication without changing
  the formulations.
- [x] Fix every accepted finding without changing dynamics, solver settings,
  or lifted-elimination mathematics.
- [x] Validate graph ownership, exact horizon construction, Python bindings,
  and quadruped execution.
- [x] Use only the graph-owned `sqp.stages` vector for stage construction.
- [x] Remove unused graph-construction interfaces from C++, Python, tests, and
  examples.
- [x] Remove repeated Python wrappers from `example/helpers.py` and use the
  nanobind surface directly.
- [x] Cover in-place constraint edits, stage-range replacement, and horizon
  shift/tail insertion with native Python/C++ stage containers.

## Workflow

1. Read the public stage and graph-composition contracts and the standard
   quadruped example from `origin/main`.
2. Compare `run.py`, `lifted_go2.py`, `mpc.py`, shared helpers, and lifted
   contact elimination by responsibility and formulation.
3. Record every finding with exact evidence and implementation difficulty.
4. Do not change dynamics, costs, constraints, phase boundaries, or solver
   settings during review.

## References

- `AGENTS.md` public modeling surface and graph composition sections
- `docs/contracts/lifted_direction_recovery.md`
- `docs/contracts/linear_graph_lowering.md`
- `docs/contracts/stage_graph_modeling.md`
- `docs/contracts/vocabulary.md`

## Findings

| id | severity | finding | evidence | suggested action | done-status | user-action |
| --- | --- | --- | --- | --- | --- | --- |
| QER-1 | high | Ordinary path-state terms are authored as phase-start boundary terms. The first copied stage's `st` is not the graph start node, so the current construction omits these terms at `x_0` and shifts the remaining copies onto predecessor `y` boundaries. The same placement also hid invalid force deactivation: a force removed from stage storage was still an argument of dynamics and was only reintroduced by the following boundary kinematics. | `run.py:163`, `lifted_go2.py:209`, `mpc.py:98`; `graph_model::build_stage_chain()` only attaches copied `st` views to preceding intervals, while the first record uses `sqp.start_node`; the composer lowers end-boundary terms from `x` to `y`. After moving path terms, the standard smoke failed in dynamics approximation with `expr f_FL_foot ... cannot be found`, confirming the hidden storage dependency. | Put joint limits, state running cost, and ordinary contact kinematics on `stage.add(...)`. Keep only genuine graph/phase boundary terms on `st`/`ed`, author distinct terminal expressions, and represent an inactive explicit contact by disabling its contact/friction constraints plus `force = 0` rather than removing a dynamics argument. | done | y |
| QER-2 | high | The requested horizon is not always the number of stages created. Equal gait/horizon lengths produce zero-length stance phases rejected by `add_phases`; an odd remainder is floored twice and silently drops one stage. | Both runners compute `stance_length = int((N_horizon - total_gait_steps) / 2)` and use it for both ends. `graph_model::add_phases()` rejects `count == 0`. | Split the remainder as `head = remainder // 2`, `tail = remainder - head`, omit zero-length phases, and assert the flattened graph has exactly `N_horizon` stages. | done | y |
| QER-3 | medium | The example helper obscures the public stage API and uses inaccurate names: `segment_start_nodes` contains stage prototypes, while `add_stage_segments()` merely zips and flattens `sqp.add_phases()`. The final `stage_proto.copy()` is redundant because graph insertion already copies every prototype. | `run.py:202-211`, `lifted_go2.py:248-257`, `helpers.py:664-671`, and `graph_model::build_stage_chain():151`. | Build a phase list explicitly, call `sqp.add_phases()` directly, and flatten the returned graph-owned phases where mutation is required. Remove the extra copy and wrapper. | done | y |
| QER-4 | medium | `lifted_go2.py` is almost a full copy of `run.py`: Git's structural diff reports only 70 inserted and 8 removed lines despite files of 398 and 336 lines. Model construction, state cost, gait setup, settings, benchmarking, profiling, and display are duplicated. | `git diff --no-index --numstat run.py lifted_go2.py` reports `70 8`; both define the same `QuadrupedModel` shell and runner workflow. | Keep one Go2 model/stage/gait runner with a small formulation configuration. Make `run.py` and `lifted_go2.py` thin entry points, or merge their CLI if preserving two scripts is unnecessary. | done | y |
| QER-5 | medium | `mpc.py` repeats the same trivial model subclass and path-stage assembly error, although its simulation loop is legitimately separate. | `mpc.py:27-57` duplicates state-cost construction; `mpc.py:95-109` repeats the stage/terminal assembly. | Share only Go2 model, cost, and stage builders with the offline runner; leave MuJoCo setup and MPC loop local. | done | y |
| QER-6 | high | Graph-owned stages can only be accessed if every caller retains the return value of `add_stage()` or `add_phases()`. The solver exposes `nodes` after composition but has no modeling-level `stages` accessor; runtime `node.prob` is not the authored mutable stage. | `graph_model` stores every owned stage in private `intervals_`; `ns_sqp` exposes only `start_node`, the add methods, and `solver_nodes()`; the Python binding exports `nodes` but no `stages`. | Add an ordered read-only `sqp.stages` view over graph-owned stages. Keep add-method return values for local convenience, but do not require callers to maintain a shadow stage registry. | done | y |
| QER-7 | medium | Stage/phase insertion methods duplicate operations already provided by the graph-owned stage vector. | Every insertion method reduces to copying a prototype and appending pointers; topology can be derived from vector order before realization. | Remove all stage/phase insertion methods. Users append independent `stage.copy()` objects directly to `sqp.stages`; expose graph `st`/`ed` separately. | done | y |
| QER-8 | medium | `add_terms`, `visit_nodes`, and `set_node_value` in `example/helpers.py` duplicate list-based `add`, Python iteration, slicing, and native node-value assignment. | Repository search finds these wrappers used by toy, arm, quadruped, and MPC examples; none adds model semantics or batching in the native runtime. | Delete the wrappers and call the nanobind APIs or ordinary Python control flow directly. | done | y |
| QER-9 | high | A retained stage pointer can be edited, but there is no compact public operation for replacing a stage range or shifting a horizon while preserving surviving pointer identity. | Graph topology is stored as interval records while Python naturally expresses both edits as list/vector mutation. | Expose the graph-owned `std::vector` directly through nanobind's standard vector binding. Mutate constraints through retained pointers and replace or shift with direct index/delete/insert/append operations; rebuild adjacency lazily before realization. | done | y |

## Refactor boundary

The smallest formulation-preserving change is:

1. Introduce one quadruped model/stage module containing the common Go2 model,
   state cost, terminal terms, solver settings, and phase assembly.
2. Represent explicit contact, lifted contact, lifted acceleration, and
   acceleration control as configuration of that builder rather than copied
   scripts.
3. Keep path terms on the interval stage, terminal terms on the final
   graph-owned `ed`, and use `sqp.start_node` only for a genuine initial-state
   term.
4. Expose graph-owned stages through `sqp.stages`; do not use runtime
   `sqp.nodes` as the modeling API.
5. Keep lifted elimination and the MuJoCo control loop in their current
   specialized modules.

This does not require changing dynamics, active-status behavior, solver
settings, or the lifted elimination graph.

## Implementation

- User authorized all findings on 2026-08-29.
- Added the stage graph modeling contract and an ordered `sqp.stages` modeling
  accessor in C++ and Python.
- Moved quadruped running state/contact terms onto interval stages and created
  distinct terminal expressions.
- Replaced invalid swing-force removal with explicit zero-force constraints
  while disabling swing contact and friction constraints.
- Replaced the segment wrapper with exact positive-length phase construction.
- Consolidated the offline runner and shared quadruped model/stage setup;
  `lifted_go2.py` is now a thin formulation entry point and MPC retains only its
  simulation loop.
- User authorized a second API cleanup on 2026-08-29: retain only tail-appending
  `add_stages` and remove example-only forwarding wrappers.
- User explicitly rejected a custom `segment`/stage-list wrapper. Stage
  collections must use the native C++ and Python containers.
- User then removed `add_stages` as redundant once `sqp.stages` became directly
  mutable.

## Verification

- Incremental Release build: `cmake --build build -j4 --target
  graph_model_compose_test moto_pywrap`, followed by `moto_pywrap_stub`.
- Full graph test: 205 assertions in 22 test cases passed.
- Full CTest suite: 17 of 17 tests passed with `ctest --test-dir build
  --output-on-failure -j4`.
- Installed Python binding exposes `sqp.stages`; returned and retrieved stages
  have identical Python object identity.
- Python syntax and Ruff checks passed for all touched example modules.
- One-iteration quadruped smoke tests completed for forward contact, explicit
  RNEA contact, lifted contact, and acceleration control.
- Exact horizon checks passed for zero-stance (`horizon=2`, two gait nodes) and
  odd-remainder (`horizon=5`, stance split `1+2`) cases, including runtime node
  realization.
- `graph_model_compose_test`: 188 assertions in 20 cases passed, including
  direct range replacement, shift, stable terminal-boundary placement, and
  identity-based runtime reuse.
- `example/toy/stage_graph_editing.py` passed all three requested native-vector
  editing scenarios; the installed binding has no `add_stage`, `add_stages`,
  or `add_phases` method.
- Full CTest: 17 of 17 tests passed with four jobs. `example/toy/run.py`
  converged in 12 iterations, and the one-iteration Go2 smoke completed with
  the direct stage-vector API.
