# Stage Graph Overhead Review

Opened 2026-08-29. Contract: `docs/contracts/stage_graph_modeling.md`.

## Checklist

- [x] Simplify topology, composition, and runtime reconciliation.
- [x] Preserve edits, replacement, shift, terminal placement, and handles after lowering.
- [x] Keep hot 500-stage construction and shift below 50 microseconds.
- [x] Reduce net line growth without removing behavior or tests.

## Findings

| id | finding | resolution |
| --- | --- | --- |
| SGO-1 | A second interval-record vector duplicated stage order. | Derive boundaries from one stage snapshot. |
| SGO-2 | Equivalent copies each allocated and finalized an OCP. | Share composed OCPs by copy-shared identity. |
| SGO-3 | Composer/runtime shift matching used quadratic scans. | Use hashed composition and sorted occurrence keys. |
| SGO-4 | Default `solver_nodes()` rebuilt a pointer vector. | Return runtime order unless a virtual initial node exists. |
| SGO-5 | Mutation callbacks duplicated scans and outlived removed stages. | Detect identity changes during revision scans. |
| SGO-6 | Distinct pointers are required for occurrence identity but expensive to author. | Prepare copies outside topology timing and share immutable artifacts. |
| SGO-7 | Lowering/status propagation was order-dependent and missed `x -> y` status. | Place active terms, union mapped status, then resolve once. |
| SGO-8 | Lowering invalidated user-defined handles for runtime data access. | Preserve remap lineage only in contextual function-data lookup. |
| SGO-9 | Initial diff was +673/-281 (net +392). | Reduced to +555/-282 (net +273); production code is net +84. |

## Verification

- Release construction median/p95: 26.120/32.056 microseconds; shift: 35.619/41.555. The test also checks three shared formulations across 500 occurrence identities.
- Mapping tests cover predicates, `x -> y` status, later pruning, and user-defined handle access to lowered value/Jacobian data.
- Python mutation/replacement/shift and `node.data(endpoint)[x]` passed; controlled-thread Go2 completed.
- Release/native build, 18/18 CTests, Python handle/editing checks, and controlled-thread Go2 passed after compaction.
