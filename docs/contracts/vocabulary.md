# Vocabulary

| term | symbol | definition | source | owner | forbidden aliases |
| --- | --- | --- | --- | --- | --- |
| lifted Hessian contraction | $R_l^T Q_{ll} R_l$ | Contribution of lifted primal Hessian blocks after mapping lifted directions to unlifted directions. | user discussion and NSP implementation | NSP presolve | lifted condensation |
| weighted Gram | $R^T D R$ | Symmetric product with a diagonal or panel-structured middle weight. | linear backend | linear backend | Gram solve |
| response matrix | $R_l=[Z_l,-l_y]$ | Lifted response columns used by input-nullspace and state-sensitivity contraction. | NSP implementation | NSP presolve | projector matrix |
| lifted direction recovery | $dl=-l_0-l_xdx-l_udu$ | Recovery of the explicit lifted primal direction after the state and input directions are available. | user instruction and NSP implementation | NSP rollout | lifted sensitivity recovery |
