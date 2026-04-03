# Validation

- Local worktree binding validation completed on `example/toy/run.py`.
- With `eq_init = false`: solved in 9 iterations.
- With `eq_init = true`: solved in 9 iterations.
- Final primal / dual / complementarity residuals matched to the expected numerical tolerance, with no observed convergence regression on that example.
- Local worktree binding validation also completed on `example/arm/run.py` with restoration enabled.
- With `eq_init = false`: completed 50 iterations without crashing and hit `iter_result_exceed_max_iter`.
- With `eq_init = true`: completed 50 iterations without crashing and hit `iter_result_exceed_max_iter`.
- The arm check was used as a runtime regression test for restoration/equality-init interaction, not as a claim of improved convergence on that problem.
