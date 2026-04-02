# Restoration Problem

This note describes the restoration phase that is **currently implemented** in:

- [src/solver/sqp_impl/restoration.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/restoration.cpp)
- [src/solver/restoration/resto_overlay.cpp](/home/harper/Documents/moto/src/solver/restoration/resto_overlay.cpp)
- [src/solver/sqp_impl/line_search.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/line_search.cpp)
- [src/solver/sqp_impl/ns_sqp_impl.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp)

The old framework-external `resto_runtime` path has been removed. Restoration
now runs on a separate **overlay graph** built from finalized composed
`edge_ocp`s, and it reuses the normal solver shell:

- `update_approximation`
- soft/ineq lifecycle
- NSP / Riccati
- iterative refinement
- line-search skeleton

The phase difference is only in the active problem definition:

- cost becomes `proximal + exact elastic penalty`
- non-dynamics constraints are replaced by elastic restoration wrappers
- globalization uses restoration phase metrics

## 1. Entry Condition

The outer SQP loop enters restoration when normal globalization returns a
tiny-step failure while primal infeasibility is still too large:

$$
\mathrm{action}_k = \mathrm{failure},
\qquad
\mathrm{failure\_reason}_k = \mathrm{tiny\_step},
\qquad
\|r_{\mathrm{prim}}(w_k)\|_\infty > \varepsilon_{\mathrm{prim}}.
$$

In code this is the `tiny_step_trigger` in
[ns_sqp_impl.cpp](/home/harper/Documents/moto/src/solver/sqp_impl/ns_sqp_impl.cpp).

At restoration entry the solver now prints:

- `=== enter restoration ===`
- one summary line for the outer iterate at entry

Restoration iterations themselves are printed by the same `print_stats(...)`
path as normal iterations, but with an `r` suffix in the iteration counter
such as `25r`, `26r`, and so on.

## 2. Restoration Overlay Graph

Restoration does **not** mutate the already-constructed normal `node_data`
objects in place.

Instead, `ns_sqp::restoration_graph()` lazily creates a second solver graph
from the active model graph:

1. each model edge is composed into a finalized interval `edge_ocp`
2. a restoration overlay problem is built on top of that composed problem
3. the overlay graph is realized into its own `node_data` / dense storage

So normal and restoration do **not** share:

- `node_data`
- `lag_data`
- merit / KKT storage
- stage Jacobian/Hessian buffers

They only synchronize selected state at restoration entry and exit.

## 3. Restoration NLP

Let

$$
w := (x,u,y),
\qquad
F(w)=0
$$

denote the hard dynamics constraints, and let

$$
c(w)=0
$$

denote the stacked equalities formed from `__eq_x` and `__eq_xu`, and

$$
g(w)\le 0
$$

denote the stacked inequalities formed from `__ineq_x` and `__ineq_xu`.

The restoration original problem is

$$
\begin{aligned}
\min_{w,t,p_c,n_c,p_d,n_d}\quad &
\operatorname{obj}_R(w)
 + \varrho_c \mathbf{1}^T (p_c+n_c)
 + \varrho_d \mathbf{1}^T (p_d+n_d) \\
\text{s.t.}\quad &
F(w)=0, \\
&
c(w)-p_c+n_c=0, \\
&
g(w)+t-p_d+n_d=0, \\
&
t \succeq 0,\quad
p_c \succeq 0,\quad n_c \succeq 0,\quad
p_d \succeq 0,\quad n_d \succeq 0 .
\end{aligned}
$$

Here:

- `F(w)=0` remains hard
- `c(w)` is elastic through `(p_c,n_c)`
- `g(w)` is elastic through `(t,p_d,n_d)`
- `\varrho_c = settings.restoration.rho_eq`
- `\varrho_d = settings.restoration.rho_ineq`
- `\operatorname{obj}_R(w)` is currently the proximal anchoring cost on `u/y`

## 4. Search Model vs Original Objective

The current implementation now makes a strict distinction between:

- `objective`
- `penalized_obj`

### 4.1 `objective`

`objective` is the **original phase objective**.

For restoration:

$$
\mathrm{objective}_R
=
\operatorname{obj}_R(w)
 + \varrho_c \mathbf{1}^T (p_c+n_c)
 + \varrho_d \mathbf{1}^T (p_d+n_d).
$$

This is what is shown in the `obj` column of `print_stats(...)`.

### 4.2 `penalized_obj`

`penalized_obj` is the **search objective** used by restoration line search.

The positivity barriers are not part of the restoration original NLP, but they
are part of the primal-dual IPM search model. The current code stores their
aggregate contribution in `search_penalty(...)` and reports

$$
\mathrm{penalized\_obj}_R
=
\mathrm{objective}_R - \mathrm{search\_barrier\_value}.
$$

Since the barrier log-sum is negative, this makes the search objective
numerically larger than the original objective.

The logs therefore show both:

- `obj` in the table = original restoration objective
- `search_obj` on the extra restoration line = line-search objective

This is why the restoration logs can show, for example:

- `objective ≈ 1.95e4`
- `search_obj ≈ 8.11e4`

at the same iterate.

## 5. Overlay Wrapper Structure

Restoration is implemented as standard `cost/constr` overlays.

### 5.1 Proximal Cost

`resto_prox_cost` is a real `__cost` function.

It contributes through the standard cost channels:

- value into `cost_`
- gradient into `cost_jac_` and `lag_jac_`
- Hessian into base `lag_hess_`

No hand-injected restoration Hessian storage remains.

### 5.2 Elastic Equality Wrapper

`resto_eq_elastic_constr` wraps a source equality and represents

$$
c(w)-p_c+n_c=0,
\qquad
p_c>0,\quad n_c>0
$$

for the search model.

Its condensed contributions go through the normal non-cost channels:

- first-order correction into `lag_jac_corr_`
- second-order correction into `hessian_modification_`

### 5.3 Elastic Inequality Wrapper

`resto_ineq_elastic_ipm_constr` wraps a source inequality and represents

$$
g(w)+t-p_d+n_d=0
$$

with positivity variables for

$$
t>0,\qquad p_d>0,\qquad n_d>0
$$

in the search model.

This is a restoration-specific double-slack IPM wrapper.
It is not the normal `ipm_constr`, but it follows the same lifecycle style:

- initialize
- finalize Newton step
- predictor bookkeeping
- line-search bounds
- backup / restore
- affine step

## 6. Iterative Refinement

Restoration reuses the same iterative-refinement shell as normal.

The important current fact is:

- iterative refinement no longer depends on a framework-external
  `refinement_local_residuals(...)`
- if restoration local residual summaries are printed, they are aggregated from
  the currently active restoration wrappers on the overlay graph

So there is now no separate runtime residual object outside the active
`cost/constr` framework.

## 7. Line Search And `alpha_min`

Restoration line search uses the same backtracking shell as normal, but with a
restoration phase metric:

- primal metric: `inf_prim_res`
- dual metric: `max(inf_dual_res, inf_comp_res)`
- objective metric: `penalized_obj`

The current restoration `alpha_min` is **not** an independent absolute floor.
It is computed as

$$
\alpha_{\min}

=
\max\!\Bigl(
\texttt{settings.ls.primal.alpha\_min},
\texttt{settings.restoration.alpha\_min\_factor}\cdot \alpha_{\mathrm{init}}
\Bigr),
$$

where `\alpha_init` is the current iteration's initial primal step bound.

This means restoration can still end up with a very small `alpha_min` if the
initial primal bound is already tiny.

The current `arm` logs show exactly that behavior: restoration does trigger its
own `alpha_min` rule, but the resulting minimum is still small because the
restoration step is already heavily clipped before backtracking starts.

## 8. Entry / Exit Synchronization

At restoration entry:

- primal state is copied from outer graph to overlay graph
- hard duals are copied from outer graph to overlay graph
- proximal references are seeded from the current overlay primal state
- overlay constraints are initialized through the standard `ineq_soft::initialize`
  path

At restoration success:

- primal state is copied back
- hard duals are copied back
- equality duals are reset according to the restoration threshold policy
- outer graph derivatives are recomputed in normal mode

At restoration failure:

- the outer graph remains the active source of truth
- restoration graph data is discarded as phase-local state

## 9. Current Numerical Behavior

As of the current implementation:

- `quadruped` normal behavior remains intact
- `arm` enters restoration cleanly on the overlay graph
- restoration prints:
  - explicit entry / exit markers
  - full per-iteration stats with `r`-suffixed iteration numbers
  - a separate line showing `objective`, `search_obj`, and barrier contribution

The current remaining failure mode in `arm` is not graph synchronization or
objective bookkeeping. It is a tiny-step / direction-quality issue:

- restoration can reduce primal infeasibility from the large entry value to
  approximately `1.112e+00`
- but only with extremely small accepted steps
- backtracking eventually reaches `alpha_primal << alpha_min`
- restoration exits with `iter_result_restoration_failed`
