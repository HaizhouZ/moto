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

The restoration overlay graph is now cached inside the active SQP graph state.
It is rebuilt only when:

- the modeled graph is dirty
- restoration overlay settings change

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

Instead, `ns_sqp::restoration_graph()` lazily creates or refreshes a second
solver graph from the active model graph:

1. each model edge is composed into a finalized interval `edge_ocp`
2. a restoration overlay problem is built on top of that composed problem
3. the overlay graph is realized into its own `node_data` / dense storage

That realized overlay runtime is cached and reused across restoration entries
until the graph or overlay settings invalidate it.

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
\min_{w,p_c,n_c,p_d,n_d}\quad &
\operatorname{obj}_R(w)
 + \varrho_c \mathbf{1}^T (p_c+n_c)
 + \varrho_d \mathbf{1}^T (p_d+n_d) \\
\text{s.t.}\quad &
F(w)=0, \\
&
c(w)-p_c+n_c=0, \\
&
g(w)-p_d+n_d \le 0, \\
&
p_c \succeq 0,\quad n_c \succeq 0,\quad
p_d \succeq 0,\quad n_d \succeq 0 .
\end{aligned}
$$

Here:

- `F(w)=0` remains hard
- `c(w)` is elastic through `(p_c,n_c)`
- `g(w)` is elastic through `(p_d,n_d)`
- `\varrho_c = settings.restoration.rho_eq`
- `\varrho_d = settings.restoration.rho_ineq`
- `\operatorname{obj}_R(w)` is currently the proximal anchoring cost on `u/y`

## 4. Restoration Lagrangian And Primal-Dual IPM Form

For the restoration NLP, introduce multipliers

$$
\lambda_F,\qquad \lambda_c
$$

for

$$
F(w)=0,\qquad c(w)-p_c+n_c=0.
$$

The restoration Lagrangian is

$$
\begin{aligned}
\mathcal{L}_R
=\;&
\operatorname{obj}_R(w)
+ \varrho_c \mathbf{1}^T (p_c+n_c)
+ \varrho_d \mathbf{1}^T (p_d+n_d) \\
&+ \lambda_F^T F(w)
+ \lambda_c^T \bigl(c(w)-p_c+n_c\bigr).
\end{aligned}
$$

This is the barrier-free original restoration problem. No log-barrier term
belongs to `\mathcal{L}_R`.

For the implemented primal-dual search model, one must additionally include the
positivity-dual products. In other words, the local search blocks are derived
from Lagrangians of the form

$$
\mathcal{L}_{R,\mathrm{search}}
=
\mathcal{L}_R
-
(\nu_p^c)^T p_c
-
(\nu_n^c)^T n_c
-
(\nu_t)^T t
-
(\nu_p^d)^T p_d
-
(\nu_n^d)^T n_d ,
$$

with the understanding that only the variables relevant to a given local block
are present. The complementarity perturbation by `\mu` then appears in the KKT
system, not as an extra term inside the original objective.

For the inequality-elastic part, the implementation does not introduce a
separate global `\lambda_d` term on top of the slackened relation. Instead, the
search model is written directly in terms of the local slack/elastic variables
and their positivity duals.

### 4.1 Equality Elastic Block

For equality elastic constraints, the search model keeps

$$
c(w)-p_c+n_c=0,\qquad
p_c>0,\qquad n_c>0.
$$

With positivity duals `\nu_p^c,\nu_n^c`, the primal-dual IPM KKT block is

$$
\mathcal{L}_{R,\mathrm{eq\text{-}search}}
=
\operatorname{obj}_R(w)
+ \varrho_c \mathbf{1}^T (p_c+n_c)
+ \lambda_F^T F(w)
+ \lambda_c^T \bigl(c(w)-p_c+n_c\bigr)
- (\nu_p^c)^T p_c
- (\nu_n^c)^T n_c .
$$

$$
\begin{aligned}
r_c   &= c(w)-p_c+n_c, \\
r_p^c &= \varrho_c - \lambda_c - \nu_p^c, \\
r_n^c &= \varrho_c + \lambda_c - \nu_n^c, \\
r_{s,p}^c &= \nu_p^c \circ p_c - \mu \mathbf{1}, \\
r_{s,n}^c &= \nu_n^c \circ n_c - \mu \mathbf{1}.
\end{aligned}
$$

This is the local system condensed by `resto_eq_elastic_constr`.

#### Equality Linearized Local KKT

Given a stage perturbation `\delta c`, the local Newton system is

$$
\begin{aligned}
\delta c - \delta p_c + \delta n_c &= -r_c, \\
-\delta \lambda_c - \delta \nu_p^c &= -r_p^c, \\
\delta \lambda_c - \delta \nu_n^c &= -r_n^c, \\
\nu_p^c \circ \delta p_c + p_c \circ \delta \nu_p^c &= -r_{s,p}^c, \\
\nu_n^c \circ \delta n_c + n_c \circ \delta \nu_n^c &= -r_{s,n}^c .
\end{aligned}
$$

With the current implementation's regularization parameter `\lambda_{\mathrm{reg}}`,
the code uses the condensed quantities

$$
\begin{aligned}
\mathrm{combo}_p^c &=
\frac{p_c \circ r_p^c + \lambda_{\mathrm{reg}} r_{s,p}^c}
     {p_c + \lambda_{\mathrm{reg}} \nu_p^c}, \\
\mathrm{combo}_n^c &=
\frac{n_c \circ r_n^c + \lambda_{\mathrm{reg}} r_{s,n}^c}
     {n_c + \lambda_{\mathrm{reg}} \nu_n^c}.
\end{aligned}
$$

Then

$$
\begin{aligned}
M_c^{-1} &=
\left(
\frac{p_c}{p_c + \lambda_{\mathrm{reg}} \nu_p^c}
+
\frac{n_c}{n_c + \lambda_{\mathrm{reg}} \nu_n^c}
\right)^{-1}, \\
b_c &= r_c + \mathrm{combo}_p^c - \mathrm{combo}_n^c,
\end{aligned}
$$

and the condensed dual step is

$$
\delta \lambda_c = M_c^{-1}(\delta c + b_c).
$$

The local recover formulas are

$$
\begin{aligned}
\delta \nu_p^c &= \frac{\nu_p^c \circ (r_p^c-\delta\lambda_c)-r_{s,p}^c}
                      {p_c+\lambda_{\mathrm{reg}}\nu_p^c}, \\
\delta \nu_n^c &= \frac{\nu_n^c \circ (r_n^c+\delta\lambda_c)-r_{s,n}^c}
                      {n_c+\lambda_{\mathrm{reg}}\nu_n^c}, \\
\delta p_c &= \delta\lambda_c + \lambda_{\mathrm{reg}}\delta\nu_p^c - r_p^c, \\
\delta n_c &= -\delta\lambda_c + \lambda_{\mathrm{reg}}\delta\nu_n^c - r_n^c .
\end{aligned}
$$

This is exactly the object condensed into `lag_jac_corr_` and
`hessian_modification_`.

### 4.2 Inequality Elastic Block With Slack `t`

For inequality elastic constraints, the restoration wrapper first rewrites the
original inequality through a positive slack:

$$
g(w)+t-p_d+n_d=0,
\qquad
t>0,\quad p_d>0,\quad n_d>0.
$$

There is **no separate** `\lambda_d` in the local restoration block. The
active local variables are

$$
t,\;p_d,\;n_d,\qquad \nu_t,\;\nu_p^d,\;\nu_n^d .
$$

The exact penalty acts only on `p_d,n_d`, while the positivity duals
`\nu_t,\nu_p^d,\nu_n^d` belong to the search model. The reduced local KKT block
is derived from

$$
\mathcal{L}_{R,\mathrm{ineq\text{-}search}}
=
\operatorname{obj}_R(w)
+ \varrho_d \mathbf{1}^T (p_d+n_d)
- (\nu_t)^T t
- (\nu_p^d)^T p_d
- (\nu_n^d)^T n_d
$$

together with the slack relation `g(w)+t-p_d+n_d=0`, and is therefore

$$
\begin{aligned}
r_d   &= g(w)+t-p_d+n_d, \\
r_p^d &= \varrho_d-\nu_t-\nu_p^d, \\
r_n^d &= \varrho_d+\nu_t-\nu_n^d, \\
r_{s,t} &= \nu_t \circ t - \mu \mathbf{1}, \\
r_{s,p}^d &= \nu_p^d \circ p_d - \mu \mathbf{1}, \\
r_{s,n}^d &= \nu_n^d \circ n_d - \mu \mathbf{1}.
\end{aligned}
$$

This is the local slack-plus-elastic search block that the restoration
implementation should condense.

#### Inequality Linearized Local KKT

Given a stage perturbation `\delta g`, the local Newton system is

$$
\begin{aligned}
\delta g + \delta t - \delta p_d + \delta n_d &= -r_d, \\
\delta p_d - \delta \nu_t - \lambda_{\mathrm{reg}} \delta \nu_p^d &= -r_p^d, \\
\delta n_d + \delta \nu_t - \lambda_{\mathrm{reg}} \delta \nu_n^d &= -r_n^d, \\
\nu_t \circ \delta t + t \circ \delta \nu_t &= -r_{s,t}, \\
\nu_p^d \circ \delta p_d + p_d \circ \delta \nu_p^d &= -r_{s,p}^d, \\
\nu_n^d \circ \delta n_d + n_d \circ \delta \nu_n^d &= -r_{s,n}^d .
\end{aligned}
$$

Here `\lambda_{\mathrm{reg}}` plays the same role as in the equality-elastic
block: it regularizes the dual recovery equations for the positive variables.

Introduce the regularized diagonal denominators

$$
\begin{aligned}
D_t^d &:= \nu_t, \\
D_p^d &:= p_d + \lambda_{\mathrm{reg}} \nu_p^d, \\
D_n^d &:= n_d + \lambda_{\mathrm{reg}} \nu_n^d .
\end{aligned}
$$

Define the condensed quantities

$$
\begin{aligned}
\mathrm{combo}_t &= \frac{r_{s,t}}{D_t^d}, \\
\mathrm{combo}_p^d &=
\frac{p_d \circ r_p^d + \lambda_{\mathrm{reg}} r_{s,p}^d}
     {D_p^d}, \\
\mathrm{combo}_n^d &=
\frac{n_d \circ r_n^d + \lambda_{\mathrm{reg}} r_{s,n}^d}
     {D_n^d}.
\end{aligned}
$$

Substituting

$$
\begin{aligned}
\delta t   &= -\mathrm{combo}_t   - \frac{t}{D_t^d}\delta\nu_t, \\
\delta p_d &= -\mathrm{combo}_p^d + \frac{p_d}{D_p^d}\delta\nu_t, \\
\delta n_d &= -\mathrm{combo}_n^d - \frac{n_d}{D_n^d}\delta\nu_t
\end{aligned}
$$

into the primal equation gives

$$
\left(
\frac{t}{D_t^d}
+
\frac{p_d}{D_p^d}
+
\frac{n_d}{D_n^d}
\right)\delta\nu_t
=
\delta g + b_d,
$$

with

$$
b_d = r_d - \mathrm{combo}_t + \mathrm{combo}_p^d - \mathrm{combo}_n^d .
$$

Therefore the condensed step is

$$
\begin{aligned}
M_d^{-1} &=
\left(
\frac{t}{D_t^d}
+
\frac{p_d}{D_p^d}
+
\frac{n_d}{D_n^d}
\right)^{-1}, \\
\delta \nu_t &= M_d^{-1}(\delta g + b_d), \\
\delta \nu_p^d &=
\frac{\nu_p^d \circ (r_p^d-\delta\nu_t)-r_{s,p}^d}
     {D_p^d}, \\
\delta \nu_n^d &=
\frac{\nu_n^d \circ (r_n^d+\delta\nu_t)-r_{s,n}^d}
     {D_n^d}.
\end{aligned}
$$

The local recover formulas are then

$$
\begin{aligned}
\delta t   &= -\frac{r_{s,t} + t \circ \delta\nu_t}{D_t^d}, \\
\delta p_d &= \delta\nu_t + \lambda_{\mathrm{reg}}\delta\nu_p^d - r_p^d, \\
\delta n_d &= -\delta\nu_t + \lambda_{\mathrm{reg}}\delta\nu_n^d - r_n^d.
\end{aligned}
$$

This is the regularized reduced local solve that the restoration inequality
wrapper is intended to implement. In code terms:

- `M_d^{-1}` corresponds to `minv_diag`
- `b_d` corresponds to the condensed right-hand side used to build `minv_bd`
- the recovered `\delta\nu_t,\delta\nu_p^d,\delta\nu_n^d,\delta t,\delta p_d,\delta n_d`
  correspond to the local Newton step recovered in `recover_local_step(...)`

### 4.3 Interpretation Of `\nu_t`

The important point is:

- there is no separate `\lambda_d` in the implemented local block
- `\nu_t` is the positivity dual associated with the slack `t`
- the search model perturbs the inequality through `t>0`
- `p_d,n_d` remain the exact-penalty elastic variables

Equivalently, the search block can be viewed as a perturbed treatment of

$$
g(w)-p_d+n_d < 0
$$

with `t` as its explicit positive slack and `\nu_t` as the corresponding
positivity dual. This is the mathematical reference for
`resto_ineq_elastic_ipm_constr`.

## 5. Search Model vs Original Objective

The current implementation now makes a strict distinction between:

- `objective`
- `penalized_obj`

### 5.1 `objective`

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

### 5.2 `penalized_obj`

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

## 6. Overlay Wrapper Structure

Restoration is implemented as standard `cost/constr` overlays.

### 6.1 Proximal Cost

`resto_prox_cost` is a real `__cost` function.

It contributes through the standard cost channels:

- value into `cost_`
- gradient into `cost_jac_` and `lag_jac_`
- Hessian into base `lag_hess_`

No hand-injected restoration Hessian storage remains.

### 6.2 Elastic Equality Wrapper

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

### 6.3 Elastic Inequality Wrapper

`resto_ineq_elastic_ipm_constr` wraps a source inequality and represents

$$
g(w)+t-p_d+n_d = 0
$$

with positivity variables

$$
t>0,\qquad p_d>0,\qquad n_d>0
$$

in the search model.

This is a restoration-specific elastic-IPM wrapper built from the slack form of
the original inequality. It is not the normal `ipm_constr`, but it follows the
same lifecycle style:

- initialize
- finalize Newton step
- predictor bookkeeping
- line-search bounds
- backup / restore
- affine step

## 7. Iterative Refinement

Restoration reuses the same iterative-refinement shell as normal.

The important current fact is:

- iterative refinement does not depend on a framework-external restoration
  residual object
- if restoration local residual summaries are printed, they are aggregated from
  the currently active restoration wrappers on the overlay graph

So there is now no separate runtime residual object outside the active
`cost/constr` framework.

## 8. Line Search And `alpha_min`

Restoration line search uses the same backtracking shell as normal, but with a
restoration phase metric:

- primal metric: `inf_prim_res`
- dual metric: `max(inf_dual_res, inf_comp_res)`
- objective metric: `penalized_obj`

Current implementation detail:

- restoration is only supported with filter line search
- `merit_backtracking` currently throws at restoration entry

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

## 9. Entry / Exit Synchronization

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
- the accepted restoration point must also satisfy the outer filter acceptance
  test and a primal-improvement threshold before restoration is declared
  successful

At restoration failure:

- the outer graph remains the active source of truth
- outer derivatives are refreshed in normal mode before returning
- the cached restoration graph remains available for future restoration entries;
  only phase-local active state is cleared

## 10. Current Numerical Behavior

As of the current implementation:

- `quadruped` normal behavior remains intact
- `arm` enters restoration on the overlay graph without mutating the normal
  graph in place
- restoration prints:
  - explicit entry / exit markers
  - full per-iteration stats with `r`-suffixed iteration numbers
  - a separate line showing `objective`, `search_obj`, and barrier contribution

The most common remaining failure mode is not graph synchronization or
objective bookkeeping. It is still a globalization / direction-quality issue:

- restoration may reduce primal infeasibility while accepting only very small
  steps
- backtracking can still terminate on the tiny-step condition
- the resulting status is either `iter_result_restoration_failed` or
  `iter_result_infeasible_stationary`, depending on the post-restoration KKT
  state
