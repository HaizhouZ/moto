# Restoration Problem

This note describes the restoration phase that is **currently implemented** in
`src/solver/sqp_impl/restoration.cpp` and the downstream Riccati path.

## Entry Condition

The outer SQP loop enters restoration exactly when the normal globalization
returns a line-search failure caused by the IPOPT-style tiny-step test:

$$
\mathrm{action}_k = \mathrm{failure},
\qquad
\mathrm{failure\_reason}_k = \mathrm{tiny\_step},
\qquad
\|r_{\mathrm{prim}}(x_k)\|_\infty > \varepsilon_{\mathrm{prim}}.
$$

Equivalently, restoration is triggered when

$$
\alpha_k \le \alpha_k^{\min},
\qquad
\|r_{\mathrm{prim}}(x_k)\|_\infty > \varepsilon_{\mathrm{prim}},
$$

where `\alpha_k^{\min}` is computed in
`src/solver/sqp_impl/line_search.cpp` from the existing line-search quantities
plus the safety factor
\[
\alpha_{\min}^{\mathrm{fac}}
=
\mathrm{settings.restoration.alpha\_min\_factor}.
\]

More explicitly, the current implementation uses

$$
\alpha_k^{\min}
=
\alpha_{\min}^{\mathrm{fac}}
\begin{cases}
\min\!\left\{
\gamma_\theta,\;
\dfrac{\gamma_\phi \,\theta(x_k)}{-\nabla \varphi_{\mu_j}(x_k)^T d_k^x},\;
\dfrac{\theta(x_k)^{s_\theta}}
{\bigl(-\nabla \varphi_{\mu_j}(x_k)^T d_k^x\bigr)^{s_\phi}}
\right\},
&
\text{if } \nabla \varphi_{\mu_j}(x_k)^T d_k^x < 0,\ \theta(x_k)\le \theta_{\min},
\\[1.2ex]
\min\!\left\{
\gamma_\theta,\;
\dfrac{\gamma_\phi \,\theta(x_k)}{-\nabla \varphi_{\mu_j}(x_k)^T d_k^x}
\right\},
&
\text{if } \nabla \varphi_{\mu_j}(x_k)^T d_k^x < 0,\ \theta(x_k)> \theta_{\min},
\\[1.2ex]
\gamma_\theta,
&
\text{otherwise.}
\end{cases}
$$

The code-level identification is

$$
\theta(x_k)=\mathrm{prim\_res\_l1},\qquad
\theta_{\min}=\mathrm{constr\_vio\_min},
$$

$$
\gamma_\theta=\mathrm{ls.primal\_gamma},\qquad
\gamma_\phi=\mathrm{ls.dual\_gamma},
$$

and

$$
-\nabla \varphi_{\mu_j}(x_k)^T d_k^x
\;\widehat{=}\;
-\,\mathrm{fullstep\_dec}_k.
$$

## Variables

At a given restoration trigger point, the solver keeps optimizing the same node
primal variables

$$
(x_k, u_k, y_k),
$$

with the same inequality/barrier machinery as the normal SQP loop.

The reference point for restoration is the iterate where restoration is entered:

$$
u_k^{\mathrm{ref}},\qquad y_k^{\mathrm{ref}}.
$$

and the per-component scaling is

$$
\sigma_{u,i}^2 = \frac{1}{\max(|u^{\mathrm{ref}}_{k,i}|,1)^2},
\qquad
\sigma_{y,i}^2 = \frac{1}{\max(|y^{\mathrm{ref}}_{k,i}|,1)^2}.
$$

## What Is Removed

During restoration,
`node_data::update_approximation(..., include_original_cost=false)`
skips every `__cost` term at the source.

Therefore the restoration model excludes the original user cost contributions

$$
f(x,u,y),\qquad \nabla f(x,u,y),\qquad \nabla^2 f(x,u,y),
$$

from the search direction construction.

Only barrier/IPM contributions, constraint linearizations, and restoration
corrections remain.

## Restoration Subproblem

Let

$$
F_k(x_k,y_k,u_k)=0
$$

denote the dynamics residual at stage `k`, and let

$$
h_k(x_k,y_k,u_k)
$$

denote the stacked hard equality residual formed from `__eq_x` and `__eq_xu`,
with Jacobian

$$
J_k = \begin{bmatrix} J_{x,k} & J_{u,k} \end{bmatrix}.
$$

Then the restoration direction is computed in `sqp_iter(..., gauss_newton=true)`
with the following structure:

### Hard Constraints

Dynamics stay as hard constraints.

At the linearized level, the step must satisfy the dynamics equations handled by
the standard Riccati recursion:

$$
F_k + F_{x,k} d_{x,k} + F_{u,k} d_{u,k} - d_{y,k} = 0.
$$

So restoration is **not** a fully unconstrained least-squares phase.

### Equality Restoration Term

The hard equalities `__eq_x` and `__eq_xu` are moved from the constrained path
into an **ALM / PMM-style local model** with parameter `rho_eq`.

This is not just a pure penalty model. The existing equality multipliers are
still present through the generic constraint path:

$$
\lambda_k^\top \bigl(h_k + J_k d_k\bigr)
$$

and restoration adds the quadratic PMM term

$$
\frac{1}{2\rho_{\mathrm{eq}}}\,\|h_k + J_k d_k\|^2.
$$

So the local restoration model for `__eq_x` and `__eq_xu` is

$$
\lambda_k^\top \bigl(h_k + J_k d_k\bigr)
\;+\;
\frac{1}{2\rho_{\mathrm{eq}}}\,\|h_k + J_k d_k\|^2,
$$

Expanding this gives exactly the terms that appear in
`src/solver/nsp_impl/presolve.cpp`:

$$
J_k^\top \lambda_k,
\qquad
\frac{1}{\rho_{\mathrm{eq}}} J_k^\top h_k,
\qquad
\frac{1}{\rho_{\mathrm{eq}}} J_k^\top J_k.
$$

So, ignoring barrier terms, dynamics coupling, and proximal anchoring, the
restoration direction is better interpreted as coming from the local
augmented-Lagrangian / PMM model

$$
\min_d\;
\sum_k \lambda_k^\top \bigl(h_k + J_k d_k\bigr)
\;+\;
\sum_k \frac{1}{2\rho_{\mathrm{eq}}}\,\|h_k + J_k d_k\|^2
$$

rather than from a pure least-squares penalty alone.

### Proximal Anchoring

In addition, restoration injects first-order anchoring terms on `u` and `y`:

$$
g^{\mathrm{prox}}_{u,k}
=
\rho_u\,\operatorname{diag}(\sigma_u^2)\,(u_k-u_k^{\mathrm{ref}}),
\qquad
g^{\mathrm{prox}}_{y,k}
=
\rho_y\,\operatorname{diag}(\sigma_y^2)\,(y_k-y_k^{\mathrm{ref}}).
$$

These are written into `lag_jac_corr_[__u]` and `lag_jac_corr_[__y]`.

So the implemented QP direction is additionally biased by the linear term

$$
\sum_k \bigl(g^{\mathrm{prox}}_{u,k}\bigr)^\top d_{u,k}
\;+\;
\sum_k \bigl(g^{\mathrm{prox}}_{y,k}\bigr)^\top d_{y,k}.
$$

Important: the current code does **not** add the matching proximal Hessian for
these `u/y` anchoring terms. Therefore the implemented restoration direction is
only first-order consistent with the quadratic proximal objective

$$
\sum_k \frac{\rho_u}{2}
\left\|
\operatorname{diag}(\sigma_u^2)^{1/2}(u_k-u_k^{\mathrm{ref}})
\right\|^2
\;+\;
\sum_k \frac{\rho_y}{2}
\left\|
\operatorname{diag}(\sigma_y^2)^{1/2}(y_k-y_k^{\mathrm{ref}})
\right\|^2
$$

but does not currently realize that full quadratic model exactly.

## Practical Interpretation

Putting the pieces together, the restoration phase currently behaves like:

$$
\begin{aligned}
\min_{x,u,y}\quad
&\sum_k \lambda_k^\top h_k(x,u,y)
\;+\;
\sum_k \frac{1}{2\rho_{\mathrm{eq}}}\,\|h_k(x,u,y)\|^2
\;+\;
\text{barrier/IPM terms}
\;+\;
\text{first-order proximal anchoring on }(u,y)
\\
\text{s.t.}\quad
&F_k(x,u,y)=0,\qquad \forall k.
\end{aligned}
$$

with the search direction formed from the corresponding linearized
Gauss-Newton/PMM model.

## Globalization During Restoration

Restoration reuses the existing filter line-search machinery, but with the
following restrictions:

$$
\mathrm{switching\_condition} = \mathrm{false},
\qquad
\mathrm{Armijo\ branch\ disabled},
\qquad
\mathcal{F}_{k+1} = \mathcal{F}_k.
$$

Because the original cost is removed at the approximation stage, the filter
pair used during restoration is effectively

$$
\Bigl(
\theta(x),
\phi_{\mathrm{bar}}(x)
\Bigr)
=
\left(
\mathrm{prim\_res\_l1},
\;
-\,\mu \sum_i \log s_i
\right),
$$

rather than by a dedicated scalar feasibility merit function.

## Success / Failure Classification

Let

$$
\eta_{\mathrm{resto}}
=
\mathrm{restoration\_improvement\_frac},
\qquad
\phi_k
=
\|r_{\mathrm{prim}}(x_k)\|_\infty.
$$

Then the current implementation classifies the restoration phase as follows.

Success:

$$
\mathrm{rest\_action}_k = \mathrm{accept},
\qquad
\phi_k < \eta_{\mathrm{resto}}\,\phi_{\mathrm{entry}}.
$$

Infeasible stationary:

$$
\|r_{\mathrm{dual}}(x_k)\|_\infty < \varepsilon_{\mathrm{dual}},
\qquad
\phi_k \not< \varepsilon_{\mathrm{prim}},
\qquad
\text{stall detected},
\qquad
\mathrm{rest\_action}_k = \mathrm{failure}.
$$

Restoration failed:

$$
\text{none of the two conditions above holds before the restoration loop exits.}
$$

So the implemented restoration phase is best viewed as a

- hard-dynamics
- equality-feasibility ALM / PMM phase
- with first-order `u/y` anchoring
- and standard barrier/filter globalization

rather than a pure standalone feasibility NLP with its own exact merit
objective.
