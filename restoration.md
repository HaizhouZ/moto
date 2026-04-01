# Restoration Problem

This note describes the restoration phase that is **currently implemented** in
[`src/solver/sqp_impl/restoration.cpp`](/home/harper/Documents/moto/src/solver/sqp_impl/restoration.cpp),
[`src/solver/nsp_impl/presolve.cpp`](/home/harper/Documents/moto/src/solver/nsp_impl/presolve.cpp),
[`src/solver/nsp_impl/rollout.cpp`](/home/harper/Documents/moto/src/solver/nsp_impl/rollout.cpp),
and the restoration runtime helpers under
[`include/moto/solver/restoration/`](/home/harper/Documents/moto/include/moto/solver/restoration).

The current implementation is an **explicit elastic restoration condensed onto
the global stage variables** $w=(x,u,y)$. The elastic variables
$(p,n,\nu_p,\nu_n,\lambda_c)$ are maintained only in a stage-local
restoration runtime and are eliminated analytically when assembling the global
Riccati system.

For the full checkpoint derivation, see
[`restoration_explicit_checkpoint.md`](/home/harper/Documents/moto/restoration_explicit_checkpoint.md).

## Entry Condition

The outer SQP loop enters restoration when the normal globalization returns a
tiny-step failure while the primal infeasibility is still too large:

$$
\mathrm{action}_k = \mathrm{failure},
\qquad
\mathrm{failure\_reason}_k = \mathrm{tiny\_step},
\qquad
\|r_{\mathrm{prim}}(x_k)\|_\infty > \varepsilon_{\mathrm{prim}}.
$$

Equivalently,

$$
\alpha_k \le \alpha_k^{\min},
\qquad
\|r_{\mathrm{prim}}(x_k)\|_\infty > \varepsilon_{\mathrm{prim}}.
$$

The current implementation uses the IPOPT-style adaptive threshold
$\alpha_k^{\min}$ based on:

$$
\theta(x_k)=\mathrm{prim\_res\_l1},
\qquad
\theta_{\min}=\mathrm{ls.constr\_vio\_min},
$$

$$
\gamma_\theta=\mathrm{ls.primal\_gamma},
\qquad
\gamma_\phi=\mathrm{ls.dual\_gamma},
\qquad
\alpha_{\min}^{\mathrm{fac}}
=
\mathrm{settings.restoration.alpha\_min\_factor}.
$$

## Restoration Initialization

At restoration entry, the barrier parameter is reset to

$$
\bar\mu_0 = \max\!\bigl(\mu_j,\ \|c(w_k)\|_\infty\bigr).
$$

The restoration reference point is simply the current iterate:

$$
w_R = w_k.
$$

The normal equality multipliers are reset:

$$
\lambda^{\mathrm{dyn}}_0 = 0,
\qquad
\lambda^{\mathrm{eq\_x}}_0 = 0,
\qquad
\lambda^{\mathrm{eq\_xu}}_0 = 0.
$$

The `u` and `y` references are snapshotted for proximal anchoring, and
componentwise proximal scaling still uses

$$
\sigma(\xi) = \frac{1}{\max(|\xi|,1)}.
$$

## Restoration NLP

Let

$$
w := (x,u,y),
\qquad
F(w)=0
$$

be the hard dynamics constraints, and let

$$
c(w)=0
$$

be the stacked equalities formed from `__eq_x` and `__eq_xu`.

The restoration subproblem is

$$
\begin{aligned}
\min_{w,p,n}\quad &
\phi_R(w)
 + \varrho \mathbf{1}^T (p+n)
 - \bar\mu \sum_i \ln p_i
 - \bar\mu \sum_i \ln n_i \\
\text{s.t.}\quad &
F(w)=0, \\
&
c(w)-p+n=0, \\
&
p \succeq 0,\quad n \succeq 0 .
\end{aligned}
$$

Here:

- `F(w)=0` stays hard throughout restoration.
- `c(w)` is restored elastically.
- `\varrho` is the elastic exact-penalty weight, currently stored in
  `settings.restoration.rho_eq`.
- `\rho_\lambda` is the Hippo-style local elastic regularization, currently stored
  in `settings.restoration.lambda_reg`.
- `\phi_R(w)` contains the restoration-only terms on `w`, currently the
  proximal anchoring on `u/y`.

## KKT Residuals

Using Hippo-style notation, define

$$
A := F_w,
\qquad
C := c_w.
$$

The local restoration residuals are

$$
r_w := \nabla_w \phi_R(w) + A^T \lambda_f + C^T \lambda_c ,
$$

$$
r_f := F(w) ,
$$

$$
r_c := c(w)-p+n ,
$$

$$
r_p := \varrho \mathbf{1} - \lambda_c - \nu_p ,
$$

$$
r_n := \varrho \mathbf{1} + \lambda_c - \nu_n ,
$$

$$
r_{s,p} := \nu_p \odot p - \bar\mu \mathbf{1} ,
$$

$$
r_{s,n} := \nu_n \odot n - \bar\mu \mathbf{1} .
$$

Introduce

$$
T_p := \operatorname{diag}(p),
\qquad
T_n := \operatorname{diag}(n),
\qquad
N_p := \operatorname{diag}(\nu_p),
\qquad
N_n := \operatorname{diag}(\nu_n).
$$

## Implemented Local Linearized System

For `lambda_reg > 0`, the current implementation uses the regularized local
linearization

$$
C\,\delta w - \delta p + \delta n = -r_c ,
$$

$$
\delta p - \delta \lambda_c - \rho_\lambda \delta \nu_p = -r_p ,
$$

$$
\delta n + \delta \lambda_c - \rho_\lambda \delta \nu_n = -r_n ,
$$

$$
N_p\,\delta p + T_p\,\delta \nu_p = -r_{s,p},
$$

$$
N_n\,\delta n + T_n\,\delta \nu_n = -r_{s,n}.
$$

This is the system actually condensed by the code. The bound blocks do not
couple directly to $w$; all coupling comes from

$$
c(w)-p+n=0.
$$

## Condensation Onto The Global Stage System

The elastic variables are **not** part of the global Riccati state. Instead,
the stage-local block is condensed exactly.

From the regularized stationarity rows,

$$
\delta \nu_p = \frac{\delta p + r_p - \delta \lambda_c}{\rho_\lambda},
\qquad
\delta \nu_n = \frac{\delta n + r_n + \delta \lambda_c}{\rho_\lambda}.
$$

Substituting into the complementarity rows gives

$$
\delta p
=
(T_p + \rho_\lambda N_p)^{-1} T_p\,\delta \lambda_c
-
(T_p + \rho_\lambda N_p)^{-1}(\rho_\lambda r_{s,p}+T_p r_p),
$$

$$
\delta n
=
-(T_n + \rho_\lambda N_n)^{-1} T_n\,\delta \lambda_c
-
(T_n + \rho_\lambda N_n)^{-1}(\rho_\lambda r_{s,n}+T_n r_n).
$$

Substituting these into

$$
C\,\delta w - \delta p + \delta n = -r_c
$$

produces

$$
M_\rho \,\delta \lambda_c = C\,\delta w + b_c,
$$

with

$$
M_\rho := T_p (T_p + \rho_\lambda N_p)^{-1} + T_n (T_n + \rho_\lambda N_n)^{-1},
$$

and

$$
b_c :=
r_c
+ (T_p + \rho_\lambda N_p)^{-1}(\rho_\lambda r_{s,p}+T_p r_p)
- (T_n + \rho_\lambda N_n)^{-1}(\rho_\lambda r_{s,n}+T_n r_n).
$$

Therefore

$$
\delta \lambda_c = M_\rho^{-1}(C\,\delta w + b_c).
$$

The base stage gradient already contains

$$
C^T \lambda_c
$$

through the normal `lag_jac_` assembly. So the explicit elastic condensation
adds only

$$
C^T M_\rho^{-1} b_c
$$

to the global stage gradient. Define

$$
\hat\eta_c := M_\rho^{-1} b_c.
$$

Then the contribution written into the global stage system is

$$
\Delta \tilde Q_R{}_{(\cdot)} = \hat\eta_c^T c_{(\cdot)} ,
$$

$$
\Delta \tilde Q_R{}_{(\cdot,\cdot)} = c_{(\cdot)}^T M_\rho^{-1} c_{(\cdot)} .
$$

So the explicit elastic restoration is solved as:

- a **global** Riccati system only in $w=(x,u,y)$,
- plus a **local** restoration KKT block for
  $(p,n,\nu_p,\nu_n,\lambda_c)$,
- connected by exact Schur-complement condensation.

### Stage-Level Mapping Used By The Code

At the nullspace/Riccati stage, the stacked restoration equality Jacobians are
stored as

$$
C_u \;\widehat{=}\; \texttt{nsp.s\_c\_stacked},
\qquad
C_x \;\widehat{=}\; \texttt{nsp.s\_c\_stacked\_0\_K},
\qquad
c_0 \;\widehat{=}\; \texttt{nsp.s\_c\_stacked\_0\_k}.
$$

The helper computes

$$
\hat\eta_c = M_\rho^{-1} b_c,
\qquad
W = M_\rho^{-1}.
$$

Then presolve writes the condensed elastic terms into the stage system as

$$
\Delta g_{R,u} := \hat\eta_c^T C_u,
\qquad
\Delta g_{R,x} := \hat\eta_c^T C_x,
$$

and

$$
Q_{zz} \mathrel{+}= C_u^T W C_u,
\qquad
u_{0,p,K} \mathrel{+}= C_u^T W C_x,
\qquad
V_{xx} \mathrel{+}= C_x^T W C_x .
$$

The first-order condensed terms are not written directly into the active stage
gradient. They are first accumulated into `lag_jac_corr_`, and
`data_base::activate_lag_jac_corr()` is then responsible for folding them into
the active `Q_x/Q_u/Q_y` seen by the next Riccati solve, together with the
other pending first-order corrections.

So the global Riccati solve only sees a correction on the existing
$(x,u,y)$-blocks; it never sees explicit stage variables for $p$ or $n$.

## Local Step Recovery

After rollout provides $\delta w$, the local restoration steps are recovered
with the same regularized condensed coefficients:

$$
\delta \lambda_c = M_\rho^{-1}(\delta c + b_c), \qquad \delta c := C \delta w .
$$

$$
\delta \nu_p =
(T_p + \rho_\lambda N_p)^{-1}\!\left(N_p (r_p - \delta \lambda_c) - r_{s,p}\right) ,
$$

$$
\delta \nu_n =
(T_n + \rho_\lambda N_n)^{-1}\!\left(N_n (r_n + \delta \lambda_c) - r_{s,n}\right) ,
$$

$$
\delta p = \delta \lambda_c + \rho_\lambda \delta \nu_p - r_p,
\qquad
\delta n = -\delta \lambda_c + \rho_\lambda \delta \nu_n - r_n .
$$

In the implementation, the local residual increment is formed from the rolled
out primal step as

$$
\delta c = C_u \,\delta u + C_x \,\delta x .
$$

The recovered $\delta \lambda_c$ is then scattered back into the existing
equality-dual storage for `__eq_x` and `__eq_xu`, while
$\delta p,\delta n,\delta \nu_p,\delta \nu_n$ stay inside the restoration
runtime.

The implementation keeps a single shared helper for both presolve condensation
and rollout step recovery so that the sign convention cannot drift between the
two phases. The code now follows the same `\delta \nu`-first recovery order as
the paper, which avoids an extra cancellation-prone subtraction when recovering
the local dual steps.

### Hippo-Style Multiplier Regularization

Following Hippo and the implementation pattern in
[`ipm_constr.cpp`](/home/harper/Documents/moto/src/solver/ipm_impl/ipm_constr.cpp),
the local elastic block regularizes the explicit elastic bound rows in the
same `t + \rho \nu` style. The regularized local bound equations are

$$
\delta p - \delta \lambda_c - \rho_\lambda \delta \nu_p = -r_p,
\qquad
\delta n + \delta \lambda_c - \rho_\lambda \delta \nu_n = -r_n.
$$

This yields the same denominator pattern as Hippo's
$T_\rho = T + \rho N$. In our explicit elastic condensation it produces

$$
M_\rho = T_p (T_p + \rho_\lambda N_p)^{-1} + T_n (T_n + \rho_\lambda N_n)^{-1},
$$

so the effective denominators are of the form
$p + \rho_\lambda \nu_p$ and $n + \rho_\lambda \nu_n$. When
`lambda_reg == 0`, the implementation falls back to the unregularized explicit
elastic local-KKT formulas. This regularized branch keeps the condensed Hessian
modification bounded when $p,n$ or their duals approach the boundary.

## First-Order Correction Lifecycle

All restoration-specific first-order terms now follow the normal correction
buffer lifecycle:

- the explicit elastic condensed gradient correction
  $$
  \Delta g_R = C^T M_\rho^{-1} b_c
  $$
  is written into `lag_jac_corr_[__u]` and `lag_jac_corr_[__x]`
- the restoration proximal terms on `u` and `y` are written into
  `lag_jac_corr_[__u]` and `lag_jac_corr_[__y]`
- `activate_lag_jac_corr()` then activates all pending first-order corrections
  before the stage factorization

So restoration no longer special-cases its first-order condensed term by
writing directly into the active gradient outside the normal correction path.

## What Is Kept And What Is Removed

During restoration:

- the original user cost is still **evaluated**
- the original filter/objective bookkeeping is still based on the normal pair
  `(prim_res_l1, barrier objective)`
- the original cost directional derivative is still available for globalization
- the restoration-local elastic variables $p,n$ do **not** contribute to the
  outer barrier objective or `log_slack_sum`

But the original user cost does **not** drive the restoration search direction.
The restoration direction is driven by:

- the explicit elastic local-KKT condensation for `__eq_x` and `__eq_xu`
- the proximal anchoring on `u` and `y`
- the hard dynamics constraints

## Dynamic Activeness

The implementation does **not** rebuild the problem graph and does **not** use
`update_active_status()` to enter restoration.

Instead, restoration is controlled by solver-phase activeness:

- restoration runtime state is attached to each stage
- when `settings.in_restoration == false`, the restoration hooks are dormant
- when `settings.in_restoration == true`, the restoration hooks are active

At the same time, the original `__eq_x` and `__eq_xu`:

- continue to be evaluated so that `c(w)` and `C=c_w` are available
- continue to contribute to primal infeasibility and the outer filter
- but no longer enter the restoration phase as hard equalities

## Globalization Semantics

Restoration still uses the **outer problem's** filter semantics:

- the objective/filter pair remains the original `(prim_res_l1, barrier objective)`
- restoration does **not** add points to the outer filter history
- restoration line-search failure alone is not an immediate restoration failure

More concretely, the outer barrier objective is still

$$
\phi(x)=f(x)-\mu\sum_i \log s_i,
$$

where the sum only runs over the normal active `ipm_constr` inequalities. The
restoration-local quantities $p,n,\nu_p,\nu_n$ are stored outside that IPM
traversal, so they do not pollute:

- `kkt.log_slack_sum`
- `kkt.barrier_dir_deriv`
- `kkt.objective`
- the outer complementarity residual

This is why the original cost still has to be evaluated during restoration,
even though it no longer defines the search direction.

## Success / Failure

The restoration phase is considered successful when an accepted restoration step
produces sufficient improvement relative to the entry infeasibility:

$$
\|r_{\mathrm{prim}}(x)\|_\infty
<
\eta_R \,
\|r_{\mathrm{prim}}(x_{\mathrm{entry}})\|_\infty.
$$

It is classified as `infeasible_stationary` only when dual stationarity is
small but feasibility is not recovered.

Otherwise, if the restoration phase does not recover feasibility before the
global iteration budget is exhausted, the solver returns

$$
\mathrm{iter\_result}=\mathrm{restoration\_failed}
$$

or eventually `exceed_max_iter`.

## Iterative Refinement Status

After restoring the normal `lag_jac_corr_ -> activate_lag_jac_corr()`
lifecycle for the explicit elastic first-order term, the ordinary SQP
iterations again show normal iterative-refinement behavior: the stationarity
residuals are typically driven down to near machine precision within a few
correction sweeps.

The first restoration factorization can still show a very large *generic*
iterative-refinement residual. At the moment this should not be interpreted as
proof that the explicit elastic local-KKT recovery is wrong. The generic
`compute_kkt_residual()` checker is still derived from the standard stage-level
stationarity layout, while restoration is solving a reduced QP that already
contains the explicit local elastic Schur-complement terms.

So the current interpretation is:

- ordinary iterative refinement is healthy again after the correction-buffer
  lifecycle fix
- a large residual on the first restoration factorization indicates a mismatch
  between the generic residual checker and the restoration-condensed QP, not by
  itself a bug in the local elastic algebra

## Current Practical Status

The explicit local-KKT implementation has replaced the older condensed-only
prototype. Current behavior observed on `example/arm/run.py` is:

- restoration is triggered consistently for `--n-job 1` and `--n-job 4`
- the implementation is thread-consistent
- the solver still stalls on the terminal `arm_ee_constr` bottleneck
- current explicit restoration does not yet fully rescue the `arm` example

So the remaining limitation is no longer the old condensed analytic shortcut;
it is now the behavior of the explicit elastic restoration on difficult
terminal equalities.
